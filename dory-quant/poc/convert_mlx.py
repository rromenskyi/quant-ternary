"""Convert an assemble_checkpoint.py output to MLX at the per-component
bits Dory was calibrated at (quant_recipe.json).

Dory-specific wrinkle: `mlx_lm.models.dory.sanitize()` fuses the
checkpoint's separate q/k/v/g_proj into one qkvg_proj, and gate/up_proj
into one gate_up_proj (dory.py, ported from ml-explore/mlx-lm#1988) --
`mlx_lm.convert.convert()` runs that fuse automatically on load, so the
RTN baseline quantizes the already-fused tensor. gptq_dory.py calibrated
the *separate* components (one Hessian per component, same as every other
model here), so writing its exact codes back needs the fuse redone on the
saved (W, scale, bias, codes) tuples: concatenated along the output axis in
sanitize()'s order (q,k,v,g and gate,up), which is safe because MLX's
affine grouping is along the input axis -- each component's own groups,
scale and bias stay independent row blocks of the fused tensor.

Embeddings and lm_head: RTN only (no GPTQ codes -- not worth a Hessian,
same as every other model's --embed-bits/--head-bits). nGPT scale vectors
(sqk, s_gate, suv_gate, suv_up, attn_alpha, mlp_alpha, logit_scale) are not
nn.Linear and the predicate never sees them: bf16, unconditionally.

Needs the ipsupport-llc/mlx-lm fork (mlx_lm.models.dory).

    python convert_mlx.py --hf /workspace/dory-ongrid --out /workspace/Dory1.0-4B-gptq-mlx-jang
"""

from __future__ import annotations

import argparse
import glob
import importlib.util
import json
from pathlib import Path

import mlx.core as mx
import numpy as np
from mlx_lm.convert import convert

# Same-named sibling module (qwen35-quant/poc/convert_mlx.py): loaded under
# its own name to avoid shadowing/circularly importing this file.
_spec = importlib.util.spec_from_file_location(
    "_qwen35_convert_mlx", Path(__file__).resolve().parents[2] / "qwen35-quant" / "poc" / "convert_mlx.py"
)
_qwen35_convert_mlx = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_qwen35_convert_mlx)
pack = _qwen35_convert_mlx.pack  # MLX's affine bit-packing, architecture-agnostic

MLX_COMPONENTS = {
    "mixer.qkvg_proj": "attn", "mixer.o_proj": "attn",
    "mixer.gate_up_proj": "mlp", "mixer.down_proj": "mlp",
}
# Checkpoint component name -> (fused MLX name, concat order) or None (kept separate).
FUSE = {
    "mixer.q_proj": ("mixer.qkvg_proj", 0), "mixer.k_proj": ("mixer.qkvg_proj", 1),
    "mixer.v_proj": ("mixer.qkvg_proj", 2), "mixer.g_proj": ("mixer.qkvg_proj", 3),
    "mixer.gate_proj": ("mixer.gate_up_proj", 0), "mixer.up_proj": ("mixer.gate_up_proj", 1),
    "mixer.o_proj": (None, 0), "mixer.down_proj": (None, 0),
}


def bits_for(component: str, recipe: dict) -> int:
    if component not in recipe:
        raise SystemExit(f"quant_recipe.json has no bits for {component}")
    return recipe[component]


def gptq_tensors(work: Path, recipe: dict) -> dict[str, tuple]:
    """MLX module path -> (W_hat, scale, bias, codes, bits), fusing q/k/v/g
    and gate/up along the output axis into qkvg_proj / gate_up_proj (the
    same order mlx_lm.models.dory.sanitize() concatenates the float
    weights in), o_proj and down_proj passed through unchanged."""
    bits_of = recipe["components"]
    parts: dict[str, dict[int, tuple]] = {}
    passthrough: dict[str, tuple] = {}
    for f in sorted(glob.glob(str(work / "layers" / "[0-9]*.safetensors"))):
        t = mx.load(f)
        for key in t:
            if ".gptq_" in key or not key.endswith(".weight"):
                continue
            mod = key.split(".layers.", 1)[1].split(".", 1)[1][: -len(".weight")]  # "mixer.q_proj"
            layer = key.split(".layers.", 1)[1].split(".", 1)[0]
            bits = bits_for(bits_of[mod], recipe["recipe"])
            W, S, B, C = t[key], t[key + ".gptq_scales"], t[key + ".gptq_biases"], t[key + ".gptq_codes"]
            fused_name, order = FUSE[mod]
            prefix = f"backbone.layers.{layer}."
            if fused_name is None:
                passthrough[prefix + mod] = (W, S, B, C, bits)
            else:
                parts.setdefault(prefix + fused_name, {})[order] = (W, S, B, C, bits)
    out = dict(passthrough)
    for prefix, by_order in parts.items():
        n = len(by_order)
        if sorted(by_order) != list(range(n)):
            raise SystemExit(f"{prefix}: expected parts 0..{n - 1}, got {sorted(by_order)}")
        items = [by_order[i] for i in range(n)]
        bits = items[0][4]
        if any(it[4] != bits for it in items):
            raise SystemExit(f"{prefix}: fused components must share one bit width, got {[it[4] for it in items]}")
        out[prefix] = (
            mx.concatenate([it[0] for it in items], axis=0),
            mx.concatenate([it[1] for it in items], axis=0),
            mx.concatenate([it[2] for it in items], axis=0),
            mx.concatenate([it[3] for it in items], axis=0),
            bits,
        )
    return out


def _gpu():
    try:
        mx.eval(mx.zeros(1, stream=mx.gpu))
        return mx.gpu
    except Exception:
        return None


def write_gptq_codes(out: Path, work: Path, recipe: dict, group_size: int) -> None:
    gpu = _gpu()
    exact = gptq_tensors(work, recipe)
    changed = total = 0
    shards = {f.name: mx.load(str(f)) for f in sorted(out.glob("model*.safetensors"))}
    for t in shards.values():
        mx.eval(t)  # loaded lazily from the files rewritten below
    where = {k: name for name, t in shards.items() for k in t}
    touched = set()
    for prefix, (W, S, B, C, bits) in exact.items():
        if f"{prefix}.weight" not in where:
            raise SystemExit(f"{prefix}.weight not in the MLX model (fuse/name mismatch)")
        n = 2**bits - 1
        with mx.stream(gpu or mx.cpu):
            Wg = W.astype(mx.float32).reshape(*W.shape[:-1], -1, group_size)
            from_bf16 = mx.clip(mx.round((Wg - B[..., None]) / mx.where(S == 0, 1, S)[..., None]), 0, n)
            from_bf16 = from_bf16.reshape(W.shape).astype(mx.uint8)
            rederived_off = int((from_bf16 != C).sum().item())
            codes = np.array(C)
        packed = mx.array(pack(codes, bits))
        old = shards[where[f"{prefix}.weight"]][f"{prefix}.weight"]
        assert packed.shape == old.shape, (prefix, packed.shape, old.shape)
        for part, value in (("scales", S), ("biases", B)):
            have = shards[where[f"{prefix}.{part}"]][f"{prefix}.{part}"]
            assert have.shape == value.shape, (prefix, part, value.shape, have.shape)
        changed += int((packed != old).sum().item())
        total += old.size
        sdt = shards[where[f"{prefix}.scales"]][f"{prefix}.scales"].dtype
        for part, value in (("weight", packed), ("scales", S.astype(sdt)), ("biases", B.astype(sdt))):
            name = where[f"{prefix}.{part}"]
            shards[name][f"{prefix}.{part}"] = value
            touched.add(name)
    for name in sorted(touched):
        mx.save_safetensors(str(out / name), shards[name], metadata={"format": "mlx"})
    print(f"GPTQ codes written for {len(exact)} tensors; "
          f"{100 * changed / max(total, 1):.2f} % of the packed words differ from MLX's own re-quantization",
          flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--embed-bits", type=int, default=8)
    ap.add_argument("--head-bits", type=int, default=8)
    ap.add_argument("--gptq-work", help="gptq_dory.py's --work: write its exact codes")
    ap.add_argument("--device", default="cpu", help="cpu: a bf16 checkpoint bigger than the GPU converts in RAM")
    args = ap.parse_args()
    mx.set_default_device(mx.cpu if args.device == "cpu" else mx.gpu)
    hf, out = Path(args.hf).expanduser(), Path(args.out).expanduser()
    recipe = json.load(open(hf / "quant_recipe.json"))
    bits, gs = recipe["recipe"], recipe["group_size"]

    def predicate(path: str, module) -> bool | dict:
        if path.endswith("backbone.embeddings"):
            return {"group_size": gs, "bits": args.embed_bits}
        if path.endswith("lm_head"):
            return {"group_size": gs, "bits": args.head_bits}
        for suffix, component in MLX_COMPONENTS.items():
            if path.endswith("." + suffix):
                return {"group_size": gs, "bits": bits_for(component, bits)}
        return False  # nGPT scale vectors etc. aren't reached here (not nn.Linear/nn.Embedding)

    convert(str(hf), str(out), quantize=True, q_group_size=gs, q_bits=min(bits.values()), quant_predicate=predicate)
    if args.gptq_work:
        write_gptq_codes(out, Path(args.gptq_work), recipe, gs)
    (out / "quant_recipe.json").write_text(json.dumps(recipe, indent=2))
    print("CONVERT_DONE", flush=True)


if __name__ == "__main__":
    main()
