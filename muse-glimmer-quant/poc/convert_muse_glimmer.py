"""Convert an assemble_checkpoint.py output (Meta's Muse Glimmer 30B,
`muse_glimmer`) to MLX at the per-component bits its decoder was calibrated
at (quant_recipe.json), via the ipsupport-llc/mlx-lm fork's upstream-synced
`muse_glimmer` support (text tower only -- the vision tower isn't ported;
HF's `vision_tower`/`vision_adapter`/`vision_projection` keys are dropped by
mlx_lm's own sanitize()).

Outside the calibrated decoder: embed_tokens (--embed-bits, RTN) and, if
the checkpoint doesn't tie embeddings, lm_head (--head-bits, RTN).

With --gptq-work (gptq_muse_glimmer.py's directory), the decoder's
quantized tensors are then rewritten with GPTQ's own codes, scales and
biases, same reasoning as qwen35-quant/docs/GPTQ_EXACT_CODES.md: GPTQ's
error feedback often leaves a group's extreme codes unused, and MLX's
re-derived min / max grid then differs from what was actually calibrated.

    python convert_muse_glimmer.py --hf /workspace/glimmer-ongrid --out /workspace/Muse-Glimmer-30B-gptq-mlx-jang
"""

from __future__ import annotations

import argparse
import glob
import json
import shutil
from pathlib import Path

import mlx.core as mx
import numpy as np
from mlx_lm.convert import convert

# MLX module path suffix -> recipe component.
MLX_COMPONENTS = {
    "self_attn.q_proj": "attn", "self_attn.k_proj": "attn", "self_attn.v_proj": "attn",
    "self_attn.o_proj": "attn", "self_attn.gate_proj": "attn",
    "mlp.gate_proj": "mlp", "mlp.up_proj": "mlp", "mlp.down_proj": "mlp",
}
# gptq_muse_glimmer.py writes keys under the HF checkpoint's own prefix.
HF_PREFIX = "model.language_model.layers."
MLX_PREFIX = "model.layers."


def pack(codes: np.ndarray, bits: int) -> np.ndarray:
    """MLX's affine packing: a little-endian bit stream of `bits`-bit codes
    in uint32 words ([..., n] -> [..., n * bits / 32])."""
    if codes.shape[-1] % 32:
        raise ValueError(f"packing needs a multiple of 32 codes per row, got {codes.shape[-1]}")
    c = codes.astype(np.uint64).reshape(*codes.shape[:-1], -1, 32)
    words = np.zeros(c.shape[:-1] + (bits,), dtype=np.uint64)
    for j in range(32):
        w, off = divmod(j * bits, 32)
        words[..., w] |= (c[..., j] << np.uint64(off)) & np.uint64(0xFFFFFFFF)
        if off + bits > 32:
            words[..., w + 1] |= c[..., j] >> np.uint64(32 - off)
    return words.reshape(*codes.shape[:-1], -1).astype(np.uint32)


def gptq_tensors(work: Path, recipe: dict) -> dict[str, tuple]:
    """MLX key prefix -> (W_hat, scale, bias, codes, bits) from
    gptq_muse_glimmer.py's layer files."""
    bits_of = recipe["components"]
    out = {}
    for f in sorted(glob.glob(str(work / "layers" / "[0-9]*.safetensors"))):
        t = mx.load(f)
        for key in t:
            if ".gptq_" in key or not key.startswith(HF_PREFIX):
                continue
            rest = key[len(HF_PREFIX):]  # "<i>.<module>.weight"
            i, mod = rest.split(".", 1)
            mod = mod[: -len(".weight")]
            bits = recipe["recipe"][bits_of[mod]]
            prefix = f"{MLX_PREFIX}{i}.{mod}"
            W, S, B = t[key], t[key + ".gptq_scales"], t[key + ".gptq_biases"]
            C = t.get(key + ".gptq_codes")
            out[prefix] = (W, S, B, C, bits)
    return out


def _gpu():
    try:
        mx.eval(mx.zeros(1, stream=mx.gpu))
        return mx.gpu
    except Exception:
        return None


GPU = None


def write_gptq_codes(out: Path, work: Path, recipe: dict, group_size: int) -> None:
    global GPU
    GPU = _gpu()
    exact = gptq_tensors(work, recipe)
    changed = total = 0
    done = set()
    shards = {f.name: mx.load(str(f)) for f in sorted(out.glob("model*.safetensors"))}
    for t in shards.values():
        mx.eval(t)  # loaded lazily from the files rewritten below
    where = {k: name for name, t in shards.items() for k in t}
    touched = set()
    saved = rederived_off = 0
    for prefix, (W, S, B, C, bits) in exact.items():
        if f"{prefix}.weight" not in where:
            continue
        n = 2**bits - 1
        with mx.stream(GPU or mx.cpu):
            Wg = W.astype(mx.float32).reshape(*W.shape[:-1], -1, group_size)
            from_bf16 = mx.clip(mx.round((Wg - B[..., None]) / mx.where(S == 0, 1, S)[..., None]), 0, n)
            from_bf16 = from_bf16.reshape(W.shape).astype(mx.uint8)
            if C is not None:  # GPTQ's own integer codes
                saved += 1
                rederived_off += int((from_bf16 != C).sum().item())
                codes = np.array(C)
            else:
                codes = np.array(from_bf16)
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
        done.add(prefix)
    for name in sorted(touched):
        mx.save_safetensors(str(out / name), shards[name], metadata={"format": "mlx"})
    missing = set(exact) - done
    if missing:
        raise SystemExit(f"GPTQ tensors not in the MLX model: {sorted(missing)[:5]}")
    print(f"GPTQ codes written for {len(done)} tensors ({saved} from saved integer codes); "
          f"{100 * changed / max(total, 1):.2f} % of the packed words differ from MLX's own re-quantization; "
          f"re-deriving the codes from the bf16 weights would have changed {rederived_off} codes", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--embed-bits", type=int, default=8)
    ap.add_argument("--head-bits", type=int, default=8)
    ap.add_argument("--gptq-work", help="gptq_muse_glimmer.py's --work: write its exact codes")
    ap.add_argument("--device", default="cpu", help="cpu: a bf16 checkpoint bigger than the GPU converts in RAM")
    args = ap.parse_args()
    mx.set_default_device(mx.cpu if args.device == "cpu" else mx.gpu)
    hf, out = Path(args.hf).expanduser(), Path(args.out).expanduser()
    recipe = json.load(open(hf / "quant_recipe.json"))
    bits, gs = recipe["recipe"], recipe["group_size"]

    def predicate(path: str, module) -> bool | dict:
        if path.endswith("embed_tokens"):
            return {"group_size": gs, "bits": args.embed_bits}
        if path.endswith("lm_head"):
            return {"group_size": gs, "bits": args.head_bits}
        for suffix, component in MLX_COMPONENTS.items():
            if path == suffix or path.endswith("." + suffix):
                return {"group_size": gs, "bits": bits[component]}
        raise ValueError(f"no bits for {path}: add it to the recipe")

    convert(str(hf), str(out), quantize=True, q_group_size=gs, q_bits=min(bits.values()), quant_predicate=predicate)
    if args.gptq_work:
        write_gptq_codes(out, Path(args.gptq_work), recipe, gs)
    shutil.copy(hf / "quant_recipe.json", out / "quant_recipe.json")
    print("CONVERT_DONE", flush=True)


if __name__ == "__main__":
    main()
