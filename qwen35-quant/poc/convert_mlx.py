"""Convert an assemble_checkpoint.py output to MLX at the per-component
bits its decoder was calibrated at (quant_recipe.json).

Outside the calibrated decoder:
  - embed_tokens: --embed-bits, RTN; an untied lm_head: --head-bits, RTN;
  - quant_recipe.json's keep_float modules (delta-rule gates, MoE router,
    shared-expert gate): bf16;
  - vision_tower: --vision-bits (0 = bf16), RTN, except its position
    embedding (Qwen3-VL interpolates it in its weight's dtype, which
    quantized is uint32: the interpolation weights round to 0 / 1).

With --gptq-work (gptq_qwen35.py's directory), the decoder's quantized
tensors are then rewritten with GPTQ's own codes, scales and biases: GPTQ's
error feedback often leaves a group's extreme codes unused, and MLX's
re-derived min / max grid then differs (docs/GPTQ_EXACT_CODES.md). The
codes are the integer ones gptq_qwen35.py saved (gptq_codes); a work
directory from before that re-derives them from the bf16 weights.

Needs the ipsupport-llc/mlx-lm fork (qwen3_5 / qwen3_5_moe with vision).

    python convert_mlx.py --hf /workspace/ornith-ongrid --out /workspace/Ornith-1.5-35B-A3B-gptq-mlx-jang
"""

from __future__ import annotations

import argparse
import glob
import json
import re
import shutil
from pathlib import Path

import mlx.core as mx
import numpy as np
from mlx_lm.convert import convert

# MLX module path suffix -> recipe component (mlx_lm names the fused
# experts switch_mlp.{gate,up,down}_proj after sanitize).
MLX_COMPONENTS = {
    "self_attn.q_proj": "attn", "self_attn.k_proj": "attn", "self_attn.v_proj": "attn", "self_attn.o_proj": "attn",
    "linear_attn.in_proj_qkv": "linear", "linear_attn.in_proj_z": "linear", "linear_attn.out_proj": "linear",
    "mlp.gate_proj": "mlp", "mlp.up_proj": "mlp", "mlp.down_proj": "mlp",
    "mlp.shared_expert.gate_proj": "shared", "mlp.shared_expert.up_proj": "shared", "mlp.shared_expert.down_proj": "shared",
    "mlp.switch_mlp.gate_proj": "experts_gate_up", "mlp.switch_mlp.up_proj": "experts_gate_up",
    "mlp.switch_mlp.down_proj": "experts_down",
}


def bits_for(component: str, recipe: dict, required: bool = True) -> int | None:
    """A component's bits, else its shorter prefixes' ("experts_down" ->
    "experts", "mtp_experts_down" -> "mtp_experts" -> "mtp"), as in
    gptq_qwen35.py."""
    parts = component.split("_")
    for n in range(len(parts), 0, -1):
        if "_".join(parts[:n]) in recipe:
            return recipe["_".join(parts[:n])]
    if required:
        raise SystemExit(f"quant_recipe.json has no bits for {component}")
    return None


def mtp_component(path: str) -> str | None:
    """The recipe component of an MTP head module path, or None."""
    if ".mtp." not in "." + path:
        return None
    if path.endswith("mtp.fc"):
        return "mtp_fc"
    for suffix, component in MLX_COMPONENTS.items():
        if path.endswith("." + suffix):
            return "mtp_" + component
    return None


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
    """MLX key prefix -> (W_hat, scale, bias, codes or None, bits) from
    gptq_qwen35.py's layer files; fused experts split into switch_mlp gate /
    up / down."""
    bits_of = recipe["components"]
    out = {}
    mtp_file = work / "layers" / "mtp.safetensors"
    if mtp_file.exists():
        out.update(mtp_gptq_tensors(mx.load(str(mtp_file)), recipe))
    for f in sorted(glob.glob(str(work / "layers" / "[0-9]*.safetensors"))):
        t = mx.load(f)
        for key in t:
            if ".gptq_" in key:
                continue
            mod = key.split(".layers.", 1)[1].split(".", 1)[1]  # "<module>.weight" or "mlp.experts.gate_up_proj"
            mod = mod[: -len(".weight")] if mod.endswith(".weight") else mod
            bits = bits_for(bits_of[mod], recipe["recipe"])
            prefix = "language_model.model.layers." + key.split(".layers.", 1)[1].split(".", 1)[0]
            W, S, B = t[key], t[key + ".gptq_scales"], t[key + ".gptq_biases"]
            C = t.get(key + ".gptq_codes")
            if mod == "mlp.experts.gate_up_proj":
                mid = W.shape[-2] // 2
                for name, rows in (("gate_proj", slice(None, mid)), ("up_proj", slice(mid, None))):
                    out[f"{prefix}.mlp.switch_mlp.{name}"] = (
                        W[..., rows, :], S[..., rows, :], B[..., rows, :], None if C is None else C[..., rows, :], bits)
            elif mod == "mlp.experts.down_proj":
                out[f"{prefix}.mlp.switch_mlp.down_proj"] = (W, S, B, C, bits)
            else:
                out[f"{prefix}.{mod}"] = (W, S, B, C, bits)
    return out


def mtp_gptq_tensors(t: dict, recipe: dict) -> dict[str, tuple]:
    """The MTP head's entries for gptq_tensors: checkpoint names (per-expert
    tensors) to MLX module paths, the experts stacked like mlx_lm's sanitize."""
    out, experts = {}, {}
    for key in t:
        if ".gptq_" in key:
            continue
        m = re.match(r"mtp\.layers\.(\d+)\.mlp\.experts\.(\d+)\.(\w+_proj)\.weight$", key)
        parts = (t[key], t[key + ".gptq_scales"], t[key + ".gptq_biases"], t.get(key + ".gptq_codes"))
        if m:
            experts.setdefault((m[1], m[3]), {})[int(m[2])] = parts
            continue
        prefix = "language_model." + key[: -len(".weight")]
        out[prefix] = parts + (bits_for(mtp_component(prefix), recipe["recipe"]),)
    for (layer, proj), by_e in experts.items():
        if sorted(by_e) != list(range(len(by_e))):
            raise SystemExit(f"MTP layer {layer} {proj}: experts {sorted(by_e)[:3]}... aren't 0..{len(by_e) - 1}")
        prefix = f"language_model.mtp.layers.{layer}.mlp.switch_mlp.{proj}"
        stacked = [mx.stack([by_e[e][i] for e in range(len(by_e))]) if by_e[0][i] is not None else None
                   for i in range(4)]
        out[prefix] = tuple(stacked) + (bits_for(mtp_component(prefix), recipe["recipe"]),)
    return out


def split_mtp(out: Path) -> int:
    """Moves the MTP head's tensors into model-mtp.safetensors: a model
    already installed gets the head by downloading that one file."""
    index_path = out / "model.safetensors.index.json"
    index = json.load(open(index_path)) if index_path.exists() else {"metadata": {}, "weight_map": {}}
    head = {}
    for f in sorted(out.glob("model*.safetensors")):
        if f.name == "model-mtp.safetensors":
            continue
        t = mx.load(str(f))
        mx.eval(t)  # loaded lazily from the file rewritten below
        moved = {k: v for k, v in t.items() if k.startswith("language_model.mtp.")}
        if moved:
            head.update(moved)
            mx.save_safetensors(str(f), {k: v for k, v in t.items() if k not in moved}, metadata={"format": "mlx"})
    if head:
        mx.save_safetensors(str(out / "model-mtp.safetensors"), head, metadata={"format": "mlx"})
        index["weight_map"].update({k: "model-mtp.safetensors" for k in head})
        json.dump(index, open(index_path, "w"), indent=2)
    return len(head)


def drop_mtp(out: Path) -> None:
    """No MTP bits in the recipe: the head isn't shipped (as before)."""
    (out / "model-mtp.safetensors").unlink(missing_ok=True)
    for f in sorted(out.glob("model*.safetensors")):
        t = mx.load(str(f))
        if any(k.startswith("language_model.mtp.") for k in t):
            mx.eval(t)  # loaded lazily from the file rewritten below
            mx.save_safetensors(str(f), {k: v for k, v in t.items() if not k.startswith("language_model.mtp.")},
                                metadata={"format": "mlx"})
    index_path = out / "model.safetensors.index.json"
    if index_path.exists():
        index = json.load(open(index_path))
        index["weight_map"] = {k: v for k, v in index["weight_map"].items() if not k.startswith("language_model.mtp.")}
        json.dump(index, open(index_path, "w"), indent=2)


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
    # A tensor's weight, scales and biases can sit in different shards.
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
        # On the GPU when there is one: MLX's CPU backend on Linux took
        # an hour for a 35B MoE's codes.
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
    ap.add_argument("--vision-bits", type=int, default=8, help="0 keeps the vision tower bf16")
    ap.add_argument("--gptq-work", help="gptq_qwen35.py's --work: write its exact codes")
    ap.add_argument("--device", default="cpu", help="cpu: a bf16 checkpoint bigger than the GPU converts in RAM")
    args = ap.parse_args()
    mx.set_default_device(mx.cpu if args.device == "cpu" else mx.gpu)
    hf, out = Path(args.hf).expanduser(), Path(args.out).expanduser()
    recipe = json.load(open(hf / "quant_recipe.json"))
    bits, gs = recipe["recipe"], recipe["group_size"]
    floats = [re.compile(p) for p in recipe.get("keep_float", [])]

    def predicate(path: str, module) -> bool | dict:
        if path.startswith("vision_tower"):
            if not args.vision_bits or path.endswith("pos_embed"):
                return False
            return {"group_size": gs, "bits": args.vision_bits}
        if path.endswith("embed_tokens"):
            return {"group_size": gs, "bits": args.embed_bits}
        if path.endswith("lm_head"):
            return {"group_size": gs, "bits": args.head_bits}
        if any(p.search(path) for p in floats):
            return False
        if (component := mtp_component(path)) is not None:
            b = bits_for(component, bits, required=False)
            return {"group_size": gs, "bits": b} if b else False  # no bits: dropped below
        for suffix, component in MLX_COMPONENTS.items():
            # Whole path components: "mlp.gate_proj" must not match "switch_mlp.gate_proj".
            if path == suffix or path.endswith("." + suffix):
                return {"group_size": gs, "bits": bits_for(component, bits)}
        raise ValueError(f"no bits for {path}: add it to the recipe or keep_float")

    convert(str(hf), str(out), quantize=True, q_group_size=gs, q_bits=min(bits.values()), quant_predicate=predicate)
    if args.gptq_work:
        write_gptq_codes(out, Path(args.gptq_work), recipe, gs)
    if bits_for("mtp_fc", bits, required=False):
        print(f"MTP head: {split_mtp(out)} tensors in model-mtp.safetensors", flush=True)
    else:
        drop_mtp(out)
    shutil.copy(hf / "quant_recipe.json", out / "quant_recipe.json")
    print("CONVERT_DONE", flush=True)


if __name__ == "__main__":
    main()
