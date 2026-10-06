"""Convert an assemble_checkpoint.py output to MLX with the SAME
per-component bits its decoder was calibrated at (quant_recipe.json), so
mlx_lm's affine quantization re-derives exactly the GPTQ codes.

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
re-derived min / max grid then differs (7.8 % of the codes of a 3-bit
tensor in the dry run).

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


def bits_for(component: str, recipe: dict) -> int:
    """A component's bits; "experts_down" etc. fall back to "experts"
    (recipes written before the split have only "experts")."""
    return recipe.get(component, recipe.get(component.split("_")[0]))


def pack(codes: np.ndarray, bits: int) -> np.ndarray:
    """MLX's affine packing: a little-endian bit stream of `bits`-bit codes
    in uint32 words ([..., n] -> [..., n * bits / 32])."""
    c = codes.astype(np.uint64).reshape(*codes.shape[:-1], -1, 32)
    words = np.zeros(c.shape[:-1] + (bits,), dtype=np.uint64)
    for j in range(32):
        w, off = divmod(j * bits, 32)
        words[..., w] |= (c[..., j] << np.uint64(off)) & np.uint64(0xFFFFFFFF)
        if off + bits > 32:
            words[..., w + 1] |= c[..., j] >> np.uint64(32 - off)
    return words.reshape(*codes.shape[:-1], -1).astype(np.uint32)


def gptq_tensors(work: Path, recipe: dict) -> dict[str, tuple]:
    """MLX key prefix -> (W_hat, scale, bias, bits) from gptq_qwen35.py's
    layer files; fused experts split into switch_mlp gate / up / down."""
    bits_of = recipe["components"]
    out = {}
    for f in sorted(glob.glob(str(work / "layers" / "*.safetensors"))):
        t = mx.load(f)
        for key in t:
            if key.endswith(("gptq_scales", "gptq_biases")):
                continue
            mod = key.split(".layers.", 1)[1].split(".", 1)[1]  # "<module>.weight" or "mlp.experts.gate_up_proj"
            mod = mod[: -len(".weight")] if mod.endswith(".weight") else mod
            bits = bits_for(bits_of[mod], recipe["recipe"])
            prefix = "language_model.model.layers." + key.split(".layers.", 1)[1].split(".", 1)[0]
            W, S, B = t[key], t[key + ".gptq_scales"], t[key + ".gptq_biases"]
            if mod == "mlp.experts.gate_up_proj":
                mid = W.shape[-2] // 2
                out[f"{prefix}.mlp.switch_mlp.gate_proj"] = (W[..., :mid, :], S[..., :mid, :], B[..., :mid, :], bits)
                out[f"{prefix}.mlp.switch_mlp.up_proj"] = (W[..., mid:, :], S[..., mid:, :], B[..., mid:, :], bits)
            elif mod == "mlp.experts.down_proj":
                out[f"{prefix}.mlp.switch_mlp.down_proj"] = (W, S, B, bits)
            else:
                out[f"{prefix}.{mod}"] = (W, S, B, bits)
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
    for shard in sorted(out.glob("model*.safetensors")):
        tensors = mx.load(str(shard))
        mx.eval(tensors)  # loaded lazily from the file this overwrites
        hit = False
        for prefix, (W, S, B, bits) in exact.items():
            if f"{prefix}.weight" not in tensors:
                continue
            n = 2**bits - 1
            # On the GPU when there is one: MLX's CPU backend on Linux took
            # an hour for a 35B MoE's codes.
            with mx.stream(GPU or mx.cpu):
                Wg = W.astype(mx.float32).reshape(*W.shape[:-1], -1, group_size)
                codes = mx.clip(mx.round((Wg - B[..., None]) / mx.where(S == 0, 1, S)[..., None]), 0, n)
                codes = np.array(codes.reshape(W.shape).astype(mx.uint8))
            packed = mx.array(pack(codes, bits))
            old = tensors[f"{prefix}.weight"]
            assert packed.shape == old.shape, (prefix, packed.shape, old.shape)
            changed += int((packed != old).sum().item())
            total += old.size
            sdt = tensors[f"{prefix}.scales"].dtype
            tensors[f"{prefix}.weight"] = packed
            tensors[f"{prefix}.scales"] = S.astype(sdt)
            tensors[f"{prefix}.biases"] = B.astype(sdt)
            done.add(prefix)
            hit = True
        if hit:
            mx.save_safetensors(str(shard), tensors, metadata={"format": "mlx"})
    missing = set(exact) - done
    if missing:
        raise SystemExit(f"GPTQ tensors not in the MLX model: {sorted(missing)[:5]}")
    print(f"GPTQ codes written for {len(done)} tensors; {100 * changed / max(total, 1):.2f} % of the packed words differ "
          "from MLX's own re-quantization", flush=True)


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
        for suffix, component in MLX_COMPONENTS.items():
            # Whole path components: "mlp.gate_proj" must not match "switch_mlp.gate_proj".
            if path == suffix or path.endswith("." + suffix):
                return {"group_size": gs, "bits": bits_for(component, bits)}
        raise ValueError(f"no bits for {path}: add it to the recipe or keep_float")

    convert(str(hf), str(out), quantize=True, q_group_size=gs, q_bits=min(bits.values()), quant_predicate=predicate)
    if args.gptq_work:
        write_gptq_codes(out, Path(args.gptq_work), recipe, gs)
    shutil.copy(hf / "quant_recipe.json", out / "quant_recipe.json")
    print("CONVERT_DONE", flush=True)


if __name__ == "__main__":
    main()
