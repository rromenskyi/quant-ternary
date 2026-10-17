"""Convert an assemble_lfm2.py output to MLX at the per-component bits its
decoder was calibrated at (quant_recipe.json), then write GPTQ's exact
integer codes (qwen35-quant/poc/convert_mlx.py's write_gptq_codes;
docs/GPTQ_EXACT_CODES.md). The token embedding (tied: also the output
layer) is RTN at --embed-bits; the MoE router stays bf16.

    python convert_lfm2.py --hf /workspace/ongrid-lfm2-8b --out /workspace/LFM2.5-8B-A1B-gptq-mlx-jang \\
        --gptq-work /workspace/gptq-lfm2-8b
"""

from __future__ import annotations

import argparse
import glob
import json
import shutil
import sys
from pathlib import Path

import mlx.core as mx
from mlx_lm.convert import convert

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "qwen35-quant" / "poc"))
import convert_mlx  # noqa: E402

# MLX module path suffix -> recipe component. mlx-lm's lfm2_moe renames the
# feed-forward w1 / w3 / w2 to gate / up / down_proj (experts stacked into
# switch_mlp); lfm2 (dense) keeps w1 / w3 / w2.
MLX_COMPONENTS = {
    "self_attn.q_proj": "attn", "self_attn.k_proj": "attn", "self_attn.v_proj": "attn", "self_attn.out_proj": "attn",
    "conv.in_proj": "conv", "conv.out_proj": "conv",
    "feed_forward.w1": "mlp", "feed_forward.w3": "mlp", "feed_forward.w2": "mlp",
    "feed_forward.gate_proj": "mlp", "feed_forward.up_proj": "mlp", "feed_forward.down_proj": "mlp",
    "feed_forward.switch_mlp.gate_proj": "experts_gate_up", "feed_forward.switch_mlp.up_proj": "experts_gate_up",
    "feed_forward.switch_mlp.down_proj": "experts_down",
}
MOE_DENSE = {"feed_forward.w1": "feed_forward.gate_proj", "feed_forward.w3": "feed_forward.up_proj",
             "feed_forward.w2": "feed_forward.down_proj"}


def gptq_tensors(work: Path, recipe: dict, moe: bool, mlx_prefix: str) -> dict[str, tuple]:
    """MLX key prefix -> (W_hat, scale, bias, codes, bits) from the layer files."""
    out = {}
    for f in sorted(glob.glob(str(work / "layers" / "*.safetensors"))):
        t = mx.load(f)
        for key in t:
            if ".gptq_" in key:
                continue
            layer, mod = key.split(".layers.", 1)[1].split(".", 1)
            mod = mod.removesuffix(".weight")
            bits = convert_mlx.bits_for(recipe["components"][mod], recipe["recipe"])
            prefix = f"{mlx_prefix}.{layer}"
            W, S, B, C = t[key], t[key + ".gptq_scales"], t[key + ".gptq_biases"], t[key + ".gptq_codes"]
            if mod == "feed_forward.experts.gate_up_proj":
                mid = W.shape[-2] // 2
                for name, rows in (("gate_proj", slice(None, mid)), ("up_proj", slice(mid, None))):
                    out[f"{prefix}.feed_forward.switch_mlp.{name}"] = (
                        W[..., rows, :], S[..., rows, :], B[..., rows, :], C[..., rows, :], bits)
            elif mod == "feed_forward.experts.down_proj":
                out[f"{prefix}.feed_forward.switch_mlp.down_proj"] = (W, S, B, C, bits)
            else:
                out[f"{prefix}.{MOE_DENSE.get(mod, mod) if moe else mod}"] = (W, S, B, C, bits)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--gptq-work", required=True, help="gptq_lfm2.py's --work")
    ap.add_argument("--embed-bits", type=int, default=8)
    ap.add_argument("--mlx-prefix", default="model.layers",
                    help="MLX module path of the decoder layers (language_model.model.layers for lfm2_vl)")
    ap.add_argument("--vision-bits", type=int, default=8,
                    help="RTN bits of the vision tower and projector Linears (lfm2_vl)")
    args = ap.parse_args()
    mx.set_default_device(mx.cpu)
    hf, out = Path(args.hf), Path(args.out)
    recipe = json.load(open(hf / "quant_recipe.json"))
    bits, gs = recipe["recipe"], recipe["group_size"]
    moe = json.load(open(hf / "config.json"))["model_type"] == "lfm2_moe"

    def predicate(path: str, module) -> bool | dict:
        if path.endswith("embed_tokens"):
            return {"group_size": gs, "bits": args.embed_bits}
        if path.endswith("feed_forward.gate"):  # MoE router
            return False
        if path.startswith(("vision_tower.", "multi_modal_projector.")):
            # Linears whose input width is not a multiple of the group stay bf16.
            if module.weight.shape[-1] % gs:
                return False
            return {"group_size": gs, "bits": args.vision_bits}
        for suffix, component in MLX_COMPONENTS.items():
            if path.endswith("." + suffix):
                return {"group_size": gs, "bits": convert_mlx.bits_for(component, bits)}
        raise ValueError(f"no bits for {path}")

    convert(str(hf), str(out), quantize=True, q_group_size=gs, q_bits=min(bits.values()), quant_predicate=predicate)
    # write_gptq_codes reads the layer files through convert_mlx.gptq_tensors
    # (Qwen3.5 names): give it this model's.
    convert_mlx.gptq_tensors = lambda work, rec: gptq_tensors(work, rec, moe, args.mlx_prefix)
    convert_mlx.write_gptq_codes(out, Path(args.gptq_work), recipe, gs)
    shutil.copy(hf / "quant_recipe.json", out / "quant_recipe.json")
    print("CONVERT_DONE", flush=True)


if __name__ == "__main__":
    main()
