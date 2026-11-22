"""Convert an assemble_checkpoint.py output to MLX at the per-component bits
its decoder was calibrated at (quant_recipe.json), then, with --gptq-work,
write GPTQ's own codes, scales and biases (qwen35-quant/poc/convert_mlx.py's
write_gptq_codes; docs/GPTQ_EXACT_CODES.md there).

Outside the calibrated decoder:
  - embed_tokens: --embed-bits, lm_head: --head-bits, RTN;
  - the attention sinks, the dynamic convolutions' weight projections and
    the loop's message gates: bf16 / fp32, as calibrated (the router and
    the block attention queries are no linear layers);
  - vision_tower: --vision-bits (0 = bf16, the default, as mlx-lm's
    inkling2 quant_predicate has it).

Needs the ipsupport-llc/mlx-lm fork (mlx_lm.models.inkling2).

    python convert_mlx.py --hf /workspace/ongrid-mini --out /workspace/Inkling-2-Mini-gptq-mlx-jang \\
        --gptq-work /workspace/gptq-mini
"""

from __future__ import annotations

import argparse
import glob
import json
import re
import shutil
import sys
from pathlib import Path

import mlx.core as mx
from mlx_lm.convert import convert

import names

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "qwen35-quant" / "poc"))
import convert_mlx as q35_convert  # noqa: E402

KEEP_FLOAT = [r"\.sink_proj$", r"sconv\.weight_proj$", r"_message_gate\."]
COMPONENTS = {mlx: {"mlp.experts.gate_up_proj": "experts_gate_up", "mlp.experts.down_proj": "experts_down"}.get(mlx, "attn")
              for _, mlx in names.MODULES.values()}


def gptq_tensors(work: Path, recipe: dict) -> dict[str, tuple]:
    """MLX module path -> (W_hat, scale, bias, codes, bits) from
    gptq_inkling2.py's layer files, the experts' gate / up rows interleaved."""
    out = {}
    for f in sorted(glob.glob(str(work / "layers" / "[0-9]*.safetensors"))):
        t = mx.load(f)
        for key in t:
            if ".gptq_" in key:
                continue
            fused = key.endswith("gate_up_proj")
            prefix = names.mlx_prefix(key)
            bits = q35_convert.bits_for(COMPONENTS[prefix.split(".", 4)[4]], recipe["recipe"])
            parts = [t[key]] + [t.get(f"{key}.gptq_{p}") for p in ("scales", "biases", "codes")]
            out[prefix] = tuple(None if p is None else names.interleave(p, fused) for p in parts) + (bits,)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--embed-bits", type=int, default=8)
    ap.add_argument("--head-bits", type=int, default=8)
    ap.add_argument("--vision-bits", type=int, default=0, help="0 keeps the vision tower bf16")
    ap.add_argument("--gptq-work", help="gptq_inkling2.py's --work: write its exact codes")
    ap.add_argument("--device", default="cpu", help="cpu: a bf16 checkpoint bigger than the GPU converts in RAM")
    args = ap.parse_args()
    mx.set_default_device(mx.cpu if args.device == "cpu" else mx.gpu)
    hf, out = Path(args.hf).expanduser(), Path(args.out).expanduser()
    recipe = json.load(open(hf / "quant_recipe.json"))
    bits, gs = recipe["recipe"], recipe["group_size"]
    floats = [re.compile(p) for p in KEEP_FLOAT]

    def predicate(path: str, module) -> bool | dict:
        if path.startswith("model.vision_tower"):
            return {"group_size": gs, "bits": args.vision_bits} if args.vision_bits else False
        if path.endswith("embed_tokens"):
            return {"group_size": gs, "bits": args.embed_bits}
        if path == "lm_head":
            return {"group_size": gs, "bits": args.head_bits}
        if any(p.search(path) for p in floats):
            return False
        for suffix, component in COMPONENTS.items():
            if path.endswith("." + suffix):
                return {"group_size": gs, "bits": q35_convert.bits_for(component, bits)}
        raise ValueError(f"no bits for {path}: add it to the recipe or KEEP_FLOAT")

    convert(str(hf), str(out), quantize=True, q_group_size=gs, q_bits=min(bits.values()), quant_predicate=predicate)
    if args.gptq_work:
        q35_convert.gptq_tensors = gptq_tensors  # its writer, with this model's names
        q35_convert.write_gptq_codes(out, Path(args.gptq_work), recipe, gs)
    shutil.copy(hf / "quant_recipe.json", out / "quant_recipe.json")
    for d in ("tiktoken",):
        if (hf / d).is_dir():
            shutil.copytree(hf / d, out / d, dirs_exist_ok=True)
    print("CONVERT_DONE", flush=True)


if __name__ == "__main__":
    main()
