"""Convert a gptq_qwen35.py output to MLX with the SAME per-component bits
its decoder was calibrated at (quant_recipe.json), so mlx_lm's affine
quantization re-derives exactly the GPTQ codes.

Outside the calibrated decoder:
  - embed_tokens (tied to lm_head; not GPTQ-calibrated): --embed-bits, RTN;
  - linear_attn.in_proj_a / in_proj_b: bf16 (32-row gates);
  - vision_tower: --vision-bits (0 = bf16), RTN.

Needs the ipsupport-llc/mlx-lm fork's qwen3_5 vision support (branch
qwen3-5-vision), which keeps model.visual.* instead of dropping it.

    python convert_mlx.py --hf ~/models-work/frognano-jang-8-6-4 \
        --out ~/models-work/FrogNano-4B-2609-gptq-mlx-jang --embed-bits 8 --vision-bits 0
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from mlx_lm.convert import convert


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--embed-bits", type=int, default=8)
    ap.add_argument("--vision-bits", type=int, default=0, help="0 keeps the vision tower bf16")
    args = ap.parse_args()
    hf = Path(args.hf).expanduser()
    recipe = json.load(open(hf / "quant_recipe.json"))
    bits, gs = recipe["recipe"], recipe["group_size"]
    components = recipe["components"]  # "mlp.gate_proj" -> "mlp"
    keep_bf16 = tuple(recipe.get("bf16", []))

    def predicate(path: str, module) -> bool | dict:
        if path.startswith("vision_tower"):
            return {"group_size": gs, "bits": args.vision_bits} if args.vision_bits else False
        if path.endswith("embed_tokens"):
            return {"group_size": gs, "bits": args.embed_bits}
        if path.endswith(keep_bf16):
            return False
        for suffix, component in components.items():
            if path.endswith(suffix):
                return {"group_size": gs, "bits": bits[component]}
        raise ValueError(f"no bits for {path}: add it to the recipe")

    convert(str(hf), str(Path(args.out).expanduser()), quantize=True, q_group_size=gs,
            q_bits=bits["mlp"], quant_predicate=predicate)
    shutil.copy(hf / "quant_recipe.json", Path(args.out).expanduser() / "quant_recipe.json")


if __name__ == "__main__":
    main()
