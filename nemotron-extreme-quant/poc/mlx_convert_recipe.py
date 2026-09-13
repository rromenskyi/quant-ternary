"""Custom mlx_lm converter for this project's mixed-precision recipes,
fixing a real naming mismatch that silently neutered EVERY mixed-precision
run so far tonight: stock mlx_lm.convert's own quant_predicate (and this
project's original mlx_convert_sensitivity.py) match paths against the
literal substrings "up_proj"/"down_proj" -- but NemotronH's MLX port names
the ROUTED (top-k-gated) experts' equivalent tensors `switch_mlp.fc1`
(up_proj-equivalent) / `switch_mlp.fc2` (down_proj-equivalent), not
`up_proj`/`down_proj`. Routed experts are ~99% of a MoE block's parameters
(128 experts vs 1 shared expert), so neither stock mlx_lm.convert nor the
original mlx_convert_sensitivity.py ever actually upgraded routed down_proj
to high_bits, regardless of recipe or layer selection -- confirmed by
inspecting real packed tensor shapes on HF (switch_mlp.fc2's packed width
matched 3-bit packing even for layers GPTQ had calibrated at 6-bit). See
docs/session_findings_2026-09-11.md section 7q.

Handles BOTH recipe modes so one verified-correct predicate covers either:
  --mode positional: replicates mlx_lm's own mixed_quant_predicate_builder
    layer-position formula exactly (mirrors gptq_stock_convert.py's
    recipe_bits_for with recipe_mode=positional), with the naming fix.
  --mode sensitivity: reads a sensitivity_manifest.json (as the original
    mlx_convert_sensitivity.py did), with the naming fix.

Usage:
    python poc/mlx_convert_recipe.py --hf-path ... --mlx-path ... \
        --group-size 64 --mode positional --recipe mixed_3_6

    python poc/mlx_convert_recipe.py --hf-path ... --mlx-path ... \
        --group-size 64 --mode sensitivity
"""

from __future__ import annotations

import argparse
import json

from mlx_lm.convert import convert

QUANT_RECIPES = {
    "mixed_2_6": {"low_bits": 2, "high_bits": 6},
    "mixed_3_4": {"low_bits": 3, "high_bits": 4},
    "mixed_3_6": {"low_bits": 3, "high_bits": 6},
    "mixed_4_6": {"low_bits": 4, "high_bits": 6},
}

# The actual fix: routed experts' down_proj-equivalent tensor is named
# switch_mlp.fc2 in this model's MLX port, not down_proj. up_proj is never
# a high_bits candidate in any recipe here, so switch_mlp.fc1 doesn't need
# an equivalent check, but is listed for clarity/future-proofing.
DOWN_PROJ_ALIASES = ("down_proj", "switch_mlp.fc2")
UP_PROJ_ALIASES = ("up_proj", "switch_mlp.fc1")


def _layer_index(path: str) -> int:
    for part in path.split("."):
        if part.isdigit():
            return int(part)
    return -1


def _is_down_proj(path: str) -> bool:
    return any(alias in path for alias in DOWN_PROJ_ALIASES)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hf-path", required=True)
    parser.add_argument("--mlx-path", required=True)
    parser.add_argument("--group-size", type=int, required=True)
    parser.add_argument("--mode", required=True, choices=["positional", "sensitivity"])
    parser.add_argument("--recipe", choices=list(QUANT_RECIPES), help="required for --mode positional")
    parser.add_argument(
        "--num-layers", type=int,
        help="required for --mode positional; must match gptq_stock_convert.py's model "
        "(len(config['layers_block_type']))",
    )
    args = parser.parse_args()

    if args.mode == "positional":
        if not args.recipe or not args.num_layers:
            raise SystemExit("--mode positional requires --recipe and --num-layers")
        low_bits, high_bits = QUANT_RECIPES[args.recipe]["low_bits"], QUANT_RECIPES[args.recipe]["high_bits"]
        num_layers = args.num_layers
        print(f"[INFO] positional mode: recipe={args.recipe} num_layers={num_layers}, with switch_mlp naming fix", flush=True)

        def quant_predicate(path: str, module) -> dict:
            if "lm_head" in path:
                return {"group_size": args.group_size, "bits": high_bits, "mode": "affine"}
            index = _layer_index(path)
            use_more_bits = (
                index < num_layers // 8
                or index >= 7 * num_layers // 8
                or (index - num_layers // 8) % 3 == 2
            )
            if use_more_bits and ("v_proj" in path or _is_down_proj(path)):
                return {"group_size": args.group_size, "bits": high_bits, "mode": "affine"}
            return {"group_size": args.group_size, "bits": low_bits, "mode": "affine"}

    else:
        manifest_path = f"{args.hf_path}/sensitivity_manifest.json"
        with open(manifest_path) as f:
            manifest = json.load(f)
        if manifest["group_size"] != args.group_size:
            raise SystemExit(
                f"{manifest_path} was written with group_size={manifest['group_size']}, "
                f"but --group-size={args.group_size} was passed -- these must match."
            )
        low_bits, high_bits = manifest["low_bits"], manifest["high_bits"]
        upgrade_set = {(i, name) for i, name in manifest["upgrades"]}
        print(
            f"[INFO] sensitivity mode: recipe={manifest['recipe']} "
            f"{len(upgrade_set)} (layer, proj) pairs upgraded to {high_bits}-bit, with switch_mlp naming fix",
            flush=True,
        )

        def quant_predicate(path: str, module) -> dict:
            if "lm_head" in path:
                return {"group_size": args.group_size, "bits": high_bits, "mode": "affine"}
            index = _layer_index(path)
            if "v_proj" in path and (index, "v_proj") in upgrade_set:
                return {"group_size": args.group_size, "bits": high_bits, "mode": "affine"}
            if _is_down_proj(path) and (index, "down_proj") in upgrade_set:
                return {"group_size": args.group_size, "bits": high_bits, "mode": "affine"}
            return {"group_size": args.group_size, "bits": low_bits, "mode": "affine"}

    convert(
        hf_path=args.hf_path,
        mlx_path=args.mlx_path,
        quantize=True,
        q_group_size=args.group_size,
        q_bits=low_bits,
        q_mode="affine",
        quant_predicate=quant_predicate,
    )


if __name__ == "__main__":
    main()
