"""Convert a GPTQ-calibrated HF checkpoint produced with
--quant-recipe-mode sensitivity (gptq_stock_convert.py) into MLX, using a
CUSTOM quant_predicate that matches the EXACT per-layer bit choices recorded
in the checkpoint's sensitivity_manifest.json.

Stock `mlx_lm.convert --quant-predicate <recipe>` can NOT be used here: its
builtin predicate (mixed_quant_predicate_builder) always assigns high_bits
by LAYER POSITION, which is exactly the guess sensitivity mode replaces with
real per-layer GPTQ-Hessian saliency. Using the stock positional predicate
here would re-quantize a different set of layers than GPTQ actually
calibrated at high_bits -- the same grid-mismatch failure mode that
motivated fixing the affine formula in the first place (see docs/
session_findings_2026-09-11.md section 7p), just at the recipe-selection
level instead of the per-value level.

Usage:
    python poc/mlx_convert_sensitivity.py \
        --hf-path /root/lightning30b-smart_3_6-g64-src \
        --mlx-path /root/lightning30b-smart_3_6-g64-mlx \
        --group-size 64
"""

from __future__ import annotations

import argparse
import json

from mlx_lm.convert import convert


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hf-path", required=True)
    parser.add_argument("--mlx-path", required=True)
    parser.add_argument("--group-size", type=int, required=True, help="must match the --group-size gptq_stock_convert.py used")
    args = parser.parse_args()

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
        f"[INFO] sensitivity_manifest.json: recipe={manifest['recipe']} "
        f"{len(upgrade_set)} (layer, proj) pairs upgraded to {high_bits}-bit, rest at {low_bits}-bit",
        flush=True,
    )

    def quant_predicate(path: str, module) -> dict:
        # Mirrors mlx_lm's own mixed_quant_predicate_builder's path parsing
        # (first purely-numeric dot-segment = layer index) -- see that
        # function in mlx_lm/convert.py for the reference this replaces.
        index = -1
        for part in path.split("."):
            if part.isdigit():
                index = int(part)
                break
        if "lm_head" in path:
            return {"group_size": args.group_size, "bits": high_bits, "mode": "affine"}
        for proj_name in ("v_proj", "down_proj"):
            if proj_name in path and (index, proj_name) in upgrade_set:
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
