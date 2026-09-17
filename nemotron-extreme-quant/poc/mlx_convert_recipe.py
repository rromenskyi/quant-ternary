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

Handles THREE recipe modes so one verified-correct predicate covers any:
  --mode positional: replicates mlx_lm's own mixed_quant_predicate_builder
    layer-position formula exactly (mirrors gptq_stock_convert.py's
    recipe_bits_for with recipe_mode=positional), with the naming fix.
  --mode sensitivity: reads a sensitivity_manifest.json (as the original
    mlx_convert_sensitivity.py did), with the naming fix.
  --mode component: assigns bits purely by component TYPE (attention/
    mamba/moe_shared/moe_routed_up/moe_routed_down/lm_head/embeddings),
    uniform across every layer, no position or per-layer score involved --
    reverse-engineered from JANG_2L-CRACK's published bit allocation (see
    docs/session_findings_2026-09-11.md section 7q).

Usage:
    python poc/mlx_convert_recipe.py --hf-path ... --mlx-path ... \
        --group-size 64 --mode positional --recipe mixed_3_6

    python poc/mlx_convert_recipe.py --hf-path ... --mlx-path ... \
        --group-size 64 --mode sensitivity

    python poc/mlx_convert_recipe.py --hf-path ... --mlx-path ... \
        --group-size 64 --mode component --component-recipe jang
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

# Mirrors gptq_stock_convert.py's COMPONENT_BIT_RECIPES exactly (duplicated
# rather than imported to avoid pulling in that module's heavy torch/
# transformers imports just for this small dict -- same reasoning as
# QUANT_RECIPES's existing duplication above).
COMPONENT_BIT_RECIPES = {
    "jang": {
        "attention": 8,
        "mamba": 6,
        "moe_shared": 8,
        "moe_routed_up": 4,
        "moe_routed_down": 3,
        "lm_head": 8,
        "embeddings": 6,
        # The MTP head (~4% of total size) only affects self-speculative
        # decoding's *accept rate* -- a low-bit draft head just gets rejected
        # more often, never produces a wrong final token (verified against
        # plain greedy decoding). But a badly-degraded draft head defeats the
        # entire point of doing this (speedup), so it gets its OWN, higher
        # bit-width tier instead of reusing moe_routed_up/down's aggressive
        # 4/3-bit -- deliberately not GPTQ-calibrated like the rest (see
        # inject_mtp_weights.py), so a bit of headroom here is cheap
        # insurance against the extra RTN quantization error.
        "mtp_attention": 8,
        "mtp_moe_shared": 8,
        "mtp_moe_routed_up": 6,
        "mtp_moe_routed_down": 6,
        "mtp_fusion": 8,
    },
    # For dense NemotronH variants with no MoE at all (e.g. Nemotron-3-Nano-4B).
    # See gptq_stock_convert.py's COMPONENT_BIT_RECIPES for the full rationale.
    "jang-dense": {
        "attention": 8,
        "mamba": 6,
        "mlp": 3,
        "lm_head": 8,
        "embeddings": 6,
    },
    "jang-dense-mlp6": {
        "attention": 8,
        "mamba": 6,
        "mlp": 6,
        "lm_head": 8,
        "embeddings": 6,
    },
    "jang-dense-mlp8": {
        "attention": 8,
        "mamba": 6,
        "mlp": 8,
        "lm_head": 8,
        "embeddings": 6,
    },
    "dense-8bit": {
        "attention": 8,
        "mamba": 8,
        "mlp": 8,
        "lm_head": 8,
        "embeddings": 8,
    },
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


def make_component_quant_predicate(cbits: dict, group_size: int):
    """Build the --mode component quant_predicate for a given component-bits
    dict. Exported (not just a main() closure) so inject_mtp_weights.py can
    apply the SAME named recipe (e.g. "jang") to a checkpoint's mtp.* head --
    every mtp.* path already matches one of these checks (mtp.layers.0 is an
    attention block, mtp.layers.1 an MoE block, same submodule names as the
    backbone) except eh_proj, the MTP-only embed/hidden fusion projection.
    """

    def quant_predicate(path: str, module) -> dict | bool:
        for alias in ("shared_experts.up_proj", "shared_experts.down_proj"):
            if alias in path:
                return {"group_size": group_size, "bits": cbits["moe_shared"], "mode": "affine"}
        if "switch_mlp.fc1" in path:
            return {"group_size": group_size, "bits": cbits["moe_routed_up"], "mode": "affine"}
        if "switch_mlp.fc2" in path:
            return {"group_size": group_size, "bits": cbits["moe_routed_down"], "mode": "affine"}
        if any(p in path for p in ("q_proj", "k_proj", "v_proj", "o_proj")):
            return {"group_size": group_size, "bits": cbits["attention"], "mode": "affine"}
        if "in_proj" in path or "out_proj" in path:
            return {"group_size": group_size, "bits": cbits["mamba"], "mode": "affine"}
        # Dense (non-MoE) MLP block, e.g. Nemotron-3-Nano-4B's "mlp" blocks --
        # only reached here because the shared_experts/switch_mlp checks above
        # (which require a more specific path prefix) already handled the MoE
        # case, so a bare up_proj/down_proj at this point is unambiguous.
        if ("up_proj" in path or "down_proj" in path) and "mlp" in cbits:
            return {"group_size": group_size, "bits": cbits["mlp"], "mode": "affine"}
        if "lm_head" in path:
            return {"group_size": group_size, "bits": cbits["lm_head"], "mode": "affine"}
        if "embeddings" in path:
            return {"group_size": group_size, "bits": cbits["embeddings"], "mode": "affine"}
        # MTP-only: fuses the drafted token's embedding with the backbone's
        # hidden state (mtp.layers.0.eh_proj). No backbone equivalent --
        # treated at the same precision as the attention block it feeds.
        if "eh_proj" in path:
            return {
                "group_size": group_size,
                "bits": cbits.get("mtp_fusion", cbits["attention"]),
                "mode": "affine",
            }
        print(f"[WARN] unrecognized quantizable path {path!r} under component mode, leaving unquantized", flush=True)
        return False

    return quant_predicate


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hf-path", required=True)
    parser.add_argument("--mlx-path", required=True)
    parser.add_argument("--group-size", type=int, required=True)
    parser.add_argument("--mode", required=True, choices=["positional", "sensitivity", "component"])
    parser.add_argument("--recipe", choices=list(QUANT_RECIPES), help="required for --mode positional")
    parser.add_argument(
        "--num-layers", type=int,
        help="required for --mode positional; must match gptq_stock_convert.py's model "
        "(len(config['layers_block_type']))",
    )
    parser.add_argument(
        "--component-recipe", choices=list(COMPONENT_BIT_RECIPES), help="required for --mode component",
    )
    args = parser.parse_args()

    if args.mode == "component":
        if not args.component_recipe:
            raise SystemExit("--mode component requires --component-recipe")
        cbits = COMPONENT_BIT_RECIPES[args.component_recipe]
        print(f"[INFO] component mode: component_recipe={args.component_recipe} {cbits}", flush=True)
        quant_predicate = make_component_quant_predicate(cbits, args.group_size)

        # convert() requires q_bits even though our predicate always overrides
        # it per-path -- any valid bit-width works (unused), so just take
        # the smallest one in THIS recipe rather than a hardcoded key that
        # only exists in the "jang" (MoE) recipe's schema.
        low_bits = min(cbits.values())

    elif args.mode == "positional":
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
