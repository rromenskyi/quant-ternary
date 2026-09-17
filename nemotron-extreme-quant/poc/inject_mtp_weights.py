"""Re-attach a previously-extracted MTP head (see extract_mtp_weights.py)
into a finished MLX model directory. The main pipeline (LoRA merge -> GPTQ
-> mlx_lm.convert) never sees mtp.* weights at all -- HF transformers'
NemotronHForCausalLM drops them on load (_keys_to_ignore_on_load_unexpected
= [r"mtp.*"]) before either PyTorch step ever runs. This step uses mlx_lm's
own model classes directly (no transformers involved) to quantize the head
and splice it into the already-converted model in place.

Quantization: either a uniform --bits, or --component-recipe to reuse the
SAME named recipe (e.g. "jang") the rest of the checkpoint used, via
mlx_convert_recipe.py's predicate builder (mtp.* paths already match nearly
every check there -- same submodule names as the backbone -- except eh_proj,
which the predicate handles directly). Recipes carry dedicated mtp_* bit
tiers rather than reusing moe_routed_up/down's aggressive 3-4 bit: the head
is only ~4% of total size and unquantized weights, and isn't GPTQ-calibrated
here (plain RTN via quantize_model), so it gets some headroom instead of the
backbone's most aggressive tier.

Usage:
    python poc/inject_mtp_weights.py \
        --mlx-model /root/lightning30b-RUN-mlx \
        --mtp-weights /root/mtp_head/mtp_weights.safetensors \
        --mtp-config /root/mtp_head/mtp_config.json \
        --group-size 64 --component-recipe jang
    # or a flat bit-width instead of a component recipe:
    python poc/inject_mtp_weights.py ... --group-size 64 --bits 8
"""

from __future__ import annotations

import argparse
import json
import os

import mlx.core as mx
from mlx.utils import tree_flatten

from mlx_convert_recipe import COMPONENT_BIT_RECIPES, make_component_quant_predicate
from mlx_lm.models.nemotron_h import Model, ModelArgs
from mlx_lm.utils import quantize_model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mlx-model", required=True)
    parser.add_argument("--mtp-weights", required=True)
    parser.add_argument("--mtp-config", required=True)
    parser.add_argument("--group-size", type=int, required=True)
    parser.add_argument("--bits", type=int, default=None, help="uniform bit-width; overridden by --component-recipe")
    parser.add_argument("--component-recipe", default=None, choices=list(COMPONENT_BIT_RECIPES))
    args = parser.parse_args()
    if args.component_recipe is None and args.bits is None:
        raise SystemExit("Either --bits or --component-recipe is required.")

    config_path = os.path.join(args.mlx_model, "config.json")
    with open(config_path) as f:
        config = json.load(f)
    with open(args.mtp_config) as f:
        mtp_config = json.load(f)

    model_args = ModelArgs.from_dict({**config, **mtp_config})
    model = Model(model_args)
    if model.mtp is None:
        raise SystemExit(
            "model.mtp is None after applying mtp_config -- check "
            "num_nextn_predict_layers/mtp_layers_block_type"
        )

    mtp_weights = mx.load(args.mtp_weights)
    sanitized = model.sanitize(dict(mtp_weights))
    mtp_only = {k[len("mtp.") :]: v for k, v in sanitized.items() if k.startswith("mtp.")}
    model.mtp.load_weights(list(mtp_only.items()), strict=True)

    quant_predicate = None
    bits_for_convert = args.bits
    if args.component_recipe is not None:
        cbits = COMPONENT_BIT_RECIPES[args.component_recipe]
        # Remap this recipe's mtp_* tier onto the plain component-name keys
        # make_component_quant_predicate expects -- same path-matching logic
        # as the backbone, different (higher) bit values.
        mtp_cbits = {
            "attention": cbits["mtp_attention"],
            "mamba": cbits["mtp_attention"],  # unreached: mtp never has mamba layers
            "moe_shared": cbits["mtp_moe_shared"],
            "moe_routed_up": cbits["mtp_moe_routed_up"],
            "moe_routed_down": cbits["mtp_moe_routed_down"],
            "mtp_fusion": cbits.get("mtp_fusion", cbits["mtp_attention"]),
        }
        quant_predicate = make_component_quant_predicate(mtp_cbits, args.group_size)
        bits_for_convert = min(mtp_cbits.values())  # unused (predicate overrides per-path), quantize_model still wants a value

    # Reuse mlx_lm's own quantize_model (not a bare nn.quantize) -- it skips
    # any weight whose last dim isn't divisible by group_size instead of
    # crashing, matching exactly how mlx_lm.convert quantized the rest of
    # this checkpoint.
    quantize_model(
        model.mtp, {}, group_size=args.group_size, bits=bits_for_convert, quant_predicate=quant_predicate
    )

    quantized_flat = dict(tree_flatten(model.mtp.parameters()))
    final_weights = {f"mtp.{k}": v for k, v in quantized_flat.items()}

    index_path = os.path.join(args.mlx_model, "model.safetensors.index.json")
    with open(index_path) as f:
        index = json.load(f)
    mtp_shard_name = "model-mtp.safetensors"
    mx.save_safetensors(os.path.join(args.mlx_model, mtp_shard_name), final_weights)
    for k, v in final_weights.items():
        index["weight_map"][k] = mtp_shard_name
    if "metadata" in index and "total_size" in index["metadata"]:
        index["metadata"]["total_size"] += sum(v.nbytes for v in final_weights.values())
    with open(index_path, "w") as f:
        json.dump(index, f, indent=2)

    config.update(mtp_config)
    quant = config.setdefault(
        "quantization", {"group_size": args.group_size, "bits": args.bits or 4, "mode": "affine"}
    )
    for name, module in model.mtp.named_modules():
        if hasattr(module, "bits"):
            quant[f"mtp.{name}"] = {
                "group_size": module.group_size,
                "bits": module.bits,
                "mode": "affine",
            }
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    print(f"injected {len(final_weights)} mtp.* tensors into {args.mlx_model}")
    print("INJECT_MTP_DONE")


if __name__ == "__main__":
    main()
