"""Re-attach a previously-extracted MTP head (see extract_mtp_weights.py)
into a finished MLX model directory. The main pipeline (LoRA merge -> GPTQ
-> mlx_lm.convert) never sees mtp.* weights at all -- HF transformers'
NemotronHForCausalLM drops them on load (_keys_to_ignore_on_load_unexpected
= [r"mtp.*"]) before either PyTorch step ever runs. This step uses mlx_lm's
own model classes directly (no transformers involved) to quantize the head
and splice it into the already-converted model in place.

Usage:
    python poc/inject_mtp_weights.py \
        --mlx-model /root/lightning30b-RUN-mlx \
        --mtp-weights /root/mtp_head/mtp_weights.safetensors \
        --mtp-config /root/mtp_head/mtp_config.json \
        --bits 4 --group-size 64
"""

from __future__ import annotations

import argparse
import json
import os

import mlx.core as mx
from mlx.utils import tree_flatten

from mlx_lm.models.nemotron_h import Model, ModelArgs
from mlx_lm.utils import quantize_model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mlx-model", required=True)
    parser.add_argument("--mtp-weights", required=True)
    parser.add_argument("--mtp-config", required=True)
    parser.add_argument("--bits", type=int, default=4)
    parser.add_argument("--group-size", type=int, default=64)
    args = parser.parse_args()

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

    # Reuse mlx_lm's own quantize_model (not a bare nn.quantize) -- it skips
    # any weight whose last dim isn't divisible by group_size instead of
    # crashing, matching exactly how mlx_lm.convert quantized the rest of
    # this checkpoint.
    quantize_model(model.mtp, {}, group_size=args.group_size, bits=args.bits)

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
        "quantization", {"group_size": args.group_size, "bits": args.bits, "mode": "affine"}
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
