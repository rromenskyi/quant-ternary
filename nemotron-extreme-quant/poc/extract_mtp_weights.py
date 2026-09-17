"""Extract mtp.* tensors (Multi-Token-Prediction head) from a NemotronH bf16
source checkpoint into a standalone side file, BEFORE the checkpoint goes
through any PyTorch/transformers-based step (LoRA merge, GPTQ) -- HF's
NemotronHForCausalLM has `_keys_to_ignore_on_load_unexpected = [r"mtp.*"]`,
so any transformers.AutoModelForCausalLM.from_pretrained() call silently
drops these weights on load, regardless of which mlx-lm version runs
afterward. inject_mtp_weights.py re-attaches this side file's contents
into the finished MLX model once the main pipeline (merge -> GPTQ ->
mlx_lm.convert) is done.

Reads only the shard(s) that actually contain mtp.* keys (identified via
the index), not the whole checkpoint.

Usage:
    python poc/extract_mtp_weights.py \
        --source /root/nemotron30b-bf16-src --output /root/mtp_head
"""

from __future__ import annotations

import argparse
import json
import os

import mlx.core as mx


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    index_path = os.path.join(args.source, "model.safetensors.index.json")
    if os.path.exists(index_path):
        with open(index_path) as f:
            weight_map = json.load(f)["weight_map"]
        mtp_keys = [k for k in weight_map if k.startswith("mtp.")]
        shard_files = sorted(set(weight_map[k] for k in mtp_keys))
    else:
        # Single-file checkpoint, no index.
        shard_files = ["model.safetensors"]
        mtp_keys = None  # filter after loading

    tensors: dict[str, mx.array] = {}
    for shard in shard_files:
        shard_weights = mx.load(os.path.join(args.source, shard))
        for k, v in shard_weights.items():
            if k.startswith("mtp.") and (mtp_keys is None or k in mtp_keys):
                tensors[k] = v

    if not tensors:
        raise SystemExit(f"No mtp.* tensors found under {args.source}")

    mx.save_safetensors(os.path.join(args.output, "mtp_weights.safetensors"), tensors)

    with open(os.path.join(args.source, "config.json")) as f:
        src_config = json.load(f)
    mtp_config = {
        k: src_config[k]
        for k in ("num_nextn_predict_layers", "mtp_layers_block_type", "mtp_hybrid_override_pattern")
        if k in src_config
    }
    with open(os.path.join(args.output, "mtp_config.json"), "w") as f:
        json.dump(mtp_config, f, indent=2)

    print(f"extracted {len(tensors)} mtp.* tensors from {len(shard_files)} shard(s) -> {args.output}")
    print("EXTRACT_MTP_DONE")


if __name__ == "__main__":
    main()
