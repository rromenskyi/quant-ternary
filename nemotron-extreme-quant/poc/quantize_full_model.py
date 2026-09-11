"""Apply the winning Stage-A method (rotated GPTQ-binary, group_size=128) to
every eligible Linear layer of the full model, using the calibration
activations from collect_acts_full.py.

This produces a *fake-quantized* checkpoint: every eligible weight is
replaced by its dequantized reconstruction (same shape/dtype as the
original), so it can be loaded and run with the normal model class to
measure real quality degradation, before any bit-packing/storage-format
work (roadmap.md Phase 3.2/4.2/18) is done.

Usage:
    python poc/quantize_full_model.py \
        --model cache/models/Nemotron-3-Nano-4B-BF16 \
        --calib-dir poc/calib_cache_full \
        --output cache/models/Nemotron-3-Nano-4B-BF16-quantized-poc
"""

from __future__ import annotations

import argparse
import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from collect_acts_full import collect_targets
from gptq import gptq_binary
from rotation import with_rotation

GROUP_SIZE = 128


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--calib-dir", default="poc/calib_cache_full")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    print(f"Loading {args.model} ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=False)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, trust_remote_code=False)
    model.eval()

    targets = collect_targets(model)
    print(f"{len(targets)} eligible linear layers")

    target_weight_names = {f"model.{name.split(' ')[0]}.weight" for name in targets}
    total_protected_weights = sum(
        p.numel() for n, p in model.named_parameters() if n not in target_weight_names
    )

    total_quantized_bits = 0.0
    total_quantized_weights = 0

    for i, (name, module) in enumerate(targets.items()):
        safe_name = name.split(" ")[0].replace(".", "_")
        calib_path = os.path.join(args.calib_dir, f"{safe_name}.pt")
        if not os.path.exists(calib_path):
            raise FileNotFoundError(f"missing calibration data for {name}: {calib_path}")
        data = torch.load(calib_path, weights_only=False)
        W = module.weight.detach().to(torch.float32)
        X = data["activations"]

        result = with_rotation(gptq_binary, W, X, GROUP_SIZE, group_size=GROUP_SIZE)
        module.weight.data.copy_(result["W_hat"].to(module.weight.dtype))

        n = W.numel()
        total_quantized_weights += n
        total_quantized_bits += result["bits_per_weight"] * n
        print(f"[{i + 1}/{len(targets)}] {name}: {tuple(W.shape)}  bpw={result['bits_per_weight']:.3f}")

    avg_quantized_bpw = total_quantized_bits / total_quantized_weights
    overall_bpw = (total_quantized_bits + total_protected_weights * 16) / (
        total_quantized_weights + total_protected_weights
    )
    print(f"\nQuantized {total_quantized_weights:,} weights at avg {avg_quantized_bpw:.3f} bpw")
    print(f"Protected (BF16) weights: {total_protected_weights:,}")
    print(f"Whole-model effective bpw: {overall_bpw:.3f}")

    os.makedirs(args.output, exist_ok=True)
    # Upstream checkpoint ships an inconsistent generation_config (top_p set without
    # do_sample=True), which fails save_pretrained's validation. Not our bug to fix —
    # just neutralize it so the quantized weights actually get persisted.
    model.generation_config.do_sample = True
    model.save_pretrained(args.output)
    tokenizer.save_pretrained(args.output)
    print(f"Saved fake-quantized checkpoint to {args.output}")


if __name__ == "__main__":
    main()
