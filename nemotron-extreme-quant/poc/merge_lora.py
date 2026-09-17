"""Merge a PEFT LoRA adapter into its base HF checkpoint and save the result
as a plain (non-adapter) HF-format model -- meant to run BEFORE
gptq_stock_convert.py, so the calibration and quantization see the final
merged weights directly (no adapter-aware code needed downstream).

The adapter's own adapter_config.json records base_model_name_or_path, but
that's a pod-local path from whenever the adapter was trained and may not
exist on a fresh pod -- --base always overrides it explicitly.

Usage:
    python poc/merge_lora.py \
        --base /root/nemotron30b-bf16-src \
        --adapter roman220220/ipsupport-code-nemotron-lora \
        --output /root/nemotron30b-bf16-ipsupport-code-merged
"""

from __future__ import annotations

import argparse

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True, help="base HF checkpoint dir or repo id")
    parser.add_argument("--adapter", required=True, help="LoRA adapter HF repo id or local dir")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    print(f"--- loading base model: {args.base} ---")
    model = AutoModelForCausalLM.from_pretrained(
        args.base, dtype=torch.bfloat16, trust_remote_code=False, device_map="cuda"
    )
    tokenizer = AutoTokenizer.from_pretrained(args.base, trust_remote_code=False)

    print(f"--- loading + merging adapter: {args.adapter} ---")
    model = PeftModel.from_pretrained(model, args.adapter)
    model = model.merge_and_unload()

    print(f"--- saving merged model: {args.output} ---")
    model.save_pretrained(args.output)
    tokenizer.save_pretrained(args.output)
    print("MERGE_DONE")


if __name__ == "__main__":
    main()
