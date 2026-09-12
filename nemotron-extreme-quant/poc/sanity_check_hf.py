"""Quick coherence sanity-check for a freshly-quantized HF checkpoint, meant to
run BEFORE the expensive mlx_lm.convert + upload + download steps -- so a
broken quantization is caught in seconds/minutes on the pod, not discovered
after a full round-trip. This is a direct response to getting burned by the
custom ternary+rotation+salient path: that one was only ever tested AFTER
full MLX packing, so its (severe) coherence break wasn't caught until much
later than it could have been.

Not a rigorous eval -- just a fast, cheap smoke test: generate a few short
completions and flag obvious degenerate-repetition breakdown (the failure
mode we actually hit) via a simple unique-token-ratio heuristic. A human
should still glance at the printed completions; this is a tripwire, not a
quality benchmark.

Usage:
    python poc/sanity_check_hf.py --model /root/lightning30b-gptq3bit-g64-src
"""

from __future__ import annotations

import argparse
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

PROMPTS = [
    "The capital of France is",
    "Write a one-sentence haiku about GPUs.",
    "2 + 2 =",
]

# Below this fraction of unique tokens in the generated continuation, flag as
# likely degenerate repetition -- the exact failure mode found in the custom
# ternary+rotation+salient models tonight (looping on near-identical text).
MIN_UNIQUE_TOKEN_RATIO = 0.35


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=80)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=False)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, trust_remote_code=False, device_map="cuda"
    )
    model.eval()

    any_degenerate = False
    for prompt in PROMPTS:
        messages = [{"role": "user", "content": prompt}]
        inputs = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt", return_dict=True
        ).to(model.device)
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=True,
                temperature=1.0,
                top_p=0.95,
                eos_token_id=model.generation_config.eos_token_id,
            )
        completion_ids = out[0, inputs["input_ids"].shape[1]:]
        text = tokenizer.decode(completion_ids, skip_special_tokens=True)

        n_tokens = completion_ids.shape[0]
        n_unique = len(set(completion_ids.tolist()))
        ratio = n_unique / max(n_tokens, 1)
        degenerate = ratio < MIN_UNIQUE_TOKEN_RATIO
        any_degenerate = any_degenerate or degenerate

        print(f"\n=== prompt: {prompt!r} ===")
        print(text)
        print(f"--- unique_token_ratio={ratio:.2f} ({n_unique}/{n_tokens}) {'DEGENERATE' if degenerate else 'ok'} ---", flush=True)

    if any_degenerate:
        print("\nSANITY_CHECK_FAILED: at least one completion looks degenerate (repetition loop) -- inspect above before proceeding to mlx_lm.convert.", flush=True)
        sys.exit(1)
    print("\nSANITY_CHECK_PASSED", flush=True)


if __name__ == "__main__":
    main()
