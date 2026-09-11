"""Quality-degradation check: compare the original and fake-quantized
checkpoints on held-out perplexity and a handful of greedy generations.

The held-out text is deliberately disjoint from poc/collect_acts.py's
calibration prompts, so this isn't just measuring calibration-set fit.

Usage:
    python poc/eval_quality.py \
        --original cache/models/Nemotron-3-Nano-4B-BF16 \
        --quantized cache/models/Nemotron-3-Nano-4B-BF16-quantized-poc
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

HELD_OUT_TEXT = """
The Byzantine fault tolerance problem asks how a distributed system can reach
agreement even when some of its components fail arbitrarily, including by
sending conflicting information to different parts of the system. Practical
Byzantine Fault Tolerance, introduced by Castro and Liskov, showed that this
could be done efficiently for state machine replication as long as fewer than
one third of the replicas are faulty.

def merge_sorted(a, b):
    result = []
    i = j = 0
    while i < len(a) and j < len(b):
        if a[i] <= b[j]:
            result.append(a[i]); i += 1
        else:
            result.append(b[j]); j += 1
    result.extend(a[i:])
    result.extend(b[j:])
    return result

A stack-based calculator reads tokens left to right, pushing numbers onto a
stack and, on encountering an operator, popping the top two values, applying
the operator, and pushing the result back. This directly implements reverse
Polish notation and avoids the need for a parser to handle operator
precedence or parentheses.
""".strip()

GENERATION_PROMPTS = [
    "def quicksort(arr):\n",
    "The capital of France is",
    "Step 1: To solve this equation, we first",
]


def perplexity(model, tokenizer, text: str, device: str) -> float:
    inputs = tokenizer(text, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model(**inputs, labels=inputs["input_ids"])
    return math.exp(out.loss.item())


def generate(model, tokenizer, prompt: str, device: str, max_new_tokens: int = 40) -> str:
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model.generate(
            **inputs, max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=tokenizer.pad_token_id
        )
    return tokenizer.decode(out[0], skip_special_tokens=True)


def evaluate_one(label: str, model_path: str, tokenizer, device: str, quick: bool) -> tuple[float | None, list[str]]:
    print(f"[{label}] loading {model_path} (device={device}) ...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.bfloat16, trust_remote_code=False)
    model = model.to(device)
    model.eval()

    ppl = None
    if not quick:
        print(f"[{label}] computing perplexity on held-out text ...", flush=True)
        ppl = perplexity(model, tokenizer, HELD_OUT_TEXT, device)
        print(f"[{label}] perplexity = {ppl:.3f}", flush=True)

    prompts = GENERATION_PROMPTS[:1] if quick else GENERATION_PROMPTS
    max_tokens = 10 if quick else 40
    gens = []
    for i, prompt in enumerate(prompts):
        print(f"[{label}] generating {i + 1}/{len(prompts)}: {prompt!r} ...", flush=True)
        gens.append(generate(model, tokenizer, prompt, device, max_tokens))
        print(f"[{label}]   -> {gens[-1]!r}", flush=True)

    del model
    return ppl, gens


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--original", required=True)
    parser.add_argument("--quantized", required=True)
    parser.add_argument("--device", default="cpu", choices=["cpu", "mps", "cuda"])
    parser.add_argument(
        "--quick", action="store_true", help="skip perplexity, only generate 10 tokens for 1 prompt per model"
    )
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.original, trust_remote_code=False)

    # The original (unquantized) checkpoint never changes between quantization
    # experiments, and greedy decoding is deterministic, so its baseline is
    # cached rather than recomputed on every comparison.
    baseline_cache = "baseline_original.json"
    if not args.quick and os.path.exists(baseline_cache):
        with open(baseline_cache) as f:
            cached = json.load(f)
        if cached.get("model_path") == args.original:
            print(f"[1/2] using cached original-model baseline from {baseline_cache}", flush=True)
            ppl_orig, gens_orig = cached["perplexity"], cached["generations"]
        else:
            print("[1/2] evaluating original model", flush=True)
            ppl_orig, gens_orig = evaluate_one("original", args.original, tokenizer, args.device, args.quick)
            with open(baseline_cache, "w") as f:
                json.dump({"model_path": args.original, "perplexity": ppl_orig, "generations": gens_orig}, f)
    else:
        print("[1/2] evaluating original model", flush=True)
        ppl_orig, gens_orig = evaluate_one("original", args.original, tokenizer, args.device, args.quick)
        if not args.quick:
            with open(baseline_cache, "w") as f:
                json.dump({"model_path": args.original, "perplexity": ppl_orig, "generations": gens_orig}, f)

    print("[2/2] evaluating quantized model", flush=True)
    ppl_quant, gens_quant = evaluate_one("quantized", args.quantized, tokenizer, args.device, args.quick)

    if args.quick:
        print("\n--quick mode: skipping perplexity ratio and report file")
        return

    print("\n=== Perplexity on held-out text ===")
    print(f"  original:  {ppl_orig:.3f}")
    print(f"  quantized: {ppl_quant:.3f}")
    print(f"  ratio:     {ppl_quant / ppl_orig:.3f}x")

    print("\n=== Greedy generations ===")
    for prompt, g_orig, g_quant in zip(GENERATION_PROMPTS, gens_orig, gens_quant):
        print(f"\n--- prompt: {prompt!r} ---")
        print(f"[original ] {g_orig!r}")
        print(f"[quantized] {g_quant!r}")

    quantized_name = os.path.basename(args.quantized.rstrip("/"))
    report_path = f"quality_report_{quantized_name}.md"
    with open(report_path, "w") as f:
        f.write(f"# Quality Degradation — {quantized_name}\n\n")
        f.write("## Perplexity (held-out text, disjoint from calibration)\n\n")
        f.write("| Model | Perplexity |\n|---|---|\n")
        f.write(f"| Original (BF16) | {ppl_orig:.3f} |\n")
        f.write(f"| Quantized | {ppl_quant:.3f} |\n")
        f.write(f"\nRatio: {ppl_quant / ppl_orig:.3f}x\n\n")
        f.write("## Greedy generations\n\n")
        for prompt, g_orig, g_quant in zip(GENERATION_PROMPTS, gens_orig, gens_quant):
            f.write(f"### Prompt: `{prompt}`\n\n")
            f.write(f"- **Original**: {g_orig!r}\n")
            f.write(f"- **Quantized**: {g_quant!r}\n\n")
    print(f"\nWrote poc/{report_path}")

    # Append-only experiment log — quality_report_*.md gets overwritten per
    # run if you reuse a name, but this history never does.
    log_path = "experiments_log.csv"
    log_exists = os.path.exists(log_path)
    with open(log_path, "a", newline="") as f:
        writer = csv.writer(f)
        if not log_exists:
            writer.writerow(["quantized_model", "original_perplexity", "quantized_perplexity", "ratio"])
        writer.writerow([quantized_name, f"{ppl_orig:.3f}", f"{ppl_quant:.3f}", f"{ppl_quant / ppl_orig:.3f}"])
    print(f"Appended to poc/{log_path}")


if __name__ == "__main__":
    main()
