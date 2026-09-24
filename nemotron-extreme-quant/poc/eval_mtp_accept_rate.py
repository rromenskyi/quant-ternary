"""Measures Nemotron-H MTP self-speculative decoding accept rate and
wall-clock generation speed for a given MLX model, using
mlx_lm.generate.nemotron_h_mtp_generate_step directly.

Accept rate never affects correctness -- a rejected draft is always
replaced by the backbone's own top-1 pick (see docs/FINDINGS.md section
4.2, bit-exact-verified) -- it only affects how much of the speedup
self-speculative decoding actually delivers. This script exists to compare
that speedup across MTP head quantization variants (RTN-affine vs.
GPTQ-affine vs. RTN-nvfp4) on real held-out text, not the synthetic/dry-run
checks gptq_mtp.py's own commit was validated against.

nemotron_h_mtp_generate_step's yield pattern (token, logprobs, from_draft)
is NOT one yield per speculative attempt: an accepted draft yields TWO
tokens (the draft itself, from_draft=True, immediately followed by a free
"bonus" token, from_draft=False) while a rejected draft yields ONE
(the backbone's own pick, from_draft=False) -- see generate.py's own
nemotron_h_mtp_generate_step for the source of this pattern. Reconstructing
attempts/accepted from the yielded stream means tracking that alternation
explicitly (skip counting the bonus token as a new attempt), not just
tallying from_draft=True/False.

Usage:
    python poc/eval_mtp_accept_rate.py --model /root/lightning30b-RUN-mlx \
        --text /root/llama.cpp/wikitext-2-raw/wiki.test.raw \
        --prompts 8 --prompt-tokens 64 --gen-tokens 200
"""

from __future__ import annotations

import argparse
import time

import mlx.core as mx
from mlx_lm.generate import nemotron_h_mtp_generate_step
from mlx_lm.utils import load


def run_one(model, prompt_ids: list[int], gen_tokens: int) -> tuple[int, int, int, float]:
    gen = nemotron_h_mtp_generate_step(mx.array(prompt_ids), model, max_tokens=gen_tokens)
    attempts = accepted = total_tokens = 0
    expecting_bonus = False
    first = True
    start = time.time()
    for _tok, _logprobs, from_draft in gen:
        total_tokens += 1
        if first:
            # The very first yield is the backbone's own next-token pick
            # before the speculative loop starts at all -- not an attempt
            # outcome.
            first = False
            continue
        if expecting_bonus:
            expecting_bonus = False
            continue
        attempts += 1
        if from_draft:
            accepted += 1
            expecting_bonus = True
    elapsed = time.time() - start
    return attempts, accepted, total_tokens, elapsed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--text", required=True, help="held-out text file, e.g. wikitext-2-raw's wiki.test.raw")
    parser.add_argument("--prompts", type=int, default=8)
    parser.add_argument("--prompt-tokens", type=int, default=64)
    parser.add_argument("--gen-tokens", type=int, default=200)
    parser.add_argument("--trust-remote-code", action="store_true")
    args = parser.parse_args()

    model, tokenizer = load(args.model, trust_remote_code=args.trust_remote_code)
    if not hasattr(model, "mtp") or model.mtp is None:
        raise SystemExit(f"{args.model}: model.mtp is None -- no MTP head to evaluate")

    text = open(args.text, "r", encoding="utf-8", errors="ignore").read()
    ids_all = tokenizer.encode(text)
    print(f"total tokens in corpus: {len(ids_all)}", flush=True)

    # Quarter-in start offset, same convention as ppl_mlx.py -- a held-out-
    # ish feel without a separate train/test split file. Non-overlapping
    # prompt+gen windows so no two prompts share generated continuation.
    start_offset = len(ids_all) // 4
    window = args.prompt_tokens + args.gen_tokens

    total_attempts = total_accepted = total_tokens_all = 0
    total_elapsed = 0.0
    for i in range(args.prompts):
        s = start_offset + i * window
        e = s + args.prompt_tokens
        if e > len(ids_all):
            break
        prompt_ids = ids_all[s:e]
        attempts, accepted, total_tokens, elapsed = run_one(model, prompt_ids, args.gen_tokens)
        rate = accepted / attempts if attempts else 0.0
        tok_s = total_tokens / elapsed if elapsed > 0 else 0.0
        print(
            f"[{i + 1}/{args.prompts}] attempts={attempts} accepted={accepted} "
            f"accept_rate={rate:.3f} tokens={total_tokens} tok/s={tok_s:.1f}",
            flush=True,
        )
        total_attempts += attempts
        total_accepted += accepted
        total_tokens_all += total_tokens
        total_elapsed += elapsed

    overall_rate = total_accepted / total_attempts if total_attempts else 0.0
    overall_tok_s = total_tokens_all / total_elapsed if total_elapsed > 0 else 0.0
    print(
        f"\nOverall: attempts={total_attempts} accepted={total_accepted} "
        f"accept_rate={overall_rate:.4f} total_tokens={total_tokens_all} "
        f"avg_tok/s={overall_tok_s:.1f}"
    )
    print("EVAL_MTP_ACCEPT_RATE_DONE")


if __name__ == "__main__":
    main()
