"""Decode and prefill speed of MLX checkpoints on this Mac, fanless-safe:
each model runs --rounds times, the order reversed every round, with
--cool seconds of idle before each run (a fanless M5 throttles under a
sustained load and would otherwise favour whichever runs first).

  python qat_speed.py --cool 60 E2B-qat-mlx mlx-community/gemma-4-E2B-it-qat-4bit ...

Prints the median generation and prompt tokens/s per model, and peak memory.
"""
from __future__ import annotations

import argparse
import gc
import statistics
import time

import mlx.core as mx
from mlx_lm import load, stream_generate

PROMPT = ("Write a short, practical guide for a new team member on how to review a pull request: "
          "what to look at first, how to leave useful comments, and when to approve.")


def run(path: str, max_tokens: int) -> tuple[float, float, float]:
    model, tok = load(path)
    prompt = tok.apply_chat_template([{"role": "user", "content": PROMPT}], add_generation_prompt=True, tokenize=False)
    for _ in stream_generate(model, tok, prompt, max_tokens=8):   # Metal kernels compiled, not timed
        pass
    mx.reset_peak_memory()
    last = None
    for last in stream_generate(model, tok, prompt, max_tokens=max_tokens):
        pass
    peak = mx.get_peak_memory() / 1e9
    del model
    gc.collect()
    mx.clear_cache()
    return last.generation_tps, last.prompt_tps, peak


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--cool", type=float, default=60)
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("models", nargs="+")
    args = ap.parse_args()
    gen = {m: [] for m in args.models}
    pre = {m: [] for m in args.models}
    peak = {m: 0.0 for m in args.models}
    for r in range(args.rounds):
        order = args.models if r % 2 == 0 else list(reversed(args.models))
        for m in order:
            time.sleep(args.cool)
            g, p, mem = run(m, args.max_tokens)
            gen[m].append(g)
            pre[m].append(p)
            peak[m] = max(peak[m], mem)
            print(f"round {r + 1} {m}: {g:.1f} tok/s decode, {p:.0f} tok/s prefill", flush=True)
    print()
    for m in args.models:
        print(f"{m}: decode {statistics.median(gen[m]):.1f} tok/s, prefill {statistics.median(pre[m]):.0f} tok/s, peak {peak[m]:.2f} GB")


if __name__ == "__main__":
    main()
