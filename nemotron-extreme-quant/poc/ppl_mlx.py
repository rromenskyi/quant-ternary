"""Perplexity for an already-MLX-converted model, computed directly (no
server, no custom kernel dependency) -- for comparing against an HF
checkpoint's PPL (poc/ppl_wikitext.py) after the mlx_lm.convert -q step,
since that conversion may not exactly reproduce a GPTQ checkpoint's codes
(see docs/session_findings_2026-09-11.md and docs/RUNBOOK.md for why).

Non-overlapping chunks starting a quarter of the way into the corpus, for a
held-out-ish feel without needing a separate train/test split file.

Usage:
    python poc/ppl_mlx.py --model /root/lightning30b-gptq3bit-g64-mlx \
        --text /root/llama.cpp/wikitext-2-raw/wiki.test.raw --chunks 20 --chunk-tokens 512
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import mlx.core as mx
import mlx_lm.tokenizer_utils as tu
from mlx_lm.utils import load_model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--text", required=True, help="path to a held-out text file (e.g. wikitext-2-raw's wiki.test.raw)")
    parser.add_argument("--chunks", type=int, default=20)
    parser.add_argument("--chunk-tokens", type=int, default=512)
    parser.add_argument("--trust-remote-code", action="store_true")
    args = parser.parse_args()

    model, config = load_model(Path(args.model))
    tokenizer = tu.load(Path(args.model), tokenizer_config_extra={"trust_remote_code": args.trust_remote_code} if args.trust_remote_code else {})

    text = open(args.text, "r", encoding="utf-8", errors="ignore").read()
    ids_all = tokenizer.encode(text)
    print(f"total tokens in corpus: {len(ids_all)}", flush=True)

    nlls = []
    start_offset = len(ids_all) // 4
    for i in range(args.chunks):
        s = start_offset + i * args.chunk_tokens
        e = s + args.chunk_tokens
        if e > len(ids_all):
            break
        chunk = ids_all[s:e]
        x = mx.array(chunk)[None]
        logits = model(x)
        logp = logits[:, :-1, :].astype(mx.float32)
        targets = x[:, 1:]
        logp = logp - mx.logsumexp(logp, axis=-1, keepdims=True)
        tok_logp = mx.take_along_axis(logp, targets[..., None], axis=-1).squeeze(-1)
        nll = -mx.mean(tok_logp)
        mx.eval(nll)
        nlls.append(float(nll))
        print(f"[{i + 1}/{args.chunks}] nll={nlls[-1]:.4f}", flush=True)

    mean_nll = sum(nlls) / len(nlls)
    ppl = math.exp(mean_nll)
    print(f"\nFinal: PPL = {ppl:.4f} (mean_nll={mean_nll:.4f}, n={len(nlls)} chunks x {args.chunk_tokens} tokens)")


if __name__ == "__main__":
    main()
