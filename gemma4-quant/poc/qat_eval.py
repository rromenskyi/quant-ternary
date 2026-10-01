"""Perplexity on wikitext-2 (test) and KL divergence to a reference model,
for MLX checkpoints of one model -- the QAT conversions against each other.

  python qat_eval.py --text wiki.test.raw --ref E2B-unq E2B-qat-mlx mlx-community/gemma-4-E2B-it-qat-4bit ...

Non-overlapping windows of --ctx tokens (the first --windows of them), the
same token ids for every model (the reference's tokenizer). PPL counts every
next-token prediction in a window; KL(ref || model) averages over the same
positions, with the reference's logits computed once per window.
"""
from __future__ import annotations

import argparse
import gc
import json
import math

import mlx.core as mx
from mlx_lm import load


def windows(tokenizer, text: str, ctx: int, n: int) -> list[list[int]]:
    """Each window starts with BOS, as llama.cpp's perplexity does: Gemma
    without it is off the rails (PPL in the thousands)."""
    ids = tokenizer.encode(text, add_special_tokens=False)
    bos = tokenizer.bos_token_id
    step = ctx - 1
    out = [[bos] + ids[i: i + step] for i in range(0, len(ids) - step, step)]
    return out[:n]


def logprobs(model, tokens: list[int]) -> mx.array:
    logits = model(mx.array(tokens)[None])[0].astype(mx.float32)
    return logits - mx.logsumexp(logits, axis=-1, keepdims=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--text", required=True)
    ap.add_argument("--ref", required=True, help="reference checkpoint (bf16): its tokenizer and logits")
    ap.add_argument("--ctx", type=int, default=512)
    ap.add_argument("--windows", type=int, default=64)
    ap.add_argument("--json")
    ap.add_argument("models", nargs="+")
    args = ap.parse_args()

    text = open(args.text).read()
    ref_model, tok = load(args.ref)
    wins = windows(tok, text, args.ctx, args.windows)
    print(f"{len(wins)} windows of {args.ctx} tokens", flush=True)
    ref_lp = []
    nll = 0.0
    for w in wins:
        lp = logprobs(ref_model, w)
        mx.eval(lp)
        ref_lp.append(lp)
        nll -= mx.take_along_axis(lp[:-1], mx.array(w[1:])[:, None], axis=-1).sum().item()
    count = sum(len(w) - 1 for w in wins)
    results = {args.ref: {"ppl": math.exp(nll / count), "kl": 0.0, "top1": 1.0}}
    print(f"{args.ref}: ppl {results[args.ref]['ppl']:.3f}", flush=True)
    del ref_model
    gc.collect()

    for path in args.models:
        model, _ = load(path)
        nll, kl, agree = 0.0, 0.0, 0
        for w, rlp in zip(wins, ref_lp):
            lp = logprobs(model, w)
            nll -= mx.take_along_axis(lp[:-1], mx.array(w[1:])[:, None], axis=-1).sum().item()
            kl += (mx.exp(rlp[:-1]) * (rlp[:-1] - lp[:-1])).sum().item()
            agree += (mx.argmax(rlp[:-1], -1) == mx.argmax(lp[:-1], -1)).sum().item()
        results[path] = {"ppl": math.exp(nll / count), "kl": kl / count, "top1": agree / count}
        r = results[path]
        print(f"{path}: ppl {r['ppl']:.3f}  KL {r['kl']:.4f}  top-1 agree {r['top1']:.2%}", flush=True)
        del model
        gc.collect()
        mx.clear_cache()
    if args.json:
        json.dump(results, open(args.json, "w"), indent=1)


if __name__ == "__main__":
    main()
