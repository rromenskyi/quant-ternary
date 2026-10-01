"""Perplexity on wikitext-2 (test) and KL divergence to a reference model,
for MLX checkpoints of one model -- the QAT conversions against each other.

  python qat_eval.py --text wiki.test.raw --ref E2B-unq E2B-qat-mlx mlx-community/gemma-4-E2B-it-qat-4bit ...

Non-overlapping windows of --ctx tokens, each starting with BOS (the first
--windows of them), the same token ids for every model (the reference's
tokenizer). The reference's log-probabilities wait in host memory (fp16). PPL counts every
next-token prediction in a window; KL(ref || model) averages over the same
positions, with the reference's logits computed once per window.
"""
from __future__ import annotations

import argparse
import gc
import json
import math

import mlx.core as mx
import numpy as np
from mlx_lm import load


def windows(tok, text: str, ctx: int, n: int, chat: bool = False) -> list[tuple[list[int], int]]:
    """(tokens, first scored position) per window. Plain: BOS + text, as
    llama.cpp's perplexity (Gemma without BOS is off the rails). --chat:
    the text as the model's reply to "Continue this text." -- for a
    chat-only checkpoint (the 12B QAT scores raw text like noise, in
    transformers too); only the text's own tokens are scored."""
    ids = tok.encode(text, add_special_tokens=False)
    prefix = [tok.bos_token_id]
    if chat:
        prompt = tok.apply_chat_template([{"role": "user", "content": "Continue this text."}],
                                         add_generation_prompt=True, tokenize=False)
        prefix = tok.encode(prompt, add_special_tokens=False)
        if prefix[0] != tok.bos_token_id:
            prefix = [tok.bos_token_id] + prefix
    step = ctx - len(prefix)
    return [(prefix + ids[i: i + step], len(prefix) - 1) for i in range(0, len(ids) - step, step)][:n]


def logprobs(model, tokens: list[int]) -> mx.array:
    logits = model(mx.array(tokens)[None])[0].astype(mx.float32)
    return logits - mx.logsumexp(logits, axis=-1, keepdims=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--text", required=True)
    ap.add_argument("--ref", required=True, help="reference checkpoint (bf16): its tokenizer and logits")
    ap.add_argument("--ctx", type=int, default=512)
    ap.add_argument("--windows", type=int, default=64)
    ap.add_argument("--chat", action="store_true", help="score the text as the model's chat reply (chat-only checkpoints)")
    ap.add_argument("--json")
    ap.add_argument("models", nargs="+")
    args = ap.parse_args()

    text = open(args.text).read()
    ref_model, tok = load(args.ref)
    wins = windows(tok, text, args.ctx, args.windows, chat=args.chat)
    print(f"{len(wins)} windows of {args.ctx} tokens", flush=True)
    ref_lp = []
    nll = 0.0
    for w, s in wins:
        lp = logprobs(ref_model, w)[s:-1]
        nll -= mx.take_along_axis(lp, mx.array(w[s + 1:])[:, None], axis=-1).sum().item()
        # Host memory, fp16: 128 windows of a 262k vocabulary don't fit a GPU.
        ref_lp.append(np.array(lp.astype(mx.float16)))
    count = sum(len(w) - 1 - s for w, s in wins)
    results = {args.ref: {"ppl": math.exp(nll / count), "kl": 0.0, "top1": 1.0}}
    print(f"{args.ref}: ppl {results[args.ref]['ppl']:.3f}", flush=True)
    del ref_model
    gc.collect()

    for path in args.models:
        model, _ = load(path)
        nll, kl, agree = 0.0, 0.0, 0
        for (w, s), rlp in zip(wins, ref_lp):
            rlp = mx.array(rlp).astype(mx.float32)
            lp = logprobs(model, w)[s:-1]
            nll -= mx.take_along_axis(lp, mx.array(w[s + 1:])[:, None], axis=-1).sum().item()
            kl += (mx.exp(rlp) * (rlp - lp)).sum().item()
            agree += (mx.argmax(rlp, -1) == mx.argmax(lp, -1)).sum().item()
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
