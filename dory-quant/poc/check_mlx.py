"""Check a converted Dory MLX (GPTQ JANG) model: perplexity on wikitext-2
test (and, with --code, held-out Python code) against reference.py's torch
bf16 numbers, plus one short greedy generation as an end-to-end sanity
check (Dory has no vision tower or MTP head to check here).

    python check_mlx.py --model /workspace/Dory1.0-4B-gptq-mlx-jang \
        --ref /workspace/ref --wikitext /workspace/data/wiki.test.raw \
        --code /workspace/data/code.test.txt --out /workspace/check-dory1.0-4b.json
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn

from mlx_lm import load, stream_generate
from mlx_lm.sample_utils import make_sampler


def perplexity(model, tok, path: str, chunks: int, n: int = 512) -> tuple[float, int]:
    text_ids = tok.encode(open(Path(path).expanduser(), encoding="utf-8").read())
    stride = max(n, (len(text_ids) - n) // chunks)
    nll, count = 0.0, 0
    for k in range(chunks):
        chunk = mx.array(text_ids[k * stride: k * stride + n])[None]
        if chunk.shape[1] < n:
            break
        logits = model(chunk).astype(mx.float32)
        nll += nn.losses.cross_entropy(logits[0, :-1], chunk[0, 1:], reduction="sum").item()
        count += n - 1
    return math.exp(nll / count), count


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--ref", required=True)
    ap.add_argument("--wikitext", required=True)
    ap.add_argument("--code")
    ap.add_argument("--out")
    ap.add_argument("--chunks", type=int, default=40)
    ap.add_argument("--max-ppl-ratio", type=float, default=1.10)
    args = ap.parse_args()
    out_path = Path(args.out) if args.out else Path(args.ref) / "check.json"
    bf16 = json.load(open(Path(args.ref) / "ref.json"))
    result: dict = {}

    model, tok = load(args.model)
    prompt = tok.encode("The capital of France is")
    answer = "".join(r.text for r in stream_generate(model, tok, mx.array(prompt), max_tokens=30,
                                                      sampler=make_sampler(0.0)))
    print(f"generation sample: {answer!r}", flush=True)
    result["generation_sample"] = answer

    sets = [("", args.wikitext, "ppl_bf16")]
    if args.code:
        sets.append(("_code", args.code, "ppl_bf16_code"))
    for suffix, path, ref_key in sets:
        ppl, count = perplexity(model, tok, path, args.chunks)
        result[f"ppl{suffix}"] = ppl
        print(f"PPL{suffix} {ppl:.4f} ({count} tokens)", flush=True)
        if ref_key in bf16:
            result[f"ppl{suffix}_bf16"] = bf16[ref_key]
            result[f"ppl{suffix}_ratio"] = ppl / bf16[ref_key]
            print(f"  vs bf16 {bf16[ref_key]:.4f}: {100 * (ppl / bf16[ref_key] - 1):+.2f} %", flush=True)

    failures = []
    for key in ("ppl_ratio", "ppl_code_ratio"):
        if result.get(key, 1.0) > args.max_ppl_ratio:
            failures.append(f"perplexity off: {key} {result[key]:.3f} > {args.max_ppl_ratio}")
    result.update(passed=not failures, max_ppl_ratio=args.max_ppl_ratio)
    json.dump(result, open(out_path, "w"), indent=2)
    if failures:
        raise SystemExit("; ".join(failures))
    print("CHECK_DONE", flush=True)


if __name__ == "__main__":
    main()
