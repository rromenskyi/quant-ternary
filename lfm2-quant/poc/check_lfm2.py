"""Perplexity of an MLX model (a local dir or an HF repo) on wikitext-2 test
and held-out Python code, 40 x 512 tokens each (the protocol of
qwen35-quant/poc/check_mlx.py), plus one greedy chat answer. Every model
-- bf16, ours, other quants -- is measured the same way, in MLX, so the
ratios compare like with like.

    python check_lfm2.py --model /workspace/hf/.../LFM2.5-8B-A1B --name bf16 --out /workspace/results-lfm2-8b.json ...
    python check_lfm2.py --model /workspace/LFM2.5-8B-A1B-gptq-mlx-jang --name ours --ref bf16 --max-ppl-ratio 1.10 ...

Results go into --out under --name (other entries kept). With --ref, the
ratios to that entry are added and checked against --max-ppl-ratio.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn

from mlx_lm import load, stream_generate
from mlx_lm.sample_utils import make_sampler


def perplexity(model, tok, path: str, chunks: int, n: int = 512) -> tuple[float, int]:
    ids = tok.encode(open(path, encoding="utf-8").read())
    stride = max(n, (len(ids) - n) // chunks)
    nll, count = 0.0, 0
    for k in range(chunks):
        chunk = mx.array(ids[k * stride: k * stride + n])[None]
        if chunk.shape[1] < n:
            break
        logits = model(chunk).astype(mx.float32)
        nll += nn.losses.cross_entropy(logits[0, :-1], chunk[0, 1:], reduction="sum").item()
        count += n - 1
    return math.exp(nll / count), count


def size_gb(path: Path) -> float:
    return sum(f.stat().st_size for f in path.glob("*.safetensors")) / 1e9


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--wikitext", required=True)
    ap.add_argument("--code", required=True)
    ap.add_argument("--chunks", type=int, default=40)
    ap.add_argument("--ref", help="entry in --out to compare with (bf16)")
    ap.add_argument("--max-ppl-ratio", type=float, default=1.10)
    args = ap.parse_args()
    out = Path(args.out)
    results = json.load(open(out)) if out.exists() else {}
    model, tok = load(args.model)
    from huggingface_hub import snapshot_download
    local = Path(args.model) if Path(args.model).exists() else Path(snapshot_download(args.model))
    r = {"model": args.model, "size_gb": round(size_gb(local), 3)}

    messages = [{"role": "user", "content": "Write a Python function that checks whether a number is prime. Code only."}]
    prompt = tok.apply_chat_template(messages, add_generation_prompt=True)
    answer, t0, n = "", time.time(), 0
    for resp in stream_generate(model, tok, prompt, max_tokens=200, sampler=make_sampler(0.0)):
        answer += resp.text
        n += 1
    r["answer"], r["tok_s"] = answer, round(n / (time.time() - t0), 1)
    print(f"answer ({r['tok_s']} tok/s): {answer!r}", flush=True)

    for key, path in (("ppl_text", args.wikitext), ("ppl_code", args.code)):
        r[key], count = perplexity(model, tok, path, args.chunks)
        print(f"{key} {r[key]:.4f} ({count} tokens)", flush=True)
    r["peak_gb"] = round(mx.get_peak_memory() / 1e9, 2)

    failures = []
    if args.ref:
        base = results[args.ref]
        for key in ("ppl_text", "ppl_code"):
            r[f"{key}_ratio"] = r[key] / base[key]
            print(f"  {key} vs {args.ref}: {100 * (r[f'{key}_ratio'] - 1):+.2f} %", flush=True)
            if r[f"{key}_ratio"] > args.max_ppl_ratio:
                failures.append(f"{key} ratio {r[f'{key}_ratio']:.3f} > {args.max_ppl_ratio}")
        r["max_ppl_ratio"] = args.max_ppl_ratio
    r["passed"] = not failures
    results[args.name] = r
    json.dump(results, open(out, "w"), indent=2)
    if failures:
        raise SystemExit("; ".join(failures))
    print("CHECK_DONE", flush=True)


if __name__ == "__main__":
    main()
