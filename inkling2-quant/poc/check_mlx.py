"""Checks of a converted Inkling 2 MLX model against hf_reference.py's
output: the vision tower's features vs transformers', an image question as
the chat template renders it, perplexity on wikitext-2 test and held-out
code (the same chunks as the reference), and one text answer.

    python check_mlx.py --model /workspace/Inkling-2-Mini-gptq-mlx-jang --ref /workspace/ref \\
        --wikitext wiki.test.raw --code code.test.txt --out check.json
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import mlx.core as mx
import numpy as np
from mlx_lm import load, stream_generate
from mlx_lm.models.inkling2.processor import IMAGE_TOKEN, Processor
from mlx_lm.sample_utils import make_sampler
from PIL import Image


def perplexity(model, tok, path: str, chunks: int, n: int) -> tuple[float, int]:
    """As hf_reference.perplexity: `chunks` windows of n tokens, evenly strided."""
    ids = tok.encode(open(path, encoding="utf-8").read())
    stride = max(n, (len(ids) - n) // chunks)
    nll, count = 0.0, 0
    for k in range(chunks):
        chunk = ids[k * stride: k * stride + n]
        if len(chunk) < n:
            break
        x = mx.array(chunk)[None]
        logits = model(x[:, :-1]).astype(mx.float32)
        nll += float(mx.sum(mx.logsumexp(logits, -1) - mx.take_along_axis(logits, x[:, 1:, None], -1)[..., 0]))
        count += n - 1
    return math.exp(nll / count), count


def answer(model, tok, messages, max_tokens: int, processor=None, images=None) -> str:
    prompt = tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    embeds = None
    if images:
        prompt, media = processor(prompt, images=images)
        ids = mx.array(tok.encode(prompt, add_special_tokens=False))
        embeds = model.embed_inputs(ids[None], pixel_values=media["pixel_values"])[0]
    else:
        ids = mx.array(tok.encode(prompt, add_special_tokens=False))
    text = ""
    for r in stream_generate(model, tok, ids, max_tokens=max_tokens, sampler=make_sampler(0.0),
                             input_embeddings=embeds):
        text += r.text
    return text


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--ref", required=True)
    ap.add_argument("--wikitext", required=True)
    ap.add_argument("--code")
    ap.add_argument("--out")
    ap.add_argument("--max-ppl-ratio", type=float, default=1.10, help="vs ref.json's bf16 perplexity")
    ap.add_argument("--min-vision-cos", type=float, default=0.98)
    ap.add_argument("--max-tokens", type=int, default=200)
    args = ap.parse_args()
    ref = Path(args.ref).expanduser()
    out_path = Path(args.out) if args.out else ref / "check.json"
    bf16 = json.load(open(ref / "ref.json"))
    model, tok = load(str(Path(args.model).expanduser()))
    result = {}

    # 1. Vision tower vs transformers.
    image = np.asarray(Image.open(ref / "ref_image.png").convert("RGB"))
    processor = Processor.from_config({"vision_config": model.args.vision_config})
    pixels, _ = processor.preprocess_image(image)
    mine = np.array(model.encode_image(pixels).astype(mx.float32))
    hf = np.load(ref / "ref_vision_feats.npy")
    cos = (mine * hf).sum(-1) / (np.linalg.norm(mine, axis=-1) * np.linalg.norm(hf, axis=-1) + 1e-9)
    print(f"vision vs HF: shape {mine.shape} / {hf.shape}, cosine min {cos.min():.4f} mean {cos.mean():.4f}", flush=True)
    result.update(vision_cos_mean=float(cos.mean()), vision_cos_min=float(cos.min()))

    # 2. An image question, then a text one, greedy.
    t0 = time.time()
    messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "Describe the image briefly."}]}]
    result["image_answer"] = answer(model, tok, messages, args.max_tokens, processor, [image])
    print(f"image answer ({time.time() - t0:.1f}s): {result['image_answer']!r}", flush=True)
    t0 = time.time()
    messages = [{"role": "user", "content": "Write a Python function that returns the n-th Fibonacci number."}]
    result["text_answer"] = answer(model, tok, messages, args.max_tokens)
    print(f"text answer ({time.time() - t0:.1f}s): {result['text_answer']!r}", flush=True)

    # 3. Perplexity, the reference's chunks.
    failures = []
    for suffix, path, ref_key in [("", args.wikitext, "ppl_bf16")] + ([("_code", args.code, "ppl_bf16_code")] if args.code else []):
        ppl, count = perplexity(model, tok, path, bf16["chunks"], bf16["chunk_tokens"])
        result[f"ppl{suffix}"] = ppl
        print(f"PPL{suffix} {ppl:.4f} ({count} tokens)", flush=True)
        if ref_key in bf16:
            result[f"ppl{suffix}_bf16"] = bf16[ref_key]
            result[f"ppl{suffix}_ratio"] = ppl / bf16[ref_key]
            print(f"  vs bf16 {bf16[ref_key]:.4f}: {100 * (ppl / bf16[ref_key] - 1):+.2f} %", flush=True)
            if ppl / bf16[ref_key] > args.max_ppl_ratio:
                failures.append(f"perplexity off: ppl{suffix}_ratio {ppl / bf16[ref_key]:.3f} > {args.max_ppl_ratio}")
    if result["vision_cos_mean"] < args.min_vision_cos:
        failures.append(f"vision features off: mean cosine {result['vision_cos_mean']:.4f}")
    result.update(passed=not failures, max_ppl_ratio=args.max_ppl_ratio)
    json.dump(result, open(out_path, "w"), indent=2)
    if failures:
        raise SystemExit("; ".join(failures))
    print("CHECK_DONE", flush=True)


if __name__ == "__main__":
    main()
