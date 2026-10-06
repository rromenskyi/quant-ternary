"""Check a converted Qwen3.5 (dense or MoE) MLX model:
  1. its vision tower against HF's on the same image (ref_vision_feats.npy,
     saved on the pod from the bf16 HF model): per-token cosine similarity;
  2. an image question end to end, the way mlx_lm.server runs it
     (Qwen35ImageInputs.build -> set_media_positions -> stream_generate with
     the fused input embeddings);
  3. perplexity, 40 x 512 tokens, as hf_reference.py measures bf16: on
     wikitext-2 test, and with --code on held-out Python code.

    python check_mlx.py --model /workspace/Ornith-1.5-35B-A3B-gptq-mlx-jang \
        --ref /workspace/ref --wikitext /workspace/data/wiki.test.raw \
        --code /workspace/data/code.test.txt --out /workspace/check-ornith-35b.json

Adds its results to --out (default <ref>/check.json) and fails when the
vision features or a perplexity are off (--min-vision-cos; --max-ppl-ratio
against hf_reference.py's ref.json).
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from PIL import Image

from mlx_lm import load, stream_generate
from mlx_lm.multimodal import load_image_inputs
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
    ap.add_argument("--ref", required=True, help="dir with ref.json, ref_vision_feats.npy, ref_image.png")
    ap.add_argument("--wikitext", required=True)
    ap.add_argument("--code", help="held-out code text: ppl_code, against ref.json's ppl_bf16_code")
    ap.add_argument("--only-code", action="store_true", help="just the code perplexity")
    ap.add_argument("--chunks", type=int, default=40)
    ap.add_argument("--out", help="results JSON (default <ref>/check.json); existing results are kept")
    ap.add_argument("--min-vision-cos", type=float, default=0.95, help="mean cosine vs HF")
    ap.add_argument("--max-ppl-ratio", type=float, default=1.10, help="vs ref.json's bf16 PPL")
    args = ap.parse_args()
    ref = Path(args.ref).expanduser()
    out_path = Path(args.out) if args.out else ref / "check.json"
    result = json.load(open(out_path)) if out_path.exists() else {}
    bf16 = json.load(open(ref / "ref.json")) if (ref / "ref.json").exists() else {}
    model, tok = load(str(Path(args.model).expanduser()))

    if not args.only_code:
        inputs = load_image_inputs(model, Path(args.model).expanduser())
        assert inputs is not None, "no image inputs: vision tower missing"

        # 1. Vision tower vs HF.
        image = Image.open(ref / "ref_image.png")
        pv, grid = inputs._patches(image)
        dtype = model.vision_tower.patch_embed.proj.weight.dtype
        feats, _ = model.vision_tower(mx.array(pv).astype(dtype), mx.array([grid], dtype=mx.int32))
        mine = np.array(feats.astype(mx.float32))
        hf = np.load(ref / "ref_vision_feats.npy")
        cos = (mine * hf).sum(-1) / (np.linalg.norm(mine, axis=-1) * np.linalg.norm(hf, axis=-1) + 1e-9)
        print(f"vision vs HF: shape {mine.shape} / {hf.shape}, cosine min {cos.min():.4f} mean {cos.mean():.4f}")
        result.update(vision_cos_mean=float(cos.mean()), vision_cos_min=float(cos.min()))

        # 2. An image question, as the server runs it.
        blob = (ref / "ref_image.png").read_bytes()
        messages = [{"role": "user", "content": [{"type": "text", "text": "Describe:"}, {"type": "image"},
                                                 {"type": "text", "text": "Short."}]}]
        prompt = tok.apply_chat_template(messages, add_generation_prompt=True)
        ids, embeds, _ = inputs.build(list(prompt), [blob])
        model.set_media_positions(inputs.media_positions)
        answer, t0 = "", time.time()
        try:
            for r in stream_generate(model, tok, mx.array(ids), max_tokens=200, sampler=make_sampler(0.0),
                                     input_embeddings=embeds):
                answer += r.text
        finally:
            model.set_media_positions(None)
        print(f"image answer ({time.time() - t0:.1f}s): {answer!r}")
        result["image_answer"] = answer

    # 3. Perplexity.
    sets = [] if args.only_code else [("", args.wikitext, "ppl_bf16")]
    if args.code:
        sets.append(("_code", args.code, "ppl_bf16_code"))
    for suffix, path, ref_key in sets:
        ppl, count = perplexity(model, tok, path, args.chunks)
        result[f"ppl{suffix}"] = ppl
        print(f"PPL{suffix} {ppl:.4f} ({count} tokens)")
        if ref_key in bf16:
            result[f"ppl{suffix}_bf16"] = bf16[ref_key]
            result[f"ppl{suffix}_ratio"] = ppl / bf16[ref_key]
            print(f"  vs bf16 {bf16[ref_key]:.4f}: {100 * (ppl / bf16[ref_key] - 1):+.2f} %")
    failures = []
    if result.get("vision_cos_mean", 1.0) < args.min_vision_cos:
        failures.append(f"vision features off: mean cosine {result['vision_cos_mean']:.4f}")
    for key in ("ppl_ratio", "ppl_code_ratio"):
        if result.get(key, 1.0) > args.max_ppl_ratio:
            failures.append(f"perplexity off: {key} {result[key]:.3f} > {args.max_ppl_ratio}")
    # The pipeline skips the check only on a pass.
    result.update(passed=not failures, max_ppl_ratio=args.max_ppl_ratio)
    json.dump(result, open(out_path, "w"), indent=2)
    if failures:
        raise SystemExit("; ".join(failures))
    print("CHECK_DONE")


if __name__ == "__main__":
    main()
