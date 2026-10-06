"""Check a converted Qwen3.5 (dense or MoE) MLX model:
  1. its vision tower against HF's on the same image (ref_vision_feats.npy,
     saved on the pod from the bf16 HF model): per-token cosine similarity;
  2. an image question end to end, the way mlx_lm.server runs it
     (Qwen35ImageInputs.build -> set_media_positions -> stream_generate with
     the fused input embeddings);
  3. wikitext-2 test perplexity (same 40 x 512 chunks as ppl_hf.py).

    python check_mlx.py --model /workspace/Ornith-1.5-35B-A3B-gptq-mlx-jang \
        --ref /workspace/ref --wikitext /workspace/data/wiki.test.raw

Writes <ref>/check.json and fails when the vision features or the
perplexity are off (--min-vision-cos; --max-ppl-ratio against
hf_reference.py's ref.json).
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--ref", required=True, help="dir with ref_vision_feats.npy, ref_image.png")
    ap.add_argument("--wikitext", required=True)
    ap.add_argument("--chunks", type=int, default=40)
    ap.add_argument("--out", help="check.json path (default: <ref>/check.json)")
    ap.add_argument("--min-vision-cos", type=float, default=0.95, help="mean cosine vs HF")
    ap.add_argument("--max-ppl-ratio", type=float, default=1.10, help="vs ref.json's bf16 PPL")
    args = ap.parse_args()
    result = {}
    model, tok = load(str(Path(args.model).expanduser()))
    ref = Path(args.ref).expanduser()
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
    messages = [{"role": "user", "content": [{"type": "text", "text": "Describe:"}, {"type": "image"}, {"type": "text", "text": "Short."}]}]
    prompt = tok.apply_chat_template(messages, add_generation_prompt=True)
    ids, embeds, _ = inputs.build(list(prompt), [blob])
    model.set_media_positions(inputs.media_positions)
    out, t0 = "", time.time()
    try:
        for r in stream_generate(model, tok, mx.array(ids), max_tokens=200, sampler=make_sampler(0.0), input_embeddings=embeds):
            out += r.text
    finally:
        model.set_media_positions(None)
    print(f"image answer ({time.time() - t0:.1f}s): {out!r}")
    result["image_answer"] = out

    # 3. Perplexity.
    text_ids = tok.encode(open(Path(args.wikitext).expanduser(), encoding="utf-8").read())
    n = 512
    stride = max(n, (len(text_ids) - n) // args.chunks)
    nll, count = 0.0, 0
    for k in range(args.chunks):
        chunk = mx.array(text_ids[k * stride : k * stride + n])[None]
        if chunk.shape[1] < n:
            break
        logits = model(chunk).astype(mx.float32)
        loss = nn.losses.cross_entropy(logits[0, :-1], chunk[0, 1:], reduction="sum")
        nll += loss.item()
        count += n - 1
    ppl = math.exp(nll / count)
    print(f"PPL {ppl:.4f} ({count} tokens)")
    result["ppl"] = ppl
    ref_ppl = json.load(open(ref / "ref.json"))["ppl_bf16"] if (ref / "ref.json").exists() else None
    if ref_ppl:
        result.update(ppl_bf16=ref_ppl, ppl_ratio=ppl / ref_ppl)
        print(f"vs bf16 {ref_ppl:.4f}: {100 * (ppl / ref_ppl - 1):+.2f} %")
    json.dump(result, open(args.out or ref / "check.json", "w"), indent=2)
    if result["vision_cos_mean"] < args.min_vision_cos:
        raise SystemExit(f"vision features off: mean cosine {result['vision_cos_mean']:.4f}")
    if ref_ppl and result["ppl_ratio"] > args.max_ppl_ratio:
        raise SystemExit(f"perplexity off: {result['ppl_ratio']:.3f}x bf16")
    print("CHECK_DONE")


if __name__ == "__main__":
    main()
