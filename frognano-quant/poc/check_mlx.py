"""Check a converted FrogNano (Qwen3.5) MLX model:
  1. its vision tower against HF's on the same image (ref_vision_feats.npy,
     saved on the pod from the bf16 HF model): per-token cosine similarity;
  2. an image question end to end, the way mlx_lm.server runs it
     (Qwen35ImageInputs.build -> set_media_positions -> stream_generate with
     the fused input embeddings);
  3. wikitext-2 test perplexity (same 40 x 512 chunks as ppl_hf.py).

    python check_mlx.py --model ~/models-work/FrogNano-4B-2609-gptq-mlx-jang \
        --ref ~/models-work --wikitext ~/models-work/wiki.test.raw
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
    args = ap.parse_args()
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
    print(f"PPL {math.exp(nll / count):.4f} ({count} tokens)")


if __name__ == "__main__":
    main()
