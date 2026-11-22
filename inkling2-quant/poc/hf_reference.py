"""transformers reference for an Inkling 2 checkpoint, one model load:
perplexity on wikitext-2 test and held-out code (the model's own logits:
final softcap, unpadded vocabulary), and the vision tower's image features
on one synthetic 640 x 480 image, for check_mlx.py.

Writes to --out: ref_image.png, ref_vision_feats.npy, ref.json. What is
already there is kept, not recomputed.

    python hf_reference.py --model SNAPSHOT --out /workspace/ref --wikitext wiki.test.raw --code code.test.txt
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw
from transformers import AutoImageProcessor, AutoModelForImageTextToText, AutoTokenizer


def reference_image() -> Image.Image:
    img = Image.new("RGB", (640, 480), (30, 120, 200))
    d = ImageDraw.Draw(img)
    d.rectangle([100, 100, 400, 300], fill=(250, 200, 40))
    d.ellipse([350, 200, 600, 460], fill=(200, 30, 60))
    return img


def perplexity(model, ids: torch.Tensor, chunks: int, n: int) -> float:
    device = model.get_input_embeddings().weight.device
    stride = max(n, (len(ids) - n) // chunks)
    nll, count = 0.0, 0
    with torch.no_grad():
        for k in range(chunks):
            chunk = ids[k * stride: k * stride + n].unsqueeze(0).to(device)
            if chunk.shape[1] < n:
                break
            logits = model(input_ids=chunk, use_cache=False).logits.float()
            nll += torch.nn.functional.cross_entropy(logits[0, :-1], chunk[0, 1:].to(logits.device), reduction="sum").item()
            count += n - 1
    return math.exp(nll / count)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--wikitext", required=True)
    ap.add_argument("--code", help="held-out code text: a second perplexity, ppl_bf16_code")
    ap.add_argument("--chunks", type=int, default=40)
    ap.add_argument("--chunk-tokens", type=int, default=512)
    ap.add_argument("--gpu-memory", default="76GiB", help="the rest of a big model goes to CPU memory")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    old = json.load(open(out / "ref.json")) if (out / "ref.json").exists() else {}
    want_text = "ppl_bf16" not in old
    want_code = bool(args.code) and "ppl_bf16_code" not in old
    want_vision = not (out / "ref_vision_feats.npy").exists()
    if not (want_text or want_code or want_vision):
        print("HF_REFERENCE_DONE (nothing to add)", flush=True)
        return
    tok = AutoTokenizer.from_pretrained(args.model)
    placement = dict(device_map="auto", max_memory={0: args.gpu_memory, "cpu": "1000GiB"}) if torch.cuda.is_available() \
        else {}
    model = AutoModelForImageTextToText.from_pretrained(args.model, dtype=torch.bfloat16, **placement).eval()

    result = {**old, "chunks": args.chunks, "chunk_tokens": args.chunk_tokens}
    for key, path, want in (("ppl_bf16", args.wikitext, want_text), ("ppl_bf16_code", args.code, want_code)):
        if want:
            ids = tok(open(path, encoding="utf-8").read(), return_tensors="pt")["input_ids"][0]
            result[key] = perplexity(model, ids, args.chunks, args.chunk_tokens)
            print(f"{key} {result[key]:.4f} ({args.chunks}x{args.chunk_tokens})", flush=True)
    if want_vision:
        image = reference_image()
        image.save(out / "ref_image.png")
        inputs = AutoImageProcessor.from_pretrained(args.model)(images=[image], return_tensors="pt")
        vt = model.model.vision_tower
        with torch.no_grad():
            feats = model.model.get_image_features(inputs["pixel_values"].to(vt.device, torch.bfloat16),
                                                   inputs["image_grid_thw"].to(vt.device), return_dict=True)
        feats = feats.pooler_output
        feats = torch.cat(list(feats)) if isinstance(feats, (list, tuple)) else feats
        np.save(out / "ref_vision_feats.npy", feats.float().cpu().numpy().reshape(-1, feats.shape[-1]))
        print(f"vision features {tuple(feats.shape)}", flush=True)
    json.dump(result, open(out / "ref.json", "w"), indent=2)
    print("HF_REFERENCE_DONE", flush=True)


if __name__ == "__main__":
    main()
