"""HF transformers reference for a Qwen3.5 checkpoint (dense or MoE), with
one model load: wikitext-2 test perplexity of the text decoder, and the
vision tower's features on one synthetic 640 x 480 image for check_mlx.py.

Writes to --out: ref_image.png, ref_vision_feats.npy, ref.json (PPL on
wikitext-2 and, with --code, on held-out Python code). What ref.json and
the features file already have is kept, not recomputed.

    python hf_reference.py --model SNAPSHOT --out /workspace/ref --wikitext /workspace/data/wiki.test.raw
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


def numpy_patches(image: Image.Image, cfg: dict) -> dict:
    """Qwen2VL image preprocessing without torchvision (as the mlx-lm fork's
    Qwen35ImageInputs does it): resize to multiples of patch * merge within
    the pixel limits, normalize, cut into temporal x patch x patch patches."""
    p, m, t = cfg.get("patch_size", 16), cfg.get("merge_size", 2), cfg.get("temporal_patch_size", 2)
    lo, hi = cfg["size"]["shortest_edge"], min(cfg["size"]["longest_edge"], 1_048_576)
    f = p * m
    w, h = image.size
    hb, wb = max(f, round(h / f) * f), max(f, round(w / f) * f)
    if hb * wb > hi:
        beta = math.sqrt(h * w / hi)
        hb, wb = max(f, math.floor(h / beta / f) * f), max(f, math.floor(w / beta / f) * f)
    elif hb * wb < lo:
        beta = math.sqrt(lo / (h * w))
        hb, wb = math.ceil(h * beta / f) * f, math.ceil(w * beta / f) * f
    x = np.asarray(image.convert("RGB").resize((wb, hb), Image.BICUBIC), dtype=np.float32) / 255.0
    x = (x - np.array(cfg["image_mean"], np.float32)) / np.array(cfg["image_std"], np.float32)
    x = np.repeat(x.transpose(2, 0, 1)[None], t, axis=0)
    gh, gw = hb // p, wb // p
    x = x.reshape(1, t, 3, gh // m, m, p, gw // m, m, p).transpose(0, 3, 6, 4, 7, 2, 1, 5, 8)
    return {"pixel_values": torch.from_numpy(x.reshape(gh * gw, 3 * t * p * p).copy()),
            "image_grid_thw": torch.tensor([[1, gh, gw]])}


def perplexity(model, ids: torch.Tensor, chunks: int, n: int) -> float:
    lm, head = model.model.language_model, model.lm_head
    device = lm.embed_tokens.weight.device
    stride = max(n, (len(ids) - n) // chunks)
    nll, count = 0.0, 0
    with torch.no_grad():
        for k in range(chunks):
            chunk = ids[k * stride: k * stride + n].unsqueeze(0).to(device)
            if chunk.shape[1] < n:
                break
            hidden = lm(input_ids=chunk, use_cache=False).last_hidden_state
            logits = head(hidden.to(head.weight.device)).float()
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
    if want_vision and getattr(model.model, "visual", None) is not None:
        img = reference_image()
        img.save(out / "ref_image.png")
        try:
            inp = AutoImageProcessor.from_pretrained(args.model)(images=[img], return_tensors="pt")
            result["preprocessing"] = "hf"
        except ImportError:  # no torchvision: the fork's numpy preprocessing (equal to HF's to 4e-9, FINDINGS)
            inp = numpy_patches(img, json.load(open(Path(args.model) / "preprocessor_config.json")))
            result["preprocessing"] = "numpy"
        dev = next(model.model.visual.parameters()).device
        with torch.no_grad():
            feats = model.model.get_image_features(inp["pixel_values"].to(dev), inp["image_grid_thw"].to(dev)).pooler_output
        feats = torch.cat(list(feats)).float().cpu().numpy()
        np.save(out / "ref_vision_feats.npy", feats)
        result["vision_feats"] = list(feats.shape)
        print(f"vision features {feats.shape}", flush=True)
    json.dump(result, open(out / "ref.json", "w"), indent=2)
    print("HF_REFERENCE_DONE", flush=True)


if __name__ == "__main__":
    main()
