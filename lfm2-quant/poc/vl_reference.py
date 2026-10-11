"""HF transformers reference for an LFM2-VL checkpoint, in fp32: for each
test image, the processor's token ids and patches, the projected image
features and a greedy answer. check_vl_lfm2.py compares MLX models with it.

    python vl_reference.py --model SNAPSHOT --out /workspace/data/vl-ref
"""

from __future__ import annotations

import argparse
import io
import json
import urllib.request
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

# Two photos large enough to be tiled, and a small image sent whole.
IMAGES = {
    "cat": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg",
    "bee": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/bee.jpg",
}
QUESTION = "Describe this image in two sentences."


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-tokens", type=int, default=60)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    torch.backends.cuda.matmul.allow_tf32 = False

    blobs = {k: urllib.request.urlopen(u).read() for k, u in IMAGES.items()}
    buf = io.BytesIO()
    Image.open(io.BytesIO(blobs["bee"])).convert("RGB").resize((300, 200)).save(buf, format="PNG")
    blobs["small"] = buf.getvalue()

    proc = AutoProcessor.from_pretrained(args.model)
    model = AutoModelForImageTextToText.from_pretrained(args.model, dtype=torch.float32).cuda().eval()
    meta = {"question": QUESTION, "images": {}}
    for name, blob in blobs.items():
        (out / f"{name}.img").write_bytes(blob)
        image = Image.open(io.BytesIO(blob)).convert("RGB")
        messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": QUESTION}]}]
        text = proc.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        enc = proc(images=[image], text=text, return_tensors="pt")
        cuda = {k: v.cuda() for k, v in enc.items()}
        with torch.no_grad():
            feats = model.model.get_image_features(
                pixel_values=cuda["pixel_values"], spatial_shapes=cuda["spatial_shapes"],
                pixel_attention_mask=cuda["pixel_attention_mask"]).pooler_output
            gen = model.generate(**cuda, max_new_tokens=args.max_tokens, do_sample=False)
        answer = proc.tokenizer.decode(gen[0, enc["input_ids"].shape[1]:], skip_special_tokens=True)
        np.savez(out / f"{name}.npz", input_ids=enc["input_ids"].numpy(),
                 pixel_values=enc["pixel_values"].numpy(), spatial_shapes=enc["spatial_shapes"].numpy(),
                 features=torch.cat(feats).cpu().numpy())
        meta["images"][name] = {"size": image.size, "text": text, "answer": answer,
                                "crops": enc["spatial_shapes"].tolist()}
        print(f"{name} {image.size} crops {enc['spatial_shapes'].tolist()}: {answer!r}", flush=True)
    json.dump(meta, open(out / "meta.json", "w"), indent=2)
    print("VL_REFERENCE_DONE", flush=True)


if __name__ == "__main__":
    main()
