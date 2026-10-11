"""Image inputs of an MLX LFM2-VL model against vl_reference.py's HF fp32
reference: the expanded token ids, the image features (cosine per image
token) and the greedy answer per test image. The result is added to the
model's entry (--name) in check_lfm2.py's results file, under "vision".

    python check_vl_lfm2.py --model OUT_DIR --ref /workspace/data/vl-ref --name lfm2-vl-3b --out results.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import mlx.core as mx
import numpy as np

from mlx_lm import load, stream_generate
from mlx_lm.multimodal import _decode, load_image_inputs
from mlx_lm.sample_utils import make_sampler


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--ref", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--min-cosine", type=float, default=0.95, help="fail below this mean feature cosine")
    args = ap.parse_args()
    ref, out = Path(args.ref), Path(args.out)
    meta = json.load(open(ref / "meta.json"))
    model, tok = load(args.model)
    inputs = load_image_inputs(model, args.model)
    if inputs is None:
        raise SystemExit(f"{args.model} takes no images (no vision tower or processor_config.json)")

    report, failures = {}, []
    for name, m in meta["images"].items():
        r = np.load(ref / f"{name}.npz")
        blob = (ref / f"{name}.img").read_bytes()
        ids, emb, _ = inputs.build(tok.encode(m["text"], add_special_tokens=False), [blob])
        crops, *_ = inputs.crops(_decode(blob))
        f = np.array(model.image_features([(mx.array(p), h, w) for p, h, w in crops]).astype(mx.float32))
        hf = r["features"]
        cos = (f * hf).sum(-1) / (np.linalg.norm(f, axis=-1) * np.linalg.norm(hf, axis=-1) + 1e-9)
        answer = "".join(x.text for x in stream_generate(
            model, tok, prompt=ids, input_embeddings=emb, max_tokens=60, sampler=make_sampler(0.0)))
        report[name] = {"ids_equal": ids == r["input_ids"][0].tolist(), "cos_mean": round(float(cos.mean()), 4),
                        "cos_min": round(float(cos.min()), 4), "answer": answer, "hf_answer": m["answer"]}
        print(f"{name}: ids equal {report[name]['ids_equal']}, cosine mean {cos.mean():.4f} min {cos.min():.4f}\n"
              f"   HF fp32: {m['answer']!r}\n   this:    {answer!r}", flush=True)
        if not report[name]["ids_equal"]:
            failures.append(f"{name}: token ids differ from HF")
        if cos.mean() < args.min_cosine:
            failures.append(f"{name}: feature cosine {cos.mean():.4f} < {args.min_cosine}")

    results = json.load(open(out)) if out.exists() else {}
    results.setdefault(args.name, {})["vision"] = report
    json.dump(results, open(out, "w"), indent=2)
    if failures:
        raise SystemExit("; ".join(failures))
    print("CHECK_VL_DONE", flush=True)


if __name__ == "__main__":
    main()
