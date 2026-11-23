"""Logits of the transformers model (--hf, the source snapshot) against an
MLX model (--mlx: the same snapshot, unquantized) on fixed token ids: the
port check of smoke_test.sh. Prints the max absolute difference, the
cosine and the top-1 agreement per position.

With --quantized (a convert_mlx.py output of --mlx), the weights instead:
every quantized decoder tensor, dequantized, against --mlx's, at the bits
of quant_recipe.json. A tiny random model's logits are no test there: the
same weights with bf16 instead of fp32 activations already flip its routing
(cosine min 0.34).

    python compare_hf_mlx.py --hf /tmp/tiny-mini --mlx /tmp/tiny-mini
    python compare_hf_mlx.py --hf /tmp/ongrid --mlx /tmp/ongrid --quantized /tmp/mlx
"""

from __future__ import annotations

import argparse
import json

import mlx.core as mx
import numpy as np
import torch
from mlx.utils import tree_flatten
from mlx_lm import load
from transformers import AutoModelForImageTextToText


def check_weights(quantized: str, ref_path: str) -> None:
    """Every quantized module of the decoder against the unquantized model:
    off by no more than the bf16 rounding of MLX's scales and biases."""
    q, _ = load(quantized)
    ref, _ = load(ref_path)
    qp, rp = dict(tree_flatten(q.parameters())), dict(tree_flatten(ref.parameters()))
    spec = json.load(open(f"{quantized}/config.json"))["quantization"]
    recipe = json.load(open(f"{quantized}/quant_recipe.json"))["recipe"]
    worst, n = 0.0, 0
    for key in sorted(k for k in qp if k.endswith(".scales") and k.startswith("model.language_model.layers.")):
        path = key[: -len(".scales")]
        s = spec[path]
        want = recipe["attn"] if ".self_attn." in path else recipe["experts_gate_up" if "gate_up" in path else "experts_down"]
        if s["bits"] != want:
            raise SystemExit(f"{path}: {s['bits']} bits, the recipe says {want}")
        w = mx.dequantize(qp[path + ".weight"], qp[key], qp[path + ".biases"], group_size=s["group_size"], bits=s["bits"])
        r = rp[path + ".weight"].astype(mx.float32)
        worst = max(worst, float(mx.abs(w.astype(mx.float32) - r).max() / mx.abs(r).max()))
        n += 1
    print(f"{n} quantized decoder tensors, worst max |diff| / max |w| {worst:.4g}", flush=True)
    if not n or worst > 2**-7:
        raise SystemExit("quantized weights are not the GPTQ grid's")
    print("WEIGHTS_OK", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf", required=True)
    ap.add_argument("--mlx", required=True)
    ap.add_argument("--quantized", help="a convert_mlx.py output of --mlx: compare weights, not logits")
    ap.add_argument("--tokens", type=int, default=96)
    ap.add_argument("--min-cos", type=float, default=0.999)
    ap.add_argument("--min-top1", type=float, default=0.95)
    args = ap.parse_args()
    if args.quantized:
        check_weights(args.quantized, args.mlx)
    ids = np.random.default_rng(0).integers(0, 200000, (1, args.tokens))
    hf = AutoModelForImageTextToText.from_pretrained(args.hf, dtype=torch.float32).eval()
    with torch.no_grad():
        ref = hf(input_ids=torch.from_numpy(ids), use_cache=False).logits[0].float().numpy()
    model, _ = load(args.mlx)
    # fp32 activations, as the reference: a quantized model's scales too.
    model.set_dtype(mx.float32)
    mine = np.array(model(mx.array(ids))[0].astype(mx.float32))
    ref = ref[:, : mine.shape[-1]]
    cos = (ref * mine).sum(-1) / (np.linalg.norm(ref, axis=-1) * np.linalg.norm(mine, axis=-1))
    top1 = float((ref.argmax(-1) == mine.argmax(-1)).mean())
    print(f"max |diff| {np.abs(ref - mine).max():.4g}, cosine min {cos.min():.6f} mean {cos.mean():.6f}, "
          f"top-1 {top1:.3f}", flush=True)
    if cos.min() < args.min_cos or top1 < args.min_top1:
        raise SystemExit(f"MLX differs from transformers: cosine min {cos.min():.6f}, top-1 {top1:.3f}")
    print("COMPARE_DONE", flush=True)


if __name__ == "__main__":
    main()
