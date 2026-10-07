"""Ternary checkpoint -> MLX 2-bit tensors (stock format, no custom kernel).

    python export_mlx.py --ckpt run/ckpt/step_000871 --out ternary_step871.safetensors --group 128

Reads only the latent weights (p.*) of a train_ternary.py checkpoint, one
tensor at a time on the CPU (safe next to a running training job), and
writes for every ternary Linear the three MLX quantized tensors:

  <name>.weight  uint32 [out, in/16]   codes {0,1,2}, code k of a word at bits 2k
  <name>.scales  bf16   [out, in/group] s = mean|w| of the group
  <name>.biases  bf16   [out, in/group] -s

so that dequantize = code * s - s = {-s, 0, +s}, the training projection
(ternary_lib.ternary_q; codes from the fp32 s, as in training, stored
scale rounded to bf16). splice_mlx.py drops these into an MLX model.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open
from safetensors.numpy import save_file


def _words(codes: torch.Tensor, bits: int) -> np.ndarray:
    """MLX packing: 32/bits codes per uint32, code k at bits k*bits."""
    out_f, in_f = codes.shape
    per = 32 // bits
    c = codes.to(torch.int64).view(out_f, in_f // per, per)
    return (c << (bits * torch.arange(per, dtype=torch.int64))).sum(-1).to(torch.uint32).numpy().astype(np.uint32)


as_np = lambda t: t.view(torch.uint16).numpy().view(np.uint16)  # bf16 bits


def pack(w: torch.Tensor, group: int, alpha: torch.Tensor | None = None):
    """Ternary on the 2-bit grid: codes {0,1,2}, scale s, bias -s; s =
    mean|w| per group, times exp(alpha) if the scales were learned."""
    out_f, in_f = w.shape
    g = w.float().view(out_f, in_f // group, group)
    s = g.abs().mean(-1, keepdim=True)
    if alpha is not None:
        s = s * alpha.float().exp()[..., None]
    s = s.clamp_min(1e-8)
    codes = ((g / s).round().clamp(-1, 1) + 1).view(out_f, in_f)
    sb = s.squeeze(-1).to(torch.bfloat16)
    return _words(codes, 2), as_np(sb), as_np(-sb)


def pack_affine(w: torch.Tensor, bits: int, group: int):
    """MLX affine n-bit, the grid ternary_lib.affine_q trains on:
    scale = (max - min) / (2^bits - 1), bias = min."""
    out_f, in_f = w.shape
    g = w.float().view(out_f, in_f // group, group)
    lo, hi = g.amin(-1, keepdim=True), g.amax(-1, keepdim=True)
    s = ((hi - lo) / (2**bits - 1)).clamp_min(1e-8)
    codes = ((g - lo) / s).round().clamp(0, 2**bits - 1).view(out_f, in_f)
    return _words(codes, bits), as_np(s.squeeze(-1).to(torch.bfloat16)), as_np(lo.squeeze(-1).to(torch.bfloat16))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--group", type=int, default=128)
    ap.add_argument("--affine-pattern", help="for checkpoints saved before per-layer grids were recorded: "
                    "regex of the Linears trained on the affine grid (train_ternary.py --affine-pattern)")
    ap.add_argument("--affine-bits", type=int, default=4)
    ap.add_argument("--affine-group", type=int, default=64)
    args = ap.parse_args()

    # each Linear's grid, as trained: from the checkpoint's meta.json
    # ([bits, group], bits 0 = ternary); checkpoints from before that field
    # are ternary everywhere at --group
    meta_f = Path(args.ckpt) / "meta.json"
    quant = json.loads(meta_f.read_text()).get("quant", {}) if meta_f.exists() else {}
    tensors, n, layers = {}, 0, {}
    alphas = {}  # learned scale multipliers (--learn-scale), if any
    for f in sorted(Path(args.ckpt).glob("part_*.safetensors")):
        with safe_open(str(f), framework="pt", device="cpu") as sf:
            for key in sf.keys():
                if key.startswith("p.") and key.endswith(".alpha"):
                    alphas[key[2:].removesuffix(".alpha")] = sf.get_tensor(key)
    for f in sorted(Path(args.ckpt).glob("part_*.safetensors")):
        with safe_open(str(f), framework="pt", device="cpu") as sf:
            for key in sf.keys():
                if not key.startswith("p.") or not key.endswith(".weight"):
                    continue
                name = key[2:].removesuffix(".weight")
                fallback = [0, args.group]
                if args.affine_pattern and re.search(args.affine_pattern, key[2:]):
                    fallback = [args.affine_bits, args.affine_group]
                bits, group = quant.get(key[2:], fallback)
                if bits:
                    w, sc, bi = pack_affine(sf.get_tensor(key), bits, group)
                else:
                    w, sc, bi = pack(sf.get_tensor(key), group, alphas.get(name))
                layers[name] = [bits or 2, group]
                tensors[f"{name}.weight"] = w
                tensors[f"{name}.scales"] = sc
                tensors[f"{name}.biases"] = bi
                n += 1
    # bf16 travels as uint16 bits; splice_mlx.py reinterprets them
    meta = {"format": "ternary-mlx-2bit", "group_size": str(args.group), "bf16_as_uint16": "scales,biases",
            "layers": json.dumps(layers),  # name -> [MLX bits, group]
            "ckpt": str(Path(args.ckpt).resolve())}
    save_file(tensors, args.out, metadata=meta)
    print(json.dumps({"linears": n, "out": args.out, "bytes": Path(args.out).stat().st_size}))


if __name__ == "__main__":
    main()
