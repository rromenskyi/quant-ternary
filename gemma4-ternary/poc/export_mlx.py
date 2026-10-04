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
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open
from safetensors.numpy import save_file


def pack(w: torch.Tensor, group: int):
    out_f, in_f = w.shape
    g = w.float().view(out_f, in_f // group, group)
    s = g.abs().mean(-1, keepdim=True).clamp_min(1e-8)
    codes = ((g / s).round().clamp(-1, 1) + 1).to(torch.int64).view(out_f, in_f // 16, 16)
    words = (codes << (2 * torch.arange(16, dtype=torch.int64))).sum(-1).to(torch.uint32)
    sb = s.squeeze(-1).to(torch.bfloat16)
    as_np = lambda t: t.view(torch.uint16).numpy().view(np.uint16)  # bf16 bits
    return words.numpy().astype(np.uint32), as_np(sb), as_np(-sb)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--group", type=int, default=128)
    args = ap.parse_args()

    tensors, n = {}, 0
    for f in sorted(Path(args.ckpt).glob("part_*.safetensors")):
        with safe_open(str(f), framework="pt", device="cpu") as sf:
            for key in sf.keys():
                if not key.startswith("p."):
                    continue
                name = key[2:].removesuffix(".weight")
                w, sc, bi = pack(sf.get_tensor(key), args.group)
                tensors[f"{name}.weight"] = w
                tensors[f"{name}.scales"] = sc
                tensors[f"{name}.biases"] = bi
                n += 1
    # bf16 travels as uint16 bits; splice_mlx.py reinterprets them
    meta = {"format": "ternary-mlx-2bit", "group_size": str(args.group), "bf16_as_uint16": "scales,biases",
            "ckpt": str(Path(args.ckpt).resolve())}
    save_file(tensors, args.out, metadata=meta)
    print(json.dumps({"linears": n, "out": args.out, "bytes": Path(args.out).stat().st_size}))


if __name__ == "__main__":
    main()
