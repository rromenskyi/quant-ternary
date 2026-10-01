"""Which tensors of a Gemma 4 `-qat-q4_0-unquantized` checkpoint sit exactly
on llama.cpp's q4_0 grid.

q4_0 (ggml-quants.c, quantize_row_q4_0_ref): blocks of 32 along a row; d =
(the block's value with the largest magnitude) / -8; q = clamp(floor(x/d +
8.5), 0, 15); x' = d * (q - 8). Google's "unquantized" QAT release is the
QAT weights dequantized to bf16, so a tensor that was QAT-trained on that
grid round-trips to itself; one that wasn't (norms, embeddings kept at
higher precision, the vision/audio towers) doesn't.

  python qat_grid_check.py <checkpoint dir> [--json out.json]

Prints, per tensor family, how many blocks round-trip exactly.
"""
import argparse
import collections
import glob
import json
import os
import re

import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
from safetensors import safe_open


def q4_0_roundtrip(w: torch.Tensor):
    """(exact-block fraction, max abs error) of `w` through q4_0, by rows."""
    rows, cols = w.shape
    x = w.to(DEVICE).float().reshape(rows, cols // 32, 32)
    idx = x.abs().argmax(dim=-1, keepdim=True)
    mx = torch.gather(x, -1, idx)
    d = mx / -8.0
    safe = torch.where(d == 0, torch.ones_like(d), d)
    q = torch.clamp(torch.floor(x / safe + 8.5), 0, 15)
    back = torch.where(d == 0, torch.zeros_like(x), (q - 8) * d)
    # Compare in bf16, the checkpoint's own precision.
    err = (back.to(w.dtype).float() - x).abs()
    exact = (err.amax(dim=-1) == 0).float().mean().item()
    return exact, err.max().item()


def family(name: str) -> str:
    return re.sub(r"\.\d+\.", ".N.", name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint")
    ap.add_argument("--json")
    args = ap.parse_args()
    stats = collections.defaultdict(lambda: [0, 0.0, 0.0, 0])  # tensors, exact sum, max err, skipped
    per_tensor = {}
    for path in sorted(glob.glob(os.path.join(args.checkpoint, "*.safetensors"))):
        with safe_open(path, "pt") as f:
            for name in f.keys():
                t = f.get_tensor(name)
                fam = family(name)
                if t.ndim != 2 or t.shape[1] % 32 != 0 or not t.is_floating_point():
                    stats[fam][3] += 1
                    continue
                exact, err = q4_0_roundtrip(t)
                s = stats[fam]
                s[0] += 1
                s[1] += exact
                s[2] = max(s[2], err)
                per_tensor[name] = {"shape": list(t.shape), "dtype": str(t.dtype), "exact": exact, "max_err": err}
    print(f"{'family':72} {'n':>4} {'exact blocks':>12} {'max err':>10}")
    for fam in sorted(stats):
        n, ex, err, skipped = stats[fam]
        if n:
            print(f"{fam:72} {n:4d} {ex / n:12.4%} {err:10.3g}")
        else:
            print(f"{fam:72} {'-':>4} {'(not 2-D/32)':>12} {skipped:>10}")
    if args.json:
        json.dump(per_tensor, open(args.json, "w"), indent=1)


if __name__ == "__main__":
    main()
