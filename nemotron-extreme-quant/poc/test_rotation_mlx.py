"""Validate rotation_mlx.py against the reference torch implementation in
rotation.py: same inputs must produce numerically matching outputs, and the
MLX version must round-trip (rotate then unrotate recovers the original).

Usage:
    python poc/test_rotation_mlx.py
"""

from __future__ import annotations

import numpy as np
import torch

import rotation as rotation_torch
import rotation_mlx
import mlx.core as mx


def check(name: str, a: np.ndarray, b: np.ndarray, atol: float = 1e-4):
    diff = np.abs(a - b).max()
    status = "OK" if diff < atol else "FAIL"
    print(f"[{status}] {name}: max abs diff = {diff:.6g}")
    if status == "FAIL":
        raise SystemExit(1)


def main():
    torch.manual_seed(0)
    np.random.seed(0)

    shapes_and_blocks = [
        ((8, 128), 128),
        ((4, 2688), 128),
        ((4, 2688), 64),
        ((16, 1856), 64),  # not a multiple of 64 -> exercises padding path
        ((3, 5, 2688), 128),  # 3D input, like a batch of expert activations
    ]

    for shape, block_size in shapes_and_blocks:
        W = torch.randn(*shape, dtype=torch.float32)
        W_np = W.numpy()
        W_mx = mx.array(W_np)

        rot_t = rotation_torch.rotate(W, block_size).numpy()
        rot_m = np.array(rotation_mlx.rotate(W_mx, block_size))
        check(f"rotate {shape} bs={block_size}", rot_t, rot_m)

        unrot_t = rotation_torch.unrotate(torch.from_numpy(rot_t), block_size, shape[-1]).numpy()
        unrot_m = np.array(rotation_mlx.unrotate(mx.array(rot_m), block_size, shape[-1]))
        check(f"unrotate {shape} bs={block_size}", unrot_t, unrot_m)

        check(f"round-trip recovers original {shape} bs={block_size}", W_np, unrot_m, atol=1e-3)

    print("\nAll rotation_mlx checks passed.")


if __name__ == "__main__":
    main()
