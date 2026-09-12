"""MLX port of rotation.py's block-diagonal fast Walsh-Hadamard rotation.

Same butterfly algorithm, same block-size/padding conventions, ported from
torch to mlx.core ops so a custom MLX inference layer can apply the same
fixed, parameter-free rotation used by the PyTorch-side rot_gptq_salient
pipeline (see poc/rotation.py's docstring for why this helps quantization
and why it costs nothing to store). Validate against the torch version with
poc/test_rotation_mlx.py before trusting this for real inference.
"""

from __future__ import annotations

import mlx.core as mx


def _fwht_blocks(x: mx.array, block_size: int) -> mx.array:
    if block_size & (block_size - 1) != 0:
        raise ValueError(f"block_size must be a power of two, got {block_size}")
    orig_shape = x.shape
    n = orig_shape[-1]
    num_blocks = n // block_size
    y = x.reshape(-1, num_blocks, block_size)

    h = 1
    while h < block_size:
        y = y.reshape(y.shape[0], num_blocks, block_size // (2 * h), 2, h)
        a = y[:, :, :, 0, :]
        b = y[:, :, :, 1, :]
        y = mx.concatenate([a + b, a - b], axis=-1)
        y = y.reshape(y.shape[0], num_blocks, block_size)
        h *= 2

    y = y * (block_size ** -0.5)
    return y.reshape(orig_shape)


def rotate(M: mx.array, block_size: int = 128) -> mx.array:
    last = M.shape[-1]
    pad = (-last) % block_size
    if pad:
        pad_width = [(0, 0)] * (M.ndim - 1) + [(0, pad)]
        M = mx.pad(M, pad_width)
    return _fwht_blocks(M, block_size)


def unrotate(M_rot: mx.array, block_size: int, original_size: int) -> mx.array:
    return _fwht_blocks(M_rot, block_size)[..., :original_size]
