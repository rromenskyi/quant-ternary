"""Block-diagonal fast Walsh-Hadamard rotation (QuIP/QuaRot-style incoherence
processing), applied via the O(n log n) butterfly algorithm rather than a
materialized dense rotation matrix.

Applying a fixed orthogonal rotation R to the input-feature axis of a linear
layer doesn't change what it computes (x @ W.T == (x @ R) @ (W @ R).T when R
is orthogonal), but it spreads outlier weight/activation magnitudes across
each block, which makes the uniform group-wise scale used by binary/ternary
quantization fit much better. Since R is a *fixed*, publicly-known transform
(no learned parameters), it costs nothing to store — both sides just need to
agree to apply it.

We rotate in block_size chunks (a power of two, matched to the quantization
group size) rather than across the full `in_features` width, since that
width is not always a power of two. Blocks are independent, so this is a
block-diagonal Hadamard matrix — but we never materialize it: an
`in_features x in_features` dense matrix would be `4 * in_features**2`
bytes (>500MB for a 12k-wide layer), while the butterfly transform below
uses O(in_features) memory.
"""

from __future__ import annotations

import torch


def _fwht_blocks(x: torch.Tensor, block_size: int) -> torch.Tensor:
    """In-place-style fast Walsh-Hadamard transform, applied independently to
    each contiguous block of size `block_size` along the last dimension.
    `x.shape[-1]` must already be a multiple of `block_size`, and
    `block_size` must be a power of two. Orthonormal (matches H/sqrt(n)).
    """
    if block_size & (block_size - 1) != 0:
        raise ValueError(f"block_size must be a power of two, got {block_size}")
    orig_shape = x.shape
    n = orig_shape[-1]
    num_blocks = n // block_size
    y = x.reshape(-1, num_blocks, block_size).clone()

    h = 1
    while h < block_size:
        y = y.reshape(y.shape[0], num_blocks, block_size // (2 * h), 2, h)
        a = y[:, :, :, 0, :]
        b = y[:, :, :, 1, :]
        y = torch.cat([a + b, a - b], dim=-1)
        y = y.reshape(y.shape[0], num_blocks, block_size)
        h *= 2

    y = y * (block_size ** -0.5)
    return y.reshape(orig_shape)


def rotate(M: torch.Tensor, block_size: int = 128) -> torch.Tensor:
    """Rotate the last dimension of M in independent Hadamard blocks,
    padding with zeros up to a multiple of block_size first.

    Self-inverse: rotate(rotate(M, bs), bs) == M (up to the same padding),
    since each block's Hadamard transform is its own inverse when
    orthonormal.
    """
    last = M.shape[-1]
    pad = (-last) % block_size
    if pad:
        M = torch.nn.functional.pad(M, (0, pad))
    return _fwht_blocks(M, block_size)


def unrotate(M_rot: torch.Tensor, block_size: int, original_size: int) -> torch.Tensor:
    """Inverse of rotate(), truncated back to the original last-dim width."""
    return _fwht_blocks(M_rot, block_size)[..., :original_size]


def with_rotation(quantize_fn, W: torch.Tensor, X: torch.Tensor | None, block_size: int = 128, **kwargs) -> dict:
    """Wrap any quantize_fn(W, [X,] **kwargs) -> {"W_hat", "bits_per_weight", "method"}
    to operate in the rotated domain, then un-rotate the result back.
    """
    in_features = W.shape[1]
    W_rot = rotate(W, block_size)
    if X is not None:
        X_rot = rotate(X, block_size)
        result = quantize_fn(W_rot, X_rot, **kwargs)
    else:
        result = quantize_fn(W_rot, **kwargs)
    result = dict(result)
    result["W_hat"] = unrotate(result["W_hat"], block_size, in_features)
    result["method"] = f"rot_{result['method']}"
    return result
