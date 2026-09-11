"""Standalone extreme-quantization methods for the PoC.

Convention: a weight matrix W has shape [out_features, in_features] (the
`nn.Linear` convention), and activations X have shape [n_samples,
in_features], so the layer computes Y = X @ W.T.

Every method returns a dict with at least:
    - "W_hat": the dequantized weight matrix, same shape/dtype as W
    - "bits_per_weight": effective bpw including scale/residual overhead
    - "method": a short string identifying the method
"""

from __future__ import annotations

import torch


def _reshape_groups(W: torch.Tensor, group_size: int) -> tuple[torch.Tensor, int, int]:
    """Reshape [out, in] into [out, num_groups, group_size], padding `in` if needed."""
    out_features, in_features = W.shape
    pad = (-in_features) % group_size
    if pad:
        W = torch.nn.functional.pad(W, (0, pad))
    num_groups = W.shape[1] // group_size
    return W.reshape(out_features, num_groups, group_size), num_groups, pad


def _unpad(W_full: torch.Tensor, in_features: int) -> torch.Tensor:
    return W_full[:, :in_features]


def binary_bpw(group_size: int) -> float:
    """1 packed bit/weight + one FP16 scale per group."""
    return 1.0 + 16.0 / group_size


def ternary_bpw(group_size: int) -> float:
    """2 packed bits/weight (naive, 4 values/byte) + one FP16 scale per group."""
    return 2.0 + 16.0 / group_size


def residual_bpw_overhead(residual_fraction: float, in_features: int) -> float:
    """Extra bpw from a per-row sparse residual: FP16 value + row-local index."""
    if residual_fraction <= 0:
        return 0.0
    index_bits = max(1, (in_features - 1).bit_length())
    return residual_fraction * (16.0 + index_bits)


def naive_binary(W: torch.Tensor, group_size: int = 128) -> dict:
    """Group-wise sign quantization: scale = mean(|W|), B = sign(W)."""
    in_features = W.shape[1]
    Wg, num_groups, pad = _reshape_groups(W, group_size)
    scale = Wg.abs().mean(dim=-1, keepdim=True)  # [out, num_groups, 1]
    B = torch.sign(Wg)
    B[B == 0] = 1.0
    W_hat = _unpad((scale * B).reshape(W.shape[0], -1), in_features)
    return {
        "W_hat": W_hat,
        "bits_per_weight": binary_bpw(group_size),
        "method": f"naive_binary_g{group_size}",
    }


def activation_aware_binary(W: torch.Tensor, X: torch.Tensor, group_size: int = 128) -> dict:
    """Binary quantization with a per-(neuron, group) scale fit by least squares
    against real calibration activations, instead of a magnitude heuristic.

    Direction is still B = sign(W). For each output neuron o and group g, we
    solve for the scalar s minimizing || X_g @ w_g - s * (X_g @ b_g) ||^2 over
    the calibration set, where w_g/b_g are the group's true/binarized weights.
    """
    in_features = W.shape[1]
    out_features = W.shape[0]
    Wg, num_groups, pad = _reshape_groups(W, group_size)
    if pad:
        X = torch.nn.functional.pad(X, (0, pad))
    Xg = X.reshape(X.shape[0], num_groups, group_size)  # [N, G, gs]

    B = torch.sign(Wg)
    B[B == 0] = 1.0

    # P[g, n, o] = X_g[n] . W_g[o] ; Q[g, n, o] = X_g[n] . B_g[o]
    Xg_t = Xg.permute(1, 0, 2)  # [G, N, gs]
    Wg_t = Wg.permute(1, 2, 0)  # [G, gs, out]
    Bg_t = B.permute(1, 2, 0)  # [G, gs, out]
    P = torch.bmm(Xg_t, Wg_t)  # [G, N, out]
    Q = torch.bmm(Xg_t, Bg_t)  # [G, N, out]

    numer = (P * Q).sum(dim=1)  # [G, out]
    denom = (Q * Q).sum(dim=1).clamp_min(1e-12)  # [G, out]
    scale = (numer / denom).t().unsqueeze(-1)  # [out, G, 1]

    # Fall back to the magnitude heuristic for groups with degenerate activations.
    magnitude_scale = Wg.abs().mean(dim=-1, keepdim=True)
    scale = torch.where(denom.t().unsqueeze(-1) > 1e-10, scale, magnitude_scale)
    scale = scale.abs()

    W_hat = _unpad((scale * B).reshape(out_features, -1), in_features)
    return {
        "W_hat": W_hat,
        "bits_per_weight": binary_bpw(group_size),
        "method": f"act_aware_binary_g{group_size}",
    }


def _select_residual_mask(
    W: torch.Tensor,
    W_hat: torch.Tensor,
    X: torch.Tensor | None,
    fraction: float,
    criterion: str,
) -> torch.Tensor:
    """Boolean mask over W selecting the top `fraction` entries to correct."""
    residual = W - W_hat
    if criterion == "magnitude":
        score = residual.abs()
    elif criterion == "activation_weighted":
        if X is None:
            raise ValueError("activation_weighted residual selection requires X")
        col_scale = X.abs().mean(dim=0)  # [in_features] — typical activation per input dim
        score = residual.abs() * col_scale.unsqueeze(0)
    else:
        raise ValueError(f"unknown residual criterion: {criterion}")

    k = max(1, int(fraction * W.numel()))
    flat = score.reshape(-1)
    threshold = torch.topk(flat, k, largest=True).values.min()
    return score >= threshold


def binary_with_residual(
    W: torch.Tensor,
    X: torch.Tensor,
    group_size: int = 128,
    residual_fraction: float = 0.01,
    criterion: str = "magnitude",
    activation_aware: bool = False,
) -> dict:
    """Binary quantization plus a sparse correction on the largest residual entries."""
    base = activation_aware_binary(W, X, group_size) if activation_aware else naive_binary(W, group_size)
    W_hat = base["W_hat"]
    mask = _select_residual_mask(W, W_hat, X, residual_fraction, criterion)
    W_hat_corrected = torch.where(mask, W, W_hat)
    base_name = "act_aware" if activation_aware else "naive"
    return {
        "W_hat": W_hat_corrected,
        "bits_per_weight": base["bits_per_weight"] + residual_bpw_overhead(residual_fraction, W.shape[1]),
        "method": f"{base_name}_binary_g{group_size}_res{residual_fraction:.3g}_{criterion}",
    }


def ternary_optimized(W: torch.Tensor, group_size: int = 128, threshold_factor: float = 0.7) -> dict:
    """Per-group ternary quantization {-s, 0, +s} using the standard
    threshold-then-mean-magnitude rule (Li et al., Ternary Weight Networks).
    """
    in_features = W.shape[1]
    Wg, num_groups, pad = _reshape_groups(W, group_size)
    threshold = threshold_factor * Wg.abs().mean(dim=-1, keepdim=True)
    mask = Wg.abs() > threshold
    masked_abs = torch.where(mask, Wg.abs(), torch.zeros_like(Wg))
    count = mask.sum(dim=-1, keepdim=True).clamp_min(1)
    scale = masked_abs.sum(dim=-1, keepdim=True) / count
    Wt = scale * torch.sign(Wg) * mask
    W_hat = _unpad(Wt.reshape(W.shape[0], -1), in_features)
    return {
        "W_hat": W_hat,
        "bits_per_weight": ternary_bpw(group_size),
        "method": f"ternary_g{group_size}",
    }
