"""Reconstruction metrics for a quantized weight matrix."""

from __future__ import annotations

import torch


def cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    a_flat, b_flat = a.reshape(-1).double(), b.reshape(-1).double()
    denom = a_flat.norm() * b_flat.norm()
    if denom == 0:
        return float("nan")
    return (a_flat @ b_flat / denom).item()


def evaluate(W: torch.Tensor, W_hat: torch.Tensor, X: torch.Tensor) -> dict:
    """W, W_hat: [out, in]. X: [n_samples, in]. Y = X @ W.T."""
    Y = (X.double() @ W.double().t())
    Y_hat = (X.double() @ W_hat.double().t())

    diff_w = (W.double() - W_hat.double())
    diff_y = (Y - Y_hat)

    return {
        "weight_cosine": cosine(W, W_hat),
        "weight_mse": diff_w.pow(2).mean().item(),
        "weight_max_abs_err": diff_w.abs().max().item(),
        "act_cosine": cosine(Y, Y_hat),
        "act_mse": diff_y.pow(2).mean().item(),
        "has_nan_or_inf": bool(torch.isnan(W_hat).any() or torch.isinf(W_hat).any()),
    }
