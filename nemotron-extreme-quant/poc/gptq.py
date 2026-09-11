"""GPTQ-style blockwise quantization with Hessian-based error compensation
(Frantar et al., "GPTQ: Accurate Post-Training Quantization for Generative
Pre-trained Transformers", 2022 — public method, reimplemented here against
the binary/ternary quantizers instead of GPTQ's original uniform-integer
grid).

Unlike naive/activation-aware quantization, which picks each weight
independently, GPTQ quantizes one column at a time and immediately spreads
the rounding error of that column onto the *not-yet-quantized* columns,
weighted by the calibration-activation Hessian. Columns are grouped into
blocks (`group_size`, also used for the per-neuron scale) so that most of
the error propagation is done as a handful of full matrix multiplies
instead of one Python-level step per column.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from quantize import binary_bpw, ternary_bpw


def _compute_hinv(X: torch.Tensor, in_features: int, percdamp: float) -> torch.Tensor:
    """Upper-triangular Cholesky factor of (X^T X + damping)^-1, float32."""
    X = X.float()
    H = 2.0 * (X.t() @ X)
    diag = torch.diagonal(H)
    dead = diag == 0
    if dead.any():
        diag[dead] = 1.0  # padded / never-activated columns: treat as identity
    damp = percdamp * diag.mean().clamp_min(1e-8)
    diag += damp
    L = torch.linalg.cholesky(H)
    Hinv = torch.cholesky_inverse(L)
    Hinv = torch.linalg.cholesky(Hinv, upper=True)
    return Hinv


def _gptq_run(
    W: torch.Tensor,
    X: torch.Tensor,
    group_size: int,
    percdamp: float,
    decide,
    device: str = "cpu",
    salient_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """decide(w_col, scale_col, salient_col) -> q_col, all [out_features].

    salient_mask, if given, is a boolean [out_features, in_features] mask:
    True where that weight should be pinned at full precision instead of
    quantized. Pinning is implemented inside `decide` (q = w there), which
    makes that entry's error exactly zero — it consumes none of the block's
    error-compensation budget and propagates nothing onto later columns.
    """
    W, X = W.to(device), X.to(device)
    out_features, in_features = W.shape
    pad = (-in_features) % group_size
    if pad:
        W = F.pad(W, (0, pad))
        X = F.pad(X, (0, pad))
        if salient_mask is not None:
            salient_mask = F.pad(salient_mask, (0, pad))
    padded_in = W.shape[1]
    num_groups = padded_in // group_size

    W = W.clone().float()
    if salient_mask is not None:
        salient_mask = salient_mask.to(device)
    Hinv = _compute_hinv(X, padded_in, percdamp)

    scale = W.reshape(out_features, num_groups, group_size).abs().mean(dim=-1)  # [out, num_groups]
    Q = torch.zeros_like(W)

    for g in range(num_groups):
        start, end = g * group_size, (g + 1) * group_size
        W_block = W[:, start:end].clone()
        Hinv_block = Hinv[start:end, start:end]
        Err_block = torch.zeros_like(W_block)
        s = scale[:, g]

        for i in range(group_size):
            w = W_block[:, i]
            d = Hinv_block[i, i].clamp_min(1e-8)
            col_mask = salient_mask[:, start + i] if salient_mask is not None else None
            q = decide(w, s, col_mask)
            Q[:, start + i] = q
            err = (w - q) / d
            Err_block[:, i] = err
            if i < group_size - 1:
                W_block[:, i + 1 :] -= torch.outer(err, Hinv_block[i, i + 1 :])

        if end < padded_in:
            W[:, end:] -= Err_block @ Hinv[start:end, end:]

    return Q[:, :in_features].to("cpu")


def gptq_binary(
    W: torch.Tensor,
    X: torch.Tensor,
    group_size: int = 128,
    percdamp: float = 0.01,
    device: str = "cpu",
    salient_mask: torch.Tensor | None = None,
) -> dict:
    def decide(w, s, col_mask):
        b = torch.sign(w)
        b[b == 0] = 1.0
        q = s * b
        return torch.where(col_mask, w, q) if col_mask is not None else q

    W_hat = _gptq_run(W, X, group_size, percdamp, decide, device, salient_mask)
    return {"W_hat": W_hat, "bits_per_weight": binary_bpw(group_size), "method": f"gptq_binary_g{group_size}"}


def gptq_ternary(
    W: torch.Tensor,
    X: torch.Tensor,
    group_size: int = 128,
    percdamp: float = 0.01,
    threshold_factor: float = 0.7,
    device: str = "cpu",
    salient_mask: torch.Tensor | None = None,
) -> dict:
    def decide(w, s, col_mask):
        threshold = threshold_factor * s
        mask = w.abs() > threshold
        b = torch.sign(w)
        q = torch.where(mask, s * b, torch.zeros_like(w))
        return torch.where(col_mask, w, q) if col_mask is not None else q

    W_hat = _gptq_run(W, X, group_size, percdamp, decide, device, salient_mask)
    return {"W_hat": W_hat, "bits_per_weight": ternary_bpw(group_size), "method": f"gptq_ternary_g{group_size}"}
