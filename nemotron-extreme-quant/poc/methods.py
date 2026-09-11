"""Composed quantization methods: rotation + GPTQ + salient-weight pinning.

Matches the "QuIP + salient" recipe from the bonsai-1bit-repro reproduction
study (incoherence rotation + GPTQ error compensation + a small fraction of
the highest-importance weights kept at full precision) — the best public-
method result in that study got within 1.1x perplexity of FP16 at 1.571 bpw,
versus 2.1x at pure 1.125 bpw without salient handling.

Salience is selected in the *rotated* domain (the representation GPTQ
actually quantizes), not the original weight space, and salient positions
are pinned *before* GPTQ runs — see gptq.py's salient_mask — rather than
patched on afterward. An earlier attempt at post-hoc residual correction
(patch the largest errors after GPTQ finishes) measurably hurt activation
cosine: GPTQ had already spent its error-compensation budget assuming those
positions would stay wrong in a particular way, and overwriting them broke
that balance instead of improving it.
"""

from __future__ import annotations

import torch

from gptq import gptq_binary, gptq_ternary
from quantize import binary_bpw, residual_bpw_overhead, ternary_bpw
from rotation import rotate, unrotate


def _select_salient_mask(W: torch.Tensor, X: torch.Tensor, fraction: float, criterion: str) -> torch.Tensor:
    if criterion == "magnitude":
        score = W.abs()
    elif criterion == "activation_weighted":
        col_scale = X.abs().mean(dim=0)
        score = W.abs() * col_scale.unsqueeze(0)
    else:
        raise ValueError(f"unknown salient criterion: {criterion}")
    k = max(1, int(fraction * W.numel()))
    threshold = torch.topk(score.reshape(-1), k, largest=True).values.min()
    return score >= threshold


def rot_gptq_salient(
    W: torch.Tensor,
    X: torch.Tensor,
    group_size: int = 128,
    salient_fraction: float = 0.03,
    criterion: str = "activation_weighted",
    percdamp: float = 0.01,
    device: str = "cpu",
    method: str = "binary",
) -> dict:
    in_features = W.shape[1]
    W_rot = rotate(W, group_size)
    X_rot = rotate(X, group_size)
    salient_mask = _select_salient_mask(W_rot, X_rot, salient_fraction, criterion)

    if method == "binary":
        result = gptq_binary(
            W_rot, X_rot, group_size=group_size, percdamp=percdamp, device=device, salient_mask=salient_mask
        )
        base_bpw = binary_bpw(group_size)
    elif method == "ternary":
        result = gptq_ternary(
            W_rot, X_rot, group_size=group_size, percdamp=percdamp, device=device, salient_mask=salient_mask
        )
        base_bpw = ternary_bpw(group_size)
    else:
        raise ValueError(f"unknown method: {method}")

    W_hat = unrotate(result["W_hat"], group_size, in_features)
    return {
        "W_hat": W_hat,
        "bits_per_weight": base_bpw + residual_bpw_overhead(salient_fraction, in_features),
        "method": f"rot_gptq_{method}_g{group_size}_salient{salient_fraction:.3g}_{criterion}",
    }
