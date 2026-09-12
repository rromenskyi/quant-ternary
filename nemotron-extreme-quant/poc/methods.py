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

from gptq import gptq_binary, gptq_binary_batched, gptq_ternary, gptq_ternary_batched, hessian_diag
from quantize import binary_bpw, residual_bpw_overhead, ternary_bpw
from rotation import rotate, unrotate


def _select_salient_mask(
    W: torch.Tensor, X: torch.Tensor, fraction: float, criterion: str, percdamp: float = 0.01
) -> torch.Tensor:
    if criterion == "magnitude":
        score = W.abs()
    elif criterion == "activation_weighted":
        col_scale = X.abs().mean(dim=0)
        score = W.abs() * col_scale.unsqueeze(0)
    elif criterion == "hessian":
        # BiLLM/PB-LLM/OBD-style sensitivity: H_ii * w_ij^2 — the same Hessian
        # GPTQ's error compensation uses, rather than a raw activation-magnitude
        # proxy. Reference: bonsai-1bit-repro's billm_1bit.py.
        h_diag = hessian_diag(X, W.shape[1], percdamp)
        score = (W.float() ** 2) * h_diag.unsqueeze(0)
    else:
        raise ValueError(f"unknown salient criterion: {criterion}")
    k = max(1, int(fraction * W.numel()))
    # A threshold-based mask (score >= kth-largest value) can select *more*
    # than k entries when multiple weights tie at the threshold -- common
    # enough with float16/rounded activation-weighted scores to matter.
    # Index into topk directly instead, which always returns exactly k
    # indices (breaking ties by position, not value) -- callers that need a
    # fixed, exact-k salient count per tensor (e.g. poc/pack_mlx.py's MLX
    # packer) depend on this being exact, not "k or more".
    flat_idx = torch.topk(score.reshape(-1), k, largest=True).indices
    mask = torch.zeros_like(score, dtype=torch.bool).reshape(-1)
    mask[flat_idx] = True
    return mask.reshape(score.shape)


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
    salient_mask = _select_salient_mask(W_rot, X_rot, salient_fraction, criterion, percdamp)

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


def rot_gptq_salient_batched(
    W: torch.Tensor,
    X_list: list[torch.Tensor],
    group_size: int = 128,
    salient_fraction: float = 0.03,
    criterion: str = "activation_weighted",
    percdamp: float = 0.01,
    device: str = "cpu",
    method: str = "binary",
    progress_label: str | None = None,
) -> dict:
    """Same recipe as rot_gptq_salient, batched across E same-shaped experts
    (NemotronH's routed experts store weights as one [E, out, in] tensor).
    X_list holds one [n_e, in] calibration tensor per expert — token counts
    are allowed to differ; each is zero-padded up to n_max before the
    batched Hessian, which is exact (zero rows don't perturb X^T X), not an
    approximation. See gptq.py's batched section for why this is faster.
    """
    E, out_features, in_features = W.shape
    W_rot = rotate(W, group_size)  # [E, out, padded_in]
    X_rot_list = [rotate(x, group_size) for x in X_list]
    padded_in = W_rot.shape[-1]

    salient_mask = torch.stack(
        [
            _select_salient_mask(W_rot[e], X_rot_list[e], salient_fraction, criterion, percdamp)
            for e in range(E)
        ],
        dim=0,
    )

    n_max = max(x.shape[0] for x in X_rot_list)
    Xp = torch.zeros(E, n_max, padded_in, dtype=W_rot.dtype)
    for e, x in enumerate(X_rot_list):
        Xp[e, : x.shape[0]] = x

    if method == "binary":
        result = gptq_binary_batched(
            W_rot, Xp, group_size=group_size, percdamp=percdamp, device=device,
            salient_mask=salient_mask, progress_label=progress_label,
        )
        base_bpw = binary_bpw(group_size)
    elif method == "ternary":
        result = gptq_ternary_batched(
            W_rot, Xp, group_size=group_size, percdamp=percdamp, device=device,
            salient_mask=salient_mask, progress_label=progress_label,
        )
        base_bpw = ternary_bpw(group_size)
    else:
        raise ValueError(f"unknown method: {method}")

    W_hat = unrotate(result["W_hat"], group_size, in_features)
    return {
        "W_hat": W_hat,
        "bits_per_weight": base_bpw + residual_bpw_overhead(salient_fraction, in_features),
        "method": f"rot_gptq_{method}_g{group_size}_salient{salient_fraction:.3g}_{criterion}_batched",
    }


def rot_salient_ternary_batched_packable(
    W: torch.Tensor,
    X_list: list[torch.Tensor],
    group_size: int = 64,
    salient_fraction: float = 0.03,
    criterion: str = "activation_weighted",
    percdamp: float = 0.01,
    device: str = "cpu",
    progress_label: str | None = None,
) -> dict:
    """Same rotate + GPTQ + salient-pinning recipe as rot_gptq_salient_batched,
    but skips the final unrotate() and also returns the salient_mask (both in
    the *rotated* domain, at the padded width) instead of collapsing
    everything into one dense W_hat.

    rot_gptq_salient_batched's W_hat is only usable as a drop-in nn.Linear
    replacement (a plain matmul with it reproduces the quantized behavior),
    because unrotate() turns the sparse {ternary, salient} structure into a
    dense linear combination -- every entry becomes a mix of many original
    values, not literally one of {-scale, 0, +scale} or a pinned original.
    A real inference-time packer needs the *rotated*, still-sparse form:
    apply the same rotation to activations at inference time (see
    rotation_mlx.py), matmul against the packed ternary weight, then
    unrotate the *output* instead -- mathematically identical (rotation is
    linear and self-inverse), but only this ordering keeps the weight
    literally packable into a fixed low-bit format plus a small sparse
    salient overlay.
    """
    E, out_features, in_features = W.shape
    W_rot = rotate(W, group_size)  # [E, out, padded_in]
    X_rot_list = [rotate(x, group_size) for x in X_list]
    padded_in = W_rot.shape[-1]

    salient_mask = torch.stack(
        [
            _select_salient_mask(W_rot[e], X_rot_list[e], salient_fraction, criterion, percdamp)
            for e in range(E)
        ],
        dim=0,
    )

    n_max = max(x.shape[0] for x in X_rot_list)
    Xp = torch.zeros(E, n_max, padded_in, dtype=W_rot.dtype)
    for e, x in enumerate(X_rot_list):
        Xp[e, : x.shape[0]] = x

    result = gptq_ternary_batched(
        W_rot, Xp, group_size=group_size, percdamp=percdamp, device=device,
        salient_mask=salient_mask, progress_label=progress_label,
    )

    return {
        "W_hat_rotated": result["W_hat"],  # [E, out, padded_in], rotated domain, NOT unrotated
        "salient_mask": salient_mask,  # [E, out, padded_in], same domain/shape
        "padded_in": padded_in,
        "in_features": in_features,
        "bits_per_weight": ternary_bpw(group_size) + residual_bpw_overhead(salient_fraction, in_features),
        "method": f"rot_salient_ternary_g{group_size}_salient{salient_fraction:.3g}_{criterion}_batched_packable",
    }
