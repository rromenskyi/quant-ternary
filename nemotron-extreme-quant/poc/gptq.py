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


def _raw_hessian(X: torch.Tensor, in_features: int, percdamp: float) -> torch.Tensor:
    """2 * X^T X + damping, float32. Shared by _compute_hinv and hessian_diag
    so the salience metric and the GPTQ error-compensation math agree on
    what "the Hessian" means for this layer."""
    X = X.float()
    H = 2.0 * (X.t() @ X)
    diag = torch.diagonal(H)
    dead = diag == 0
    if dead.any():
        diag[dead] = 1.0  # padded / never-activated columns: treat as identity
    damp = percdamp * diag.mean().clamp_min(1e-8)
    diag += damp
    return H


def hessian_diag(X: torch.Tensor, in_features: int, percdamp: float = 0.01) -> torch.Tensor:
    """diag(H) per input column — the Optimal-Brain-Damage-style sensitivity
    weight used by BiLLM/PB-LLM salience (H_ii * w_ij^2), exposed so
    methods.py can select salient weights with the same Hessian GPTQ uses,
    instead of the cruder activation-magnitude heuristic.
    """
    return torch.diagonal(_raw_hessian(X, in_features, percdamp)).clone()


def _compute_hinv(X: torch.Tensor, in_features: int, percdamp: float) -> torch.Tensor:
    """Upper-triangular Cholesky factor of (X^T X + damping)^-1, float32."""
    H = _raw_hessian(X, in_features, percdamp)
    L = torch.linalg.cholesky(H)
    # torch.cholesky_inverse (LAPACK potri) measured 24s+ on a 2688x2688 matrix
    # regardless of conditioning; cholesky_solve against the identity uses the
    # same factor L but stays ~30x faster (0.7s) on this hardware's BLAS.
    Hinv = torch.cholesky_solve(torch.eye(L.shape[0], dtype=L.dtype, device=L.device), L)
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
    scale_mode: str = "mean_abs",
    return_scale: bool = False,
):
    """decide(w_col, scale_col, salient_col) -> q_col, all [out_features].

    salient_mask, if given, is a boolean [out_features, in_features] mask:
    True where that weight should be pinned at full precision instead of
    quantized. Pinning is implemented inside `decide` (q = w there), which
    makes that entry's error exactly zero — it consumes none of the block's
    error-compensation budget and propagates nothing onto later columns.

    scale_mode: "mean_abs" (binary/ternary's magnitude-heuristic scale) or
    "max_abs" (standard symmetric-uniform-quant convention, e.g. Q8_0-style
    N-bit quantization, where scale = absmax / (2^(bits-1) - 1)).
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

    W_groups = W.reshape(out_features, num_groups, group_size)
    if scale_mode == "max_abs":
        scale = W_groups.abs().amax(dim=-1)  # [out, num_groups]
    else:
        scale = W_groups.abs().mean(dim=-1)  # [out, num_groups]
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

    Q = Q[:, :in_features].to("cpu")
    if return_scale:
        return Q, scale.to("cpu")  # [out_features, num_groups], the *original* per-group scale
    return Q


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


def gptq_nbit(
    W: torch.Tensor,
    X: torch.Tensor,
    bits: int,
    group_size: int = 32,
    percdamp: float = 0.01,
    device: str = "cpu",
    salient_mask: torch.Tensor | None = None,
) -> dict:
    """Standard symmetric uniform quantization (Q8_0/Q4_0-style: scale =
    absmax / (2^(bits-1) - 1), 2^bits evenly-spaced levels) with GPTQ's
    Hessian-based error compensation, instead of binary/ternary's sign-only
    decision — for exploring whether a structurally-critical tensor (e.g.
    Mamba's SSM projections) needs more than 1-2 bits to stay stable, without
    the salient-pinning mechanism that a fixed-format GGUF export can't use.
    """
    levels = 2 ** (bits - 1) - 1

    def decide(w, s, col_mask):
        step = (s / levels).clamp_min(1e-12)
        code = torch.clamp(torch.round(w / step), -levels, levels)
        q = code * step
        return torch.where(col_mask, w, q) if col_mask is not None else q

    W_hat, scale = _gptq_run(
        W, X, group_size, percdamp, decide, device, salient_mask, scale_mode="max_abs", return_scale=True
    )
    return {
        "W_hat": W_hat,
        "scale": scale,  # [out_features, num_groups]; step = scale / (2**(bits-1) - 1), code = round(W_hat / step)
        "bits_per_weight": bits + 16.0 / group_size,
        "method": f"gptq_{bits}bit_g{group_size}",
    }


# --- Batched-across-experts variants -----------------------------------
#
# Same math as above, but with a leading [E] (expert) dimension threaded
# through every op. The single-expert path's Python column loop launches
# 2688 tiny CUDA kernels *per expert* (128 experts -> ~344k launches for a
# 2688-wide layer); batching collapses that to one loop of 2688 iterations
# where each iteration's kernel processes all E experts at once. Requires
# every expert's W to share the same shape (true for NemotronH's routed
# experts) and X to be zero-padded to a common sample count (zero rows
# don't perturb X^T X, so this is exact, not an approximation).


def _raw_hessian_batched(Xp: torch.Tensor, percdamp: float) -> torch.Tensor:
    """Batched version of _raw_hessian. Xp: [E, n_max, in_features],
    zero-padded per expert along the sample axis."""
    Xp = Xp.float()
    H = 2.0 * torch.bmm(Xp.transpose(1, 2), Xp)  # [E, d, d]
    diag = torch.diagonal(H, dim1=-2, dim2=-1)  # writable view, [E, d]
    dead = diag == 0
    diag[dead] = 1.0
    damp = percdamp * diag.mean(dim=-1, keepdim=True).clamp_min(1e-8)
    diag += damp
    return H


def _compute_hinv_batched(Xp: torch.Tensor, percdamp: float) -> torch.Tensor:
    H = _raw_hessian_batched(Xp, percdamp)
    L = torch.linalg.cholesky(H)
    E, d, _ = H.shape
    eye = torch.eye(d, dtype=L.dtype, device=L.device).expand(E, d, d)
    Hinv = torch.cholesky_solve(eye, L)
    Hinv = torch.linalg.cholesky(Hinv, upper=True)
    return Hinv


def _gptq_run_batched(
    W: torch.Tensor,
    Xp: torch.Tensor,
    group_size: int,
    percdamp: float,
    decide,
    device: str = "cpu",
    salient_mask: torch.Tensor | None = None,
    progress_label: str | None = None,
    scale_mode: str = "mean_abs",
    return_scale: bool = False,
):
    """Batched analogue of _gptq_run. W: [E, out_features, in_features],
    Xp: [E, n_max, in_features] zero-padded. decide(w, s, col_mask) operates
    on [E, out_features] tensors. Returns Q: [E, out_features, in_features].
    """
    import time

    W, Xp = W.to(device), Xp.to(device)
    E, out_features, in_features = W.shape
    pad = (-in_features) % group_size
    if pad:
        W = F.pad(W, (0, pad))
        Xp = F.pad(Xp, (0, pad))
        if salient_mask is not None:
            salient_mask = F.pad(salient_mask, (0, pad))
    padded_in = W.shape[-1]
    num_groups = padded_in // group_size

    W = W.clone().float()
    if salient_mask is not None:
        salient_mask = salient_mask.to(device)
    Hinv = _compute_hinv_batched(Xp, percdamp)  # [E, padded_in, padded_in]

    W_groups_init = W.reshape(E, out_features, num_groups, group_size)
    if scale_mode == "max_abs":
        scale = W_groups_init.abs().amax(dim=-1)  # [E, out, num_groups]
    else:
        scale = W_groups_init.abs().mean(dim=-1)  # [E, out, num_groups]
    scale_orig = scale.clone()
    Q = torch.zeros_like(W)

    t0 = time.time()
    for g in range(num_groups):
        start, end = g * group_size, (g + 1) * group_size
        W_block = W[:, :, start:end].clone()  # [E, out, gs]
        Hinv_block = Hinv[:, start:end, start:end]  # [E, gs, gs]
        Err_block = torch.zeros_like(W_block)
        s = scale[:, :, g]  # [E, out]

        for i in range(group_size):
            w = W_block[:, :, i]  # [E, out]
            d = Hinv_block[:, i, i].clamp_min(1e-8)  # [E]
            col_mask = salient_mask[:, :, start + i] if salient_mask is not None else None
            q = decide(w, s, col_mask)
            Q[:, :, start + i] = q
            err = (w - q) / d.unsqueeze(-1)  # [E, out]
            Err_block[:, :, i] = err
            if i < group_size - 1:
                W_block[:, :, i + 1 :] -= err.unsqueeze(-1) * Hinv_block[:, i, i + 1 :].unsqueeze(1)

        if end < padded_in:
            W[:, :, end:] -= torch.bmm(Err_block, Hinv[:, start:end, end:])

        if progress_label:
            print(
                f"  [{progress_label}] group {g}/{num_groups - 1} done, "
                f"elapsed {time.time() - t0:.1f}s",
                flush=True,
            )

    Q = Q[:, :, :in_features].to("cpu")
    if return_scale:
        return Q, scale_orig.to("cpu")  # [E, out_features, num_groups], the *original* per-group scale
    return Q


def gptq_binary_batched(
    W: torch.Tensor,
    Xp: torch.Tensor,
    group_size: int = 128,
    percdamp: float = 0.01,
    device: str = "cpu",
    salient_mask: torch.Tensor | None = None,
    progress_label: str | None = None,
) -> dict:
    def decide(w, s, col_mask):
        b = torch.sign(w)
        b[b == 0] = 1.0
        q = s * b
        return torch.where(col_mask, w, q) if col_mask is not None else q

    W_hat = _gptq_run_batched(W, Xp, group_size, percdamp, decide, device, salient_mask, progress_label)
    return {"W_hat": W_hat, "bits_per_weight": binary_bpw(group_size), "method": f"gptq_binary_batched_g{group_size}"}


def gptq_ternary_batched(
    W: torch.Tensor,
    Xp: torch.Tensor,
    group_size: int = 128,
    percdamp: float = 0.01,
    threshold_factor: float = 0.7,
    device: str = "cpu",
    salient_mask: torch.Tensor | None = None,
    progress_label: str | None = None,
) -> dict:
    def decide(w, s, col_mask):
        threshold = threshold_factor * s
        mask = w.abs() > threshold
        b = torch.sign(w)
        q = torch.where(mask, s * b, torch.zeros_like(w))
        return torch.where(col_mask, w, q) if col_mask is not None else q

    W_hat = _gptq_run_batched(W, Xp, group_size, percdamp, decide, device, salient_mask, progress_label)
    return {
        "W_hat": W_hat,
        "bits_per_weight": ternary_bpw(group_size),
        "method": f"gptq_ternary_batched_g{group_size}",
    }


def gptq_nbit_batched(
    W: torch.Tensor,
    Xp: torch.Tensor,
    bits: int,
    group_size: int = 32,
    percdamp: float = 0.01,
    device: str = "cpu",
    salient_mask: torch.Tensor | None = None,
    progress_label: str | None = None,
) -> dict:
    """Batched analogue of gptq_nbit -- see that function's docstring.
    W: [E, out_features, in_features], Xp: [E, n_max, in_features]
    zero-padded per expert. Used for this project's routed MoE experts
    (128 same-shaped experts per block), where the single-expert path's
    Python column loop would otherwise launch group_size iterations *per
    expert* -- batching collapses that to one loop of group_size iterations
    where each iteration's kernel processes all E experts at once.
    """
    levels = 2 ** (bits - 1) - 1

    def decide(w, s, col_mask):
        step = (s / levels).clamp_min(1e-12)
        code = torch.clamp(torch.round(w / step), -levels, levels)
        q = code * step
        return torch.where(col_mask, w, q) if col_mask is not None else q

    W_hat, scale = _gptq_run_batched(
        W, Xp, group_size, percdamp, decide, device, salient_mask, progress_label,
        scale_mode="max_abs", return_scale=True,
    )
    return {
        "W_hat": W_hat,
        "scale": scale,  # [E, out_features, num_groups]; step = scale / (2**(bits-1) - 1)
        "bits_per_weight": bits + 16.0 / group_size,
        "method": f"gptq_{bits}bit_g{group_size}_batched",
    }
