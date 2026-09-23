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
    """Upper-triangular Cholesky factor of (X^T X + damping)^-1, float32.

    When there are fewer calibration rows than in_features (e.g. a wide
    feed-forward down-proj calibrated on a few thousand diffusion-step
    activations), X^T X is exactly rank-deficient -- adding `percdamp *
    mean(diag)` to every diagonal entry is mathematically enough to make
    H positive-definite (H + damp*I has min eigenvalue >= damp > 0), but
    in float32 that margin can still be swamped by round-off once Cholesky
    is deep into a near-null direction (confirmed live: failed at leading
    minor 7278/10240 with the default percdamp on a genuinely rank-4224,
    10240-wide matrix). Retrying with 10x the damping is the standard GPTQ
    fix for this and only ever fires on already-ill-conditioned layers --
    it does not change behavior for the common case where the first
    Cholesky attempt succeeds.
    """
    damp_mult = 1.0
    for attempt in range(6):
        H = _raw_hessian(X, in_features, percdamp * damp_mult)
        try:
            L = torch.linalg.cholesky(H)
            break
        except torch._C._LinAlgError:
            if attempt == 5:
                raise
            damp_mult *= 10
    # torch.cholesky_inverse (LAPACK potri) measured 24s+ on a 2688x2688 matrix
    # regardless of conditioning; cholesky_solve against the identity uses the
    # same factor L but stays ~30x faster (0.7s) on this hardware's BLAS.
    Hinv = torch.cholesky_solve(torch.eye(L.shape[0], dtype=L.dtype, device=L.device), L)
    Hinv = torch.linalg.cholesky(Hinv, upper=True)
    return Hinv


def _affine_scale_bias(W_groups: torch.Tensor, bits: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Exact reimplementation of mlx's affine_quantize Metal kernel's scale/
    bias derivation (mlx/backend/metal/kernels/quantized.h) -- verified to
    reproduce mx.quantize's actual output bit-for-bit via direct probing (its
    docstring's alpha=max/beta=min/(2^bits-1) description does NOT match the
    real kernel). The real formula:
      1. scale = max((w_max - w_min) / (2^bits - 1), eps)  -- range-based, not absmax-based
      2. side = |w_min| > |w_max|; flip scale's sign to match whichever side "wins"
      3. edge = the winning (larger-magnitude) extreme
      4. q0 = round(edge / scale); if q0 != 0, rescale so scale = edge / q0 --
         this makes `edge` land exactly on an integer code with zero rounding
         error, at the cost of `scale` no longer being exactly (max-min)/n_bins.
      5. bias = edge (or 0 in the degenerate q0==0 case)
    W_groups: [..., group_size]. Returns (scale, bias), each [...] (last dim reduced).
    """
    eps = 1e-7
    n_bins = float(2**bits - 1)
    w_min = W_groups.amin(dim=-1)
    w_max = W_groups.amax(dim=-1)
    scale = ((w_max - w_min) / n_bins).clamp_min(eps)
    side = w_min.abs() > w_max.abs()
    scale = torch.where(side, scale, -scale)
    edge = torch.where(side, w_min, w_max)
    q0 = torch.round(edge / scale)
    at_zero = q0 == 0
    scale = torch.where(at_zero, scale, edge / q0)
    bias = torch.where(at_zero, torch.zeros_like(edge), edge)
    return scale, bias


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
    affine_bits: int | None = None,
):
    """decide(w_col, scale_col, salient_col) -> q_col, all [out_features].

    salient_mask, if given, is a boolean [out_features, in_features] mask:
    True where that weight should be pinned at full precision instead of
    quantized. Pinning is implemented inside `decide` (q = w there), which
    makes that entry's error exactly zero — it consumes none of the block's
    error-compensation budget and propagates nothing onto later columns.

    scale_mode: "mean_abs" (binary/ternary's magnitude-heuristic scale),
    "max_abs" (symmetric-uniform-quant convention, e.g. Q8_0-style N-bit
    quantization, where scale = absmax / (2^(bits-1) - 1)), or "affine"
    (matches MLX's mx.quantize exactly -- see _affine_scale_bias and
    gptq_nbit's docstring for why this specific match matters; requires
    affine_bits to be set).
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
    if scale_mode == "affine":
        scale, bias = _affine_scale_bias(W_groups, affine_bits)  # [out, num_groups] each
    elif scale_mode == "max_abs":
        scale = W_groups.abs().amax(dim=-1)  # [out, num_groups]
        bias = torch.zeros_like(scale)
    else:
        scale = W_groups.abs().mean(dim=-1)  # [out, num_groups]
        bias = torch.zeros_like(scale)
    Q = torch.zeros_like(W)

    for g in range(num_groups):
        start, end = g * group_size, (g + 1) * group_size
        W_block = W[:, start:end].clone()
        Hinv_block = Hinv[start:end, start:end]
        Err_block = torch.zeros_like(W_block)
        s = scale[:, g]
        b = bias[:, g]

        for i in range(group_size):
            w = W_block[:, i]
            d = Hinv_block[i, i].clamp_min(1e-8)
            col_mask = salient_mask[:, start + i] if salient_mask is not None else None
            q = decide(w, s, b, col_mask)
            Q[:, start + i] = q
            err = (w - q) / d
            Err_block[:, i] = err
            if i < group_size - 1:
                W_block[:, i + 1 :] -= torch.outer(err, Hinv_block[i, i + 1 :])

        if end < padded_in:
            W[:, end:] -= Err_block @ Hinv[start:end, end:]

    Q = Q[:, :in_features].to("cpu")
    if return_scale:
        return Q, scale.to("cpu"), bias.to("cpu")  # [out_features, num_groups] each
    return Q


def gptq_binary(
    W: torch.Tensor,
    X: torch.Tensor,
    group_size: int = 128,
    percdamp: float = 0.01,
    device: str = "cpu",
    salient_mask: torch.Tensor | None = None,
) -> dict:
    def decide(w, s, bias, col_mask):
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
    def decide(w, s, bias, col_mask):
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
    scheme: str = "symmetric",
) -> dict:
    """N-bit uniform quantization with GPTQ's Hessian-based error
    compensation, instead of binary/ternary's sign-only decision.

    scheme="symmetric" (default, preserves this function's original
    behavior for callers that pack the result themselves, e.g. pack_mlx.py's
    custom ternary+rotation+salient MLX export): Q8_0/Q4_0-style, scale =
    absmax / (2^(bits-1) - 1), signed codes in [-levels, levels].

    scheme="affine": matches MLX's mx.quantize(mode="affine") exactly --
    see _affine_scale_bias's docstring for the (non-obvious, empirically
    reverse-engineered from mlx's actual Metal kernel source) formula.
    Use this when the caller will hand these already-on-grid bf16 weights to
    STOCK mlx_lm.convert for a second quantization pass (this project's
    gptq_stock_convert.py) -- if that pass's grid doesn't match the one GPTQ
    actually calibrated against, it silently re-derives different codes,
    discarding the calibration (confirmed empirically: PPL 27.7 with a
    subtly-wrong grid formula, vs a 6.5 naive-RTN baseline). With matching
    grids, mlx_lm.convert's own re-derived per-group scale/bias exactly
    reproduces this function's, so it re-derives the SAME codes instead of
    silently re-quantizing -- no second-pass distortion.
    """
    if scheme not in ("symmetric", "affine"):
        raise ValueError(f"scheme must be 'symmetric' or 'affine', got {scheme!r}")

    if scheme == "affine":
        n_bins = 2**bits - 1

        def decide(w, s, bias, col_mask):
            # s here IS the final per-code step (see _affine_scale_bias) --
            # no further division needed, unlike the symmetric branch below.
            code = torch.clamp(torch.round((w - bias) / s), 0, n_bins)
            q = code * s + bias
            return torch.where(col_mask, w, q) if col_mask is not None else q

        scale_mode = "affine"
    else:
        levels = 2 ** (bits - 1) - 1

        def decide(w, s, bias, col_mask):
            step = (s / levels).clamp_min(1e-12)
            code = torch.clamp(torch.round(w / step), -levels, levels)
            q = code * step
            return torch.where(col_mask, w, q) if col_mask is not None else q

        scale_mode = "max_abs"

    W_hat, scale, bias = _gptq_run(
        W, X, group_size, percdamp, decide, device, salient_mask, scale_mode=scale_mode,
        return_scale=True, affine_bits=bits if scheme == "affine" else None,
    )
    return {
        "W_hat": W_hat,
        "scale": scale,  # [out_features, num_groups]; absmax (symmetric) or the final per-code step (affine)
        "bias": bias,  # [out_features, num_groups]; zeros (symmetric) or the affine bias (see _affine_scale_bias)
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
    affine_bits: int | None = None,
):
    """Batched analogue of _gptq_run. W: [E, out_features, in_features],
    Xp: [E, n_max, in_features] zero-padded. decide(w, s, bias, col_mask)
    operates on [E, out_features] tensors. Returns Q: [E, out_features, in_features].
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
    if scale_mode == "affine":
        scale, bias = _affine_scale_bias(W_groups_init, affine_bits)  # [E, out, num_groups] each
    elif scale_mode == "max_abs":
        scale = W_groups_init.abs().amax(dim=-1)  # [E, out, num_groups]
        bias = torch.zeros_like(scale)
    else:
        scale = W_groups_init.abs().mean(dim=-1)  # [E, out, num_groups]
        bias = torch.zeros_like(scale)
    scale_orig = scale.clone()
    bias_orig = bias.clone()
    Q = torch.zeros_like(W)

    t0 = time.time()
    for g in range(num_groups):
        start, end = g * group_size, (g + 1) * group_size
        W_block = W[:, :, start:end].clone()  # [E, out, gs]
        Hinv_block = Hinv[:, start:end, start:end]  # [E, gs, gs]
        Err_block = torch.zeros_like(W_block)
        s = scale[:, :, g]  # [E, out]
        b = bias[:, :, g]  # [E, out]

        for i in range(group_size):
            w = W_block[:, :, i]  # [E, out]
            d = Hinv_block[:, i, i].clamp_min(1e-8)  # [E]
            col_mask = salient_mask[:, :, start + i] if salient_mask is not None else None
            q = decide(w, s, b, col_mask)
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
        return Q, scale_orig.to("cpu"), bias_orig.to("cpu")  # [E, out_features, num_groups] each
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
    def decide(w, s, bias, col_mask):
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
    def decide(w, s, bias, col_mask):
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
    scheme: str = "symmetric",
) -> dict:
    """Batched analogue of gptq_nbit -- see that function's docstring for the
    symmetric vs affine scheme choice. W: [E, out_features, in_features],
    Xp: [E, n_max, in_features] zero-padded per expert. Used for this
    project's routed MoE experts (128 same-shaped experts per block), where
    the single-expert path's Python column loop would otherwise launch
    group_size iterations *per expert* -- batching collapses that to one
    loop of group_size iterations where each iteration's kernel processes
    all E experts at once.
    """
    if scheme not in ("symmetric", "affine"):
        raise ValueError(f"scheme must be 'symmetric' or 'affine', got {scheme!r}")

    if scheme == "affine":
        n_bins = 2**bits - 1

        def decide(w, s, bias, col_mask):
            # s here IS the final per-code step (see _affine_scale_bias) --
            # no further division needed, unlike the symmetric branch below.
            code = torch.clamp(torch.round((w - bias) / s), 0, n_bins)
            q = code * s + bias
            return torch.where(col_mask, w, q) if col_mask is not None else q

        scale_mode = "affine"
    else:
        levels = 2 ** (bits - 1) - 1

        def decide(w, s, bias, col_mask):
            step = (s / levels).clamp_min(1e-12)
            code = torch.clamp(torch.round(w / step), -levels, levels)
            q = code * step
            return torch.where(col_mask, w, q) if col_mask is not None else q

        scale_mode = "max_abs"

    W_hat, scale, bias = _gptq_run_batched(
        W, Xp, group_size, percdamp, decide, device, salient_mask, progress_label,
        scale_mode=scale_mode, return_scale=True, affine_bits=bits if scheme == "affine" else None,
    )
    return {
        "W_hat": W_hat,
        "scale": scale,  # [E, out_features, num_groups]; absmax (symmetric) or the final per-code step (affine)
        "bias": bias,  # [E, out_features, num_groups]; zeros (symmetric) or the affine bias (see _affine_scale_bias)
        "bits_per_weight": bits + 16.0 / group_size,
        "method": f"gptq_{bits}bit_g{group_size}_batched",
    }
