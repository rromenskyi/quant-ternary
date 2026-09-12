"""Custom MLX quantized-linear layer for routed MoE experts: applies the
same rotate -> GPTQ-ternary -> salient-pin recipe from
poc/methods.py's rot_salient_ternary_batched_packable at inference time,
instead of the stock mx.gather_qmm's plain affine quantized matmul.

Key trick for the ternary part: our ternary codes {-scale, 0, +scale} per
group map exactly onto MLX's 2-bit affine convention (4 codes per group,
we only ever emit 3 of them):
    code 0 -> -scale       (scale_mlx = scale, bias_mlx = -scale)
    code 1 ->  0
    code 2 -> +scale
    code 3 -> unused
so the ternary matmul reuses mx.gather_qmm's own optimized kernel directly
-- we just need to hand it the right (packed codes, scales, biases) tensors
instead of running mx.quantize's RTN affine search on random data.

Rotation is a fixed, parameter-free orthogonal transform (poc/rotation_mlx.py,
validated bit-identical to the PyTorch reference in poc/rotation.py): apply
it to the input activations before the ternary matmul, and to the *output*
afterward instead of unrotating the weight -- mathematically equivalent
since rotation is linear, and this is what keeps the stored weight literally
packable (see poc/methods.py's rot_salient_ternary_batched_packable
docstring for why unrotate()-ing the weight instead loses this property).

Salient correction: the ~3-10% of weights pinned at full precision during
GPTQ (poc/gptq.py's salient_mask) are applied via poc/sparse_salient_mlx's
custom Metal scatter-add kernel, NOT a dense correction matrix (which would
defeat the whole point of ternary compression) and NOT a naive gather +
one-hot-matmul in plain array ops (cost scales with salient count x
output_dims, intractable at the ~150k-entries-per-expert scale this
project's salient_fraction produces -- see sparse_salient_mlx's docstring
for the full story, incl. why SpQR-style prior art needs a real kernel).

Drop-in shape convention: matches mlx_lm.models.switch_layers.SwitchLinear
exactly (same __call__(x, indices, sorted_indices) signature and the same
expand_dims/[n_tokens, top_k, 1, output_dims] shapes), so this can replace
SwitchLinear inside the existing SwitchGLU/SwitchMLP wrapper used by
mlx_lm's nemotron_h.py without touching the MoE routing logic itself --
only the two projections' construction needs to change (see the packer /
model-loading code that wires this in).
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn

from rotation_mlx import rotate, unrotate
from sparse_salient_mlx import salient_correction_bitmap


class RotatedTernarySwitchLinear(nn.Module):
    """Drop-in replacement for mlx_lm.models.switch_layers.SwitchLinear,
    for one projection (e.g. up_proj or down_proj) across all routed
    experts, using our rotate+ternary+salient recipe instead of a plain
    affine quantized matmul.

    Expected weight buffers (set externally by the packer, not learned):
      - weight: [num_experts, output_dims, padded_in * bits // 32] uint32,
        2-bit affine-packed ternary codes in the ROTATED domain (padded_in
        is input_dims rounded up to a multiple of rotation_block_size)
      - scales, biases: [num_experts, output_dims, padded_in // group_size]
        float16, per-group affine params (biases = -scales for our ternary
        convention, see module docstring)
      - salient_bitmap: [num_experts, ceil(output_dims*padded_in/32)]
        uint32 (poc/pack_mlx.py's extract_salient format), salient_checkpoint:
        [num_experts, num_chunks+1] int32 (poc/sparse_salient_mlx.
        build_rank_index's output, built once at load time from
        salient_bitmap), salient_val: [num_experts, k] float16 -- the sparse
        full-precision overlay, k fixed across experts, all in the ROTATED
        domain. Reads positions directly from the bitmap at inference time
        (no persistent explicit (row, col) arrays) -- see
        sparse_salient_mlx.salient_correction_bitmap's docstring for why
        (the explicit-decode alternative blows past a 24GB Mac's usable
        memory at this project's salient fractions). salient_k=0 (default)
        disables the salient correction entirely (e.g. for a
        salient_fraction=0 ablation).
    """

    CHUNK_WORDS = 32  # must match whatever poc/mlx_model_ternary.py's sanitize() used to build salient_checkpoint

    def __init__(
        self,
        input_dims: int,
        output_dims: int,
        num_experts: int,
        rotation_block_size: int = 64,
        group_size: int = 64,
        bits: int = 2,
        salient_k: int = 0,
    ):
        super().__init__()
        self.input_dims = input_dims
        self.output_dims = output_dims
        self.num_experts = num_experts
        self.rotation_block_size = rotation_block_size
        self.group_size = group_size
        self.bits = bits

        pad = (-input_dims) % rotation_block_size
        self.padded_in = input_dims + pad

        # Placeholders; overwritten by the packer when loading real weights.
        self.weight = mx.zeros((num_experts, output_dims, self.padded_in * bits // 32), dtype=mx.uint32)
        self.scales = mx.zeros((num_experts, output_dims, self.padded_in // group_size), dtype=mx.float16)
        self.biases = mx.zeros((num_experts, output_dims, self.padded_in // group_size), dtype=mx.float16)

        self.salient_k = salient_k
        if salient_k > 0:
            numel = output_dims * self.padded_in
            bitmap_words = (numel + 31) // 32
            num_chunks = (bitmap_words + self.CHUNK_WORDS - 1) // self.CHUNK_WORDS
            self.salient_bitmap = mx.zeros((num_experts, bitmap_words), dtype=mx.uint32)
            self.salient_checkpoint = mx.zeros((num_experts, num_chunks + 1), dtype=mx.int32)
            self.salient_val = mx.zeros((num_experts, salient_k), dtype=mx.int8)
            self.salient_val_scale = mx.zeros((num_experts,), dtype=mx.float16)

        self.freeze()

    def __call__(self, x: mx.array, indices: mx.array, sorted_indices: bool = False) -> mx.array:
        """x: [..., 1, 1, input_dims] (already expand_dims'd by the caller,
        matching SwitchGLU/SwitchMLP's convention -- "..." is typically
        [batch, seq_len] but mx.gather_qmm itself is agnostic to how many
        leading dims there are). indices: [..., top_k], same leading dims as
        x. Returns [..., top_k, 1, output_dims], matching SwitchLinear's
        output shape exactly.
        """
        x_rot = rotate(x, self.rotation_block_size)
        y_rot = mx.gather_qmm(
            x_rot,
            self.weight,
            self.scales,
            self.biases,
            rhs_indices=indices,
            transpose=True,
            group_size=self.group_size,
            bits=self.bits,
            mode="affine",
            sorted_indices=sorted_indices,
        )
        y = unrotate(y_rot, self.rotation_block_size, self.output_dims)

        if self.salient_k > 0:
            # salient_correction's kernel expects flat [n_rows, top_k] /
            # [n_rows, padded_in]. "n_rows" and what "top_k" means here
            # differ between fc1 and fc2: fc1's x is one row per token (the
            # kernel itself loops over all top_k experts per row), but fc2's
            # x already has a distinct row per (token, expert-slot) pair --
            # each fed through its own expert's down_proj by the preceding
            # gather_qmm/activation, so each row maps to exactly *one*
            # expert, not top_k of them. Rather than special-case fc1 vs
            # fc2, infer it from sizes: if indices already has exactly one
            # entry per x-row, treat it as top_k=1 (fc2); otherwise indices
            # carries its own trailing top_k axis on top of x's per-token
            # rows (fc1). Either way, n_rows * top_k_eff * output_dims
            # equals y.size, so reshaping the flat correction straight into
            # y's own shape is always correct without tracking which case
            # we're in.
            n_rows = x_rot.size // self.padded_in
            if indices.size == n_rows:
                idx_flat = indices.reshape(n_rows, 1)
            else:
                idx_flat = indices.reshape(n_rows, indices.shape[-1])
            x_flat = x_rot.reshape(n_rows, self.padded_in)
            corr = salient_correction_bitmap(
                x_flat, idx_flat, self.salient_bitmap, self.salient_checkpoint,
                self.salient_val, self.salient_val_scale,
                self.padded_in, self.output_dims, chunk_words=self.CHUNK_WORDS,
            )  # [n_rows, top_k_eff, output_dims], top_k_eff*n_rows*output_dims == y.size
            y = y + corr.reshape(y.shape).astype(y.dtype)

        return y
