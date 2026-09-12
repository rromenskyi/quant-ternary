"""Custom MLX Metal kernel for the "salient weight" sparse correction used
by our extreme-quantization recipe (poc/gptq.py's salient_mask, poc/
methods.py's rot_salient_ternary_batched_packable): ~3-10% of a routed
MoE expert's weights are pinned at full precision during GPTQ instead of
quantized to ternary, and at inference their contribution has to be added
back on top of the ternary matmul's output.

This is the same problem SpQR (Dettmers et al., "SpQR: A Sparse-Quantized
Representation for Near-Lossless LLM Weight Compression") solves with a
hand-written CUDA kernel: per-weight outlier positions are too numerous
(tens of thousands per expert matrix at a few percent sparsity) for either
(a) a dense correction matrix (defeats the whole point of compression) or
(b) a naive gather+one-hot-matmul in plain array ops (cost scales with
salient count x output_dims, intractable at this scale). MLX has no sparse
tensor type and no scatter-add in its standard array ops (mx.put_along_axis
overwrites, it doesn't accumulate), but *does* expose
mx.fast.metal_kernel(..., atomic_outputs=True) -- JIT-compiled Metal
source with atomic<float> writes, confirmed working via a minimal round-
trip test (scatter-add of 5 values into 3 buckets, verified against the
expected sums) -- this is the missing primitive.

Note this only runs on Apple Silicon (Metal backend) -- there is no CUDA
equivalent invoked here, so this module is unimportable/unusable on the
Linux+CUDA pod used elsewhere in this project for the PyTorch-side GPTQ
math. It's meant to run inside an MLX inference process on the user's Mac.

Fixed-k design: unlike SpQR's variable-per-row outlier count, this
project's salient selection (poc/methods.py's _select_salient_mask) picks
a global top-k *within each expert's whole weight matrix*, so every
expert has *exactly* the same salient count k = round(salient_fraction *
out_features * in_features) -- no ragged/offset bookkeeping needed, which
simplifies both the packer and this kernel considerably compared to a
general sparse-outlier scheme.

Known scope cuts vs. SpQR's production kernel (a first correct version,
not the final word -- see docs/session_findings_2026-09-11.md):
  - No load-balancing across threads by nonzero density (SpQR explicitly
    calls this out as necessary; here every (token, salient-slot) pair is
    one thread regardless of local density).
  - No batch-size-dependent kernel variant (SpQR uses different kernels
    for single-token decode vs. larger batches); this is one path for all
    batch sizes.
  - Accumulates in float32 (atomic<half> is not portably supported);
    callers should cast down afterward if the model runs in float16.
"""

from __future__ import annotations

import numpy as np
import mlx.core as mx

# Bounds salient_correction_bitmap's per-call transient memory (see its
# docstring) -- its resolve/gather/scatter arrays are all [chunk, top_k, k],
# and k (this project's per-expert salient count) is hundreds of thousands,
# so processing the full prompt length at once during prefill spiked peak
# memory to ~36GB on a 601-token prompt before this existed. 8 tokens/chunk
# keeps each call's transient footprint under ~1GB even at k~500k.
_TOKEN_CHUNK = 8


def _popcount32(x: np.ndarray) -> np.ndarray:
    """Vectorized SWAR popcount for uint32 numpy arrays (no per-element
    Python loop) -- used to build the rank checkpoint index below at model
    load time."""
    x = x.astype(np.uint32)
    x = x - ((x >> np.uint32(1)) & np.uint32(0x55555555))
    x = (x & np.uint32(0x33333333)) + ((x >> np.uint32(2)) & np.uint32(0x33333333))
    x = (x + (x >> np.uint32(4))) & np.uint32(0x0F0F0F0F)
    return ((x * np.uint32(0x01010101)) & np.uint32(0xFFFFFFFF)) >> np.uint32(24)


def build_touched_rows_mask(bitmap: np.ndarray, out_features: int, padded_in: int) -> np.ndarray:
    """Per-expert boolean mask, [num_experts, out_features]: does this row
    (of the [out_features, padded_in] weight matrix) contain at least one
    salient bit?

    This exists to work around a real gotcha in mx.fast.metal_kernel with
    atomic_outputs=True: its output buffer is NOT guaranteed zero on a
    fresh call -- empirically, the underlying physical buffer gets reused
    across separate calls to the same compiled kernel, and since our
    salient_correction_bitmap kernel only ever atomic-adds into rows that
    are actually salient for a given expert (a small fraction of
    `out_features` rows), every *other* row in the kernel's raw output is
    whatever was left over from a PREVIOUS call -- confirmed by a synthetic
    probe (a trivial atomic-add kernel touching only index 0 of a 100k
    array, called repeatedly: index 0 kept accumulating across calls,
    proving buffer reuse without re-zeroing). Left unfixed, this
    accumulates garbage across a model's ~50+ forward passes during
    generation and eventually produces NaN.

    Multiplying salient_correction_bitmap's raw output by this mask
    (broadcast per expert, per output row) zeroes exactly the rows that
    were never legitimately written this call, at negligible extra cost
    (one multiply, no kernel changes needed) -- see
    docs/session_findings_2026-09-11.md for the full writeup.

    Cheap to compute: `padded_in` is always a multiple of 32 (a multiple of
    the rotation block size), so each row occupies a whole, word-aligned
    span of `padded_in // 32` bitmap words with no cross-row word-sharing --
    a plain reshape + any-nonzero-reduce, no bit-level unpacking needed.
    """
    num_experts, bitmap_words = bitmap.shape
    words_per_row = padded_in // 32
    assert words_per_row * out_features == bitmap_words, (
        f"bitmap_words={bitmap_words} doesn't match out_features={out_features} x "
        f"words_per_row={words_per_row} -- padded_in must be a multiple of 32"
    )
    return (bitmap.reshape(num_experts, out_features, words_per_row) != 0).any(axis=-1)


def build_rank_index(bitmap: np.ndarray, chunk_words: int = 32) -> np.ndarray:
    """Build a per-expert rank checkpoint array over poc/pack_mlx.py's
    packed bitmap, for O(log num_chunks) "find the flat position of the
    j-th set bit" queries *without* ever decoding the bitmap into explicit
    (row, col) coordinates -- see salient_correction_bitmap's docstring for
    why the naive decode-everything approach (poc/sparse_salient_mlx.py's
    older decode_salient_bitmap/salient_correction) blows past a 24GB Mac's
    usable memory at realistic salient_fraction values (measured: ~27.6GB of
    resident decoded indices/values at salient_fraction=0.10 across this
    project's 30B model, see docs/session_findings_2026-09-11.md).

    checkpoint[e, c] = number of set bits in bitmap[e, 0 : c*chunk_words]
    (checkpoint[e, 0] = 0; checkpoint[e, -1] = k, the expert's total
    salient count -- guaranteed identical across experts by the packer's
    fixed-k design, see poc/methods.py's _select_salient_mask). Query time
    for a target rank j: binary-search checkpoint[e] for the chunk c with
    checkpoint[e,c] <= j < checkpoint[e,c+1], then linearly scan at most
    chunk_words 32-bit words (using Metal's builtin popcount) to find the
    exact word, then scan at most 32 bits within that word.

    chunk_words trades index size against per-query scan length:
    chunk_words=32 (1024 bits/chunk, the default) costs one int32 checkpoint
    per 1024 bitmap bits -- numel/256 bytes, about 1/8th the bitmap's own
    numel/8 bytes, and negligible next to the values array. This is a
    standard succinct-bitvector rank/select technique (e.g. Jacobson's
    rank/select), not a novel structure -- what's specific to this project
    is applying it to avoid ever materializing the decoded (row, col) form
    at all, given how large k gets here (hundreds of thousands per expert).
    """
    num_experts, bitmap_words = bitmap.shape
    num_chunks = (bitmap_words + chunk_words - 1) // chunk_words
    pad = num_chunks * chunk_words - bitmap_words
    bitmap_padded = np.pad(bitmap, ((0, 0), (0, pad)), constant_values=0) if pad else bitmap
    word_popcounts = _popcount32(bitmap_padded)
    chunk_popcounts = word_popcounts.reshape(num_experts, num_chunks, chunk_words).sum(axis=-1)
    checkpoint = np.zeros((num_experts, num_chunks + 1), dtype=np.int32)
    checkpoint[:, 1:] = np.cumsum(chunk_popcounts, axis=-1)
    return checkpoint


def decode_salient_bitmap(bitmap: np.ndarray, values: np.ndarray, padded_in: int) -> tuple[mx.array, mx.array, mx.array]:
    """Unpack poc/pack_mlx.py's on-disk (bitmap, values) salient storage
    into the explicit (row, col, value) arrays salient_correction's kernel
    needs. bitmap: [num_experts, ceil(out*padded_in/32)] uint32 (see
    pack_mlx.py's extract_salient for the bit-packing convention: bit i of
    the flattened [out, padded_in] tensor is stored at bitmap[i//32],
    shifted by i%32). values: [num_experts, k] float16, in ascending
    flat-index order per expert (matches torch.nonzero's default order,
    which is how pack_mlx.py generated them).

    This is a one-time cost paid at model *load*, not per forward pass --
    a Python/numpy loop over ~128 experts x a few hundred-thousand bits is
    a few hundred ms at most, dwarfed by the seconds-scale cost of loading
    a multi-GB safetensors file in the first place.
    """
    # int16 not int32: row/col are always < out_features/padded_in (<= a few
    # thousand for this project's shapes, well within int16's +-32767
    # range), and this halves the resident memory cost of the decoded
    # indices (10 -> 6 bytes/entry) -- at salient_fraction=0.10 that is the
    # difference between ~27.6GB and ~16.5GB of decoded salient data across
    # all MoE blocks, which is what actually blew past a 24GB Mac's usable
    # memory (the on-disk bitmap+values form is compact; decoding into
    # explicit coordinates is not, see docs/session_findings_2026-09-11.md).
    num_experts, bitmap_words = bitmap.shape
    k = values.shape[1]
    row = np.zeros((num_experts, k), dtype=np.int16)
    col = np.zeros((num_experts, k), dtype=np.int16)
    for e in range(num_experts):
        bits = np.unpackbits(bitmap[e].view(np.uint8), bitorder="little")
        flat_idx = np.nonzero(bits)[0]
        if flat_idx.shape[0] != k:
            raise ValueError(f"expert {e}: bitmap has {flat_idx.shape[0]} set bits, expected {k}")
        row[e] = flat_idx // padded_in
        col[e] = flat_idx % padded_in
    return mx.array(row), mx.array(col), mx.array(values)

_kernel = mx.fast.metal_kernel(
    name="salient_scatter_add",
    input_names=["x", "expert_ids", "salient_row", "salient_col", "salient_val"],
    output_names=["out"],
    source="""
        uint token = thread_position_in_grid.y;
        uint slot = thread_position_in_grid.z;
        uint j = thread_position_in_grid.x;
        int e = expert_ids[token * top_k_ + slot];
        uint base = (uint)e * k_ + j;
        int row = salient_row[base];
        int col = salient_col[base];
        T val = salient_val[base];
        T xval = x[token * padded_in_ + (uint)col];
        // Per-(token, slot) output cell: no cross-slot accumulation here,
        // only cross-j (within one expert's k salient entries) can
        // collide -- see module docstring on why slots stay separate
        // (routing scores are applied per-slot by the caller, same as the
        // ternary matmul path, before the top-k axis gets summed).
        size_t out_idx = ((size_t)token * top_k_ + slot) * output_dims_ + (uint)row;
        atomic_fetch_add_explicit(&out[out_idx], (float)(xval * val), memory_order_relaxed);
    """,
    atomic_outputs=True,
)


def salient_correction(
    x_rot: mx.array,
    expert_ids: mx.array,
    salient_row: mx.array,
    salient_col: mx.array,
    salient_val: mx.array,
    output_dims: int,
) -> mx.array:
    """x_rot: [n_tokens, padded_in] float16/float32, already rotated (see
    rotation_mlx.rotate). expert_ids: [n_tokens, top_k] int, the router's
    chosen experts per token (top_k=1 is fine, just shape it [n_tokens, 1]).
    One kernel call covers every top-k slot (instead of calling this
    function top_k times), amortizing fixed per-dispatch overhead, which
    dominates over actual compute at the batch sizes MLX inference on
    Apple Silicon typically runs (single-token decode) -- measured ~5x
    fewer effective per-token launches for top_k=6 (see
    docs/session_findings_2026-09-11.md for the before/after benchmark).
    salient_row/col: [num_experts, k] int16 (fixed k across experts, see
    module docstring -- int16 not int32 to halve the resident memory cost of
    the decoded indices). salient_val: [num_experts, k], same dtype as x_rot.

    Returns: [n_tokens, top_k, output_dims] float32 correction, NOT summed
    across top_k -- the caller must weight each slot by its routing score
    and sum, exactly like it already does for the ternary matmul's output
    (see NemotronHMoE.__call__'s `(y * scores[..., None]).sum(axis=-2)`),
    since scores are applied per-slot before the top-k reduction.
    """
    n_tokens, padded_in = x_rot.shape
    num_experts, k = salient_row.shape
    top_k = expert_ids.shape[1]

    (out,) = _kernel(
        inputs=[
            x_rot,
            expert_ids.astype(mx.int32),
            salient_row,
            salient_col,
            salient_val.astype(x_rot.dtype),
        ],
        template=[
            ("T", x_rot.dtype),
            ("k_", k),
            ("top_k_", top_k),
            ("padded_in_", padded_in),
            ("output_dims_", output_dims),
        ],
        grid=(k, n_tokens, top_k),
        threadgroup=(min(k, 256), 1, 1),
        output_shapes=[(n_tokens, top_k, output_dims)],
        output_dtypes=[mx.float32],
    )
    return out


def _make_kernel_resolve(chunk_words: int):
    # Non-atomic: each thread writes to a UNIQUE (token, slot, j) cell of
    # row_out/col_out, so there is no accumulation and therefore no
    # dependence on the output buffer's initial contents -- see
    # salient_correction_bitmap's docstring for why this replaced an
    # earlier atomic-accumulate design (mx.fast.metal_kernel's atomic
    # output buffer is not guaranteed zero across separate calls, and
    # our kernel only ever wrote a small fraction of the output space, so
    # leftover values from a previous call silently accumulated into
    # further garbage -- eventually NaN -- across a model's many forward
    # passes during generation; confirmed via a synthetic probe kernel
    # whose single written cell kept incrementing across repeated calls).
    # This kernel only *resolves positions*; the actual gather + scatter-
    # add uses mx.array.at[...].add(...), MLX's own correct primitive.
    return mx.fast.metal_kernel(
        name=f"salient_resolve_positions_cw{chunk_words}",
        input_names=["expert_ids", "bitmap", "checkpoint"],
        output_names=["row_out", "col_out"],
        source="""
        uint token = thread_position_in_grid.y;
        uint slot = thread_position_in_grid.z;
        uint j = thread_position_in_grid.x;   // target rank within expert e's k salient entries
        int e = expert_ids[token * top_k_ + slot];

        // Binary search checkpoint[e, 0..num_chunks_] for the chunk c with
        // checkpoint[c] <= j < checkpoint[c+1] (see build_rank_index's
        // docstring for the checkpoint array's exact semantics). Indexed
        // directly (no intermediate `device`/`constant`-qualified pointer
        // variable) -- MLX's auto-generated buffer address space for a
        // given input isn't guaranteed to be `device`, and redeclaring a
        // pointer with an explicit (wrong) address space is a hard Metal
        // compile error, not just a warning.
        uint cp_base = (uint)e * (num_chunks_ + 1);
        uint lo = 0;
        uint hi = num_chunks_;
        while (lo + 1 < hi) {
            uint mid = (lo + hi) / 2;
            if ((uint)checkpoint[cp_base + mid] <= j) { lo = mid; } else { hi = mid; }
        }
        uint remaining = j - (uint)checkpoint[cp_base + lo];

        // Scan words within the chunk (popcount is an MSL builtin) to find
        // the exact word containing the (remaining)-th set bit.
        uint bmp_base = (uint)e * bitmap_words_;
        uint word_start = lo * chunk_words_;
        uint w = 0;
        uint word = 0;
        for (; w < chunk_words_; w++) {
            uint widx = word_start + w;
            word = (widx < bitmap_words_) ? bitmap[bmp_base + widx] : 0u;
            uint wc = popcount(word);
            if (remaining < wc) break;
            remaining -= wc;
        }

        // Find the position of the (remaining)-th set bit within `word`.
        uint bit_pos = 0;
        uint seen = 0;
        for (uint b = 0; b < 32; b++) {
            if ((word >> b) & 1u) {
                if (seen == remaining) { bit_pos = b; break; }
                seen++;
            }
        }

        uint flat_pos = (word_start + w) * 32u + bit_pos;
        size_t out_idx = ((size_t)token * top_k_ + slot) * k_ + j;
        row_out[out_idx] = (short)(flat_pos / (uint)padded_in_);
        col_out[out_idx] = (short)(flat_pos % (uint)padded_in_);
        """,
        atomic_outputs=False,
    )


_kernel_resolve_cache: dict[int, object] = {}


def salient_correction_bitmap(
    x_rot: mx.array,
    expert_ids: mx.array,
    bitmap: mx.array,
    checkpoint: mx.array,
    salient_val: mx.array,
    salient_val_scale: mx.array,
    padded_in: int,
    output_dims: int,
    chunk_words: int = 32,
) -> mx.array:
    """Same contract as salient_correction (same grid, same output shape),
    but reads salient positions directly from poc/pack_mlx.py's packed
    bitmap via build_rank_index's checkpoint array, instead of requiring the
    caller to have pre-decoded everything into explicit (row, col) arrays.
    This is the memory-fixing alternative to decode_salient_bitmap +
    salient_correction: no per-entry row/col storage at all, only the
    bitmap itself (fixed cost, independent of salient_fraction) + the small
    rank checkpoint index + the values array. See build_rank_index's
    docstring for the sizing argument and docs/session_findings_2026-09-11.md
    for the measured before/after memory numbers.

    Implementation note: an earlier version of this function used a single
    atomic-accumulate Metal kernel to do position-resolve + gather + scatter
    all in one dispatch. That kernel's output buffer turned out to NOT be
    reliably zero on a fresh call -- confirmed via a synthetic probe (a
    trivial atomic-add kernel touching one fixed index, called repeatedly:
    the value kept incrementing across calls instead of resetting), because
    our kernel only ever atomic-adds into the small fraction of rows that
    are actually salient for a given expert, leaving the rest of the output
    buffer's *previous* contents in place -- and since MLX appears to reuse
    the same physical buffer across separate calls to the same kernel, that
    "previous contents" includes the last call's own (correct) results,
    causing values to double, triple, etc. on every repeated call, and
    eventually NaN in a real 50+-layer model over a generation run. Fixed
    by splitting into (a) a non-atomic Metal kernel that only *resolves*
    positions (each thread writes a unique cell, so no accumulation issue
    is possible), and (b) gather + scatter-add via mx.array.at[...].add(...),
    MLX's own correct primitive, starting from a guaranteed-fresh
    mx.zeros(...) every call.

    bitmap: [num_experts, bitmap_words] uint32 (poc/pack_mlx.py's
    extract_salient format). checkpoint: [num_experts, num_chunks+1] int32
    (build_rank_index's output, built once at load time from `bitmap` with
    the same `chunk_words`). salient_val: [num_experts, k] int8, ascending
    flat-index order per expert. salient_val_scale: [num_experts] --
    per-expert dequantization scale (poc/pack_mlx.py's extract_salient:
    `value.abs().max() / 127`), so the values array's resident cost stays 1
    byte/entry instead of float16's 2.

    Processes tokens in fixed-size chunks internally (see _TOKEN_CHUNK):
    the resolve+gather+scatter arrays below are all shaped
    [chunk, top_k, k], and k (this project's per-expert salient count) is
    hundreds of thousands -- at the full prompt length (n_tokens in the
    thousands during prefill) these arrays would be tens of GB each,
    confirmed by a real OOM-adjacent spike (~36GB peak for a 601-token
    prompt) before this chunking was added. Chunking bounds peak transient
    memory to a fixed size regardless of prompt length, at the cost of a
    few more (cheap) loop iterations during a long prefill; single-token
    decode (the common case) is unaffected (one chunk, same as before).
    """
    n_tokens, _ = x_rot.shape
    num_experts, k = salient_val.shape
    bitmap_words = bitmap.shape[1]
    num_chunks = checkpoint.shape[1] - 1
    top_k = expert_ids.shape[1]

    if chunk_words not in _kernel_resolve_cache:
        _kernel_resolve_cache[chunk_words] = _make_kernel_resolve(chunk_words)
    kernel = _kernel_resolve_cache[chunk_words]

    expert_ids_i32 = expert_ids.astype(mx.int32)
    outputs = []
    for start in range(0, n_tokens, _TOKEN_CHUNK):
        end = min(start + _TOKEN_CHUNK, n_tokens)
        chunk = end - start
        x_chunk = x_rot[start:end]
        eid_chunk = expert_ids_i32[start:end]

        row_out, col_out = kernel(
            inputs=[
                eid_chunk,
                bitmap.astype(mx.uint32),
                checkpoint.astype(mx.int32),
            ],
            template=[
                ("k_", k),
                ("top_k_", top_k),
                ("padded_in_", padded_in),
                ("bitmap_words_", bitmap_words),
                ("num_chunks_", num_chunks),
                ("chunk_words_", chunk_words),
            ],
            grid=(k, chunk, top_k),
            threadgroup=(min(k, 256), 1, 1),
            output_shapes=[(chunk, top_k, k), (chunk, top_k, k)],
            output_dtypes=[mx.int16, mx.int16],
        )  # row_out/col_out: [chunk, top_k, k] int16, transient (freed
        # after this iteration) -- see _make_kernel_resolve's docstring for
        # why int16 (halves this array's cost vs int32).

        token_idx = mx.broadcast_to(mx.arange(chunk).reshape(chunk, 1, 1), (chunk, top_k, k))
        slot_idx = mx.broadcast_to(mx.arange(top_k).reshape(1, top_k, 1), (chunk, top_k, k))
        expert_b = mx.broadcast_to(eid_chunk[:, :, None], (chunk, top_k, k))
        j_idx = mx.broadcast_to(mx.arange(k).reshape(1, 1, k), (chunk, top_k, k))

        xval = x_chunk[token_idx, col_out].astype(mx.float32)
        codes = salient_val[expert_b, j_idx].astype(mx.float32)
        scale = salient_val_scale[expert_b].astype(mx.float32)
        contribution = xval * codes * scale

        out_chunk = mx.zeros((chunk, top_k, output_dims), dtype=mx.float32)
        out_chunk = out_chunk.at[token_idx, slot_idx, row_out].add(contribution)
        mx.eval(out_chunk)  # force this chunk's transients to free before the next
        outputs.append(out_chunk)

    return outputs[0] if len(outputs) == 1 else mx.concatenate(outputs, axis=0)
