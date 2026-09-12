"""Validate sparse_salient_mlx.salient_correction against a plain (slow,
obviously-correct) reference implementation using numpy loops.

Usage:
    python poc/test_sparse_salient_mlx.py
"""

from __future__ import annotations

import numpy as np
import mlx.core as mx

from sparse_salient_mlx import build_rank_index, salient_correction, salient_correction_bitmap


def reference(x, expert_ids, salient_row, salient_col, salient_val, output_dims):
    n_tokens, padded_in = x.shape
    top_k = expert_ids.shape[1]
    out = np.zeros((n_tokens, top_k, output_dims), dtype=np.float32)
    for t in range(n_tokens):
        for slot in range(top_k):
            e = int(expert_ids[t, slot])
            for j in range(salient_row.shape[1]):
                row, col, val = int(salient_row[e, j]), int(salient_col[e, j]), float(salient_val[e, j])
                out[t, slot, row] += float(x[t, col]) * val
    return out


def main():
    np.random.seed(0)
    n_tokens, padded_in, output_dims = 17, 64, 32
    num_experts, k, top_k = 5, 20, 6

    x_np = np.random.randn(n_tokens, padded_in).astype(np.float32)
    expert_ids_np = np.random.randint(0, num_experts, size=(n_tokens, top_k)).astype(np.int32)
    salient_row_np = np.random.randint(0, output_dims, size=(num_experts, k)).astype(np.int32)
    salient_col_np = np.random.randint(0, padded_in, size=(num_experts, k)).astype(np.int32)
    salient_val_np = np.random.randn(num_experts, k).astype(np.float32)

    expected = reference(x_np, expert_ids_np, salient_row_np, salient_col_np, salient_val_np, output_dims)

    got = salient_correction(
        mx.array(x_np),
        mx.array(expert_ids_np),
        mx.array(salient_row_np),
        mx.array(salient_col_np),
        mx.array(salient_val_np),
        output_dims,
    )
    got_np = np.array(got)

    diff = np.abs(got_np - expected).max()
    status = "OK" if diff < 1e-3 else "FAIL"
    print(f"[{status}] max abs diff = {diff:.6g}")
    if status == "FAIL":
        raise SystemExit(1)

    # Also test duplicate (row, col) collisions within one expert -- makes
    # sure atomic accumulation, not overwrite, is really happening.
    salient_row_np2 = np.zeros((num_experts, k), dtype=np.int32)  # all -> row 0, and top_k slots also collide
    expected2 = reference(x_np, expert_ids_np, salient_row_np2, salient_col_np, salient_val_np, output_dims)
    got2 = np.array(
        salient_correction(
            mx.array(x_np), mx.array(expert_ids_np), mx.array(salient_row_np2),
            mx.array(salient_col_np), mx.array(salient_val_np), output_dims,
        )
    )
    diff2 = np.abs(got2 - expected2).max()
    status2 = "OK" if diff2 < 1e-3 else "FAIL"
    print(f"[{status2}] collision (all rows=0) max abs diff = {diff2:.6g}")
    if status2 == "FAIL":
        raise SystemExit(1)

    # salient_correction_bitmap, across chunk_words values (7 deliberately
    # doesn't divide bitmap_words evenly, exercising the tail-pad path) --
    # see _run_bitmap_test. An earlier atomic-accumulate kernel design could
    # not run this loop safely in one process (see docs/session_findings_
    # 2026-09-11.md's §7m); the current resolve-then-scatter design has no
    # such issue (verified by the repeat-call check inside _run_bitmap_test).
    for chunk_words in (32, 7):
        _run_bitmap_test(chunk_words)


def _run_bitmap_test(chunk_words: int):
    np.random.seed(0)
    n_tokens, padded_in, output_dims = 17, 64, 32
    num_experts, k, top_k = 5, 20, 6
    x_np = np.random.randn(n_tokens, padded_in).astype(np.float32)
    expert_ids_np = np.random.randint(0, num_experts, size=(n_tokens, top_k)).astype(np.int32)
    # Advance past the same three np.random draws main() makes before this
    # point, so this subprocess's x_np/expert_ids_np match exactly (the
    # legacy global RNG is deterministic given the same call sequence).
    np.random.randint(0, output_dims, size=(num_experts, k))
    np.random.randint(0, padded_in, size=(num_experts, k))
    np.random.randn(num_experts, k)

    rng = np.random.default_rng(0)
    numel = output_dims * padded_in
    bitmap_words = (numel + 31) // 32
    bitmap_np = np.zeros((num_experts, bitmap_words), dtype=np.uint32)
    row3 = np.zeros((num_experts, k), dtype=np.int32)
    col3 = np.zeros((num_experts, k), dtype=np.int32)
    val3_int8 = np.zeros((num_experts, k), dtype=np.int8)
    val3_scale = np.zeros((num_experts,), dtype=np.float16)
    for e in range(num_experts):
        flat_idx = np.sort(rng.choice(numel, size=k, replace=False))
        words = flat_idx // 32
        bits = flat_idx % 32
        np.bitwise_or.at(bitmap_np[e], words, (np.uint32(1) << bits.astype(np.uint32)))
        row3[e] = flat_idx // padded_in
        col3[e] = flat_idx % padded_in
        # Mirror poc/pack_mlx.py's extract_salient int8 quantization exactly,
        # so the reference below compares against the *dequantized* values
        # the kernel is actually supposed to reproduce -- isolates kernel
        # correctness from int8 quantization error, which is a separate,
        # already-accepted lossy step (not something this test should flag).
        vals_e = rng.standard_normal(k).astype(np.float32)
        scale_e = max(float(np.abs(vals_e).max()) / 127.0, 1e-8)
        val3_int8[e] = np.round(vals_e / scale_e).clip(-127, 127).astype(np.int8)
        val3_scale[e] = np.float16(scale_e)
    val3_dequant = val3_int8.astype(np.float32) * val3_scale.astype(np.float32)[:, None]

    expected3 = reference(x_np, expert_ids_np, row3, col3, val3_dequant, output_dims)

    checkpoint_np = build_rank_index(bitmap_np, chunk_words=chunk_words)

    def run_once():
        return np.array(
            salient_correction_bitmap(
                mx.array(x_np), mx.array(expert_ids_np), mx.array(bitmap_np),
                mx.array(checkpoint_np), mx.array(val3_int8), mx.array(val3_scale),
                padded_in, output_dims, chunk_words=chunk_words,
            )
        )

    got3 = run_once()
    diff3 = np.abs(got3 - expected3).max()
    status3 = "OK" if diff3 < 1e-2 else "FAIL"
    print(f"[{status3}] bitmap-kernel (chunk_words={chunk_words}) max abs diff = {diff3:.6g}")
    if status3 == "FAIL":
        raise SystemExit(1)

    # Regression test for a real bug found the hard way: mx.fast.metal_
    # kernel's atomic-output buffer is NOT guaranteed zero on a fresh call
    # (its physical buffer gets reused across calls without re-zeroing) --
    # untouched rows silently accumulated leftover garbage across repeated
    # calls until it produced NaN in the real model. Calling the same
    # kernel several times in a row and checking every result matches
    # `expected3` (not just the first call) is exactly what would have
    # caught this before it reached a live model.
    for rep in range(4):
        got_rep = run_once()
        diff_rep = np.abs(got_rep - expected3).max()
        status_rep = "OK" if diff_rep < 1e-2 else "FAIL"
        print(f"[{status_rep}] bitmap-kernel repeat-call #{rep} (chunk_words={chunk_words}) max abs diff = {diff_rep:.6g}")
        if status_rep == "FAIL":
            raise SystemExit(1)


if __name__ == "__main__":
    main()
    print("\nAll sparse_salient_mlx checks passed.")
