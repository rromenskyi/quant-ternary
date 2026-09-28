#!/usr/bin/env python3
"""[pod] Sequential GPTQ of the VoiceChat LLM (NemotronH 9B hybrid) in MLX.

Runs on mlx[cuda] with the same mlx-audio fork model code the Mac uses.
Calibration inputs are the fused per-frame LLM inputs captured by
calib_capture.py (real duplex sessions), replayed layer by layer in prefill
mode. Each layer: collect Hessians of its Linears on the (already quantized)
previous layers' outputs, GPTQ each Linear onto MLX's own affine grid
(scales/biases from mx.quantize, so the packed result is exactly what
mx.dequantize reproduces), then recompute the layer output with the
quantized weights for the next layer.

Bits are chosen by CLI only:
  --bits 3 --group-size 64                    default for every LLM Linear
  --override 'REGEX=BITS[:GROUP]'             first match wins, e.g.
                                              --override 'mixer\\.(q|k|v|o)_proj=4'
  --head-bits 4 / --embed-bits 4              lm_head+function_head (GPTQ), embed_tokens (RTN)
  --rtn                                       skip GPTQ, plain round-to-nearest (reference)

Output (--out): llm.safetensors (only stt_model.{llm,lm_head,function_head,
embed_tokens} tensors, quantized) + quant.json (per-module config entries)
+ gptq_log.json. splice_llm.py merges them into a full model dir.
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

import os
import sys

# --cpu-threads (default: every core) must reach BLAS/OpenMP before numpy and
# torch load: the host-side Hessian solve and code packing ran on ~2 of the
# pod's 128 cores (GPU at ~28%) without it.
def _early_threads(argv) -> int:
    for i, a in enumerate(argv):
        if a == "--cpu-threads" and i + 1 < len(argv):
            return max(1, int(argv[i + 1]))
        if a.startswith("--cpu-threads="):
            return max(1, int(a.split("=", 1)[1]))
    return os.cpu_count() or 1


CPU_THREADS = _early_threads(sys.argv)
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_var, str(CPU_THREADS))

import mlx.core as mx
import mlx.nn as nn
import numpy as np

# ---------------------------------------------------------------- packing


def pack_codes(q, bits: int, chunk: int = 2048) -> mx.array:
    """Pack integer codes (rows, cols) as MLX affine quantized weights.

    MLX stores codes as a little-endian bit stream per row (element k at bit
    k*bits), viewed as uint32. Verified against mx.quantize by self_test().
    """
    q = np.asarray(q).astype(np.uint32)
    rows, cols = q.shape
    assert (cols * bits) % 32 == 0
    weights = (np.uint64(1) << np.arange(32, dtype=np.uint64))
    out = np.empty((rows, cols * bits // 32), dtype=np.uint32)
    shifts = np.arange(bits, dtype=np.uint32)
    for r in range(0, rows, chunk):
        blk = q[r : r + chunk]
        bitsarr = ((blk[..., None] >> shifts) & 1).astype(np.uint64)  # (r, cols, bits)
        bitsarr = bitsarr.reshape(blk.shape[0], -1, 32)
        out[r : r + chunk] = (bitsarr * weights).sum(-1).astype(np.uint32)
    return mx.array(out)


def self_test() -> None:
    rng = np.random.default_rng(0)
    for bits in (2, 3, 4, 5, 6, 8):
        w = mx.array(rng.standard_normal((8, 128)).astype(np.float32))
        packed, scales, biases = mx.quantize(w, group_size=64, bits=bits)
        ref = mx.dequantize(packed, scales, biases, group_size=64, bits=bits)
        codes = mx.round((ref - mx.repeat(biases, 64, axis=1)) / mx.repeat(scales, 64, axis=1))
        mine = pack_codes(codes.astype(mx.uint32), bits)
        assert mx.array_equal(mine, packed).item(), f"pack mismatch bits={bits}"
    print("pack self-test ok (2,3,4,5,6,8 bits)")


# ---------------------------------------------------------------- GPTQ


class Catcher(nn.Module):
    def __init__(self, module):
        super().__init__()
        self.module = module
        self.H = None
        self.n = 0

    def __call__(self, x, *a, **k):
        xf = x.reshape(-1, x.shape[-1]).astype(mx.float32)
        h = xf.T @ xf
        self.H = h if self.H is None else self.H + h
        self.n += xf.shape[0]
        return self.module(x, *a, **k)


def inverse_hessian_chol(H: mx.array, damp: float):
    """Upper Cholesky factor of H^-1 (with damping, dead columns fixed).

    Done in float64 with torch on CUDA when available (the pod), else scipy
    LAPACK on the host. MLX's own linalg is CPU-only and took 78 s for one
    4480x4480 Cholesky on the pod; 15680x15680 (down_proj) would take hours."""
    Hn = np.array(H.astype(mx.float32), dtype=np.float64)
    try:
        import torch
        torch.set_num_threads(CPU_THREADS)

        if torch.cuda.is_available():
            Ht = torch.from_numpy(Hn).cuda()
            d = torch.diagonal(Ht)
            dead_t = d == 0
            Ht[dead_t, dead_t] = 1.0
            Ht += damp * torch.mean(torch.diagonal(Ht)) * torch.eye(Ht.shape[0], device="cuda", dtype=Ht.dtype)
            Hinv = torch.cholesky_inverse(torch.linalg.cholesky(Ht))
            U = torch.linalg.cholesky(Hinv, upper=True)
            out = mx.array(U.float().cpu().numpy()), mx.array(dead_t.cpu().numpy())
            del Ht, Hinv, U
            torch.cuda.empty_cache()  # MLX and torch share the GPU; don't let torch's cache grow
            return out
    except ImportError:
        pass
    import scipy.linalg

    diag = np.diag(Hn).copy()
    dead = diag == 0
    Hn[dead, dead] = 1.0
    Hn[np.diag_indices_from(Hn)] += damp * np.mean(np.diag(Hn))
    L = scipy.linalg.cholesky(Hn, lower=True)
    Hinv = scipy.linalg.cho_solve((L, True), np.eye(Hn.shape[0]))
    U = scipy.linalg.cholesky(Hinv, lower=False)
    return mx.array(U.astype(np.float32)), mx.array(dead)


def _group_step(Wg, s, b, Ug, n_bins):
    """GPTQ over one group of columns (unrolled; compiled per shape)."""
    Err, Qg = [], []
    for c in range(Wg.shape[1]):
        w = Wg[:, c]
        q = mx.clip(mx.round((w - b[:, 0]) / s[:, 0]), 0, n_bins)
        e = (w - (q * s[:, 0] + b[:, 0])) / Ug[c, c]
        Wg = Wg - e[:, None] * Ug[c][None, :]  # U is upper: columns < c untouched
        Err.append(e)
        Qg.append(q)
    return mx.stack(Qg, axis=1), mx.stack(Err, axis=1)


_compiled_steps = {}


def group_grid(wg: mx.array, bits: int, group: int, scales_dtype):
    _, s, b = mx.quantize(wg, group_size=group, bits=bits)
    return s.astype(scales_dtype).astype(mx.float32), b.astype(scales_dtype).astype(mx.float32)


def gptq_matrix(W: mx.array, H: mx.array, bits: int, group: int, damp: float, scales_dtype):
    """Standard GPTQ (no act-order, static per-group grid chosen when the
    group is reached, as AutoGPTQ/mlx-lm do). Returns codes, scales, biases."""
    U, dead = inverse_hessian_chol(H, damp)
    W = W.astype(mx.float32)
    W = mx.where(dead[None, :], 0.0, W)
    rows, cols = W.shape
    n_bins = 2**bits - 1
    codes, all_s, all_b = [], [], []
    for i in range(0, cols, group):
        j = i + group
        Wg = W[:, i:j]
        s, b = group_grid(Wg, bits, group, scales_dtype)
        Ug = U[i:j, i:j]
        if n_bins not in _compiled_steps:
            _compiled_steps[n_bins] = mx.compile(lambda W_, s_, b_, U_, nb=n_bins: _group_step(W_, s_, b_, U_, nb))
        Q, E = _compiled_steps[n_bins](Wg, s, b, Ug)
        if j < cols:
            W = mx.concatenate([W[:, :j], W[:, j:] - E @ U[i:j, j:]], axis=1)
        codes.append(Q)
        all_s.append(s)
        all_b.append(b)
        mx.eval(W, codes[-1])
    return (
        mx.concatenate(codes, axis=1).astype(mx.uint32),
        mx.concatenate(all_s, axis=1),
        mx.concatenate(all_b, axis=1),
    )


def rtn_matrix(W, bits, group, scales_dtype):
    s, b = group_grid(W.astype(mx.float32), bits, group, scales_dtype)
    n_bins = 2**bits - 1
    Wr = W.astype(mx.float32).reshape(W.shape[0], -1, group)
    q = mx.clip(mx.round((Wr - b[..., None]) / s[..., None]), 0, n_bins)
    return q.reshape(W.shape).astype(mx.uint32), s, b


def dequant(codes, s, b, group):
    rows, cols = codes.shape
    c = codes.astype(mx.float32).reshape(rows, -1, group)
    return (c * s[..., None] + b[..., None]).reshape(rows, cols)


# ---------------------------------------------------------------- driver


def parse_overrides(items):
    out = []
    for item in items:
        pat, spec = item.rsplit("=", 1)
        bits, _, group = spec.partition(":")
        out.append((re.compile(pat), int(bits), int(group) if group else None))
    return out


def bits_for(path, args, overrides):
    for pat, bits, group in overrides:
        if pat.search(path):
            return bits, group or args.group_size
    return args.bits, args.group_size


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="unquantized MLX VoiceChat dir (bf16)")
    ap.add_argument("--cpu-threads", type=int, default=CPU_THREADS, help="host threads for BLAS/torch (default: every core)")
    ap.add_argument("--calib", required=True, help="calib_capture.py output dir")
    ap.add_argument("--out", required=True)
    ap.add_argument("--bits", type=int, default=3)
    ap.add_argument("--group-size", type=int, default=64)
    ap.add_argument("--override", action="append", default=[])
    ap.add_argument("--head-bits", type=int, default=None, help="default 4, or the recipe's lm_head")
    ap.add_argument("--embed-bits", type=int, default=None, help="default 4, or the recipe's embeddings")
    ap.add_argument("--component-recipe", default=None,
                    help="named per-component bit recipe from recipes.py (e.g. jang-voicechat); "
                         "explicit --override/--head-bits/--embed-bits still win")
    ap.add_argument("--damp", type=float, default=0.01)
    ap.add_argument("--rtn", action="store_true")
    ap.add_argument("--max-clips", type=int, default=None)
    ap.add_argument("--scales-dtype", default="bf16", choices=["bf16", "fp32"])
    ap.add_argument("--mlx-cache-gb", type=float, default=8.0,
                    help="cap on MLX's buffer cache (variable clip lengths otherwise fill the GPU)")
    args = ap.parse_args()
    mx.set_cache_limit(int(args.mlx_cache_gb * 1e9))
    self_test()
    sdt = mx.bfloat16 if args.scales_dtype == "bf16" else mx.float32
    head_bits, embed_bits, recipe_over = 4, 4, []
    if args.component_recipe:
        import recipes

        recipe_over, head_bits, embed_bits = recipes.recipe_overrides(args.component_recipe)
    args.head_bits = head_bits if args.head_bits is None else args.head_bits
    args.embed_bits = embed_bits if args.embed_bits is None else args.embed_bits
    overrides = parse_overrides(args.override + recipe_over)  # explicit overrides first: first match wins
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    from mlx_audio.lm.models.base import create_attention_mask, create_ssm_mask
    from mlx_audio.sts import load

    model = load(args.model, lazy=True)
    stt = model.stt_model
    llm = stt.llm

    files = sorted(Path(args.calib).glob("c*.npy"))[: args.max_clips]
    xs = [mx.array(np.load(f)).astype(mx.bfloat16)[None] for f in files]
    n_tok = sum(x.shape[1] for x in xs)
    print(f"calibration: {len(xs)} clips, {n_tok} frames", flush=True)

    tensors, qconfig, log = {}, {}, []

    def store(path, codes, s, b, bits, group):
        tensors[f"{path}.weight"] = pack_codes(codes, bits)
        tensors[f"{path}.scales"] = s.astype(sdt)
        tensors[f"{path}.biases"] = b.astype(sdt)
        qconfig[path] = {"group_size": group, "bits": bits}

    def quantize_linears(named, prefix):
        for name, (mod, H, nrows) in named.items():
            path = f"{prefix}.{name}"
            bits, group = bits_for(path, args, overrides)
            t = time.time()
            W = mod.weight
            if args.rtn:
                codes, s, b = rtn_matrix(W, bits, group, sdt)
            else:
                codes, s, b = gptq_matrix(W, H, bits, group, args.damp, sdt)
            Wq = dequant(codes, s, b, group)
            Wf = W.astype(mx.float32)
            # output error proxy: trace((W-Wq) H (W-Wq)^T) / trace(W H W^T)
            D = Wf - Wq
            rel = (mx.sum((D @ H) * D) / mx.maximum(mx.sum((Wf @ H) * Wf), 1e-12)).item() if H is not None else None
            mod.weight = Wq.astype(W.dtype)
            store(path, codes, s, b, bits, group)
            mx.eval(mod.weight)
            log.append({"module": path, "bits": bits, "group": group, "rel_out_err": rel,
                        "tokens": nrows, "s": round(time.time() - t, 1)})
            print(f"  {path} {tuple(W.shape)} {bits}b rel_err={rel:.3e} {time.time() - t:.1f}s", flush=True)

    for li, layer in enumerate(llm.layers):
        t0 = time.time()
        mx.eval(layer.parameters())
        mixer = layer.mixer
        lin_names = [k for k, v in mixer.children().items() if isinstance(v, nn.Linear)]
        catchers = {k: Catcher(getattr(mixer, k)) for k in lin_names}
        for k, c in catchers.items():
            setattr(mixer, k, c)

        def run(x):
            mask = (create_attention_mask(x, None) if layer.block_type == "*" else create_ssm_mask(x, None))
            return layer(x, mask=mask, cache=None)

        for x in xs:
            y = run(x)
            mx.eval(y, [c.H for c in catchers.values()])
        for k, c in catchers.items():
            setattr(mixer, k, c.module)
        named = {k: (catchers[k].module, catchers[k].H, catchers[k].n) for k in lin_names}
        quantize_linears(named, f"stt_model.llm.layers.{li}.mixer")
        del catchers, named
        new_xs = []
        for x in xs:
            y = run(x)
            mx.eval(y)
            new_xs.append(y)
        xs = new_xs
        del new_xs
        mx.clear_cache()
        print(f"layer {li} ({layer.block_type}) done in {time.time() - t0:.0f}s "
              f"mlx_active={mx.get_active_memory() / 1e9:.1f}GB mlx_cache={mx.get_cache_memory() / 1e9:.1f}GB", flush=True)

    # heads: shared Hessian of the final-norm hidden states
    hs = [llm.norm_f(x) for x in xs]
    H = None
    for h in hs:
        hf = h.reshape(-1, h.shape[-1]).astype(mx.float32)
        H = hf.T @ hf if H is None else H + hf.T @ hf
        mx.eval(H)
    for name in ("lm_head", "function_head"):
        if not hasattr(stt, name):
            continue
        mod = getattr(stt, name)
        mx.eval(mod.parameters())
        bits, group = args.head_bits, args.group_size
        t = time.time()
        codes, s, b = (rtn_matrix(mod.weight, bits, group, sdt) if args.rtn
                       else gptq_matrix(mod.weight, H, bits, group, args.damp, sdt))
        store(f"stt_model.{name}", codes, s, b, bits, group)
        log.append({"module": f"stt_model.{name}", "bits": bits, "s": round(time.time() - t, 1)})
        print(f"  stt_model.{name} {bits}b {time.time() - t:.0f}s", flush=True)

    # embeddings: RTN (a lookup table has no input Hessian)
    emb = stt.embed_tokens.weight
    codes, s, b = rtn_matrix(emb, args.embed_bits, args.group_size, sdt)
    store("stt_model.embed_tokens", codes, s, b, args.embed_bits, args.group_size)

    # float leftovers of the LLM (norms, conv1d, A_log, D, dt_bias) pass through unchanged
    from mlx.utils import tree_flatten

    for k, v in tree_flatten(llm.parameters()):
        key = f"stt_model.llm.{k}"
        if not any(key.startswith(p + ".") for p in qconfig):
            tensors[key] = v
    mx.save_safetensors(str(out / "llm.safetensors"), tensors)
    n_w = sum(mod_n for mod_n in (tensors[f"{p}.scales"].size * c["group_size"] for p, c in qconfig.items()))
    n_bits = sum(tensors[f"{p}.scales"].size * c["group_size"] * (c["bits"] + 2 * tensors[f"{p}.scales"].itemsize * 8 / c["group_size"])
                 for p, c in qconfig.items())
    print(f"average bits/weight (incl. scales+biases) over quantized modules: {n_bits / n_w:.3f}", flush=True)
    (out / "quant.json").write_text(json.dumps({"argv": vars(args), "modules": qconfig}, indent=1))
    (out / "gptq_log.json").write_text(json.dumps(log, indent=1))
    print(f"saved {len(tensors)} tensors to {out}", flush=True)


if __name__ == "__main__":
    main()
