#!/usr/bin/env python3
"""The TTS backbone's per-frame step alone: ms per frame for the eager
backbone (growing KV caches) vs the compiled step over static K/V buffers
(--chunk), from the same warmed-up cache, with the hidden states compared.

  PYTHONPATH=<fork checkout> python bench_tts_step.py --model models/vc-gptq3 --frames 300 --chunk 512
"""
import argparse
import copy
import time

import mlx.core as mx
from mlx_audio.sts import load

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--frames", type=int, default=300)
ap.add_argument("--chunk", type=int, default=512, help="static buffer growth in frames")
ap.add_argument("--batch", type=int, default=2, help="2: with CFG guidance (the session default), 1: without")
ap.add_argument("--bits", type=int, default=0, help="RTN the backbone's Linears to this many bits first (0: as loaded)")
ap.add_argument("--group", type=int, default=64)
ap.add_argument("--skip", type=int, default=10, help="first steps not timed (compiles)")
a = ap.parse_args()

model = load(a.model)
session = model.create_duplex_session()
stream = getattr(session, "_stream", session)
tts = model.tts_model.tts_model
if a.bits:
    import mlx.nn as nn
    nn.quantize(tts.backbone, group_size=a.group, bits=a.bits,
                class_predicate=lambda _, m: isinstance(m, nn.Linear) and m.weight.shape[-1] % a.group == 0)
cache0 = stream._tts_cache
if a.batch == 1:
    cache0 = [copy.copy(c) for c in cache0]
    for c in cache0:
        c.keys, c.values = c.keys[:1], c.values[:1]
mx.random.seed(0)
dtype = cache0[0].keys.dtype
xs = [mx.random.normal((a.batch, 1, tts.config.hidden_size)).astype(dtype) * 0.5 for _ in range(a.frames)]
mx.eval(xs)


def fresh(c):
    # New array objects: the caches' in-place slice writes must not reach cache0.
    c = copy.copy(c)
    c.keys, c.values = c.keys[:], c.values[:]
    return c


def run(chunk):
    tts.backbone_buffer_chunk = chunk
    cache = [fresh(c) for c in cache0]
    out, times = [], []
    for i, x in enumerate(xs):
        t = time.perf_counter()
        h, cache = tts._backbone_step(x, cache)
        mx.eval(h)
        if i >= a.skip:
            times.append(time.perf_counter() - t)
        out.append(h)
    print("chunk", chunk, "->", type(cache).__name__)
    return mx.concatenate(out, axis=1), 1000 * sum(times) / len(times)


eager, eager_ms = run(0)
static, static_ms = run(a.chunk)
diff = mx.abs(eager.astype(mx.float32) - static.astype(mx.float32))
cos = mx.sum(eager * static) / (mx.linalg.norm(eager.astype(mx.float32)) * mx.linalg.norm(static.astype(mx.float32)))
print(f"bits {a.bits or 'as loaded'}, batch {a.batch}, {a.frames} frames after a {cache0[0].offset}-frame prompt")
print(f"eager  {eager_ms:.2f} ms/frame")
print(f"static {static_ms:.2f} ms/frame (chunk {a.chunk})")
print(f"max_abs {diff.max().item():.4g} cos {cos.item():.6f}")
