#!/usr/bin/env python3
"""Perception (streaming FastConformer) alone: ms per 80 ms frame and the
encoded outputs, for comparing code variants of the fork.

  PYTHONPATH=<fork checkout> python bench_perception.py --audio eval/audio/q01.wav --out /tmp/a.npz
  python bench_perception.py --compare /tmp/a.npz /tmp/b.npz
"""
import argparse, json, sys, time, wave
from pathlib import Path
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--model", default=None, help="VoiceChat dir; default: the mlx-community 4-bit snapshot")
ap.add_argument("--audio", nargs="*", default=[])
ap.add_argument("--silence-s", type=float, default=2.0, help="silence after each clip")
ap.add_argument("--warmup-frames", type=int, default=10)
ap.add_argument("--out", default=None)
ap.add_argument("--rtn-bits", type=int, default=0, help="round-to-nearest the perception Linears to this many bits (0: off)")
ap.add_argument("--rtn-group", type=int, default=64)
ap.add_argument("--compare", nargs=2, default=None)
a = ap.parse_args()

if a.compare:
    x, y = np.load(a.compare[0]), np.load(a.compare[1])
    for k in x.files:
        u, v = x[k].astype(np.float32), y[k].astype(np.float32)
        cos = float((u * v).sum() / (np.linalg.norm(u) * np.linalg.norm(v) + 1e-12))
        print(k, "max_abs", float(np.abs(u - v).max()), "cos", round(cos, 6))
    sys.exit()

import mlx.core as mx
from huggingface_hub import snapshot_download
from mlx_audio.sts import load
from mlx_audio.sts.models.nemotron_voicechat.streaming import VoiceChatStreamingSession  # noqa: F401

model_dir = a.model or snapshot_download("mlx-community/NemotronLabs-VoiceChat-11B-4bit", local_files_only=True)
model = load(model_dir)
if a.rtn_bits:
    import mlx.nn as nn
    nn.quantize(model.stt_model.perception, group_size=a.rtn_group, bits=a.rtn_bits,
                class_predicate=lambda _, m: isinstance(m, nn.Linear) and m.weight.shape[-1] % a.rtn_group == 0)
session = model.create_duplex_session()
stream = session._stream if hasattr(session, "_stream") else session
fs = stream.frame_samples

def read(path):
    with wave.open(path) as w:
        pcm = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32) / 32768
    return pcm

audio = [read(p) for p in a.audio] or [np.zeros(16000, np.float32)]
frames = []
for pcm in audio:
    pcm = np.concatenate([pcm, np.zeros(int(a.silence_s * 16000), np.float32)])
    frames += [pcm[i:i + fs] for i in range(0, len(pcm) - fs + 1, fs)]

outs, times = [], []
for i, f in enumerate(frames):
    t = time.perf_counter()
    projected, encoded = stream._perception_step(mx.array(f))
    mx.eval(projected, encoded)
    dt = (time.perf_counter() - t) * 1000
    if i >= a.warmup_frames:
        times.append(dt)
    outs.append(np.array(projected.astype(mx.float32)))
t = np.array(times)
print(json.dumps({"frames": len(times), "ms_mean": round(float(t.mean()), 2), "ms_p50": round(float(np.median(t)), 2),
                  "ms_p95": round(float(np.percentile(t, 95)), 2)}))
if a.out:
    np.savez(a.out, projected=np.concatenate(outs, axis=1))
