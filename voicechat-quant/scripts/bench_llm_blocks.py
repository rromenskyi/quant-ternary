#!/usr/bin/env python3
"""Where one LLM frame step goes, by block type (Mamba / attention / MLP),
with a sync after every block (so the shares, not the absolute total, count).

  PYTHONPATH=<fork> python bench_llm_blocks.py --model models/vc-gptq3 --steps 60
"""
import argparse, collections, json, time
import mlx.core as mx

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--steps", type=int, default=60)
ap.add_argument("--warmup", type=int, default=10)
ap.add_argument("--no-sync", action="store_true", help="time the whole step only")
a = ap.parse_args()

from mlx_audio.sts import load
from mlx_audio.lm.models import nemotron_h as nh

model = load(a.model)
stt = model.stt_model
cache = stt.make_cache()
hidden = stt.llm.args.hidden_size if hasattr(stt.llm, "args") else 4480
times = collections.defaultdict(float)
counts = collections.Counter()
orig = nh.NemotronHBlock.__call__
if not a.no_sync:
    def timed(self, *args, **kw):
        t = time.perf_counter(); out = orig(self, *args, **kw); mx.eval(out)
        kind = type(self.mixer).__name__
        times[kind] += time.perf_counter() - t; counts[kind] += 1
        return out
    nh.NemotronHBlock.__call__ = timed
x = (mx.random.normal((1, 1, hidden)) * 0.02).astype(mx.bfloat16)
total = []
for i in range(a.warmup + a.steps):
    if i == a.warmup:
        times.clear(); counts.clear()
    t = time.perf_counter()
    out = stt(x, cache=cache)
    mx.eval(out)
    if i >= a.warmup:
        total.append(time.perf_counter() - t)
res = {"step_ms": round(1000 * sum(total) / len(total), 2)}
for k in times:
    res[k] = {"ms_per_step": round(1000 * times[k] / a.steps, 2), "layers": counts[k] // a.steps}
print(json.dumps(res))
