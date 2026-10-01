#!/usr/bin/env python3
"""LLM one-frame step time, eager vs the compiled decode path, and whether
the outputs match (same inputs, fresh caches for each run).

  PYTHONPATH=<fork> python bench_llm_step.py --model models/vc-gptq3 --steps 80
"""
import argparse, dataclasses, json, time
import mlx.core as mx

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--steps", type=int, default=80)
ap.add_argument("--warmup", type=int, default=10)
a = ap.parse_args()

from mlx_audio.sts import load
from mlx_audio.lm.models import nemotron_h as nh

model = load(a.model)
stt = model.stt_model
hidden = 4480
mx.random.seed(0)
xs = [(mx.random.normal((1, 1, hidden)) * 0.05).astype(mx.bfloat16) for _ in range(a.warmup + a.steps)]

def arrays(out):
    return [v for v in vars(out).values() if isinstance(v, mx.array)] if dataclasses.is_dataclass(out) else [out]

def run(compiled):
    nh.NemotronHModel.compile_decode = compiled
    cache = stt.make_cache()
    ts, last = [], None
    for i, x in enumerate(xs):
        t = time.perf_counter()
        out = arrays(stt(x, cache=cache))
        mx.eval(out)
        if i >= a.warmup:
            ts.append(time.perf_counter() - t)
        last = out
    return 1000 * sum(ts) / len(ts), [o.astype(mx.float32) for o in last]

eager_ms, eager_out = run(False)
comp_ms, comp_out = run(True)
diff = max(float(mx.abs(u - v).max()) for u, v in zip(eager_out, comp_out))
print(json.dumps({"eager_ms": round(eager_ms, 2), "compiled_ms": round(comp_ms, 2), "max_abs_diff_last_step": diff}))
