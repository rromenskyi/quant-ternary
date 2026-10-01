"""Which of a QAT model's q4_0 Linears are worth 8 bits: a sensitivity scan,
then a size/quality curve.

Starting point: every text Linear on Google's q4_0 grid (exact, as in
qat_aligned_convert.py), embeddings at --embed-bits. Each candidate (one
Linear, or an embedding at 8-bit) is raised alone and scored by how much it
cuts KL to the bf16 QAT master weights on --scan-windows windows, per MB
it adds. The candidates are then taken greedily by that score and the
model is measured for real (PPL, KL, top-1 on --windows windows) at each
size budget.

Built in memory from the master weights; nothing is written but the JSON.

  python qat_sensitivity.py --master E2B-unq --text wiki.test.raw --budgets-mb 50 100 200 400 --json E2B-sens.json
"""
from __future__ import annotations

import argparse
import json
import math
import re

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx.utils import tree_flatten, tree_unflatten
from mlx_lm import load

LINEAR = re.compile(r"^language_model\.model\.layers\.\d+\.(self_attn\.[qkvo]_proj|mlp\.(gate|up|down)_proj"
                    r"|per_layer_input_gate|per_layer_projection)$")
EMBED = re.compile(r"^language_model\.model\.embed_tokens(_per_layer)?$")


def q4_0_module(linear: nn.Linear) -> nn.QuantizedLinear:
    """The Linear on q4_0's grid, as MLX affine 4-bit group 32 (scale=d,
    bias=-8d; d rounded fp16 then bf16, as the converter stores it)."""
    w = np.array(linear.weight.astype(mx.float32))
    rows, cols = w.shape
    x = w.reshape(rows, cols // 32, 32)
    idx = np.abs(x).argmax(-1)[..., None]
    d = (np.take_along_axis(x, idx, -1) / -8.0).astype(np.float32)
    inv = np.where(d != 0, 1.0 / np.where(d == 0, 1, d), 0)
    q = np.minimum(15, (x * inv + 8.5).astype(np.int32)).reshape(rows, cols // 8, 8).astype(np.uint32)
    packed = (q << np.arange(0, 32, 4, dtype=np.uint32)).sum(-1).astype(np.uint32)
    d16 = mx.array(d.squeeze(-1).astype(np.float16)).astype(mx.bfloat16)
    m = nn.QuantizedLinear(cols, rows, bias=False, group_size=32, bits=4)
    m.weight, m.scales, m.biases = mx.array(packed), d16, d16 * -8.0
    return m


def windows(tok, text: str, ctx: int, n: int) -> list[list[int]]:
    ids = tok.encode(text, add_special_tokens=False)
    step = ctx - 1
    return [[tok.bos_token_id] + ids[i: i + step] for i in range(0, len(ids) - step, step)][:n]


def logprobs(model, tokens):
    logits = model(mx.array(tokens)[None])[0].astype(mx.float32)
    return logits - mx.logsumexp(logits, axis=-1, keepdims=True)


def measure(model, wins, ref):
    nll = kl = agree = 0.0
    for w, r in zip(wins, ref):
        r = mx.array(r).astype(mx.float32)
        lp = logprobs(model, w)
        nll -= mx.take_along_axis(lp[:-1], mx.array(w[1:])[:, None], axis=-1).sum().item()
        kl += (mx.exp(r[:-1]) * (r[:-1] - lp[:-1])).sum().item()
        agree += (mx.argmax(r[:-1], -1) == mx.argmax(lp[:-1], -1)).sum().item()
    n = sum(len(w) - 1 for w in wins)
    return {"ppl": math.exp(nll / n), "kl": kl / n, "top1": agree / n}


def nbytes(module) -> int:
    return sum(v.nbytes for _, v in tree_flatten(module.parameters()))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--master", required=True)
    ap.add_argument("--text", required=True)
    ap.add_argument("--ctx", type=int, default=512)
    ap.add_argument("--scan-windows", type=int, default=16)
    ap.add_argument("--windows", type=int, default=128)
    ap.add_argument("--embed-bits", type=int, default=6)
    ap.add_argument("--high-bits", type=int, default=8)
    ap.add_argument("--budgets-mb", type=float, nargs="+", default=[50, 100, 200, 400])
    ap.add_argument("--json", required=True)
    args = ap.parse_args()

    model, tok = load(args.master)
    text = open(args.text).read()
    wins_all = windows(tok, text, args.ctx, max(args.windows, args.scan_windows))
    scan_wins, eval_wins = wins_all[: args.scan_windows], wins_all[: args.windows]
    ref = []
    for w in eval_wins:
        ref.append(np.array(logprobs(model, w).astype(mx.float16)))
    print(f"reference: {len(eval_wins)} windows", flush=True)

    leaves = dict(tree_flatten(model.leaf_modules(), is_leaf=nn.Module.is_module))
    low, high, cost = {}, {}, {}
    for path, m in leaves.items():
        if LINEAR.match(path) and isinstance(m, nn.Linear):
            low[path] = q4_0_module(m)
            high[path] = nn.QuantizedLinear.from_linear(m, group_size=64, bits=args.high_bits)
        elif EMBED.match(path) and isinstance(m, nn.Embedding):
            low[path] = nn.QuantizedEmbedding.from_embedding(m, group_size=64, bits=args.embed_bits)
            high[path] = nn.QuantizedEmbedding.from_embedding(m, group_size=64, bits=args.high_bits)
        else:
            continue
        mx.eval(low[path].parameters(), high[path].parameters())
        cost[path] = nbytes(high[path]) - nbytes(low[path])
    print(f"{len(low)} candidates", flush=True)

    def build(raised: set[str]):
        model.update_modules(tree_unflatten([(p, high[p] if p in raised else low[p]) for p in low]))
        return model

    base_size = sum(nbytes(m) for m in low.values())
    base_scan = measure(build(set()), scan_wins, ref)
    print(f"base (all q4_0 grid): scan KL {base_scan['kl']:.4f}", flush=True)
    gains = {}
    for i, p in enumerate(sorted(low)):
        kl = measure(build({p}), scan_wins, ref)["kl"]
        gains[p] = {"dkl": base_scan["kl"] - kl, "mb": cost[p] / 1e6}
        if i % 20 == 0:
            print(f"  {i}/{len(low)} {p}: dKL {gains[p]['dkl']:.5f} for {gains[p]['mb']:.1f} MB", flush=True)
    order = sorted(gains, key=lambda p: gains[p]["dkl"] / max(gains[p]["mb"], 1e-6), reverse=True)

    curve = [{"budget_mb": 0, "raised": [], "size_gb": base_size / 1e9, **measure(build(set()), eval_wins, ref)}]
    print(f"budget 0: {curve[-1]}", flush=True)
    for budget in args.budgets_mb:
        raised, used = [], 0.0
        for p in order:
            if gains[p]["dkl"] <= 0:
                break
            if used + gains[p]["mb"] <= budget:
                raised.append(p)
                used += gains[p]["mb"]
        r = measure(build(set(raised)), eval_wins, ref)
        curve.append({"budget_mb": budget, "raised": raised, "size_gb": (base_size + used * 1e6) / 1e9, **r})
        print(f"budget {budget} MB ({len(raised)} raised, +{used:.0f} MB): ppl {r['ppl']:.3f} KL {r['kl']:.4f} top-1 {r['top1']:.2%}", flush=True)
    json.dump({"gains": gains, "order": order, "curve": curve}, open(args.json, "w"), indent=1)


if __name__ == "__main__":
    main()
