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
import gc
import json
import os
import math
import re

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx.utils import tree_flatten, tree_unflatten
from mlx_lm import load

from qat_text import windows

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


def logprobs(model, tokens):
    logits = model(mx.array(tokens)[None])[0].astype(mx.float32)
    return logits - mx.logsumexp(logits, axis=-1, keepdims=True)


def measure(model, wins, ref):
    nll = kl = agree = 0.0
    for (w, s), r in zip(wins, ref):
        r = mx.array(r).astype(mx.float32)
        lp = logprobs(model, w)[s:-1]
        nll -= mx.take_along_axis(lp, mx.array(w[s + 1:])[:, None], axis=-1).sum().item()
        kl += (mx.exp(r) * (r - lp)).sum().item()
        agree += (mx.argmax(r, -1) == mx.argmax(lp, -1)).sum().item()
    n = sum(len(w) - 1 - s for w, s in wins)
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
    ap.add_argument("--chat", action="store_true", help="score the text as the model's chat reply (chat-only checkpoints)")
    ap.add_argument("--json", required=True)
    args = ap.parse_args()

    model, tok = load(args.master)
    wins_all = windows(tok, args.text, args.ctx, max(args.windows, args.scan_windows), chat=args.chat)
    scan_wins, eval_wins = wins_all[: args.scan_windows], wins_all[: args.windows]
    ref = []
    for w, s in eval_wins:
        ref.append(np.array(logprobs(model, w)[s:-1].astype(mx.float16)))
    print(f"reference: {len(eval_wins)} windows", flush=True)

    # The master weights wait in host memory; the GPU holds the low-bit
    # model and, during a probe, one candidate at high bits (a 31B's
    # master + both versions of every Linear would not fit).
    leaves = dict(tree_flatten(model.leaf_modules(), is_leaf=nn.Module.is_module))
    master, low, cost = {}, {}, {}
    for path, m in leaves.items():
        if LINEAR.match(path) and isinstance(m, nn.Linear):
            lo = q4_0_module(m)
            kind = "linear"
        elif EMBED.match(path) and isinstance(m, nn.Embedding):
            lo = nn.QuantizedEmbedding.from_embedding(m, group_size=64, bits=args.embed_bits)
            kind = "embed"
        else:
            continue
        mx.eval(lo.parameters())
        master[path] = (kind, np.array(m.weight.astype(mx.float16)))
        low[path] = lo
        model.update_modules(tree_unflatten([(path, lo)]))
        # bits*n/8 + a scale and a bias (bf16) per group
        n = master[path][1].size
        cost[path] = n * args.high_bits / 8 + n / 64 * 4 - nbytes(lo)
    # The bf16 master modules are referenced only here now: drop them, or
    # the GPU keeps the whole master (59 GB on a 31B) next to the low model.
    del leaves, m
    gc.collect()
    mx.clear_cache()
    print(f"{len(low)} candidates", flush=True)

    def high(path: str):
        kind, w = master[path]
        w = mx.array(w).astype(mx.bfloat16)
        if kind == "linear":
            src = nn.Linear(w.shape[1], w.shape[0], bias=False)
            src.weight = w
            return nn.QuantizedLinear.from_linear(src, group_size=64, bits=args.high_bits)
        src = nn.Embedding(w.shape[0], w.shape[1])
        src.weight = w
        return nn.QuantizedEmbedding.from_embedding(src, group_size=64, bits=args.high_bits)

    def build(raised: set[str]):
        model.update_modules(tree_unflatten([(p, high(p) if p in raised else low[p]) for p in low]))
        return model

    base_size = sum(nbytes(m) for m in low.values())
    base_scan = measure(build(set()), scan_wins, ref)
    print(f"base (all q4_0 grid): scan KL {base_scan['kl']:.4f}", flush=True)
    # The scan takes hours on a 31B and a pod can be stopped under it: the
    # gains so far go to <json>.partial every 20 probes, and a rerun with
    # the same scan (same base KL: windows, reference, --chat) goes on from
    # there.
    partial = args.json + ".partial"
    # What a scan's gains depend on: a rerun with any of it changed (the
    # raised width, say, which the base KL doesn't see) starts over.
    settings = {"master": os.path.abspath(args.master), "text": os.path.abspath(args.text), "ctx": args.ctx,
                "scan_windows": args.scan_windows, "windows": args.windows, "embed_bits": args.embed_bits,
                "high_bits": args.high_bits, "chat": bool(args.chat)}
    gains = {}
    if os.path.exists(partial):
        saved = json.load(open(partial))
        if saved.get("settings") == settings and abs(saved.get("base_kl", float("nan")) - base_scan["kl"]) < 1e-9:
            gains = {p: g for p, g in saved["gains"].items() if p in low}
            print(f"resuming: {len(gains)}/{len(low)} probes from {partial}", flush=True)
        else:
            print(f"{partial} is another scan's (settings {saved.get('settings')}, base KL {saved.get('base_kl')}): starting over", flush=True)
    for i, p in enumerate(sorted(low)):
        if p in gains:
            continue
        model.update_modules(tree_unflatten([(p, high(p))]))
        kl = measure(model, scan_wins, ref)["kl"]
        model.update_modules(tree_unflatten([(p, low[p])]))
        gains[p] = {"dkl": base_scan["kl"] - kl, "mb": cost[p] / 1e6}
        if i % 20 == 0:
            print(f"  {i}/{len(low)} {p}: dKL {gains[p]['dkl']:.5f} for {gains[p]['mb']:.1f} MB", flush=True)
            json.dump({"settings": settings, "base_kl": base_scan["kl"], "gains": gains}, open(partial + ".tmp", "w"))
            os.replace(partial + ".tmp", partial)
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
    json.dump({"settings": settings, "gains": gains, "order": order, "curve": curve}, open(args.json, "w"), indent=1)
    if os.path.exists(partial):
        os.remove(partial)


if __name__ == "__main__":
    main()
