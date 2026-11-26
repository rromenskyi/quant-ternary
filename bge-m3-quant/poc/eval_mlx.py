"""bge-m3 MLX builds (mlx_build.py) through LLMTray's own embedder runtime
(llmtray_embed.core.Embedder: its loader, quantization, tokenizer,
batching, CLS pooling and normalization), against the transformers
reference, with eval_gguf.py's metrics.

    python eval_mlx.py --llmtray-runtime ~/llmtray/runtime --eval eval.jsonl --ref reference/ \\
        --model bge-m3-mlx-fp16 --model bge-m3-mlx-8bit --out results-mlx.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

import eval_gguf


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--llmtray-runtime", required=True)
    ap.add_argument("--eval", required=True)
    ap.add_argument("--ref", required=True)
    ap.add_argument("--model", action="append", required=True)
    ap.add_argument("--compute-dtype", default="float16", help="LLMTray's; float32 on a CPU without fp16 math")
    ap.add_argument("--pairs", type=int, default=0,
                    help="only the first N pairs per bitext language and of code, and N // 60 long texts (0: all)")
    ap.add_argument("--cache-gb", type=float, default=1.0, help="MLX buffer cache limit (a shared host)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    import mlx.core as mx

    # Unbounded, the cache grew to 6 GB a process on a CPU.
    mx.set_cache_limit(int(args.cache_gb * (1 << 30)))
    sys.path.insert(0, args.llmtray_runtime)
    from llmtray_embed.core import Embedder

    registry = json.load(open(Path(args.llmtray_runtime) / "embedders.json"))
    entries = registry if isinstance(registry, list) else registry.get("embedders", list(registry.values()))
    base = next(e for e in entries if e["id"] == "bge-m3")
    rows = [json.loads(line) for line in open(args.eval, encoding="utf-8")]
    ref = np.load(Path(args.ref) / "dense.npy")
    assert json.load(open(Path(args.ref) / "ids.json")) == [r["id"] for r in rows]
    tokens = json.load(open(Path(args.ref) / "tokens.json"))
    # transformers truncated these at 8192 tokens: out, as in eval_gguf.py
    keep = [i for i, r in enumerate(rows) if len(tokens.get(r["id"], [])) < 8192]
    if args.pairs:
        long_rows = [i for i in keep if rows[i]["task"] == "long"][: max(1, args.pairs // 60)]
        keep = [i for i in keep if (rows[i]["task"] != "long" and rows[i]["pair"] < args.pairs) or i in long_rows]
    rows, ref = [rows[i] for i in keep], ref[keep]
    langs = sorted({r["lang"] for r in rows if r["task"] == "bitext"})
    ref_m, ref_tops = eval_gguf.metrics(ref, rows, langs)
    results = {"reference": ref_m, "builds": {}, "texts": len(rows), "pairs": args.pairs}
    for model_dir in args.model:
        cfg = json.load(open(Path(model_dir) / "config.json"))
        entry = json.loads(json.dumps(base))
        entry["source"]["files"] = {f: "" for f in ("config.json", "tokenizer.json", "model.safetensors")}
        entry["weights"] = {"compute_dtype": args.compute_dtype, "quantization": cfg.get("quantization")}
        t0 = time.time()
        emb = Embedder(entry, model_dir)
        e = emb.embed([r["text"] for r in rows], "document")
        name = Path(model_dir).name
        cos = (e * ref).sum(1)
        r = {"size_bytes": (Path(model_dir) / "model.safetensors").stat().st_size,
             "compute_dtype": args.compute_dtype, "seconds": round(time.time() - t0)}
        for task in ("bitext", "code", "long"):
            c = cos[[i for i, x in enumerate(rows) if x["task"] == task]]
            r[f"cos_ref_{task}_mean"], r[f"cos_ref_{task}_min"] = float(c.mean()), float(c.min())
        m, tops = eval_gguf.metrics(e, rows, langs)
        r.update(m)
        r["top10_agreement"] = float(np.mean([len(set(a) & set(b)) / 10 for k in tops
                                              for a, b in zip(tops[k], ref_tops[k])]))
        results["builds"][name] = r
        print(name, json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()}), flush=True)
    json.dump(results, open(args.out, "w"), indent=2)
    print("EVAL_MLX_DONE", args.out, flush=True)


if __name__ == "__main__":
    main()
