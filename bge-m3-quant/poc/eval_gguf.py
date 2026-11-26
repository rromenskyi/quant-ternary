"""bge-m3 GGUF builds against the transformers reference (reference.py),
each through llama-server --embedding, as apps use it (/v1/embeddings):

  - tokenizer: /tokenize ids equal the HF tokenizer's on a sample;
  - cosine of every embedding to the reference (by task; min and mean);
  - bitext: top-1 accuracy finding the English translation (and back),
    per language, against the reference's accuracy;
  - code: top-1 accuracy finding a function body from its docstring;
  - top-10 agreement with the reference's ranking;
  - cosine to the --baseline GGUF (the ggml-org Q8_0 LLMTray uses today).

Embeddings are cached per GGUF (<name>.npy next to --out).

    python eval_gguf.py --server llama-server --eval eval.jsonl --ref ref/ \\
        --gguf out/bge-m3-q8_0.gguf ... --baseline ref/bge-m3-q8_0.gguf --out results.json
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
import urllib.request
from pathlib import Path

import numpy as np


def post(port: int, path: str, body: dict) -> dict:
    req = urllib.request.Request(f"http://127.0.0.1:{port}{path}", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=3600))


def serve(server: str, gguf: str, port: int, threads: int) -> subprocess.Popen:
    p = subprocess.Popen([server, "-m", gguf, "--embedding", "--port", str(port), "-c", "8192", "-b", "8192",
                          "-ub", "8192", "-np", "1", "-t", str(threads), "--log-disable"],
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    for _ in range(300):
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2)
            return p
        except Exception:
            if p.poll() is not None:
                raise SystemExit(f"llama-server exited ({p.returncode}) for {gguf}")
            time.sleep(1)
    p.kill()
    raise SystemExit(f"llama-server did not come up for {gguf}")


def embed(port: int, texts: list[str], partial: Path, batch: int = 32) -> np.ndarray:
    """Embeds in batches, saving every 20 batches to `partial` (.npy), which
    a rerun continues from."""
    out = list(np.load(partial)) if partial.exists() else []
    for i in range(len(out) - len(out) % batch, len(texts), batch):
        data = post(port, "/v1/embeddings", {"input": texts[i: i + batch]})["data"]
        out = out[:i] + [d["embedding"] for d in sorted(data, key=lambda d: d["index"])]
        if (i // batch) % 20 == 19:
            np.save(partial, np.asarray(out, np.float32))
    e = np.asarray(out, np.float32)
    return e / np.linalg.norm(e, axis=1, keepdims=True)


def retrieval(e: np.ndarray, rows: list[dict], task: str, q_side: str, d_side: str, group=None) -> tuple[float, list]:
    """Top-1 accuracy of the q_side texts finding their pair among the
    d_side texts (of the same group), and each query's top-10."""
    sel = [i for i, r in enumerate(rows) if r["task"] == task and (group is None or r.get("lang") == group)]
    qs = [i for i in sel if rows[i]["side"] == q_side]
    ds = [i for i in sel if rows[i]["side"] == d_side]
    pair_of = {rows[i]["pair"]: k for k, i in enumerate(ds)}
    sims = e[qs] @ e[ds].T
    top = np.argsort(-sims, axis=1)[:, :10]
    hits = [top[k, 0] == pair_of[rows[i]["pair"]] for k, i in enumerate(qs)]
    return float(np.mean(hits)), top.tolist()


def metrics(e: np.ndarray, rows: list[dict], langs: list[str]) -> tuple[dict, dict]:
    m, tops = {}, {}
    for lang in langs:
        m[f"bitext_{lang}_to_eng"], tops[f"{lang}>"] = retrieval(e, rows, "bitext", "src", "eng", lang)
        m[f"bitext_eng_to_{lang}"], tops[f"<{lang}"] = retrieval(e, rows, "bitext", "eng", "src", lang)
    m["code_doc_to_body"], tops["code"] = retrieval(e, rows, "code", "doc", "body")
    return m, tops


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--server", required=True)
    ap.add_argument("--eval", required=True)
    ap.add_argument("--ref", required=True)
    ap.add_argument("--gguf", action="append", required=True)
    ap.add_argument("--baseline", help="the GGUF the others are also compared to")
    ap.add_argument("--out", required=True)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--port", type=int, default=18190)
    args = ap.parse_args()
    rows = [json.loads(line) for line in open(args.eval, encoding="utf-8")]
    ref_dir = Path(args.ref)
    ref = np.load(ref_dir / "dense.npy")
    assert json.load(open(ref_dir / "ids.json")) == [r["id"] for r in rows], "reference made from another set"
    ref_tokens = json.load(open(ref_dir / "tokens.json"))
    # The reference truncated these at 8192 tokens; llama-server refuses them
    # (HTTP 500) instead of truncating: out of the comparison.
    cut = {k for k, v in ref_tokens.items() if len(v) >= 8192}
    keep = [i for i, r in enumerate(rows) if r["id"] not in cut]
    rows, ref = [rows[i] for i in keep], ref[keep]
    ref_tokens = {k: v for k, v in ref_tokens.items() if k not in cut}
    langs = sorted({r["lang"] for r in rows if r["task"] == "bitext"})
    by_id = {r["id"]: r for r in rows}
    cache = Path(args.out).parent / "embeddings"
    cache.mkdir(parents=True, exist_ok=True)

    ref_m, ref_tops = metrics(ref, rows, langs)
    results = {"reference": ref_m, "builds": {}, "excluded_over_8192_tokens": sorted(cut)}
    ggufs = args.gguf + ([args.baseline] if args.baseline and args.baseline not in args.gguf else [])
    emb = {}
    for gguf in ggufs:
        # ours and ggml-org's Q8_0 share a file name
        name = f"{Path(gguf).parent.name}-{Path(gguf).stem}"
        npy = cache / f"{name}.npy"
        proc = serve(args.server, gguf, args.port, args.threads)
        try:
            bad = []
            for tid, want in ref_tokens.items():
                got = post(args.port, "/tokenize", {"content": by_id[tid]["text"], "add_special": True})["tokens"]
                if got != want:
                    bad.append(tid)
            if npy.exists():
                e = np.load(npy)
            else:
                t0 = time.time()
                e = embed(args.port, [r["text"] for r in rows], cache / f"{name}.partial.npy")
                np.save(npy, e)
                print(f"{name}: embedded {len(rows)} texts in {time.time() - t0:.0f}s", flush=True)
        finally:
            proc.terminate()
            proc.wait()
        emb[name] = e
        cos = (e * ref).sum(1)
        r = {"size_bytes": Path(gguf).stat().st_size, "tokenizer_mismatches": bad,
             "tokenizer_checked": len(ref_tokens)}
        for task in ("bitext", "code", "long"):
            c = cos[[i for i, x in enumerate(rows) if x["task"] == task]]
            r[f"cos_ref_{task}_mean"], r[f"cos_ref_{task}_min"] = float(c.mean()), float(c.min())
        m, tops = metrics(e, rows, langs)
        r.update(m)
        r["top10_agreement"] = float(np.mean([len(set(a) & set(b)) / 10 for k in tops
                                              for a, b in zip(tops[k], ref_tops[k])]))
        results["builds"][name] = r
        print(name, json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()
                                if k != "tokenizer_mismatches"}), "tokenizer mismatches:", len(bad), flush=True)
    if args.baseline:
        base = f"{Path(args.baseline).parent.name}-{Path(args.baseline).stem}"
        for name, e in emb.items():
            c = (e * emb[base]).sum(1)
            results["builds"][name]["cos_baseline_mean"] = float(c.mean())
            results["builds"][name]["cos_baseline_min"] = float(c.min())
        results["baseline"] = base
    json.dump(results, open(args.out, "w"), indent=2)
    print("EVAL_DONE", args.out, flush=True)


if __name__ == "__main__":
    main()
