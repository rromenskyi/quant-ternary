"""The evaluation texts for bge-m3 GGUF builds, one JSON per line
{"id", "task", "text", ...}:

  bitext   Tatoeba test pairs (mteb/tatoeba-bitext-mining, CC BY 2.0),
           rus / ukr / cmn / jpn / deu against eng: a sentence must find its
           translation among all the English ones (MTEB's bitext mining).
  code     Python standard library functions with a docstring: the
           docstring must find its function body among all the bodies.
  long     whole standard-library modules cut to about --long-chars
           characters (several thousand tokens): only the cosine to the
           reference is measured on them (bge-m3's 8192-token context).

    python make_eval_set.py --out eval.jsonl
"""

from __future__ import annotations

import argparse
import ast
import gzip
import json
import random
import sysconfig
from pathlib import Path

from huggingface_hub import hf_hub_download

LANGS = ("rus", "ukr", "cmn", "jpn", "deu")


def bitext(per_lang: int) -> list[dict]:
    rows = []
    for lang in LANGS:
        path = hf_hub_download("mteb/tatoeba-bitext-mining", f"test/{lang}-eng.jsonl.gz", repo_type="dataset")
        pairs = [json.loads(line) for line in gzip.open(path, "rt", encoding="utf-8")][:per_lang]
        for i, p in enumerate(pairs):
            rows.append({"id": f"{lang}-{i}", "task": "bitext", "lang": lang, "side": "src", "pair": i,
                         "text": p["sentence1"]})
            rows.append({"id": f"{lang}-{i}-eng", "task": "bitext", "lang": lang, "side": "eng", "pair": i,
                         "text": p["sentence2"]})
    return rows


def code(n: int, seed: int) -> list[dict]:
    std = Path(sysconfig.get_paths()["stdlib"])
    found = []
    for path in sorted(std.rglob("*.py")):
        if {"test", "tests", "idlelib", "site-packages"} & set(path.parts):
            continue
        try:
            source = path.read_text(encoding="utf-8")
            tree = ast.parse(source)
        except (SyntaxError, UnicodeDecodeError, ValueError):
            continue
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and ast.get_docstring(node):
                doc = ast.get_docstring(node)
                body = node.body[1:]
                if len(doc) < 40 or not body:
                    continue
                lines = source.splitlines()
                text = "\n".join(lines[body[0].lineno - 1: node.end_lineno])
                if 200 < len(text) < 4000:
                    found.append((f"{path.relative_to(std)}:{node.name}", doc, text))
    random.Random(seed).shuffle(found)
    rows = []
    for i, (name, doc, text) in enumerate(found[:n]):
        rows.append({"id": f"code-{i}-doc", "task": "code", "side": "doc", "pair": i, "name": name, "text": doc})
        rows.append({"id": f"code-{i}-body", "task": "code", "side": "body", "pair": i, "name": name, "text": text})
    return rows


def long(n: int, chars: int, seed: int) -> list[dict]:
    std = Path(sysconfig.get_paths()["stdlib"])
    files = [p for p in sorted(std.glob("*.py")) if p.stat().st_size > chars]
    random.Random(seed).shuffle(files)
    return [{"id": f"long-{i}", "task": "long", "name": p.name, "text": p.read_text(encoding="utf-8")[:chars]}
            for i, p in enumerate(files[:n])]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--per-lang", type=int, default=1000)
    ap.add_argument("--code", type=int, default=1000)
    ap.add_argument("--long", type=int, default=40)
    ap.add_argument("--long-chars", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rows = bitext(args.per_lang) + code(args.code, args.seed) + long(args.long, args.long_chars, args.seed)
    with open(args.out, "w", encoding="utf-8") as f:
        for r in rows:
            # Tatoeba lines end in "\n": BAAI's tokenizer.json turns trailing
            # whitespace into an extra token before </s>, other copies don't.
            r["text"] = r["text"].strip()
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    counts = {}
    for r in rows:
        counts[r["task"]] = counts.get(r["task"], 0) + 1
    print(f"EVAL_SET_DONE {args.out} {counts}", flush=True)


if __name__ == "__main__":
    main()
