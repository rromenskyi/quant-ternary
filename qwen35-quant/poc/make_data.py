"""wikitext-2 (train / test) and a Python code corpus into <dir>:
wiki.train.raw, wiki.test.raw, code.txt (calibration: the installed
transformers / torch sources) and code.test.txt (held out: the Python
standard library). Shared by the qwen35 and lfm2 pipelines.

    python make_data.py /workspace/data
"""
import sys, pathlib, urllib.request, zipfile, io, sysconfig, transformers, torch
out = pathlib.Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
# Held-out code for the code perplexity: the Python standard library (no
# test suites), which the calibration corpus below doesn't contain.
std = pathlib.Path(sysconfig.get_paths()["stdlib"])
test = [p for p in sorted(std.rglob("*.py")) if not {"test", "tests", "idlelib", "site-packages", "dist-packages"} & set(p.parts)]
(out / "code.test.txt").write_text("\n\n".join(p.read_text(encoding="utf-8", errors="ignore") for p in test), encoding="utf-8")
for split in ("train", "test"):
    if (out / f"wiki.{split}.raw").exists():
        continue
    url = f"https://huggingface.co/datasets/Salesforce/wikitext/resolve/main/wikitext-2-raw-v1/{split}-00000-of-00001.parquet"
    import pandas as pd
    df = pd.read_parquet(io.BytesIO(urllib.request.urlopen(url).read()))
    (out / f"wiki.{split}.raw").write_text("".join(df["text"]), encoding="utf-8")
# Code calibration: the installed libraries' own Python sources (a fixed,
# license-clean corpus that is on every pod), sorted for a stable order.
files = sorted(p for root in (transformers.__path__[0], torch.__path__[0]) for p in pathlib.Path(root).rglob("*.py"))
text, size = [], 0
for p in files:
    t = p.read_text(encoding="utf-8", errors="ignore")
    if len(t) > 2000:
        text.append(t); size += len(t)
    if size > 40_000_000:
        break
(out / "code.txt").write_text("\n\n".join(text), encoding="utf-8")
print({f.name: f.stat().st_size for f in out.iterdir()})
