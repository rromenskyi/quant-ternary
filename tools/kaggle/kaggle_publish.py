"""Mirror our Hugging Face releases to Kaggle Models, one variation each.

    KAGGLE_API_TOKEN=... python kaggle_publish.py releases.json [--only <hf_repo> ...]

releases.json lists {hf, model, framework, variation}: the HF repo
(roman220220/<name>) is downloaded file by file into a scratch dir, uploaded
as <user>/<model>/<framework>/<variation> (kagglehub creates the model and
the variation if they don't exist; a rerun adds a new version), then the
scratch copy is deleted. Finished uploads are recorded in done.json next to
the spec, so a rerun skips them. The Kaggle credentials come from the
environment only; nothing is written to disk.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import time
from pathlib import Path

import kagglehub
from huggingface_hub import HfApi, snapshot_download

IGNORE = [".gitattributes", "*.png", "*.jpg"]  # banners stay on HF


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("spec")
    ap.add_argument("--only", nargs="*", help="HF repos to do (default: all in the spec)")
    ap.add_argument("--scratch", default="/workspace/scratch")
    args = ap.parse_args()

    user = os.environ.get("KAGGLE_USERNAME") or kagglehub.whoami()["username"]
    spec_f = Path(args.spec)
    done_f = spec_f.with_name("done.json")
    done = json.loads(done_f.read_text()) if done_f.exists() else {}
    hf = HfApi()
    for r in json.loads(spec_f.read_text()):
        if args.only and r["hf"] not in args.only:
            continue
        handle = f"{user}/{r['model']}/{r['framework']}/{r['variation']}"
        sha = hf.model_info(r["hf"]).sha
        if done.get(handle) == sha:
            print(f"skip {handle} (HF {sha[:7]} already uploaded)", flush=True)
            continue
        local = Path(args.scratch) / r["hf"].replace("/", "__")
        t0 = time.time()
        snapshot_download(r["hf"], local_dir=local, ignore_patterns=IGNORE)
        size = sum(f.stat().st_size for f in local.rglob("*") if f.is_file()) / 1e9
        print(f"{r['hf']}: {size:.1f} GB downloaded in {time.time() - t0:.0f}s", flush=True)
        t1 = time.time()
        kagglehub.model_upload(handle, str(local), license_name=r.get("license", "Apache 2.0"),
                               version_notes=f"Mirror of https://huggingface.co/{r['hf']} at {sha[:7]}",
                               ignore_patterns=[".cache/"])
        print(f"{handle}: uploaded in {time.time() - t1:.0f}s", flush=True)
        shutil.rmtree(local, ignore_errors=True)
        done[handle] = sha
        done_f.write_text(json.dumps(done, indent=1))


if __name__ == "__main__":
    main()
