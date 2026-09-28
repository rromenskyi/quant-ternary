#!/usr/bin/env python3
"""[pod] Upload one result folder to a PRIVATE Hugging Face repo and verify it.

  python hf_upload.py --repo roman220220/NemotronLabs-VoiceChat-11B-quant-experiments \
      --folder /workspace/out/gptq3 --path-in-repo gptq3

The repo is created private if missing (an existing repo's visibility is
never changed here -- going public is a manual decision). Authentication is
the pod's ~/.cache/huggingface/token (copied by pod_pipeline.sh via stdin,
never printed). Exits non-zero unless every local file is on the Hub with
the same size.
"""

import argparse
import sys
from pathlib import Path

from huggingface_hub import HfApi


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--folder", required=True)
    ap.add_argument("--path-in-repo", required=True)
    args = ap.parse_args()
    api = HfApi()
    api.create_repo(args.repo, private=True, exist_ok=True, repo_type="model")
    info = api.model_info(args.repo)
    print(f"repo {args.repo} private={info.private}")
    if not info.private:
        sys.exit("refusing: repo is public")
    folder = Path(args.folder)
    api.upload_folder(
        repo_id=args.repo,
        folder_path=str(folder),
        path_in_repo=args.path_in_repo,
        commit_message=f"voicechat-quant: {args.path_in_repo}",
    )
    remote = {
        f.path: f.size
        for f in api.list_repo_tree(args.repo, path_in_repo=args.path_in_repo, recursive=True)
        if hasattr(f, "size")
    }
    ok = True
    for p in sorted(folder.rglob("*")):
        if p.is_file():
            key = f"{args.path_in_repo}/{p.relative_to(folder)}"
            status = "ok" if remote.get(key) == p.stat().st_size else "MISSING/SIZE MISMATCH"
            ok &= status == "ok"
            print(f"{status} {key} {p.stat().st_size}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
