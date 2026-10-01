#!/usr/bin/env python3
"""[mac] Stage and publish a VoiceChat checkpoint as a public model repo.

  python hf_publish.py --model models/vc-gptq3 \
      --card cards/NemotronLabs-VoiceChat-11B-gptq-mlx-3bit.md \
      --notices ../scratch/nvbase --banner ../../cards_assets/llmtray-banner.png \
      --repo roman220220/NemotronLabs-VoiceChat-11B-gptq-mlx-3bit [--public]

Staging (release/<repo name>/) hard-links the weights, so it takes no extra
disk: the checkpoint without its source README, the card as README.md, the
base model's LICENSE and notice files unchanged, and the banner. The repo is
created private and uploaded; it is made public only with --public, and only
once every staged file is on the Hub with the same size.
"""

import argparse
import os
import shutil
import sys
from pathlib import Path

from huggingface_hub import HfApi

NOTICES = ("LICENSE", "bias.md", "explainability.md", "privacy.md", "safety.md")


def stage(args) -> Path:
    out = Path("release") / args.repo.split("/")[-1]
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    model = Path(args.model)
    for p in sorted(model.rglob("*")):
        rel = p.relative_to(model)
        if p.is_dir() or rel.as_posix() in ("README.md", ".gitattributes"):
            continue
        dst = out / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        os.link(p, dst)
    shutil.copyfile(args.card, out / "README.md")
    for name in NOTICES:
        src = Path(args.notices) / name
        if not src.is_file():
            sys.exit(f"missing notice file {src}")
        shutil.copyfile(src, out / name)
    shutil.copyfile(args.banner, out / Path(args.banner).name)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--card", required=True)
    ap.add_argument("--notices", required=True, help="dir with the base repo's LICENSE and notice .md files")
    ap.add_argument("--banner", required=True)
    ap.add_argument("--repo", required=True)
    ap.add_argument("--public", action="store_true", help="flip to public after a verified upload")
    args = ap.parse_args()

    folder = stage(args)
    api = HfApi()
    api.create_repo(args.repo, private=True, exist_ok=True, repo_type="model")
    api.upload_folder(repo_id=args.repo, folder_path=str(folder), commit_message="Upload model, card, license and notices")
    remote = {f.path: f.size for f in api.list_repo_tree(args.repo, recursive=True) if hasattr(f, "size")}
    ok = True
    for p in sorted(folder.rglob("*")):
        if p.is_file():
            key = p.relative_to(folder).as_posix()
            status = "ok" if remote.get(key) == p.stat().st_size else "MISSING/SIZE MISMATCH"
            ok &= status == "ok"
            print(f"{status} {key} {p.stat().st_size}")
    if not ok:
        sys.exit("upload incomplete; repo left private")
    if args.public:
        api.update_repo_settings(args.repo, private=False)
    print(f"https://huggingface.co/{args.repo} private={api.model_info(args.repo).private}")


if __name__ == "__main__":
    main()
