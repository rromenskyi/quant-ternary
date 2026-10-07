"""Model cards for the Kaggle mirror (run after kaggle_publish.py).

    KAGGLE_API_TOKEN=... python kaggle_cards.py releases.json cards.json

cards.json gives each Kaggle model its title, subtitle and description
(the family page). Each variation gets the README of its Hugging Face repo
as its overview (YAML header dropped, relative links and images made
absolute to the HF repo, so the banners show), a usage section for its
format, and the Google base model from the HF card's `base_model` as its
external base model.
"""
from __future__ import annotations

import argparse
import json
import os
import re

import kagglehub
from google.protobuf.field_mask_pb2 import FieldMask
from huggingface_hub import HfApi, hf_hub_download
from kagglehub.clients import build_kaggle_client
from kagglesdk.models.types.model_api_service import ApiUpdateModelInstanceRequest, ApiUpdateModelRequest
from kagglesdk.models.types.model_enums import ModelInstanceType

USAGE = {
    "gguf": """```python
import kagglehub
path = kagglehub.model_download("{handle}")
```
Run with llama.cpp (`llama-server -m <path>/<file>.gguf`) or Ollama (the repo's `Modelfile`). The same files are on Hugging Face: [{hf}](https://huggingface.co/{hf}).""",
    "other": """MLX build for Apple Silicon Macs (does not run on Kaggle's CUDA/TPU machines).
```python
import kagglehub
path = kagglehub.model_download("{handle}")
```
Then `mlx_lm.generate --model <path> --prompt "Hello"`, or open it in [LLMTray](https://www.ipsupport.us/llmtray/). The same files are on Hugging Face: [{hf}](https://huggingface.co/{hf}).""",
}


def hf_readme(repo: str) -> str:
    text = open(hf_hub_download(repo, "README.md")).read()
    text = re.sub(r"\A---\n.*?\n---\n", "", text, flags=re.S)  # YAML front matter
    base = f"https://huggingface.co/{repo}"
    # relative images / links -> absolute (resolve/ for files, blob/ for docs)
    text = re.sub(r"(!\[[^\]]*\]\()(?!https?://|#)([^)]+)\)", lambda m: f"{m[1]}{base}/resolve/main/{m[2]})", text)
    text = re.sub(r"(?<!!)(\[[^\]]*\]\()(?!https?://|#|mailto:)([^)]+)\)", lambda m: f"{m[1]}{base}/blob/main/{m[2]})", text)
    text = re.sub(r'(<img[^>]*src=")(?!https?://)([^"]+)"', lambda m: f'{m[1]}{base}/resolve/main/{m[2]}"', text)
    return text.strip()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("releases")
    ap.add_argument("cards")
    args = ap.parse_args()
    user = os.environ.get("KAGGLE_USERNAME") or kagglehub.whoami()["username"]
    releases = json.load(open(args.releases))
    cards = json.load(open(args.cards))
    hf = HfApi()
    with build_kaggle_client() as client:
        api = client.models.model_api_client
        for slug, c in cards.items():
            r = ApiUpdateModelRequest()
            r.owner_slug, r.model_slug = user, slug
            r.title, r.subtitle, r.description = c["title"], c["subtitle"], c["description"]
            r.update_mask = FieldMask(paths=["title", "subtitle", "description"])
            try:
                api.update_model(r)
                print(f"model {slug}: card set", flush=True)
            except Exception as e:  # not uploaded yet
                print(f"model {slug}: skipped ({str(e)[:120]})", flush=True)
        for rel in releases:
            info = hf.model_info(rel["hf"])
            base = (info.card_data or {}).get("base_model") if info.card_data else None
            base = base[0] if isinstance(base, list) else base
            handle = f"{user}/{rel['model']}/{rel['framework']}/{rel['variation']}"
            r = ApiUpdateModelInstanceRequest()
            r.owner_slug, r.model_slug = user, rel["model"]
            r.framework = kagglehub.handle.parse_model_handle(handle).framework_enum()
            r.instance_slug = rel["variation"]
            r.overview = hf_readme(rel["hf"])
            r.usage = USAGE[rel["framework"]].format(handle=handle, hf=rel["hf"])
            paths = ["overview", "usage"]
            if base:
                r.model_instance_type = ModelInstanceType.MODEL_INSTANCE_TYPE_EXTERNAL_VARIANT
                r.external_base_model_url = f"https://huggingface.co/{base}"
                paths += ["model_instance_type", "external_base_model_url"]
            r.update_mask = FieldMask(paths=paths)
            try:
                api.update_model_instance(r)
                print(f"{handle}: card set (base {base})", flush=True)
            except Exception as e:  # a variation not uploaded yet
                print(f"{handle}: skipped ({str(e)[:120]})", flush=True)


if __name__ == "__main__":
    main()
