"""A checkpoint with a LoRA merged in, tensor by tensor in its own
safetensors shards: W += (B @ A) * alpha / r for every adapted weight,
everything else (vision tower, MTP head, embeddings, configs) copied as it
is -- a model class that drops tensors it doesn't use (the MTP head) never
sees them. The output is a checkpoint like the input, for the quant
pipeline (MODEL_ID=<this folder>).

    python merge_lora.py --model SNAPSHOT --adapter adapter/ --out merged/
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    src, out = Path(args.model), Path(args.out)
    cfg = json.load(open(Path(args.adapter) / "adapter_config.json"))
    scale = cfg["lora_alpha"] / cfg["r"]
    lora = load_file(str(Path(args.adapter) / "adapter_model.safetensors"))
    # base_model.model.<module>.lora_A.weight -> <module>.weight
    deltas = {}
    for key in lora:
        if ".lora_A." not in key:
            continue
        module = key.split(".lora_A.")[0].removeprefix("base_model.model.")
        a = lora[key].float().cuda()
        b = lora[key.replace(".lora_A.", ".lora_B.")].float().cuda()
        deltas[module + ".weight"] = (b @ a) * scale
    out.mkdir(parents=True, exist_ok=True)
    index = json.load(open(src / "model.safetensors.index.json"))
    left = set(deltas)
    for shard in sorted(set(index["weight_map"].values())):
        tensors = load_file(str(src / shard))
        for key in list(tensors):
            if key in deltas:
                w = tensors[key]
                assert w.shape == deltas[key].shape, (key, w.shape, deltas[key].shape)
                tensors[key] = (w.float().cuda() + deltas[key]).to(w.dtype).cpu()
                left.discard(key)
        save_file(tensors, str(out / shard), metadata={"format": "pt"})
        print(f"{shard} written", flush=True)
    if left:
        raise SystemExit(f"adapted weights not in the checkpoint: {sorted(left)[:5]}")
    for f in src.iterdir():
        if f.is_file() and not f.name.endswith(".safetensors"):
            shutil.copy(f, out / f.name)
    print(f"MERGE_DONE {len(deltas)} weights", flush=True)


if __name__ == "__main__":
    main()
