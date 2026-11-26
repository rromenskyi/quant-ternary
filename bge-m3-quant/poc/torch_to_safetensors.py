"""BAAI/bge-m3's pytorch_model.bin as an fp32 safetensors file in the HF
layout (embeddings.*, encoder.layer.N.*), without the unused pooler head
and position_ids buffer: the input of mlx_build.py.

    python torch_to_safetensors.py --model SNAPSHOT --out bge-m3-f32.safetensors
"""

from __future__ import annotations

import argparse

import torch
from safetensors.torch import save_file


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    state = torch.load(f"{args.model}/pytorch_model.bin", map_location="cpu", weights_only=True)
    out = {}
    for k, v in state.items():
        for p in ("roberta.", "model."):
            if k.startswith(p):
                k = k[len(p):]
        if k.startswith("pooler.") or k.endswith("position_ids"):
            continue
        out[k] = v.float().contiguous()
    save_file(out, args.out, metadata={"format": "pt"})
    print(f"F32_DONE {args.out} ({len(out)} tensors)", flush=True)


if __name__ == "__main__":
    main()
