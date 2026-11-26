"""MLX builds of bge-m3 for LLMTray's embedder runtime (llmtray_embed,
XLM-RoBERTa family: HF-layout model.safetensors, config.json with one
{"group_size", "bits"} "quantization", the modules stored quantized are the
ones with .scales). Variants: fp16, and qN: the linear layers and the word
embeddings (250k x 1024, about half the model) at N bits, group 64; the
position and token-type embeddings and the layer norms stay fp16.

    python mlx_build.py --f32 bge-m3-f32.safetensors --config SNAPSHOT/config.json \\
        --tokenizer tokenizer.json --llmtray-runtime ~/llmtray/runtime --bits 8 --out bge-m3-mlx-8bit
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--f32", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--llmtray-runtime", required=True, help="LLMTray's runtime/ (llmtray_embed)")
    ap.add_argument("--bits", type=int, default=0, help="0: fp16")
    ap.add_argument("--group-size", type=int, default=64)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    sys.path.insert(0, args.llmtray_runtime)
    from llmtray_embed.xlmr import XLMRobertaModel

    config = json.load(open(args.config))
    model = XLMRobertaModel(config)
    weights = XLMRobertaModel.sanitize(mx.load(args.f32))
    model.load_weights(list(weights.items()))
    if args.bits:
        nn.quantize(model, group_size=args.group_size, bits=args.bits,
                    class_predicate=lambda p, m: isinstance(m, nn.Linear) or p == "embeddings.word_embeddings")
        config["quantization"] = {"group_size": args.group_size, "bits": args.bits}
    params = {k: (v.astype(mx.float16) if mx.issubdtype(v.dtype, mx.floating) else v)
              for k, v in tree_flatten(model.parameters())}
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    mx.save_safetensors(str(out / "model.safetensors"), params, metadata={"format": "mlx"})
    config["torch_dtype"] = "float16"
    json.dump(config, open(out / "config.json", "w"), indent=2)
    shutil.copy(args.tokenizer, out / "tokenizer.json")
    size = (out / "model.safetensors").stat().st_size
    print(f"MLX_BUILD_DONE {out} bits={args.bits or 16} {size / 1e6:.0f} MB", flush=True)


if __name__ == "__main__":
    main()
