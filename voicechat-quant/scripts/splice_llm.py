#!/usr/bin/env python3
"""Splice a pod-quantized LLM (gptq_llm.py output) into a full VoiceChat MLX dir.

Takes every non-LLM tensor (perception, RNNT, TTS, codec, ...) and all
tokenizer/config files from --base (e.g. the mlx-community 4-bit snapshot,
whose non-LLM parts are unquantized bf16), replaces stt_model.{llm,
embed_tokens, lm_head, function_head} with --llm's tensors, and rewrites the
config's per-module quantization entries. No compute beyond a copy.

  python splice_llm.py --base <snapshot> --llm pod_out/gptq3 --out models/vc-gptq3
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import shutil
from pathlib import Path

import mlx.core as mx

LLM_PREFIXES = ("stt_model.llm.", "stt_model.embed_tokens.", "stt_model.lm_head.", "stt_model.function_head.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--llm", required=True, help="dir with llm.safetensors + quant.json")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    base = Path(glob.glob(os.path.expanduser(args.base))[0])
    llm_dir, out = Path(args.llm), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    for f in base.iterdir():
        if f.suffix == ".safetensors" or f.name in ("model.safetensors.index.json", "config.json"):
            continue
        dst = out / f.name
        if f.is_dir():
            shutil.copytree(f.resolve(), dst, dirs_exist_ok=True)
        else:
            shutil.copy2(f.resolve(), dst)

    # Only the quantized modules are swapped. The LLM's float leftovers (norms,
    # mamba conv1d, A_log, D, dt_bias) stay the base's own stored tensors: gptq_llm
    # saves them post-sanitize (MLX layout), while checkpoints store them pre-sanitize
    # and the loader transposes 3-D conv weights again -- mixing them breaks conv1d.
    modules = json.loads((llm_dir / "quant.json").read_text())["modules"]
    replaced = {f"{m}.{p}" for m in modules for p in ("weight", "scales", "biases")}
    tensors = {}
    for f in sorted(base.glob("*.safetensors")):
        for k, v in mx.load(str(f)).items():
            if k not in replaced:
                tensors[k] = v
    new = mx.load(str(llm_dir / "llm.safetensors"))
    tensors.update({k: v for k, v in new.items() if k in replaced})
    missing = replaced - set(new)
    assert not missing, f"llm.safetensors lacks {sorted(missing)[:5]}"

    cfg = json.loads((base / "config.json").read_text())
    q = cfg["quantization"]
    for k in list(q):
        if k.startswith(LLM_PREFIXES) or (isinstance(q[k], dict) and any(k.startswith(p.rstrip(".")) for p in LLM_PREFIXES)):
            q.pop(k)
    q.update(modules)
    cfg["quantization"] = q
    if "quantization_config" in cfg:
        cfg["quantization_config"] = q
    (out / "config.json").write_text(json.dumps(cfg, indent=2))

    # shard at ~5 GB like the base
    shards, cur, size = [], {}, 0
    for k in sorted(tensors):
        v = tensors[k]
        if cur and size + v.nbytes > 5e9:
            shards.append(cur)
            cur, size = {}, 0
        cur[k] = v
        size += v.nbytes
    shards.append(cur)
    weight_map = {}
    for i, shard in enumerate(shards, 1):
        name = f"model-{i:05d}-of-{len(shards):05d}.safetensors"
        mx.save_safetensors(str(out / name), shard, metadata={"format": "mlx"})
        weight_map.update({k: name for k in shard})
    total = sum(v.nbytes for v in tensors.values())
    (out / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {"total_size": total}, "weight_map": weight_map}, indent=1)
    )
    print(f"{out}: {len(tensors)} tensors, {total / 1e9:.2f} GB, {len(shards)} shards")


if __name__ == "__main__":
    main()
