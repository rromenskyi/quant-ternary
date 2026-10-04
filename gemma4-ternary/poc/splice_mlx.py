"""Build a runnable MLX model from exported ternary tensors.

    python splice_mlx.py --base ~/.lmstudio/models/roman220220/gemma-4-12B-it-qat-mlx \\
        --ternary build/ternary_step_000871.safetensors --out build/gemma-4-12B-it-ternary-step871

Takes everything else from --base (our 12B QAT MLX build: its embeddings,
norms, vision/audio embedders, tokenizer, chat template), replaces the
text-decoder Linears with the ternary ones, and gives each of them a
{bits: 2, group_size: <group>} override in config.json's quantization map
(MLX module path language_model.model.layers.N.<proj>).
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path

import mlx.core as mx
from safetensors import safe_open


def hf_to_module(name: str) -> str:
    # model.language_model.layers.N.x -> language_model.model.layers.N.x
    m = re.fullmatch(r"model\.language_model\.(layers\.\d+\..+)", name)
    if not m:
        raise ValueError(name)
    return f"language_model.model.{m[1]}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--ternary", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--shard-gb", type=float, default=2.0)
    args = ap.parse_args()

    base, out = Path(args.base).expanduser(), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    with safe_open(args.ternary, framework="numpy") as sf:
        meta = sf.metadata()
        group = int(meta["group_size"])
        tern = {}
        for k in sf.keys():
            a = sf.get_tensor(k)
            arr = mx.array(a)
            if k.endswith((".scales", ".biases")):
                arr = arr.view(mx.bfloat16)  # stored as raw bf16 bits
            tern[k] = arr
    linears = sorted({k.rsplit(".", 1)[0] for k in tern})

    weights = {}
    for f in sorted(base.glob("model-*.safetensors")):
        for k, v in mx.load(str(f)).items():
            if k.rsplit(".", 1)[0] in linears and k.endswith((".weight", ".scales", ".biases")):
                continue
            weights[k] = v
    weights.update(tern)

    cfg = json.loads((base / "config.json").read_text())
    for key in ("quantization", "quantization_config"):
        q = cfg.get(key)
        if q is None:
            continue
        for lin in linears:
            q[hf_to_module(lin)] = {"group_size": group, "bits": 2}
    cfg["ternary"] = {"source": meta.get("ckpt"), "group_size": group, "linears": len(linears)}
    (out / "config.json").write_text(json.dumps(cfg, indent=2))

    shards, cur, size = [], {}, 0
    for k in sorted(weights):
        cur[k] = weights[k]
        size += weights[k].nbytes
        if size >= args.shard_gb * 1e9:
            shards.append(cur); cur, size = {}, 0
    if cur:
        shards.append(cur)
    index = {"metadata": {"total_size": sum(v.nbytes for v in weights.values())}, "weight_map": {}}
    for i, sh in enumerate(shards):
        name = f"model-{i + 1:05d}-of-{len(shards):05d}.safetensors"
        mx.save_safetensors(str(out / name), sh, metadata={"format": "mlx"})
        index["weight_map"].update({k: name for k in sh})
    (out / "model.safetensors.index.json").write_text(json.dumps(index, indent=2))
    for f in base.iterdir():
        if f.suffix in (".json", ".jinja") and not f.name.startswith(".") \
                and f.name not in ("config.json", "model.safetensors.index.json") \
                or f.name in ("tokenizer.json",):
            shutil.copy2(f, out / f.name)
    print(json.dumps({"linears": len(linears), "tensors": len(weights), "bytes": index["metadata"]["total_size"],
                      "out": str(out)}))


if __name__ == "__main__":
    main()
