#!/usr/bin/env python3
"""[mac] Bake round-to-nearest quantization of chosen parts into a VoiceChat
checkpoint, so it loads quantized with no flags (the fork's loader quantizes
every layer that has `<path>.scales`, with the bits of its per-layer entry
in config.json `quantization`).

  python bake_rtn.py --model models/vc-gptq3 --out models/vc-mixed \
      --rtn tts_model.tts_model.backbone:8:64 --rtn tts_model.tts_model.mog_head:8:64

Works on the checkpoint's own tensors, not a load/save round trip (the
codec's sanitize would convert its layouts twice). The model is loaded only
to find which weights are nn.Linear (with their shapes); a Linear already
quantized, one whose input width isn't a multiple of the group, or one
matched by --skip is left as is. The source's other files are copied.
"""

import argparse
import json
import re
import shutil
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten

SHARD_BYTES = 5 << 30


def linear_paths(model_dir: str) -> dict:
    from mlx_audio.sts import load

    model = load(model_dir)
    return {
        path: tuple(module.weight.shape)
        for path, module in tree_flatten(model.leaf_modules(), is_leaf=nn.Module.is_module)
        if type(module) is nn.Linear
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--rtn", action="append", required=True, metavar="PREFIX:BITS:GROUP")
    ap.add_argument("--skip", action="append", default=[], metavar="REGEX")
    args = ap.parse_args()

    src, out = Path(args.model), Path(args.out)
    specs = []
    for spec in args.rtn:
        prefix, bits, group = spec.rsplit(":", 2)
        specs.append((prefix, int(bits), int(group)))
    skips = [re.compile(s) for s in args.skip]

    linears = linear_paths(str(src))
    weights = {}
    for f in sorted(src.glob("*.safetensors")):
        weights.update(mx.load(str(f)))
    config = json.loads((src / "config.json").read_text())
    quant = config.setdefault("quantization", {"group_size": 64, "bits": 4})

    report = {}
    for path, shape in sorted(linears.items()):
        spec = next((s for s in specs if path.startswith(s[0])), None)
        if spec is None or any(r.search(path) for r in skips):
            continue
        _, bits, group = spec
        key = f"{path}.weight"
        w = weights.get(key)
        if w is None or tuple(w.shape) != shape or f"{path}.scales" in weights or shape[-1] % group:
            report.setdefault("skipped", []).append(path)
            continue
        q, scales, biases = mx.quantize(w, group_size=group, bits=bits)
        weights[key] = q
        weights[f"{path}.scales"] = scales.astype(w.dtype)
        weights[f"{path}.biases"] = biases.astype(w.dtype)
        quant[path] = {"group_size": group, "bits": bits}
        report[spec[0]] = report.get(spec[0], 0) + 1
    config["quantization_config"] = quant

    out.mkdir(parents=True, exist_ok=True)
    for f in src.iterdir():
        if f.suffix == ".safetensors" or f.name in ("model.safetensors.index.json", "config.json"):
            continue
        dst = out / f.name
        shutil.copytree(f, dst, dirs_exist_ok=True) if f.is_dir() else shutil.copyfile(f, dst)
    shards, cur, size = [], {}, 0
    for k in sorted(weights):
        n = weights[k].nbytes
        if cur and size + n > SHARD_BYTES:
            shards.append(cur)
            cur, size = {}, 0
        cur[k] = weights[k]
        size += n
    shards.append(cur)
    index = {"metadata": {"total_size": sum(v.nbytes for v in weights.values())}, "weight_map": {}}
    for i, shard in enumerate(shards, 1):
        name = f"model-{i:05d}-of-{len(shards):05d}.safetensors"
        mx.save_safetensors(str(out / name), shard, metadata={"format": "mlx"})
        index["weight_map"].update({k: name for k in shard})
    (out / "model.safetensors.index.json").write_text(json.dumps(index, indent=2))
    (out / "config.json").write_text(json.dumps(config, indent=2))
    total = index["metadata"]["total_size"] / 1e9
    print(json.dumps({"quantized": {k: v for k, v in report.items() if k != "skipped"},
                      "skipped": len(report.get("skipped", [])), "total_gb": round(total, 3)}))


if __name__ == "__main__":
    main()
