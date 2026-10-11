"""The source LFM2 checkpoint with gptq_lfm2.py's quantized weights swapped
in (everything else copied byte for byte), plus quant_recipe.json, for
convert_lfm2.py. The fused experts of the layer files go back to the
checkpoint's per-expert names: gate_up_proj[e] = [w1; w3], down_proj[e] = w2.
Single-file and sharded checkpoints.

    python assemble_lfm2.py --model SNAPSHOT --work /workspace/gptq-lfm2-8b --out /workspace/ongrid-lfm2-8b
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from safetensors.torch import load_file, save_file


def layer_tensors(f: Path) -> dict:
    """A layer file's weights under the checkpoint's names."""
    out = {}
    for key, t in load_file(f).items():
        if ".gptq_" in key:
            continue
        base, _, name = key.rpartition(".experts.")
        if name == "gate_up_proj":
            mid = t.shape[1] // 2
            for e in range(t.shape[0]):
                out[f"{base}.experts.{e}.w1.weight"] = t[e, :mid]
                out[f"{base}.experts.{e}.w3.weight"] = t[e, mid:]
        elif name == "down_proj":
            for e in range(t.shape[0]):
                out[f"{base}.experts.{e}.w2.weight"] = t[e]
        else:
            out[key] = t
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--work", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    src, work, out = Path(args.model), Path(args.work), Path(args.out)
    progress = json.load(open(work / "progress.json"))
    if progress["done"] + 1 != progress["layers"]:
        raise SystemExit(f"calibration not finished: {progress}")

    new = {}
    for f in sorted((work / "layers").glob("*.safetensors")):
        new.update(layer_tensors(f))
    out.mkdir(parents=True, exist_ok=True)
    index = src / "model.safetensors.index.json"
    shards = sorted(set(json.load(open(index))["weight_map"].values())) if index.exists() else ["model.safetensors"]
    left = set(new)
    for shard in shards:
        tensors = load_file(src / shard)
        for key in tensors:
            if key in new:
                assert new[key].shape == tensors[key].shape and new[key].dtype == tensors[key].dtype, key
                tensors[key] = new[key].contiguous()
                left.discard(key)
        save_file(tensors, out / shard, metadata={"format": "pt"})
        print(f"{shard} written", flush=True)
    if left:
        raise SystemExit(f"not in any source shard: {sorted(left)[:5]}")
    for f in src.iterdir():
        if f.is_file() and not f.name.endswith(".safetensors"):
            shutil.copy(f, out / f.name)
    shutil.copy(work / "quant_recipe.json", out / "quant_recipe.json")
    print(f"ASSEMBLE_DONE ({len(new)} tensors replaced)", flush=True)


if __name__ == "__main__":
    main()
