"""The source checkpoint with gptq_qwen35.py's quantized decoder weights
swapped in, shard by shard (the vision tower, MTP head, embeddings, norms,
router and everything else copied byte for byte), plus quant_recipe.json.
convert_mlx.py turns it into MLX at the same bits and, with --gptq-work,
writes GPTQ's own codes (MLX's re-quantization alone changes some:
docs/GPTQ_EXACT_CODES.md).

    python assemble_checkpoint.py --model SNAPSHOT --work /workspace/ornith-gptq --out /workspace/ornith-ongrid
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from safetensors import safe_open
from safetensors.torch import load_file, save_file


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

    where: dict[str, Path] = {}
    for f in sorted((work / "layers").glob("*.safetensors")):
        with safe_open(f, "pt") as s:
            where.update({k: f for k in s.keys() if ".gptq_" not in k})
    out.mkdir(parents=True, exist_ok=True)
    if not (src / "model.safetensors.index.json").exists():
        raise SystemExit(f"{src}: a sharded checkpoint (model.safetensors.index.json) is expected")
    index = json.load(open(src / "model.safetensors.index.json"))
    cache: dict[Path, dict] = {}
    left = set(where)
    for shard in sorted(set(index["weight_map"].values())):
        tensors = load_file(src / shard)
        for key in list(tensors):
            if key in where:
                f = where[key]
                if f not in cache:
                    cache = {f: load_file(f)}  # one layer file at a time
                new = cache[f][key]
                assert new.shape == tensors[key].shape and new.dtype == tensors[key].dtype, key
                tensors[key] = new
                left.discard(key)
        save_file(tensors, out / shard, metadata={"format": "pt"})
        print(f"{shard} written", flush=True)
    if left:
        raise SystemExit(f"not in any source shard: {sorted(left)[:5]}")
    for f in src.iterdir():
        if f.is_file() and not f.name.endswith(".safetensors"):
            shutil.copy(f, out / f.name)
    shutil.copy(work / "quant_recipe.json", out / "quant_recipe.json")
    print("ASSEMBLE_DONE", flush=True)


if __name__ == "__main__":
    main()
