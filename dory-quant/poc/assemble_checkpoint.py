"""The source Dory checkpoint (a single model.safetensors, no shard index)
with gptq_dory.py's quantized layer weights swapped in, plus
quant_recipe.json. convert_mlx.py turns it into MLX at the same bits and,
with --gptq-work, writes GPTQ's own codes.

    python assemble_checkpoint.py --model SNAPSHOT --work /workspace/dory-gptq --out /workspace/dory-ongrid
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

    tensors = load_file(src / "model.safetensors")
    cache: dict[Path, dict] = {}
    left = set(where)
    for key in list(tensors):
        if key in where:
            f = where[key]
            if f not in cache:
                cache = {f: load_file(f)}  # one layer file at a time
            new = cache[f][key]
            assert new.shape == tensors[key].shape and new.dtype == tensors[key].dtype, key
            tensors[key] = new
            left.discard(key)
    if left:
        raise SystemExit(f"not in the source checkpoint: {sorted(left)[:5]}")

    out.mkdir(parents=True, exist_ok=True)
    save_file(tensors, out / "model.safetensors", metadata={"format": "pt"})
    for f in src.iterdir():
        if f.name == "model.safetensors":
            continue
        if f.is_file() and not f.name.startswith("."):
            (out / f.name).unlink(missing_ok=True)
            shutil.copyfile(f, out / f.name)
    shutil.copy(work / "quant_recipe.json", out / "quant_recipe.json")
    print("ASSEMBLE_DONE", flush=True)


if __name__ == "__main__":
    main()
