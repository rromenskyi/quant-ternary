"""The source checkpoint with gptq_inkling2.py's quantized decoder weights
swapped in, under the checkpoint's own names (names.py), plus
quant_recipe.json. The NVFP4 experts become bf16 tensors on the GPTQ grid
(w13 / w2, gate / up rows interleaved as published) and lose their
scale / amax / outer_scale_denominator; config.json loses its
quantization_config. Everything else is copied byte for byte; the DSpark
draft (mtp.safetensors) goes to model-mtp.safetensors under mtp.*.

    python assemble_checkpoint.py --model SNAPSHOT --work /workspace/gptq-mini --out /workspace/ongrid-mini
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from safetensors import safe_open
from safetensors.torch import load_file, save_file

import names

NVFP4_PARTS = ("_scale", "_amax", "_outer_scale_denominator")
SKIP_FILES = ("hf_quant_config.json",)


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

    where: dict[str, tuple[Path, str]] = {}
    for f in sorted((work / "layers").glob("*.safetensors")):
        with safe_open(f, "pt") as s:
            where.update({names.checkpoint_name(k): (f, k) for k in s.keys() if ".gptq_" not in k})
    out.mkdir(parents=True, exist_ok=True)
    index = json.load(open(src / "model.safetensors.index.json"))
    weight_map = {}
    cache: dict[Path, dict] = {}
    left = set(where)
    for shard in sorted(set(index["weight_map"].values())):
        tensors = load_file(src / shard)
        for key in list(tensors):
            if key.endswith(tuple(f"experts.{w}_weight{p}" for w in ("w13", "w2") for p in NVFP4_PARTS)):
                del tensors[key]
            elif key in where:
                f, name = where[key]
                if f not in cache:
                    cache = {f: load_file(f)}  # one layer file at a time
                new = names.interleave(cache[f][name], fused=name.endswith("gate_up_proj"))
                old = tensors[key]
                if ".experts." not in key:
                    assert new.shape == old.shape and new.dtype == old.dtype, key
                elif old.dtype.is_floating_point:
                    assert new.shape == old.shape, key
                tensors[key] = new.contiguous()
                left.discard(key)
        save_file(tensors, out / shard, metadata={"format": "pt"})
        weight_map.update({k: shard for k in tensors})
        print(f"{shard} written", flush=True)
    if left:
        raise SystemExit(f"not in any source shard: {sorted(left)[:5]}")
    if (src / "mtp.safetensors").exists():
        # The DSpark draft, as is: mlx-lm loads model*.safetensors, mtp.* keys.
        draft = {f"mtp.{k}": v for k, v in load_file(src / "mtp.safetensors").items()}
        save_file(draft, out / "model-mtp.safetensors", metadata={"format": "pt"})
        weight_map.update({k: "model-mtp.safetensors" for k in draft})
        print(f"model-mtp.safetensors written ({len(draft)} draft tensors)", flush=True)
    json.dump({"metadata": {}, "weight_map": weight_map}, open(out / "model.safetensors.index.json", "w"), indent=2)
    for f in src.iterdir():
        if f.is_file() and not f.name.endswith((".safetensors", ".bak")) and not f.name.startswith(".") \
                and f.name not in SKIP_FILES + ("model.safetensors.index.json",):
            (out / f.name).unlink(missing_ok=True)
            shutil.copyfile(f, out / f.name)
    for d in ("tiktoken",):
        if (src / d).is_dir():
            shutil.copytree(src / d, out / d, dirs_exist_ok=True)
    config = json.load(open(src / "config.json"))
    config.pop("quantization_config", None)
    json.dump(config, open(out / "config.json", "w"), indent=2)
    shutil.copy(work / "quant_recipe.json", out / "quant_recipe.json")
    print("ASSEMBLE_DONE", flush=True)


if __name__ == "__main__":
    main()
