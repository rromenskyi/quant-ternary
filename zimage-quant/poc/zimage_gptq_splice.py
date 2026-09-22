"""Splices GPTQ-corrected weights (produced by zimage_gptq_calibrate.py,
possibly on a remote CUDA pod) into an mflux-saved MLX checkpoint. This is
the mlx-dependent half of the split -- see zimage_gptq_calibrate.py's
docstring for why the split exists (mlx is Apple-only, calibration is much
faster on a CUDA pod).

Reads every batch_*.safetensors file under --corrected-dir (each a dict of
"layers.N.submodule" -> bf16 tensor, written by the calibrate script) and
patches each into a copy of --mflux-saved-dir, re-quantizing with
mx.quantize the same way zimage_gptq.py's monolithic version did. Tracks
progress the same way (gptq_progress.json in --output-dir) so this can be
re-run incrementally as more batch_*.safetensors files arrive from the pod
without re-processing ones already spliced.

Usage:
    python zimage_gptq_splice.py \
        --corrected-dir /path/to/rsynced/zimage-corrected \
        --mflux-saved-dir /path/to/zimage-8bit-rtn \
        --output-dir /path/to/zimage-8bit-gptq-full \
        --bits 8 --group-size 64
"""

from __future__ import annotations

import argparse
import json
import shutil
import struct
from pathlib import Path

import mlx.core as mx
import torch
from safetensors import safe_open


def read_header(path: Path) -> dict:
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        return json.loads(f.read(n))


def progress_path(output_dir: Path) -> Path:
    return output_dir / "gptq_progress.json"


def load_progress(output_dir: Path) -> set[int]:
    p = progress_path(output_dir)
    if not p.exists():
        return set()
    return set(json.loads(p.read_text())["done_layers"])


def save_progress(output_dir: Path, done_layers: set[int]) -> None:
    progress_path(output_dir).write_text(json.dumps({"done_layers": sorted(done_layers)}))


def layer_of(key: str) -> int:
    # "layers.N.submodule..." -> N
    return int(key.split(".")[1])


def apply_corrections_to_checkpoint(
    output_dir: Path, corrected: dict, bits: int, group_size: int
) -> None:
    transformer_dir = output_dir / "transformer"
    index_path = transformer_dir / "model.safetensors.index.json"
    index = json.loads(index_path.read_text())
    weight_map = index["weight_map"]

    by_shard: dict[str, list[str]] = {}
    for key in corrected:
        shard = weight_map[f"{key}.weight"]
        by_shard.setdefault(shard, []).append(key)

    for shard_name, keys in by_shard.items():
        shard_path = transformer_dir / shard_name
        # mx.load() silently drops the safetensors header's own
        # `__metadata__` entry -- mflux's own weight loader reads that
        # per-shard metadata to decide which modules to reconstruct as
        # QuantizedLinear at all, so it must be read manually and passed
        # back through mx.save_safetensors explicitly.
        original_metadata = read_header(shard_path).get("__metadata__")
        tensors = dict(mx.load(str(shard_path)))
        for key in keys:
            w = corrected[key]  # already mx.array, bf16
            wq, scales, biases = mx.quantize(w, group_size=group_size, bits=bits, mode="affine")
            tensors[f"{key}.weight"] = wq
            tensors[f"{key}.scales"] = scales
            tensors[f"{key}.biases"] = biases
        mx.eval(*tensors.values())
        mx.save_safetensors(str(shard_path), tensors, metadata=original_metadata)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corrected-dir", required=True, help="dir of batch_*.safetensors from zimage_gptq_calibrate.py")
    parser.add_argument("--mflux-saved-dir", required=True, help="template, used only if --output-dir doesn't exist yet")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--bits", type=int, default=8)
    parser.add_argument("--group-size", type=int, default=64)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    if not output_dir.exists():
        print(f"Copying template checkpoint {args.mflux_saved_dir} -> {output_dir} ...", flush=True)
        shutil.copytree(args.mflux_saved_dir, output_dir)

    done_layers = load_progress(output_dir)
    corrected_dir = Path(args.corrected_dir)
    batch_files = sorted(corrected_dir.glob("batch_*.safetensors"))
    if not batch_files:
        print(f"No batch_*.safetensors found under {corrected_dir}", flush=True)
        return

    applied = 0
    for batch_path in batch_files:
        with safe_open(str(batch_path), framework="pt") as f:
            keys = list(f.keys())
            batch_layers = {layer_of(k) for k in keys}
            if batch_layers <= done_layers:
                continue  # already spliced in a previous run of this script
            # bfloat16 isn't a native numpy dtype -- bridge through float32,
            # same as zimage_gptq.py's original write_corrected_checkpoint.
            corrected = {
                k: mx.array(f.get_tensor(k).to(torch.float32).numpy()).astype(mx.bfloat16)
                for k in keys
            }
        print(f"Splicing {batch_path.name} (layers {sorted(batch_layers)}) ...", flush=True)
        apply_corrections_to_checkpoint(output_dir, corrected, args.bits, args.group_size)
        done_layers |= batch_layers
        save_progress(output_dir, done_layers)
        applied += 1

    print(f"Done: applied {applied} new batch file(s); {len(done_layers)} layer(s) total corrected on disk.")


if __name__ == "__main__":
    main()
