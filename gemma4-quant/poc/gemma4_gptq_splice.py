"""Splices GPTQ-corrected Gemma 4 E4B weights (produced by
gemma4_gptq_calibrate.py, run on a rented CUDA pod) into a fresh MLX-lm
-loadable quantized checkpoint. This is the mlx-dependent half of the
split (mlx is Apple-only, calibration needs CUDA for real speed) -- same
two-stage pattern as zimage-quant's calibrate/splice scripts.

Unlike Z-Image/mflux (whose on-disk convention already matched mlx's own
quantization format), the original google/gemma-4-E4B-it checkpoint is
a single 16GB unsharded bf16 safetensors file with no index.json. This
script:
  1. Reads the ORIGINAL checkpoint's tensors lazily via safetensors'
     safe_open (memory-mapped, one tensor materialized at a time) --
     never loads the full 16GB into RAM at once.
  2. For every key that was GPTQ-corrected (present in the downloaded
     batch files from the pod), re-quantizes the corrected bf16 weight
     via mx.quantize and substitutes the (weight, scales, biases) triple
     for the original single bf16 tensor.
  3. Writes the result back out RE-SHARDED into several smaller
     safetensors files (bounding peak memory the same way Z-Image's
     batched calibration did) plus a model.safetensors.index.json,
     since mlx-lm's loader expects either a single small file or a
     properly indexed multi-shard checkpoint.
  4. Writes a modified config.json with a top-level "quantization" dict
     (mlx-lm's own convention -- see mlx_lm/utils.py's load_model():
     `class_predicate` checks `f"{p}.scales" in weights` to decide which
     modules were actually quantized, so only patching TARGETED modules'
     triples in, and leaving everything else as plain bf16, is enough --
     no need to enumerate every quantized path explicitly for a uniform
     bit-width recipe). For the JANG-mixed recipe, per-path overrides are
     added for the subset of paths whose bits differ from the default.

Real key naming confirmed directly against the checkpoint's own
safetensors header (not guessed) -- see gemma4_gptq_calibrate.py's own
docstring for the full list. Corrected batch files store keys as
"layers.{i}.{submodule}" per component (no "model.<component>." prefix);
this script re-adds that prefix per component when matching against the
real checkpoint's raw keys.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import mlx.core as mx
import torch
from safetensors import safe_open
from safetensors.torch import save_file as save_file_torch

from splice_common import add_leftover_args, compile_patterns, is_dead_kv_shared, is_leftover, module_path_of, quantizable_module_paths

# Raw checkpoint key prefix vs. internal (sanitized) MLX module path prefix,
# per component. These DIFFER for text (language_model.model. adds an extra
# ".model" hop that gemma4.py's sanitize() inserts) but are otherwise the
# same string -- vision genuinely needs "encoder." on BOTH sides since the
# real checkpoint nests vision layers under vision_tower.encoder.layers.N
# (confirmed directly against the real model.safetensors header; verified
# separately that the earlier "model.vision_tower." prefix, missing
# "encoder.", caused 112/112 vision-corrected tensors to silently fail to
# match any real checkpoint key -- meaning the previously published 8-bit
# release's vision tower was NEVER actually GPTQ-corrected or even
# quantized, just copied through as plain bf16).
RAW_PREFIX = {
    "text": "model.language_model.",
    "vision": "model.vision_tower.encoder.",
    "audio": "model.audio_tower.",
}
INTERNAL_PREFIX = {
    "text": "language_model.model.",
    "vision": "vision_tower.encoder.",
    "audio": "audio_tower.",
}


def load_corrected(corrected_dir: Path) -> tuple[dict[str, torch.Tensor], dict[str, int], dict[str, str]]:
    """Returns {full_checkpoint_key: corrected_bf16_tensor},
    {full_checkpoint_key: bits}, and {full_checkpoint_key: internal_module_path}
    across all batch_*.safetensors files from every component."""
    corrected: dict[str, torch.Tensor] = {}
    key_bits: dict[str, int] = {}
    internal_path: dict[str, str] = {}
    for path in sorted(corrected_dir.glob("*_batch_*.safetensors")):
        component = path.name.split("_batch_")[0]
        with safe_open(str(path), framework="pt") as f:
            metadata = f.metadata() or {}
            batch_key_bits = json.loads(metadata.get("key_bits", "{}"))
            for k in f.keys():
                full_key = RAW_PREFIX[component] + k + ".weight"
                corrected[full_key] = f.get_tensor(k)
                key_bits[full_key] = batch_key_bits.get(k, 8)
                internal_path[full_key] = INTERNAL_PREFIX[component] + k
    return corrected, key_bits, internal_path


def read_header(path: Path) -> tuple[dict, int]:
    with open(path, "rb") as fh:
        n = int.from_bytes(fh.read(8), "little")
        header = json.loads(fh.read(n))
    return header, 8 + n


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hf-checkpoint-dir", required=True, help="local HF snapshot dir with the original bf16 model.safetensors")
    parser.add_argument("--corrected-dir", required=True, help="dir of {text,vision,audio}_batch_*.safetensors from the pod")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--group-size", type=int, default=64)
    parser.add_argument("--shard-size-gb", type=float, default=2.0, help="approx uncompressed size per output shard, bounds peak memory")
    parser.add_argument("--embedding-bits", type=int, default=8, help="RTN bit-width for embed_tokens/embed_tokens_per_layer/per_layer_* weights (never GPTQ-calibrated, no bf16 left behind)")
    add_leftover_args(parser)
    args = parser.parse_args()

    hf_dir = Path(args.hf_checkpoint_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading GPTQ-corrected weights ...", flush=True)
    corrected, key_bits, internal_path = load_corrected(Path(args.corrected_dir))
    print(f"  {len(corrected)} corrected tensors across text/vision/audio", flush=True)

    # Every 2D+ weight matrix NOT covered by GPTQ calibration (embeddings,
    # per-layer-embedding gating/projection Linears, small multimodal
    # embedding projections) still needs to end up <=8-bit -- there is no
    # Hessian-based correction for these (embedding lookups have no shared
    # calibration-input matmul the way GPTQ needs), so they get plain RTN
    # (round-to-nearest) mx.quantize instead. Matched by suffix against the
    # RAW checkpoint key; not GPTQ-corrected, called out honestly in the
    # model card.
    RTN_SUFFIXES = (
        "embed_tokens.weight",
        "embed_tokens_per_layer.weight",
        "per_layer_input_gate.weight",
        "per_layer_projection.weight",
        "per_layer_model_projection.weight",
        "embedding_projection.weight",
    )

    default_bits_hint = max(set(key_bits.values()), key=list(key_bits.values()).count) if key_bits else 8

    orig_path = hf_dir / "model.safetensors"
    header, _ = read_header(orig_path)
    all_keys = [k for k in header if k != "__metadata__"]
    print(f"Original checkpoint has {len(all_keys)} tensors total", flush=True)

    real_config = json.loads((hf_dir / "config.json").read_text())
    if args.drop_kv_shared_dead:
        n_before = len(all_keys)
        all_keys = [k for k in all_keys if not is_dead_kv_shared(k, real_config)]
        print(f"Dropping {n_before - len(all_keys)} dead KV-shared-layer tensor(s)", flush=True)
    quantizable = quantizable_module_paths(real_config) if args.rtn_leftovers_bits else set()
    keep_float = compile_patterns(args.keep_float)
    leftover_keys = {
        k for k in all_keys
        if k not in corrected and not k.endswith(RTN_SUFFIXES)
        and is_leftover(k, header[k]["shape"], header[k]["dtype"], quantizable, args.leftover_min_elems, keep_float)
    }
    print(f"RTN {args.rtn_leftovers_bits}-bit for {len(leftover_keys)} uncalibrated leftover Linear(s): {sorted(leftover_keys)[:6]}", flush=True)

    # Bucket keys into shards bounded by approximate uncompressed byte size
    # (using the header's own dtype+shape, no data read yet) -- same
    # memory-safety reasoning as Z-Image's --layers-per-batch: peak RAM
    # scales with shard size, not total model size.
    def tensor_nbytes(info: dict) -> int:
        import math

        dtype_bytes = {"BF16": 2, "F16": 2, "F32": 4, "I64": 8, "I32": 4, "U8": 1, "BOOL": 1}
        n = math.prod(info["shape"]) if info["shape"] else 1
        return n * dtype_bytes.get(info["dtype"], 2)

    shard_limit = int(args.shard_size_gb * 1024**3)
    shards: list[list[str]] = []
    current: list[str] = []
    current_bytes = 0
    for k in all_keys:
        sz = tensor_nbytes(header[k])
        if current and current_bytes + sz > shard_limit:
            shards.append(current)
            current, current_bytes = [], 0
        current.append(k)
        current_bytes += sz
    if current:
        shards.append(current)
    print(f"Re-sharding into {len(shards)} output shard(s)", flush=True)

    weight_map: dict[str, str] = {}
    quantization_overrides: dict[str, dict] = {}

    with safe_open(str(orig_path), framework="pt") as f:
        for shard_idx, shard_keys in enumerate(shards):
            shard_name = f"model-{shard_idx:05d}-of-{len(shards):05d}.safetensors"
            print(f"  Shard {shard_idx + 1}/{len(shards)}: {len(shard_keys)} tensors -> {shard_name}", flush=True)
            out_tensors: dict[str, mx.array] = {}
            for k in shard_keys:
                weight_map[k] = shard_name
                is_gptq_key = k.endswith(".weight") and k in corrected
                is_rtn_key = (not is_gptq_key) and (k.endswith(RTN_SUFFIXES) or k in leftover_keys)
                rtn_bits = args.rtn_leftovers_bits if k in leftover_keys else args.embedding_bits
                if is_gptq_key:
                    bf16 = corrected[k]
                    bits = key_bits[k]
                    w = mx.array(bf16.to(torch.float32).numpy()).astype(mx.bfloat16)
                    wq, scales, biases = mx.quantize(w, group_size=args.group_size, bits=bits, mode="affine")
                    out_tensors[k] = wq
                    out_tensors[k[: -len(".weight")] + ".scales"] = scales
                    out_tensors[k[: -len(".weight")] + ".biases"] = biases
                    weight_map[k[: -len(".weight")] + ".scales"] = shard_name
                    weight_map[k[: -len(".weight")] + ".biases"] = shard_name
                    if bits != default_bits_hint:
                        module_path = internal_path[k]
                        quantization_overrides[module_path] = {"group_size": args.group_size, "bits": bits}
                elif is_rtn_key:
                    t = f.get_tensor(k)
                    w = mx.array(t.to(torch.float32).numpy()).astype(mx.bfloat16)
                    wq, scales, biases = mx.quantize(w, group_size=args.group_size, bits=rtn_bits, mode="affine")
                    out_tensors[k] = wq
                    out_tensors[k[: -len(".weight")] + ".scales"] = scales
                    out_tensors[k[: -len(".weight")] + ".biases"] = biases
                    weight_map[k[: -len(".weight")] + ".scales"] = shard_name
                    weight_map[k[: -len(".weight")] + ".biases"] = shard_name
                    if rtn_bits != default_bits_hint:
                        # Never calibrated, so no per-component prefix table
                        # entry: see splice_common.module_path_of.
                        quantization_overrides[module_path_of(k)] = {"group_size": args.group_size, "bits": rtn_bits}
                else:
                    t = f.get_tensor(k)
                    if t.dtype == torch.bfloat16:
                        arr = mx.array(t.to(torch.float32).numpy()).astype(mx.bfloat16)
                    else:
                        arr = mx.array(t.numpy())
                    out_tensors[k] = arr
            mx.eval(*out_tensors.values())
            mx.save_safetensors(str(output_dir / shard_name), out_tensors)
            del out_tensors

    index = {"metadata": {"total_size": sum(tensor_nbytes(header[k]) for k in all_keys)}, "weight_map": weight_map}
    (output_dir / "model.safetensors.index.json").write_text(json.dumps(index, indent=2))

    # Copy every non-weight file (config, tokenizer, processor, chat
    # template, generation config) verbatim, then patch config.json with
    # the quantization dict mlx-lm's loader expects.
    for item in hf_dir.iterdir():
        if item.name in ("model.safetensors", "model.safetensors.index.json"):
            continue
        if item.is_file():
            shutil.copy2(item, output_dir / item.name)

    config = json.loads((hf_dir / "config.json").read_text())
    default_bits = max(set(key_bits.values()), key=list(key_bits.values()).count)
    quant_dict = {"group_size": args.group_size, "bits": default_bits}
    quant_dict.update(quantization_overrides)
    config["quantization"] = quant_dict
    (output_dir / "config.json").write_text(json.dumps(config, indent=2))

    print(f"Done. Default bits={default_bits}, {len(quantization_overrides)} per-path override(s).", flush=True)
    print("GEMMA4_SPLICE_DONE")


if __name__ == "__main__":
    main()
