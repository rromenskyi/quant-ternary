"""Splices GPTQ-corrected Gemma 4 26B-A4B (MoE) weights into a fresh
MLX-lm-loadable quantized checkpoint. Unlike gemma4_gptq_splice.py (E4B,
dense-only), this model has:
  - No audio tower (audio_config is null on the real checkpoint) -- only
    "text" and "vision" components.
  - No per-layer-embeddings (hidden_size_per_layer_input=0 on the real
    config) -- embed_tokens is the only embedding table, much smaller
    (1.48GB bf16) than E4B's embed_tokens_per_layer monster.
  - MoE routed experts: real checkpoint keys `experts.gate_up_proj` /
    `experts.down_proj` are raw nn.Parameter tensors (NO ".weight" suffix,
    confirmed directly against the real safetensors header), shape
    [128, 1408, 2816] / [128, 2816, 704]. mlx-lm's own gemma4_text.py
    sanitize() SPLITS gate_up_proj into two separate SwitchGLU submodules
    (`experts.switch_glu.gate_proj` / `.up_proj`) at LOAD time -- but that
    split logic assumes a plain (unquantized) weight tensor. Since this
    script pre-quantizes weights before writing them to disk, it must
    perform the SAME split itself, on the quantized (weight, scales,
    biases) triple, and write the POST-split key names directly --
    verified safe because mx.quantize's per-group scale/bias is per OUTPUT
    ROW (splitting along the output axis is an exact row-subset, not an
    approximation). The router (`router.*`) is never touched.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import mlx.core as mx
import numpy as np
import torch
from safetensors import safe_open
from safetensors.torch import save_file as save_file_torch

from splice_common import add_leftover_args, compile_patterns, is_dead_kv_shared, is_leftover, module_path_of, quantizable_module_paths

# MLX's GPU/CUDA stream is thread-local; arrays created on a
# ThreadPoolExecutor worker thread can't be mx.eval()'d from the main
# thread ("no Stream(gpu, 0) in current thread"). All MLX-touching work
# (mx.array/mx.quantize/mx.eval) is now confined to the main thread only
# (see build_key/prefetch_key split below), so this is no longer a
# cross-thread issue either way -- GPU device chosen because CPU's
# mx.quantize was measured to be extremely slow on this pod (217s for one
# ~4GB shard containing MoE expert tensors, vs. near-instant graph
# building), whereas GPU should have real quantize kernels.
mx.set_default_device(mx.gpu)

RAW_PREFIX = {
    "text": "model.language_model.",
    "vision": "model.vision_tower.encoder.",
}
INTERNAL_PREFIX = {
    "text": "language_model.model.",
    "vision": "vision_tower.encoder.",
}

RTN_SUFFIXES = ("embed_tokens.weight",)


def load_corrected(corrected_dir: Path):
    """Returns {raw_key: corrected_bf16_tensor}, {raw_key: bits},
    {raw_key: internal_module_path or None (MoE keys get None -- handled
    specially since one raw key produces TWO internal paths for
    gate_up_proj)}, and a set of raw_keys that are MoE expert keys."""
    corrected: dict[str, torch.Tensor] = {}
    key_bits: dict[str, int] = {}
    internal_path: dict[str, str] = {}
    moe_keys: set[str] = set()

    for path in sorted(corrected_dir.glob("*_batch_*.safetensors")) + sorted(
        corrected_dir.glob("*_moe_batch_*.safetensors")
    ):
        name = path.name
        if "_moe_batch_" in name:
            component = name.split("_moe_batch_")[0]
        else:
            component = name.split("_batch_")[0]
        with safe_open(str(path), framework="pt") as f:
            metadata = f.metadata() or {}
            batch_key_bits = json.loads(metadata.get("key_bits", "{}"))
            for k in f.keys():
                is_moe = k.endswith(".experts.gate_up_proj") or k.endswith(".experts.down_proj")
                if is_moe:
                    raw_key = RAW_PREFIX["text"] + k  # NO ".weight" suffix -- raw nn.Parameter
                    corrected[raw_key] = f.get_tensor(k)
                    key_bits[raw_key] = batch_key_bits.get(k, 4)
                    moe_keys.add(raw_key)
                else:
                    raw_key = RAW_PREFIX[component] + k + ".weight"
                    corrected[raw_key] = f.get_tensor(k)
                    key_bits[raw_key] = batch_key_bits.get(k, 8)
                    internal_path[raw_key] = INTERNAL_PREFIX[component] + k
    return corrected, key_bits, internal_path, moe_keys


def read_header(path: Path):
    with open(path, "rb") as fh:
        n = int.from_bytes(fh.read(8), "little")
        header = json.loads(fh.read(n))
    return header, 8 + n


def torch_to_f32_numpy(t: torch.Tensor) -> np.ndarray:
    """The slow, eager, thread-safe (pure torch/numpy, no MLX) part -- runs
    on a ThreadPoolExecutor worker. Real numpy/torch C code releases the
    GIL during large bulk conversions, so this is where the actual
    multi-core speedup comes from."""
    return t.to(torch.float32).numpy()


def quantize_moe_gate_up(w_f32: np.ndarray, bits: int, group_size: int):
    """w_f32: [E, 2*intermediate, hidden] float32 numpy. Returns two dicts
    (gate, up), each {"weight":..., "scales":..., "biases":...}, split
    along axis=-2 (the 2*intermediate axis) -- exact since mx.quantize's
    scale/bias are per-output-row. MLX-touching, so must run on the main
    thread only (MLX streams are thread-local -- see mx.set_default_device
    comment above)."""
    w = mx.array(w_f32).astype(mx.bfloat16)
    wq, scales, biases = mx.quantize(w, group_size=group_size, bits=bits, mode="affine")
    half = wq.shape[-2] // 2
    gate = {"weight": wq[:, :half], "scales": scales[:, :half], "biases": biases[:, :half]}
    up = {"weight": wq[:, half:], "scales": scales[:, half:], "biases": biases[:, half:]}
    return gate, up


def compatible_group_size(in_features: int, preferred: int) -> int | None:
    """mx.quantize only accepts group_size in {32, 64, 128}, and requires
    it to evenly divide the last (input) dimension -- no remainder/padding
    support. Most Gemma 4 dims divide cleanly by 64, but vision's
    mlp.down_proj input dim (moe_intermediate_size-unrelated; the VISION
    tower's own intermediate_size, 4304 on the 26B model) factors as
    2^4*269 and isn't divisible by 32, 64, OR 128 -- no group_size choice
    fixes it. Returns None when nothing works, signaling "leave this one
    tensor in bf16" rather than crashing or silently padding (which would
    need matching inference-side padding logic)."""
    for gs in (preferred, 128, 64, 32):
        if in_features % gs == 0:
            return gs
    return None


def quantize_plain(w_f32: np.ndarray, bits: int, group_size: int):
    """Returns {"weight","scales","biases"} normally, or {"weight": <plain
    bf16 mx.array>} (no scales/biases) if this tensor's input dim isn't
    divisible by any group_size mx.quantize supports -- see
    compatible_group_size."""
    w = mx.array(w_f32).astype(mx.bfloat16)
    gs = compatible_group_size(w.shape[-1], group_size)
    if gs is None:
        return {"weight": w}
    wq, scales, biases = mx.quantize(w, group_size=gs, bits=bits, mode="affine")
    return {"weight": wq, "scales": scales, "biases": biases}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hf-checkpoint-dir", required=True)
    parser.add_argument("--corrected-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--group-size", type=int, default=64)
    parser.add_argument("--shard-size-gb", type=float, default=4.0)
    parser.add_argument("--embedding-bits", type=int, default=8)
    add_leftover_args(parser)
    args = parser.parse_args()

    hf_dir = Path(args.hf_checkpoint_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading GPTQ-corrected weights ...", flush=True)
    corrected, key_bits, internal_path, moe_keys = load_corrected(Path(args.corrected_dir))
    print(f"  {len(corrected)} corrected tensors ({len(moe_keys)} MoE expert tensors)", flush=True)

    default_bits_hint = max(set(key_bits.values()), key=list(key_bits.values()).count) if key_bits else 8

    # Vision's mlp intermediate_size (4304 on the 26B model) factors as
    # 2^4*269 -- not divisible by 32, 64, OR 128, mx.quantize's only
    # supported group sizes. Padding gate_proj/up_proj's OUTPUT dim and
    # down_proj's INPUT dim with zero rows/columns up to the next multiple
    # of group_size is exact, not an approximation: GELU(0)*0=0, and those
    # zero activations hit down_proj's correspondingly-zero-weighted extra
    # input columns, contributing exactly 0 to the real output. Applied
    # uniformly (declared in config.json's vision_config.intermediate_size)
    # so the model is CONSTRUCTED with the padded width from the start --
    # no inference-code changes needed.
    real_config = json.loads((hf_dir / "config.json").read_text())
    vision_intermediate = real_config.get("vision_config", {}).get("intermediate_size")
    vision_padded_intermediate = None
    vision_pad_amount = 0
    if vision_intermediate is not None and vision_intermediate % args.group_size != 0:
        vision_padded_intermediate = (
            (vision_intermediate + args.group_size - 1) // args.group_size
        ) * args.group_size
        vision_pad_amount = vision_padded_intermediate - vision_intermediate
        print(
            f"Vision intermediate_size {vision_intermediate} not divisible by "
            f"group_size {args.group_size} -- padding to {vision_padded_intermediate} "
            f"({vision_pad_amount} zero rows/cols, mathematically exact, see comment above)",
            flush=True,
        )

    quantizable = quantizable_module_paths(real_config) if args.rtn_leftovers_bits else set()
    keep_float = compile_patterns(args.keep_float)

    # Real checkpoint is 2 shards -- read both headers, merge key->shard map
    # so lazy safe_open can find any key regardless of which shard it's in.
    shard_paths = sorted(hf_dir.glob("model-*-of-*.safetensors"))
    all_keys: dict[str, dict] = {}
    key_to_orig_shard: dict[str, Path] = {}
    for sp in shard_paths:
        header, _ = read_header(sp)
        for k, info in header.items():
            if k == "__metadata__":
                continue
            all_keys[k] = info
            key_to_orig_shard[k] = sp
    print(f"Original checkpoint has {len(all_keys)} tensors across {len(shard_paths)} shard(s)", flush=True)
    if args.drop_kv_shared_dead:
        dead = [k for k in all_keys if is_dead_kv_shared(k, real_config)]
        for k in dead:
            del all_keys[k]
        print(f"Dropping {len(dead)} dead KV-shared-layer tensor(s)", flush=True)
    leftover_keys = {
        k for k, info in all_keys.items()
        if k not in corrected and not k.endswith(RTN_SUFFIXES)
        and is_leftover(k, info["shape"], info["dtype"], quantizable, args.leftover_min_elems, keep_float)
    }
    print(f"RTN {args.rtn_leftovers_bits}-bit for {len(leftover_keys)} uncalibrated leftover Linear(s): {sorted(leftover_keys)[:6]}", flush=True)

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
        sz = tensor_nbytes(all_keys[k])
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
    open_handles: dict[Path, "safe_open"] = {sp: safe_open(str(sp), framework="pt") for sp in shard_paths}

    def internal_moe_path(raw_key: str, suffix: str) -> str:
        # raw_key like "model.language_model.layers.{i}.experts.gate_up_proj"
        # -> "language_model.model.layers.{i}.experts.switch_glu.<suffix>"
        base = raw_key[len("model.language_model."):]
        base = base.rsplit(".", 1)[0]  # drop "gate_up_proj"/"down_proj"
        return f"language_model.model.{base}.switch_glu.{suffix}"

    n_workers = min(32, (os.cpu_count() or 8))
    print(f"Using {n_workers} worker threads per shard for the CPU-bound torch->numpy pre-conversion", flush=True)

    def prefetch_key(k: str):
        """Phase 1, runs on a worker thread: ONLY pure torch/numpy work (no
        MLX at all, so no stream/thread-affinity issues). Returns a plain
        float32 numpy array -- either the already-in-memory GPTQ-corrected
        tensor (MoE/attn/mlp), or a freshly-read tensor from the original
        checkpoint (RTN embeddings, or passthrough for untouched keys).
        This eager torch->float32->numpy conversion is what dominates
        wall-clock for large MoE tensors, and torch's C-level ops release
        the GIL during it, so real multi-core speedup is possible here."""
        if k in corrected:
            return torch_to_f32_numpy(corrected[k])
        f = open_handles[key_to_orig_shard[k]]
        t = f.get_tensor(k)
        return torch_to_f32_numpy(t) if t.dtype == torch.bfloat16 else t.numpy()

    def build_key(k: str, w_f32: np.ndarray, shard_name: str):
        """Phase 2, runs on the MAIN thread only: every MLX-touching op
        (mx.array/mx.quantize/mx.eval) lives here, since MLX streams are
        thread-local and eval'ing an array from a different thread than
        the one that built its graph crashes regardless of device."""
        tensors: dict[str, "mx.array"] = {}
        overrides: dict[str, dict] = {}

        if vision_pad_amount and k.startswith("model.vision_tower.encoder."):
            if k.endswith((".mlp.gate_proj.linear.weight", ".mlp.up_proj.linear.weight")):
                # [intermediate, hidden] -- pad OUTPUT (axis 0) rows with zeros.
                w_f32 = np.pad(w_f32, ((0, vision_pad_amount), (0, 0)))
            elif k.endswith(".mlp.down_proj.linear.weight"):
                # [hidden, intermediate] -- pad INPUT (last axis) columns with
                # zeros; this is the one that actually fixes group-size
                # divisibility, the other two just have to match its shape.
                w_f32 = np.pad(w_f32, ((0, 0), (0, vision_pad_amount)))

        if k in moe_keys and k.endswith(".experts.gate_up_proj"):
            bits = key_bits[k]
            gate, up = quantize_moe_gate_up(w_f32, bits, args.group_size)
            base = k[: -len(".gate_up_proj")]
            for name, triple in (("gate_proj", gate), ("up_proj", up)):
                prefix = f"{base}.switch_glu.{name}"
                for suffix, arr in triple.items():
                    tensors[f"{prefix}.{suffix}"] = arr
                if bits != default_bits_hint:
                    overrides[internal_moe_path(k, name)] = {"group_size": args.group_size, "bits": bits}
            return tensors, overrides

        if k in moe_keys and k.endswith(".experts.down_proj"):
            bits = key_bits[k]
            triple = quantize_plain(w_f32, bits, args.group_size)
            prefix = f"{k[: -len('.down_proj')]}.switch_glu.down_proj"
            for suffix, arr in triple.items():
                tensors[f"{prefix}.{suffix}"] = arr
            if bits != default_bits_hint:
                overrides[internal_moe_path(k, "down_proj")] = {"group_size": args.group_size, "bits": bits}
            return tensors, overrides

        is_gptq_key = k.endswith(".weight") and k in corrected
        is_rtn_key = (not is_gptq_key) and (k.endswith(RTN_SUFFIXES) or k in leftover_keys)
        rtn_bits = args.rtn_leftovers_bits if k in leftover_keys else args.embedding_bits

        if is_gptq_key:
            bits = key_bits[k]
            triple = quantize_plain(w_f32, bits, args.group_size)
            base = k[: -len(".weight")]
            tensors[k] = triple["weight"]
            tensors[f"{base}.scales"] = triple["scales"]
            tensors[f"{base}.biases"] = triple["biases"]
            if bits != default_bits_hint:
                overrides[internal_path[k]] = {"group_size": args.group_size, "bits": bits}
        elif is_rtn_key:
            triple = quantize_plain(w_f32, rtn_bits, args.group_size)
            base = k[: -len(".weight")]
            for suffix, arr in triple.items():
                tensors[f"{base}.{suffix}"] = arr
            if "scales" in triple and rtn_bits != default_bits_hint:
                overrides[module_path_of(k)] = {"group_size": args.group_size, "bits": rtn_bits}
        else:
            tensors[k] = mx.array(w_f32).astype(mx.bfloat16)

        return tensors, overrides

    for shard_idx, shard_keys in enumerate(shards):
        shard_name = f"model-{shard_idx:05d}-of-{len(shards):05d}.safetensors"
        print(f"  Shard {shard_idx + 1}/{len(shards)}: {len(shard_keys)} tensors -> {shard_name}", flush=True)
        out_tensors: dict[str, mx.array] = {}
        t0 = time.monotonic()

        with ThreadPoolExecutor(max_workers=n_workers) as pool:
            prefetched = list(pool.map(prefetch_key, shard_keys))

        t1 = time.monotonic()
        print(f"    prefetch (parallel): {t1 - t0:.1f}s", flush=True)

        for k, w_f32 in zip(shard_keys, prefetched):
            tensors, overrides = build_key(k, w_f32, shard_name)
            for full_key, arr in tensors.items():
                out_tensors[full_key] = arr
                weight_map[full_key] = shard_name
            quantization_overrides.update(overrides)

        t2 = time.monotonic()
        print(f"    quantize (sequential): {t2 - t1:.1f}s", flush=True)

        mx.eval(*out_tensors.values())
        t3 = time.monotonic()
        print(f"    mx.eval: {t3 - t2:.1f}s", flush=True)
        mx.save_safetensors(str(output_dir / shard_name), out_tensors)
        t4 = time.monotonic()
        print(f"    save: {t4 - t3:.1f}s", flush=True)
        del out_tensors, prefetched

    total_size = sum(tensor_nbytes(all_keys[k]) for k in all_keys)
    index = {"metadata": {"total_size": total_size}, "weight_map": weight_map}
    (output_dir / "model.safetensors.index.json").write_text(json.dumps(index, indent=2))

    for item in hf_dir.iterdir():
        if item.name.startswith("model-") and item.name.endswith(".safetensors"):
            continue
        if item.name == "model.safetensors.index.json":
            continue
        if item.is_file():
            shutil.copy2(item, output_dir / item.name)

    config = json.loads((hf_dir / "config.json").read_text())
    if vision_padded_intermediate is not None:
        config["vision_config"]["intermediate_size"] = vision_padded_intermediate
    default_bits = max(set(key_bits.values()), key=list(key_bits.values()).count)
    quant_dict = {"group_size": args.group_size, "bits": default_bits}
    quant_dict.update(quantization_overrides)
    config["quantization"] = quant_dict
    (output_dir / "config.json").write_text(json.dumps(config, indent=2))

    print(f"Done. Default bits={default_bits}, {len(quantization_overrides)} per-path override(s).", flush=True)
    print("GEMMA4_26B_SPLICE_DONE")


if __name__ == "__main__":
    main()
