"""Google's mobile QAT Gemma 4 (google/gemma-4-{E2B,E4B}-it-qat-mobile-transformers,
"wNa8o8") to MLX, on its own grid.

The mobile checkpoints were trained for their own mixed scheme: per-row
(per-block for the per-layer embeddings) symmetric int2 / int4 / int8, MLP
layers past the first fifteen and the token embeddings / lm_head at 2-bit
(module_quant_configs in their config). Stored as:
- int4: two per byte, low nibble first, w = (q - 8) * scale;
- int2: four per byte, low bits first, w = (q - 2) * scale;
- int8: signed bytes, w = q * scale;
with `<module>.weight_scale` [out, 1] (embeddings: `embedding_quantized` and
`embedding_scale` [rows, blocks]).

MLX's affine quantization stores w = scale * q + bias per group of a row, its
q packed low bits first into uint32 words. With every group's scale the
row's (block's) and bias = -2^(b-1) * scale that is the same grid, and the
packed bytes are MLX's packed words as they are: nothing is re-rounded except
the scale, to bf16 (MLX's scale dtype). int8 becomes unsigned (+128).

The static activation scales (input/output_activation_scale) and the KV-cache
scales are for mobile NPUs (the "a8o8"): dropped -- activations stay bf16.

  python qat_mobile_convert.py --hf-checkpoint-dir e2b-mobile --output-dir E2B-it-qat-mobile-mlx
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path

import mlx.core as mx
import numpy as np

from splice_common import is_dead_kv_shared, module_path_of, quantizable_module_paths

DROPPED = ("input_activation_scale", "output_activation_scale", "k_cache_scale", "v_cache_scale")


def bits_matcher(qconfig: dict):
    """The bits transformers' gemma quantizer gives a module name: the patterns
    joined into one regex, the first listed one that matches wins
    (integrations/gemma_quant.replace_with_gemma_linear)."""
    default = qconfig.get("num_bits", 4)
    overrides = list((qconfig.get("module_quant_configs") or {}).items())
    joined = re.compile("|".join(f"(?P<g{i}>{p})" for i, (p, _) in enumerate(overrides))) if overrides else None

    def bits(name: str) -> int:
        if joined is not None and (m := joined.search(name)):
            i = next(int(g[1:]) for g, v in m.groupdict().items() if v is not None)
            return overrides[i][1].get("num_bits", default)
        return default
    return bits


def mlx_path(raw_module: str) -> str:
    """HF module name -> mlx-lm's (gemma4.sanitize: lm_head goes under language_model)."""
    return "language_model.lm_head" if raw_module == "lm_head" else module_path_of(raw_module + ".weight")


def to_mlx(packed: np.ndarray, row_scales: np.ndarray, bits: int, cols: int, group: int):
    """(weight uint32, scales, biases) for MLX from Google's packed rows and
    per-row (or per-block) scales."""
    rows = packed.shape[0]
    if bits == 8:
        packed = (packed.view(np.int8).astype(np.int16) + 128).astype(np.uint8)
    assert packed.shape[1] * 8 == cols * bits, (packed.shape, cols, bits)
    words = np.ascontiguousarray(packed).view(np.uint32).reshape(rows, cols * bits // 32)
    block = cols // row_scales.shape[1]
    assert block % group == 0, (block, group)
    scales = np.repeat(row_scales.astype(np.float32), block // group, axis=1)
    biases = -(2 ** (bits - 1)) * scales
    return mx.array(words), mx.array(scales).astype(mx.bfloat16), mx.array(biases).astype(mx.bfloat16)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf-checkpoint-dir", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--group-size", type=int, default=128, help="MLX group (the row's scale repeated in each): the widest dividing a row costs the least")
    ap.add_argument("--shard-size-gb", type=float, default=4.0)
    args = ap.parse_args()

    src, out = Path(args.hf_checkpoint_dir), Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    config = json.loads((src / "config.json").read_text())
    qconfig = config.pop("quantization_config")
    bits_of = bits_matcher(qconfig)
    # mx.load: uint8 / int8 / bf16 / f32 as stored (numpy has no bf16).
    loaded: dict[str, mx.array] = {}
    for f in sorted(src.glob("*.safetensors")):
        loaded.update(mx.load(str(f)))
    keys = set(loaded)
    # For the module paths: the model built from the config without its quantization.
    quantizable = quantizable_module_paths(config)

    tensors: dict[str, mx.array] = {}
    overrides: dict[str, dict] = {}
    counts: dict[str, int] = {}
    for k in sorted(keys):
        if k.endswith(DROPPED) or k.endswith(("weight_scale", "embedding_scale")):
            continue
        # The KV-shared layers' own k/v/k_norm: never used (they read an
        # earlier layer's KV), and the mlx-lm model has no place for them.
        if is_dead_kv_shared(k.removesuffix(".weight_scale") if k.endswith("weight_scale") else k, config):
            counts["dead kv"] = counts.get("dead kv", 0) + 1
            continue
        a = loaded[k]
        if k.endswith(".embedding_quantized") or (k.endswith(".weight") and k[: -len(".weight")] + ".weight_scale" in keys):
            embedding = k.endswith(".embedding_quantized")
            module = k[: -len(".embedding_quantized" if embedding else ".weight")]
            row_scales = np.array(loaded[module + (".embedding_scale" if embedding else ".weight_scale")])
            bits = 8 if a.dtype == mx.int8 else bits_of(module)
            cols = a.shape[1] * 8 // bits
            path = mlx_path(module)
            if path not in quantizable:
                raise SystemExit(f"{module}: {path} is no quantizable module of the mlx-lm model")
            group = next((g for g in dict.fromkeys((args.group_size, 128, 64, 32)) if (cols // row_scales.shape[1]) % g == 0), None)
            if group is None:
                raise SystemExit(f"{module}: no MLX group divides its {cols // row_scales.shape[1]}-wide blocks")
            w, s, b = to_mlx(np.array(a), row_scales, bits, cols, group)
            # lm_head by its mlx-lm path: gemma4.sanitize leaves a bare
            # "lm_head.*" at the top, where the model has none (untied heads
            # live at language_model.lm_head).
            key = "language_model.lm_head" if module == "lm_head" else module
            tensors[key + ".weight"], tensors[key + ".scales"], tensors[key + ".biases"] = w, s, b
            overrides[path] = {"group_size": group, "bits": bits}
            counts[f"{bits}-bit"] = counts.get(f"{bits}-bit", 0) + 1
        else:
            tensors[k] = a   # norms, scalars, the modules_to_not_convert (bf16 as stored)
            counts["copied"] = counts.get("copied", 0) + 1

    # Shards, as mlx-lm writes them.
    limit = int(args.shard_size_gb * 1024**3)
    shards, cur, size = [], {}, 0
    for k in sorted(tensors):
        n = tensors[k].nbytes
        if cur and size + n > limit:
            shards.append(cur)
            cur, size = {}, 0
        cur[k] = tensors[k]
        size += n
    if cur:
        shards.append(cur)
    weight_map = {}
    for i, shard in enumerate(shards):
        fname = f"model-{i + 1:05d}-of-{len(shards):05d}.safetensors"
        mx.save_safetensors(str(out / fname), shard)
        weight_map.update({k: fname for k in shard})
    total = sum(v.nbytes for v in tensors.values())
    (out / "model.safetensors.index.json").write_text(json.dumps({"metadata": {"total_size": total}, "weight_map": weight_map}, indent=2))
    for item in src.iterdir():
        if item.is_file() and not item.name.endswith(".safetensors") and item.name not in ("model.safetensors.index.json", "config.json"):
            shutil.copy2(item, out / item.name)
    config["quantization"] = {"group_size": args.group_size, "bits": 4, "mode": "affine", **overrides}
    (out / "config.json").write_text(json.dumps(config, indent=2))
    print(f"{', '.join(f'{v} {k}' for k, v in sorted(counts.items()))}; {total / 1e9:.2f} GB")
    print("QAT_MOBILE_DONE")


if __name__ == "__main__":
    main()
