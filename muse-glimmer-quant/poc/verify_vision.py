"""Numeric check of the MLX muse_glimmer vision port against HF transformers,
on one real image. Loads ONLY the vision tensors from the checkpoint's
safetensors shards (safe_open, selective) -- never materializes the 30B
text decoder, so this runs on a modest pod/Mac.

    python verify_vision.py --model SNAPSHOT --image cats.jpg
"""
import argparse
import json
from pathlib import Path

import mlx.core as mx
import numpy as np
import requests
import torch
from PIL import Image
from safetensors import safe_open
from transformers.models.muse_glimmer.configuration_muse_glimmer import MuseGlimmerVisionConfig
from transformers.models.muse_glimmer.image_processing_muse_glimmer import MuseGlimmerImageProcessor
from transformers.models.muse_glimmer.modeling_muse_glimmer import MuseGlimmerVisionModel

import sys
sys.path.insert(0, "/workspace/mlx-lm")
from mlx_lm.models.muse_glimmer_vision import VisionConfig as MlxVisionConfig, VisionModel as MlxVisionModel


def load_vision_state_dict(model_dir: Path) -> dict:
    index = json.load(open(model_dir / "model.safetensors.index.json"))
    by_shard: dict[str, list[str]] = {}
    for key, shard in index["weight_map"].items():
        bare = key[len("model."):] if key.startswith("model.") else key
        if bare.startswith("vision_tower."):
            by_shard.setdefault(shard, []).append(key)
    state = {}
    for shard, keys in by_shard.items():
        with safe_open(model_dir / shard, "pt") as f:
            for key in keys:
                state[key[len("model.vision_tower."):]] = f.get_tensor(key)
    return state


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--image", default="/workspace/cats.jpg")
    args = ap.parse_args()
    model_dir = Path(args.model)

    if not Path(args.image).exists():
        img_bytes = requests.get("http://images.cocodataset.org/val2017/000000039769.jpg").content
        Path(args.image).write_bytes(img_bytes)
    image = Image.open(args.image).convert("RGB")

    config = json.load(open(model_dir / "config.json"))
    vision_cfg_dict = config["vision_config"]

    print("Loading vision state dict (selective, no text decoder)...")
    state = load_vision_state_dict(model_dir)
    print(f"{len(state)} vision tensors loaded")

    print("Running HF reference (fp32)...")
    hf_config = MuseGlimmerVisionConfig(**vision_cfg_dict)
    hf_model = MuseGlimmerVisionModel(hf_config).eval().float()
    missing, unexpected = hf_model.load_state_dict({k: v.float() for k, v in state.items()}, strict=False)
    print("missing:", missing[:10], "unexpected:", unexpected[:10])

    processor = MuseGlimmerImageProcessor()
    inputs = processor(images=[image], return_tensors="pt")
    pixel_values, grid_thw = inputs["pixel_values"], inputs["image_grid_thw"][0:1]
    print("pixel_values:", pixel_values.shape, "grid_thw:", grid_thw.tolist())

    with torch.no_grad():
        hf_out = hf_model(pixel_values=pixel_values, grid_thw=grid_thw).last_hidden_state
    print("HF out:", hf_out.shape)

    print("Running MLX port...")
    mlx_cfg = MlxVisionConfig(
        hidden_size=vision_cfg_dict["hidden_size"],
        intermediate_size=vision_cfg_dict["intermediate_size"],
        num_hidden_layers=vision_cfg_dict["num_hidden_layers"],
        num_attention_heads=vision_cfg_dict["num_attention_heads"],
        patch_size=vision_cfg_dict["patch_size"],
        patch_temporal=vision_cfg_dict.get("patch_temporal", 1),
        merge_size=vision_cfg_dict["merge_size"],
        pos_emb_height=vision_cfg_dict["pos_emb_height"],
        pos_emb_width=vision_cfg_dict["pos_emb_width"],
        layer_norm_eps=vision_cfg_dict.get("layer_norm_eps", 1e-5),
        layer_types=vision_cfg_dict.get("layer_types") or [],
        rope_theta=vision_cfg_dict.get("rope_parameters", {}).get("rope_theta", 10000.0),
    )
    mlx_model = MlxVisionModel(mlx_cfg)
    mlx_weights = {}
    for k, v in state.items():
        mlx_weights[k] = mx.array(v.float().numpy())
    mlx_weights = mlx_model.sanitize(mlx_weights)
    mlx_model.load_weights(list(mlx_weights.items()), strict=True)
    mx.eval(mlx_model.parameters())

    mlx_pixel_values = mx.array(pixel_values.numpy().astype(np.float32))
    mlx_grid_thw = grid_thw.numpy()
    mlx_out = mlx_model(mlx_pixel_values, mlx_grid_thw)
    mx.eval(mlx_out)
    print("MLX out:", mlx_out.shape)

    hf_np = hf_out.numpy()
    mlx_np = np.array(mlx_out)
    np.save("/workspace/verify/hf_out.npy", hf_np)
    np.save("/workspace/verify/mlx_out.npy", mlx_np)
    np.save("/workspace/verify/grid_thw.npy", grid_thw.numpy())
    if hf_np.shape != mlx_np.shape:
        print("SHAPE_MISMATCH", hf_np.shape, mlx_np.shape)
        return
    cos = (hf_np * mlx_np).sum(-1) / (np.linalg.norm(hf_np, axis=-1) * np.linalg.norm(mlx_np, axis=-1) + 1e-8)
    print(f"cosine similarity: mean={cos.mean():.6f} min={cos.min():.6f}")
    worst = np.argsort(cos)[:20]
    print("worst token indices:", worst.tolist())
    print("worst cosines:", cos[worst].tolist())
    print(f"tokens below 0.99: {(cos < 0.99).sum()} / {len(cos)}")
    print(f"tokens below 0.9: {(cos < 0.9).sum()} / {len(cos)}")
    print("VERIFY_OK" if cos.min() > 0.99 else "VERIFY_FAILED")


if __name__ == "__main__":
    main()
