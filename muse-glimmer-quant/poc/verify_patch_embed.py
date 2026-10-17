"""Isolate the patch embedder (position-embedding interpolation) to find
whether the small cosine gap at two tokens originates there or later
(window attention / pixel shuffle)."""
import json
from pathlib import Path

import mlx.core as mx
import numpy as np
import torch
from safetensors import safe_open

import sys
sys.path.insert(0, "/workspace/mlx-lm")
from mlx_lm.models.muse_glimmer_vision import PatchEmbed, VisionConfig as MlxVisionConfig

MODEL = Path("/workspace/hf/hub/models--meta-models--Muse-Glimmer-30B/snapshots/a4e59da52a7bc87ae7251dd5545c0dd437c44b68")
grid_thw = np.load("/workspace/verify/grid_thw.npy")
print("grid_thw:", grid_thw)

cfg = json.load(open(MODEL / "config.json"))["vision_config"]
index = json.load(open(MODEL / "model.safetensors.index.json"))
want = {"patch_embedding.weight", "position_embedding_table.weight"}
state = {}
for key, shard in index["weight_map"].items():
    bare = key[len("model.vision_tower."):] if key.startswith("model.vision_tower.") else None
    if bare and bare.startswith("patch_embedder.") and bare[len("patch_embedder."):] in want:
        with safe_open(MODEL / shard, "pt") as f:
            state[bare[len("patch_embedder."):]] = f.get_tensor(key)

patch_values = torch.from_numpy(np.load("/workspace/verify/pixel_values.npy")) if Path("/workspace/verify/pixel_values.npy").exists() else None
if patch_values is None:
    from PIL import Image
    from transformers.models.muse_glimmer.image_processing_muse_glimmer import MuseGlimmerImageProcessor
    image = Image.open("/workspace/cats.jpg").convert("RGB")
    inputs = MuseGlimmerImageProcessor()(images=[image], return_tensors="pt")
    patch_values = inputs["pixel_values"]
    np.save("/workspace/verify/pixel_values.npy", patch_values.numpy())

# --- HF reference: just the interpolation + embedding lookup math -----------
from transformers.models.muse_glimmer.modeling_muse_glimmer import MuseGlimmerVisionPatchEmbedder
from transformers.models.muse_glimmer.configuration_muse_glimmer import MuseGlimmerVisionConfig

hf_cfg = MuseGlimmerVisionConfig(**cfg)
hf_embedder = MuseGlimmerVisionPatchEmbedder(hf_cfg).eval().float()
hf_embedder.load_state_dict({k: v.float() for k, v in state.items()}, strict=True)
with torch.no_grad():
    hf_out = hf_embedder(patch_values, torch.from_numpy(grid_thw))
print("HF patch embed out:", hf_out.shape)

# --- MLX port -----------------------------------------------------------------
mlx_cfg = MlxVisionConfig(
    hidden_size=cfg["hidden_size"], patch_size=cfg["patch_size"],
    patch_temporal=cfg.get("patch_temporal", 1),
    pos_emb_height=cfg["pos_emb_height"], pos_emb_width=cfg["pos_emb_width"],
)
mlx_embedder = PatchEmbed(mlx_cfg)
mlx_weights = {k: mx.array(v.float().numpy()) for k, v in state.items()}
mlx_embedder.load_weights(list(mlx_weights.items()), strict=True)
mx.eval(mlx_embedder.parameters())
mlx_out = mlx_embedder(mx.array(patch_values.numpy().astype(np.float32)), grid_thw)
mx.eval(mlx_out)
print("MLX patch embed out:", mlx_out.shape)

hf_np, mlx_np = hf_out.numpy(), np.array(mlx_out)
cos = (hf_np * mlx_np).sum(-1) / (np.linalg.norm(hf_np, axis=-1) * np.linalg.norm(mlx_np, axis=-1) + 1e-8)
print(f"patch-embed cosine: mean={cos.mean():.6f} min={cos.min():.6f}")
worst = np.argsort(cos)[:10]
print("worst (raw, un-merged) patch indices:", worst.tolist())
h, w = int(grid_thw[0, 1]), int(grid_thw[0, 2])
for idx in worst[:10]:
    print(f"  idx {idx}: row={idx // w} col={idx % w} cos={cos[idx]:.4f}")
print(f"tokens below 0.99: {(cos < 0.99).sum()} / {len(cos)}")
