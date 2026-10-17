"""Per-layer cosine trace for tokens 358/363 (the two outliers) to find
where HF and MLX diverge inside the 50-layer encoder."""
import json
from pathlib import Path

import mlx.core as mx
import numpy as np
import torch
from safetensors import safe_open

import sys
sys.path.insert(0, "/workspace/mlx-lm")
from mlx_lm.models.muse_glimmer_vision import VisionConfig as MlxVisionConfig, VisionModel as MlxVisionModel, get_cu_seqlens, get_window_index, get_position_ids

MODEL = Path("/workspace/hf/hub/models--meta-models--Muse-Glimmer-30B/snapshots/a4e59da52a7bc87ae7251dd5545c0dd437c44b68")
grid_thw = np.load("/workspace/verify/grid_thw.npy")
pixel_values_np = np.load("/workspace/verify/pixel_values.npy")

cfg = json.load(open(MODEL / "config.json"))["vision_config"]
index = json.load(open(MODEL / "model.safetensors.index.json"))
state = {}
for key, shard in index["weight_map"].items():
    bare = key[len("model.vision_tower."):] if key.startswith("model.vision_tower.") else None
    if bare:
        with safe_open(MODEL / shard, "pt") as f:
            state[bare] = f.get_tensor(key)

# --- HF model, capturing hidden states after every layer --------------------
from transformers.models.muse_glimmer.modeling_muse_glimmer import MuseGlimmerVisionModel
from transformers.models.muse_glimmer.configuration_muse_glimmer import MuseGlimmerVisionConfig

hf_model = MuseGlimmerVisionModel(MuseGlimmerVisionConfig(**cfg)).eval().float()
hf_model.load_state_dict({k: v.float() for k, v in state.items()}, strict=True)

hf_hidden = {}
def make_hook(i):
    def hook(module, inp, out):
        hf_hidden[i] = out.detach().clone()
    return hook
for i, layer in enumerate(hf_model.layers):
    layer.register_forward_hook(make_hook(i))

pixel_values = torch.from_numpy(pixel_values_np)
with torch.no_grad():
    hf_model(pixel_values=pixel_values, grid_thw=torch.from_numpy(grid_thw))

# window_index used by HF internally -- recompute the same way to map
# post-layer hidden states (in window order) back to raster patch order.
window_index_np = np.array(get_window_index(grid_thw, cfg["pos_emb_height"] * cfg["patch_size"], cfg["patch_size"])[0])
reverse = np.argsort(window_index_np)

# raster-order indices of our two outlier *merged* tokens -> un-merged patch
# indices (each merged token is `merge_size**2` raw patches; we check all 4).
merge = cfg["merge_size"]
h, w = int(grid_thw[0, 1]), int(grid_thw[0, 2])
merged_w = w // merge
targets_merged = [358, 363]
raw_targets = []
for m in targets_merged:
    br, bc = m // merged_w, m % merged_w
    for dr in range(merge):
        for dc in range(merge):
            raw_targets.append((br * merge + dr) * w + (bc * merge + dc))
print("raw (raster-order) patch targets:", raw_targets)
# position of those raw patches within window order:
window_pos = [int(np.where(window_index_np == t)[0][0]) for t in raw_targets]
print("window-order positions:", window_pos)

# --- MLX model, capturing hidden states after every layer -------------------
mlx_cfg = MlxVisionConfig(
    hidden_size=cfg["hidden_size"], intermediate_size=cfg["intermediate_size"],
    num_hidden_layers=cfg["num_hidden_layers"], num_attention_heads=cfg["num_attention_heads"],
    patch_size=cfg["patch_size"], patch_temporal=cfg.get("patch_temporal", 1),
    merge_size=cfg["merge_size"], pos_emb_height=cfg["pos_emb_height"], pos_emb_width=cfg["pos_emb_width"],
    layer_norm_eps=cfg.get("layer_norm_eps", 1e-5), layer_types=cfg.get("layer_types") or [],
    rope_theta=cfg.get("rope_parameters", {}).get("rope_theta", 10000.0),
)
mlx_model = MlxVisionModel(mlx_cfg)
mlx_weights = mlx_model.sanitize({k: mx.array(v.float().numpy()) for k, v in state.items()})
mlx_model.load_weights(list(mlx_weights.items()), strict=True)
mx.eval(mlx_model.parameters())

cu_seqlens = get_cu_seqlens(grid_thw)
window_index, cu_window_seqlens = get_window_index(grid_thw, mlx_model.window_size, mlx_model.config.patch_size)
x = mlx_model.patch_embedder(mx.array(pixel_values_np.astype(np.float32)), grid_thw)
x = mlx_model.ln_pre(x)
x = x[window_index]
position_ids = get_position_ids(grid_thw)[:, ::-1].copy() + 1
position_ids = mx.array(position_ids)[window_index]
cos, sin = mlx_model.rotary_emb(position_ids)
layer_types = mlx_cfg.layer_types
mlx_hidden = {}
for i, (layer, lt) in enumerate(zip(mlx_model.layers, layer_types)):
    seqlens = cu_seqlens if lt == "full_attention" else cu_window_seqlens
    x = layer(x, seqlens, cos, sin)
    mx.eval(x)
    mlx_hidden[i] = np.array(x)

print(f"{'layer':>5} {'type':>17} " + " ".join(f"cos@{p}" for p in window_pos))
for i in range(cfg["num_hidden_layers"]):
    hf_h = hf_hidden[i].numpy()
    mlx_h = mlx_hidden[i]
    cosines = []
    for p in window_pos:
        a, b = hf_h[p], mlx_h[p]
        cosines.append(float((a * b).sum() / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)))
    print(f"{i:>5} {layer_types[i]:>17} " + " ".join(f"{c:.5f}" for c in cosines))
