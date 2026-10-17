"""Add Muse Glimmer's vision tower (RTN, 8-bit) onto the already-published
GPTQ-quantized text MLX model, no text recalibration needed.

    python build_vl.py --hf-checkpoint /workspace/hf/.../Muse-Glimmer-30B \\
        --text-mlx /workspace/Muse-Glimmer-30B-gptq-mlx-jang \\
        --out /workspace/Muse-Glimmer-30B-VL-gptq-mlx-jang
"""
import argparse
import json
import shutil
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten
from safetensors import safe_open

import sys
sys.path.insert(0, "/workspace/mlx-lm")
from mlx_lm.models.muse_glimmer_vision import VisionAdapter, VisionConfig, VisionModel


def load_vision_state_dict(model_dir: Path) -> dict:
    index = json.load(open(model_dir / "model.safetensors.index.json"))
    by_shard: dict[str, list[str]] = {}
    for key, shard in index["weight_map"].items():
        bare = key[len("model."):] if key.startswith("model.") else key
        if bare.startswith("vision_tower.") or bare.startswith("vision_adapter.") or bare.startswith("vision_projection."):
            by_shard.setdefault(shard, []).append(key)
    state = {}
    for shard, keys in by_shard.items():
        with safe_open(model_dir / shard, "pt") as f:
            for key in keys:
                bare = key[len("model."):] if key.startswith("model.") else key
                state[bare] = f.get_tensor(key)
    return state


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf-checkpoint", required=True, help="original bf16 HF snapshot (has vision weights)")
    ap.add_argument("--text-mlx", required=True, help="published text-only quantized MLX dir")
    ap.add_argument("--out", required=True)
    ap.add_argument("--vision-bits", type=int, default=8)
    ap.add_argument("--group-size", type=int, default=64)
    args = ap.parse_args()
    hf_dir, text_dir, out = Path(args.hf_checkpoint), Path(args.text_mlx), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    config = json.load(open(hf_dir / "config.json"))
    vision_cfg_dict = config["vision_config"]
    state = load_vision_state_dict(hf_dir)
    print(f"{len(state)} vision tensors loaded from {hf_dir}")

    vision_cfg = VisionConfig(
        hidden_size=vision_cfg_dict["hidden_size"], intermediate_size=vision_cfg_dict["intermediate_size"],
        num_hidden_layers=vision_cfg_dict["num_hidden_layers"], num_attention_heads=vision_cfg_dict["num_attention_heads"],
        patch_size=vision_cfg_dict["patch_size"], patch_temporal=vision_cfg_dict.get("patch_temporal", 1),
        merge_size=vision_cfg_dict["merge_size"], pos_emb_height=vision_cfg_dict["pos_emb_height"],
        pos_emb_width=vision_cfg_dict["pos_emb_width"], layer_norm_eps=vision_cfg_dict.get("layer_norm_eps", 1e-5),
        layer_types=vision_cfg_dict.get("layer_types") or [],
        rope_theta=vision_cfg_dict.get("rope_parameters", {}).get("rope_theta", 10000.0),
    )
    vision_tower = VisionModel(vision_cfg)
    vision_tower_weights = {k[len("vision_tower."):]: mx.array(v.float().numpy())
                            for k, v in state.items() if k.startswith("vision_tower.")}
    vision_tower.load_weights(list(vision_tower_weights.items()), strict=True)
    mx.eval(vision_tower.parameters())

    out_hidden_size = config["out_hidden_size"]
    adapter = VisionAdapter(out_hidden_size, config["projector_hidden_size"], config.get("projector_hidden_act", "gelu"))
    adapter_weights = {k[len("vision_adapter."):]: mx.array(v.float().numpy())
                       for k, v in state.items() if k.startswith("vision_adapter.")}
    adapter.load_weights(list(adapter_weights.items()), strict=True)

    projection = nn.Linear(config["projector_hidden_size"], config["text_config"]["hidden_size"], bias=False)
    proj_w = state["vision_projection.weight"]
    projection.load_weights([("weight", mx.array(proj_w.float().numpy()))], strict=True)

    print(f"quantizing vision tower + adapter + projection at {args.vision_bits}-bit...")
    def quantizable(path, module):
        # patch_embedding's input dim (patch_temporal*3*patch_size**2) isn't
        # a multiple of the group size: keep it bf16, as LFM2.5-VL's tower
        # fc2 was for the same reason (docs/FINDINGS.md).
        if "patch_embedding" in path:
            return False
        return isinstance(module, (nn.Linear, nn.Embedding))
    nn.quantize(vision_tower, group_size=args.group_size, bits=args.vision_bits, class_predicate=quantizable)
    nn.quantize(adapter, group_size=args.group_size, bits=args.vision_bits)
    nn.quantize(projection, group_size=args.group_size, bits=args.vision_bits)
    mx.eval(vision_tower.parameters(), adapter.parameters(), projection.parameters())

    vision_tensors = {f"vision_tower.{k}": v for k, v in tree_flatten(vision_tower.parameters())}
    vision_tensors.update({f"vision_adapter.{k}": v for k, v in tree_flatten(adapter.parameters())})
    vision_tensors.update({f"vision_projection.{k}": v for k, v in tree_flatten(projection.parameters())})
    print(f"{len(vision_tensors)} quantized vision tensors")

    # --- copy the text-only MLX dir, add the vision shard, patch config ----
    for f in text_dir.iterdir():
        if f.is_file():
            shutil.copyfile(f, out / f.name)
    mx.save_safetensors(str(out / "model-vision.safetensors"), vision_tensors, metadata={"format": "mlx"})
    index_path = out / "model.safetensors.index.json"
    index = json.load(open(index_path))
    index["weight_map"].update({k: "model-vision.safetensors" for k in vision_tensors})
    index["metadata"]["total_size"] = index["metadata"].get("total_size", 0) + sum(v.nbytes for v in vision_tensors.values())
    json.dump(index, open(index_path, "w"), indent=2)

    out_config = json.load(open(out / "config.json"))
    out_config["vision_config"] = vision_cfg_dict
    out_config["image_token_id"] = config["image_token_id"]
    out_config["out_hidden_size"] = out_hidden_size
    out_config["projector_hidden_size"] = config["projector_hidden_size"]
    out_config["projector_hidden_act"] = config.get("projector_hidden_act", "gelu")
    json.dump(out_config, open(out / "config.json", "w"), indent=2)
    # the HF image processor needs these files to run client-side
    for f in ("preprocessor_config.json", "processor_config.json", "chat_template.jinja"):
        src = hf_dir / f
        if src.exists():
            shutil.copyfile(src, out / f)

    print("BUILD_VL_DONE", out)


if __name__ == "__main__":
    main()
