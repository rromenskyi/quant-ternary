"""Converts an official ACE-Step 1.5 DiT (turbo, sft, base, xl-*) plus the
shared VAE and text encoder into mlx-audio's ACE-Step checkpoint layout --
mlx-audio's own convert.py is hard-wired to the turbo DiT. Same key and
layout conversions as that script (decoder proj_in Conv1d and proj_out
ConvTranspose1d transposed to MLX order, rotary caches dropped); the DiT is
kept in bfloat16, the dtype it's published in (mlx-community's conversion
stores float32, twice the size for nothing).

Needs torch + diffusers + mlx (the official ACE-Step checkout's venv has all
three):

    python acestep_convert_mlx.py --checkpoints <official checkpoints dir> --dit acestep-v15-sft --out <dir>
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import mlx.core as mx
import numpy as np
import safetensors.torch
import torch


def to_mx(value: torch.Tensor, dtype) -> mx.array:
    return mx.array(value.detach().cpu().float().numpy()).astype(dtype)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoints", required=True, help="official checkpoints dir (vae/, Qwen3-Embedding-0.6B/, <dit>/)")
    ap.add_argument("--dit", default="acestep-v15-sft")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    ck, out = Path(args.checkpoints), Path(args.out)
    dit_dir, vae_dir, text_dir = ck / args.dit, ck / "vae", ck / "Qwen3-Embedding-0.6B"
    out.mkdir(parents=True, exist_ok=True)

    state = {}
    for f in sorted(dit_dir.glob("*.safetensors")):
        state.update(safetensors.torch.load_file(str(f)))
    weights = {}
    for key, value in state.items():
        if "rotary_emb" in key:
            continue
        new_key, v = key, value
        if "decoder.proj_in.1." in key:
            new_key = key.replace("proj_in.1.", "proj_in_")
            if new_key.endswith("_weight"):
                v = v.swapaxes(1, 2)
        elif "decoder.proj_out.1." in key:
            new_key = key.replace("proj_out.1.", "proj_out_")
            if new_key.endswith("_weight"):
                v = v.permute(1, 2, 0)
        weights[new_key] = to_mx(v, mx.bfloat16)
    mx.save_safetensors(str(out / "model.safetensors"), weights)
    config = json.loads((dit_dir / "config.json").read_text())
    (out / "config.json").write_text(json.dumps(config, indent=4))
    silence = dit_dir / "silence_latent.pt"
    if silence.exists():
        # Kept [1, 64, T]: mlx-audio's loader transposes it itself.
        np.save(out / "silence_latent.npy", torch.load(silence, map_location="cpu", weights_only=True).numpy())
    print(f"DiT: {len(weights)} tensors, {sum(w.nbytes for w in weights.values()) / 1e9:.2f} GB", flush=True)

    from diffusers.models import AutoencoderOobleck

    vae = AutoencoderOobleck.from_pretrained(str(vae_dir))
    (out / "vae").mkdir(exist_ok=True)
    mx.save_safetensors(str(out / "vae" / "diffusion_pytorch_model.safetensors"),
                        {k: to_mx(v, mx.float32) for k, v in vae.state_dict().items()})
    shutil.copy2(vae_dir / "config.json", out / "vae" / "config.json")

    text_out = out / "Qwen3-Embedding-0.6B"
    text_out.mkdir(exist_ok=True)
    text_state = safetensors.torch.load_file(str(text_dir / "model.safetensors"))
    mx.save_safetensors(str(text_out / "model.safetensors"),
                        {(k[6:] if k.startswith("model.") else k): to_mx(v, mx.float32) for k, v in text_state.items()})
    for name in ("config.json", "tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt"):
        if (text_dir / name).exists():
            shutil.copy2(text_dir / name, text_out / name)
    print("converted ->", out)


if __name__ == "__main__":
    main()
