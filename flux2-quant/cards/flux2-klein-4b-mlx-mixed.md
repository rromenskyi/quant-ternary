---
license: apache-2.0
base_model: black-forest-labs/FLUX.2-klein-4B
pipeline_tag: image-to-image
tags:
  - mlx
  - mflux
  - flux2
  - flux2-klein
  - gptq
  - quantized
  - image-generation
  - image-editing
  - text-to-image
  - apple-silicon
---

# FLUX.2 klein 4B: GPTQ 4-bit for MLX / mflux (generate and edit)

> ### ▶ Run it locally in [LLMTray](https://www.ipsupport.us/llmtray/)
> A free, native macOS app for local AI on Apple Silicon — chat, images,
> music, agents and an OpenAI-compatible API. Point it at this model; nothing
> leaves your Mac.
>
> [![Download LLMTray](https://img.shields.io/badge/Download-LLMTray%20for%20Mac-2f7d4f?style=for-the-badge&logo=apple&logoColor=white)](https://github.com/ipsupport-llc/llmtray/releases/latest/download/LLMTray-Full.dmg)
> [![GitHub stars](https://img.shields.io/github/stars/ipsupport-llc/llmtray?style=for-the-badge&logo=github)](https://github.com/ipsupport-llc/llmtray)

<p align="center">
  <img src="klein-gptq-generate-edit.jpg" alt="Generated with this checkpoint, then edited with instructions: winter, night, a red canoe">
</p>

This is [FLUX.2 klein 4B](https://huggingface.co/black-forest-labs/FLUX.2-klein-4B) (Black Forest Labs, Apache 2.0) quantized for Apple Silicon and
[mflux](https://github.com/filipstrand/mflux). It does two things:

- **text to image**, in 4 steps;
- **image editing from an instruction**: "make it winter", "turn it into a watercolor", "give him glasses". You pass a reference image and the model changes what you asked while keeping the rest.

It fits comfortably on a 16 GB Mac:

- **5.3 GB on disk** (bf16 original: ~16 GB);
- **~7.3 GB peak** for text to image at 1024²;
- **~9.4 GB peak** for an edit with one reference image.

## What's inside

| component | precision | size |
|---|---|---|
| transformer (3.9B) | **GPTQ 4-bit**, group 64, Hessian-corrected | 2.2 GB |
| text encoder (Qwen3-4B) | 8-bit, **cut to the 27 layers klein actually reads** | 3.3 GB |
| VAE | as in the original | 0.16 GB |

- **GPTQ, not plain rounding.** The transformer was calibrated on the model's real 4-step denoising loop: text to image *and* edits with a reference image. Each Linear's input Hessian was summed over every token of every step. The weights are the same size as mflux's own `--quantize 4`, but closer to bf16.
- **Text encoder cut to its used layers.** klein conditions only on Qwen3 hidden states 9, 18 and 27, so layers 27–35 never affect the image. They're dropped from the checkpoint (−1 GB). Prompt embeddings and the final image are bit-identical to the full encoder.
- **8-bit text encoder, 4-bit transformer.** An earlier sweep found this split works best. The text encoder is the conditioning signal: at 4-bit it shifted text layout and faces; at 8-bit it matched bf16.

## GPTQ vs plain 4-bit (same size)

Test set: 6 text-to-image prompts and 3 edits, none of them in the calibration set, 1024², seed 7, same text encoder for every model.

| transformer 4-bit g64 | velocity error, text to image | velocity error, edit | PSNR vs bf16, text to image | PSNR vs bf16, edit |
|---|---|---|---|---|
| RTN (mflux `--quantize 4`) | 0.244 | 0.171 | 18.7 dB | 25.7 dB |
| **GPTQ (this repo)** | **0.202 (−17%)** | **0.136 (−20%)** | **19.6 dB** | **26.2 dB** |

*Velocity error* is ‖v_q − v_bf16‖ / ‖v_bf16‖ with teacher forcing: at every step, the quantized model gets exactly the bf16 run's latents. It measures how far the denoiser itself is from bf16, without trajectory drift compounding into it. GPTQ is lower on **all 9** samples, by 12–22%.

<p align="center">
  <img src="klein-gptq-vs-rtn-grid.jpg" width="600" alt="Columns: bf16, RTN 4-bit, GPTQ 4-bit; rows: held-out prompts and edits">
</p>

## Usage

### In LLMTray (easiest)

[**LLMTray**](https://www.ipsupport.us/llmtray/) runs this model for you.

1. Open *Settings › Image generation › Image editing* and choose **FLUX.2 klein 4B**.
2. Attach a photo in the chat, or generate one, and ask: "make it night", "put a hat on the cat". The chat model calls the edit tool for you.

It shows live step previews and keeps nothing on disk for temporary chats.

### With mflux (Python)

```bash
pip install mflux==0.20.0
hf download roman220220/flux2-klein-4b-mlx-mixed --local-dir klein4b
```

```python
from mflux.models.flux2.variants.txt2img.flux2_klein import Flux2Klein
from mflux.models.flux2.variants.edit.flux2_klein_edit import Flux2KleinEdit
from mflux.models.common.vae.tiling_config import TilingConfig

# Text to image
model = Flux2Klein(model_path="klein4b")
model.text_encoder.layers = model.text_encoder.layers[:27]   # the layers this checkpoint carries
model.tiling_config = TilingConfig(vae_decode_tile_size=256)  # halves the VAE decode peak
model.generate_image(seed=1, prompt="A cozy cabin by a mountain lake at golden hour",
                     num_inference_steps=4, width=1024, height=1024).image.save("cabin.png")

# Edit (a path or a PIL image as the reference)
edit = Flux2KleinEdit(model_path="klein4b")
edit.text_encoder.layers = edit.text_encoder.layers[:27]
edit.tiling_config = TilingConfig(vae_decode_tile_size=256)
edit.generate_image(seed=1, prompt="Make it winter, keep everything else", image_paths=["cabin.png"],
                    num_inference_steps=4, width=1024, height=1024).image.save("cabin-winter.png")
```

- **Keep the `layers[:27]` line.** mflux builds all 36 text-encoder layers and leaves the missing ones uninitialized. They don't change the output, but cutting them saves ~1 GB of memory and some time.
- **Speed** (M5, 1024²): text to image ~25–30 s, an edit ~1 min.

## How it was made

The code and the full write-up (method, pitfalls, every number above) are in the
[flux2-quant project of quant-ternary](https://github.com/rromenskyi/quant-ternary/tree/main/flux2-quant):

- `klein_gptq.py`: the calibration and GPTQ passes;
- `klein_gptq_eval.py`: the evaluation;
- `klein_truncate_te.py`: cuts the text encoder to its used layers;
- `docs/FINDINGS.md`: the write-up.

The GPTQ core is the same `gptq_nbit` used for [our Z-Image-Turbo releases](https://huggingface.co/roman220220/z-image-turbo-gptq-mlx-mixed), on MLX's exact affine grid.

---

## The IPSupport local-AI stack

Local-first AI tools for macOS by [IPSupport](https://www.ipsupport.us) — nothing
leaves your Mac.

- **[LLMTray](https://www.ipsupport.us/llmtray/)** — your local AI
  workstation for macOS: chat with local LLMs, generate and edit images, make
  music, run agents, and serve an OpenAI-compatible API. Downloads models from
  Hugging Face in-app, with per-model profiles.
- **[IPSupport Code](https://ipsupport-llc.github.io/ipsupport-code/)** — your
  AI coding agent for real repositories: analyze, fix, test, report.

LLMTray uses this model for in-chat image generation and editing (it drives mflux under the hood).

---

## License

Licensed under the **Apache License 2.0**, the same license as the base model. See [`LICENSE`](LICENSE).

This is a modified version of [black-forest-labs/FLUX.2-klein-4B](https://huggingface.co/black-forest-labs/FLUX.2-klein-4B):

- the transformer's Linear layers were GPTQ-corrected and quantized to 4-bit;
- the text encoder was quantized to 8-bit, and its layers 27–35 were removed;
- everything was saved in mflux's MLX checkpoint format.

The weights and configuration files in this repo are therefore modified versions of the original, not the original files.
