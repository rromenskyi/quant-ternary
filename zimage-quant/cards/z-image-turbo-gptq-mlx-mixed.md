---
license: apache-2.0
base_model: Tongyi-MAI/Z-Image-Turbo
pipeline_tag: text-to-image
tags:
  - mlx
  - mflux
  - z-image
  - gptq
  - quantized
  - mixed-precision
  - image-generation
---

# Z-Image-Turbo — GPTQ, mixed precision (attention 8-bit + feed-forward 4-bit, MLX / mflux)

> ### ▶ Run it locally in [LLMTray](https://www.ipsupport.us/llmtray/)
> A free, native macOS app for local AI on Apple Silicon — chat, images,
> music, agents and an OpenAI-compatible API. Point it at this model; nothing
> leaves your Mac.
>
> [![Download LLMTray](https://img.shields.io/badge/Download-LLMTray%20for%20Mac-2f7d4f?style=for-the-badge&logo=apple&logoColor=white)](https://github.com/ipsupport-llc/llmtray/releases/latest/download/LLMTray-Full.dmg)
> [![GitHub stars](https://img.shields.io/github/stars/ipsupport-llc/llmtray?style=for-the-badge&logo=github)](https://github.com/ipsupport-llc/llmtray)

A component-type mixed-precision GPTQ quantization of
[Z-Image-Turbo](https://huggingface.co/Tongyi-MAI/Z-Image-Turbo) for
[mflux](https://github.com/filipstrand/mflux): **attention linears
(`to_q/to_k/to_v/to_out.0`) at 8-bit, feed-forward linears
(`w1/w2/w3`) at 4-bit**, across all 30 transformer blocks. Same GPTQ
Hessian-correction method as the
[uniform 8-bit](https://huggingface.co/roman220220/z-image-turbo-gptq-mlx-8bit) and
[uniform 4-bit](https://huggingface.co/roman220220/z-image-turbo-gptq-mlx-4bit)
releases, just a different bit-allocation across component types.

## Why mixed precision

This mirrors the "JANG"-style component-type bit allocation this project's
sibling [`nemotron-extreme-quant`](https://github.com/rromenskyi/quant-ternary/tree/main/nemotron-extreme-quant)
uses for LLMs (`jang`/`jang-dense` recipes) — but JANG's original
justification (protect components that are *cheap in total params but
active on every token*, e.g. dense attention vs. sparse MoE experts)
doesn't literally apply here: Z-Image-Turbo has no MoE, every block runs
on every diffusion step regardless of type. The adapted idea is simpler:
attention is the smaller of the two component types by parameter count
(~1.77B vs. feed-forward's ~3.54B across all 30 layers), so giving it more
bits costs relatively little extra size while protecting the part of the
model that (empirically, see below) matters more for keeping the
generation stable under aggressive quantization.

## Does it actually help? Yes — same seed, same prompt, 9 steps:

| RTN 4-bit (uniform) | GPTQ 4-bit (uniform) | GPTQ mixed (this repo) |
|---|---|---|
| ![rtn4](compare-rtn4bit-seed42.png) | ![gptq4](compare-gptq4bit-seed42.png) | ![mixed](compare-mixed-seed42.png) |

Measured against the [8-bit GPTQ release](https://huggingface.co/roman220220/z-image-turbo-gptq-mlx-8bit)
as a stable high-precision reference (same seed):

| variant | PSNR vs. 8-bit ref | size |
|---|---|---|
| RTN 4-bit | 16.6 dB | 5.5GB |
| GPTQ 4-bit (uniform) | 17.1 dB | 5.5GB |
| **GPTQ mixed (this repo)** | **17.4 dB** | **6.3GB** |

More telling than the PSNR gap: **composition**. RTN-4bit and even
uniform GPTQ-4bit visibly drift to a different camera angle/framing than
what the 8-bit model produces for the same seed. This mixed-precision
release keeps the *same* composition as the 8-bit reference (same porch,
same window layout, same framing) — for +0.8GB over uniform 4-bit, you get
a result whose overall structure survives quantization, not just
marginally-better pixels.

## Usage

```bash
pip install mflux
mflux-generate-z-image-turbo \
    --model /path/to/local/checkout/of/this/repo \
    --base-model z-image-turbo \
    --prompt "your prompt" --steps 9
```

Or use it directly from [LLMTray](https://www.ipsupport.us/llmtray/)'s
built-in image generation (the model chooses to call `generate_image`
mid-conversation; LLMTray drives mflux under the hood) — point LLMTray's
image-model setting at this repo.

## Method / code

[zimage-quant](https://github.com/rromenskyi/quant-ternary/tree/main/zimage-quant) —
`zimage_gptq_calibrate.py --attn-bits 8 --ffn-bits 4` (calibrates on real
denoising-loop activations, CUDA/CPU/MPS, no mlx dependency) +
`zimage_gptq_splice.py` (splices corrected weights into mflux's native MLX
format, Apple-only). Same `gptq_nbit` implementation as
`nemotron-extreme-quant`'s LLM quantization, unmodified.

## Size

~6.3GB total (transformer + text encoder + VAE + tokenizer) — between the
uniform 8-bit (~10GB) and uniform 4-bit (~5.5GB) releases. Peak MLX memory
during generation: ~6.5GB.

## The IPSupport local-AI stack

Local-first AI tools for macOS by [IPSupport](https://www.ipsupport.us) — nothing
leaves your Mac.

- **[LLMTray](https://www.ipsupport.us/llmtray/)** — your local AI
  workstation for macOS: chat with local LLMs, generate and edit images, make
  music, run agents, and serve an OpenAI-compatible API. Downloads models from
  Hugging Face in-app, with per-model profiles.
- **[IPSupport Code](https://ipsupport-llc.github.io/ipsupport-code/)** — your
  AI coding agent for real repositories: analyze, fix, test, report.

LLMTray uses this model for in-chat image generation (it drives mflux under the hood).

## License

Licensed under the **Apache License 2.0**, the same license as the base model — see [`LICENSE`](LICENSE).

Modified from [Tongyi-MAI/Z-Image-Turbo](https://huggingface.co/Tongyi-MAI/Z-Image-Turbo): the attention linears (`to_q/to_k/to_v/to_out.0`) of all 30 transformer blocks were GPTQ-corrected and quantized to 8-bit, the feed-forward linears (`w1/w2/w3`) to 4-bit, and the model was saved in mflux's MLX checkpoint format. The weights and configuration files in this repo are therefore modified versions of the original, not the original files.
