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
  - image-generation
---

# Z-Image-Turbo — GPTQ, 4-bit (MLX / mflux)

> ### ▶ Run it locally in [LLMTray](https://www.ipsupport.us/llmtray/)
> A free, native macOS app for local AI on Apple Silicon — chat, images,
> music, agents and an OpenAI-compatible API. Point it at this model; nothing
> leaves your Mac.
>
> [![Download LLMTray](https://img.shields.io/badge/Download-LLMTray%20for%20Mac-2f7d4f?style=for-the-badge&logo=apple&logoColor=white)](https://github.com/ipsupport-llc/llmtray/releases/latest/download/LLMTray-Full.dmg)
> [![GitHub stars](https://img.shields.io/github/stars/ipsupport-llc/llmtray?style=for-the-badge&logo=github)](https://github.com/ipsupport-llc/llmtray)

A 4-bit MLX quantization of [Z-Image-Turbo](https://huggingface.co/Tongyi-MAI/Z-Image-Turbo)
(Tongyi-MAI, 6.15B diffusion transformer, 9-step distilled), for
[mflux](https://github.com/filipstrand/mflux) on Apple Silicon — using
**GPTQ Hessian-based error correction** instead of mflux's own
round-to-nearest (RTN) `--quantize 4`. Same on-disk size and speed as plain
`mflux-save --quantize 4`; this is where GPTQ's correction actually earns
its keep (see comparison below — at 8-bit the gap is small, at 4-bit it's
not).

## Method

Same pipeline as the [8-bit release](https://huggingface.co/roman220220/z-image-turbo-gptq-mlx-8bit):
GPTQ-correct all 30 transformer blocks' attention and feed-forward linears
using activations captured from the model's own real denoising loop, then
re-quantize the corrected weights and splice them into mflux's own MLX
checkpoint format (zero format changes vs. plain `--quantize 4`). Full
method + code: [zimage-quant](https://github.com/rromenskyi/quant-ternary/tree/main/zimage-quant).

## Honest quality comparison

Same prompt, same seed (42), same 9 steps, RTN-4bit vs this GPTQ-4bit:

| RTN (plain `--quantize 4`) | GPTQ (this repo) |
|---|---|
| ![rtn](compare-rtn4bit-seed42.png) | ![gptq](compare-gptq4bit-seed42.png) |

At 4 bits, RTN's larger per-weight rounding error is enough to visibly
shift the *entire generation* to a different composition/camera angle for
the same prompt+seed, compared to what the model reliably produces at
higher precision (see the [8-bit card](https://huggingface.co/roman220220/z-image-turbo-gptq-mlx-8bit)'s
matching image). GPTQ-4bit stays much closer to that same composition —
measured against the 8-bit GPTQ output as a stable reference, GPTQ-4bit
scores modestly higher PSNR (17.1dB vs 16.6dB for RTN-4bit), but the
qualitative difference (same picture vs. a different picture) is the more
telling signal here than the pixel-level number. **Judge for yourself from
the images** — quantization-induced composition drift is inherently a
qualitative failure mode, not just a noise-level one.

This is a genuinely aggressive quantization (4 bits + a 64-wide group
scale, ~4.25 bits/weight effective) — expect visibly softer detail than
8-bit or full precision even with GPTQ's correction. Use this where memory
is the binding constraint, not where you need the sharpest possible output.

**If +0.8GB is affordable**, [z-image-turbo-gptq-mlx-mixed](https://huggingface.co/roman220220/z-image-turbo-gptq-mlx-mixed)
(attention at 8-bit, feed-forward at 4-bit, 6.3GB) noticeably better
preserves the model's generation composition than this uniform 4-bit
release — worth checking if this one's outputs feel unstable for your
prompts.

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

## Size

~5.5GB total (transformer + text encoder + VAE + tokenizer), same as plain
`mflux-save --quantize 4` — GPTQ's win here is quality-at-this-size, not a
smaller file. Peak MLX memory during generation: ~5.6GB (vs. ~8.7GB for the
8-bit release).

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

Modified from [Tongyi-MAI/Z-Image-Turbo](https://huggingface.co/Tongyi-MAI/Z-Image-Turbo): the attention and feed-forward linears of all 30 transformer blocks were GPTQ-corrected and quantized to 4-bit (group size 64), and the model was saved in mflux's MLX checkpoint format (same layout as `mflux-save --quantize 4`). The weights and configuration files in this repo are therefore modified versions of the original, not the original files.
