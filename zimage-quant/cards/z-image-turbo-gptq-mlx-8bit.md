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

# Z-Image-Turbo — GPTQ, 8-bit (MLX / mflux)

> ### ▶ Run it locally in [LLMTray](https://www.ipsupport.us/llmtray/)
> A free, native macOS app for local AI on Apple Silicon — chat, images,
> music, agents and an OpenAI-compatible API. Point it at this model; nothing
> leaves your Mac.
>
> [![Download LLMTray](https://img.shields.io/badge/Download-LLMTray%20for%20Mac-2f7d4f?style=for-the-badge&logo=apple&logoColor=white)](https://github.com/ipsupport-llc/llmtray/releases/latest/download/LLMTray-Full.dmg)
> [![GitHub stars](https://img.shields.io/github/stars/ipsupport-llc/llmtray?style=for-the-badge&logo=github)](https://github.com/ipsupport-llc/llmtray)

An 8-bit MLX quantization of [Z-Image-Turbo](https://huggingface.co/Tongyi-MAI/Z-Image-Turbo)
(Tongyi-MAI, 6.15B diffusion transformer, 9-step distilled), for
[mflux](https://github.com/filipstrand/mflux) on Apple Silicon — but using
**GPTQ Hessian-based error correction** instead of mflux's own
round-to-nearest (RTN) `--quantize 8`, so it's the same on-disk size and
speed as a plain `mflux-save --quantize 8`, with better-preserved weights.

## What's actually different from plain `--quantize 8`

Nothing about the format — same shard layout, same key names, same
`mx.quantize(mode="affine")` grid mflux itself uses. What's different is
*how the quantized values were chosen*: instead of rounding every weight to
the nearest grid point independently (RTN), each of the 30 transformer
blocks' attention (`to_q/to_k/to_v/to_out.0`) and feed-forward
(`w1/w2/w3`) linears was corrected with GPTQ — quantize one column, measure
the rounding error, spread it onto the not-yet-quantized columns weighted
by that layer's activation Hessian, repeat. The Hessian was estimated from
real activations captured while running the model's own denoising loop
(hooks on every target Linear, several prompts, all 9 steps) — not
token/image data, since this is a diffusion transformer, not an LLM.

Full method + code: [zimage-quant](https://github.com/rromenskyi/quant-ternary/tree/main/zimage-quant)
(sibling of `nemotron-extreme-quant`, which does the same GPTQ correction
for LLMs — `gptq_nbit` is shared, unmodified code between the two).

## Honest quality comparison

Same prompt, same seed (42), same 9 steps, RTN-8bit vs this GPTQ-8bit:

| RTN (plain `--quantize 8`) | GPTQ (this repo) |
|---|---|
| ![rtn](compare-rtn8bit-seed42.png) | ![gptq](compare-gptq8bit-seed42.png) |

At 8 bits the two are visually near-identical (PSNR ≈ 35.3dB between them,
mean abs pixel diff ≈ 2.2/255) — 8 bits alone already has enough precision
that RTN's per-weight rounding error is small. **The 8-bit release is
mostly a correctness baseline and a drop-in you can trust**; the more
interesting quality delta shows up at 4-bit — see
[z-image-turbo-gptq-mlx-4bit](https://huggingface.co/roman220220/z-image-turbo-gptq-mlx-4bit)
and the smaller-but-more-stable [z-image-turbo-gptq-mlx-mixed](https://huggingface.co/roman220220/z-image-turbo-gptq-mlx-mixed)
(attention 8-bit + feed-forward 4-bit, 6.3GB).

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

~10GB total (transformer + text encoder + VAE + tokenizer), same as plain
`mflux-save --quantize 8` — GPTQ's win here is quality-at-this-size, not a
smaller file.

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

Modified from [Tongyi-MAI/Z-Image-Turbo](https://huggingface.co/Tongyi-MAI/Z-Image-Turbo): the attention (`to_q/to_k/to_v/to_out.0`) and feed-forward (`w1/w2/w3`) linears of all 30 transformer blocks were GPTQ-corrected and quantized to 8-bit, and the model was saved in mflux's MLX checkpoint format (same layout as `mflux-save --quantize 8`). The weights and configuration files in this repo are therefore modified versions of the original, not the original files.
