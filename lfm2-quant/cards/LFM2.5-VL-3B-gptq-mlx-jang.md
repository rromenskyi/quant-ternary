---
license: other
license_name: lfm1.0
license_link: LICENSE
base_model: LiquidAI/LFM2.5-VL-3B
pipeline_tag: image-text-to-text
library_name: mlx
tags:
  - mlx
  - quantized
  - gptq
  - jang
  - lfm2
  - liquid
  - vision
---

<p align="center">
  <img src="llmtray-banner.png" alt="LLMTray" width="100%">
</p>

# LFM2.5-VL-3B, GPTQ JANG 8/6/4 (MLX), vision kept

> ### ▶ Run it locally in [LLMTray](https://www.ipsupport.us/llmtray/)
> A free, native macOS app for local AI on Apple Silicon — chat, images,
> music, agents and an OpenAI-compatible API. Point it at this model; nothing
> leaves your Mac.
>
> [![Download LLMTray](https://img.shields.io/badge/Download-LLMTray%20for%20Mac-2f7d4f?style=for-the-badge&logo=apple&logoColor=white)](https://github.com/ipsupport-llc/llmtray/releases/latest/download/LLMTray-Full.dmg)
> [![GitHub stars](https://img.shields.io/github/stars/ipsupport-llc/llmtray?style=for-the-badge&logo=github)](https://github.com/ipsupport-llc/llmtray)

[LiquidAI/LFM2.5-VL-3B](https://huggingface.co/LiquidAI/LFM2.5-VL-3B) —
Liquid AI's 3B on-device vision-language model (a SigLIP2 NaFlex vision
encoder on the LFM2.5-2.6B hybrid decoder) — quantized for MLX with **GPTQ
and a per-component bit recipe**, image input included: **2.36 GB instead of
6.25 GB, +2.3 % perplexity on text, +1.5 % on code**, image features at
cosine 0.997 to the original. As far as we know, the first MLX build of this
model.

## The recipe

Bits by component, the decoder GPTQ-calibrated (Hessian error compensation,
layer by layer on the outputs of the already-quantized layers) on MLX's exact
affine grid, group 64: the integer codes GPTQ chose are written into the MLX
files as they are, not re-rounded by `mlx_lm.convert`. Calibration: 128 × 512
tokens, wikitext-2 train mixed with Python source code, on one L40S.

| Component | Bits (group 64) | Parameters |
|---|---|---|
| Full attention (q, k, v, out; 8 of 30 layers) | 8 | 0.08 B |
| Gated short convolution (in_proj, out_proj; 22 layers) | 6 | 0.37 B |
| MLP (w1, w3, w2; 30 layers) | 4 | 1.98 B |
| Embeddings (tied with the output head; round-to-nearest) | 8 | 0.26 B |
| Vision tower (SigLIP2, 27 layers; round-to-nearest) | 8 | 0.28 B |
| Vision MLP fc2 (input width 4304, not a multiple of 64) | bf16 | 0.13 B |
| Projector (pixel unshuffle + 2-layer MLP) | 8 | 0.01 B |
| Vision position grid, conv kernels, norms | bf16 | < 0.001 B |

## Measurements

Text: perplexity on wikitext-2 test and on held-out Python code (standard
library), 40 × 512 tokens each, both rows in MLX.

| | Size | PPL text | vs bf16 | PPL code | vs bf16 |
|---|---|---|---|---|---|
| bf16 (original) | 6.25 GB | 51.77 | — | 13.451 | — |
| **this model** | **2.36 GB** | **52.94** | **+2.3 %** | **13.647** | **+1.5 %** |

Images: against HF transformers in fp32 on three test images (two photos cut
into 6 tiles + a thumbnail, one small image sent whole). The expanded image
tokens are identical to HF's for all three; the projected image features
(cosine per image token, mean):

| | Tiled photo 1 | Tiled photo 2 | Small image |
|---|---|---|---|
| bf16 in MLX | 0.998 | 0.998 | 0.9995 |
| **this model** | **0.997** | **0.997** | **0.998** |

bf16 itself sits at 0.998 against fp32 (HF's own bf16 run does the same), so
quantization adds about 0.1 %. Greedy answers to "Describe this image in two
sentences." stay the same in substance, e.g. HF fp32: *"…flowers, plants, and
an insect on the flower"*, this model: *"…flowers, plants, and a bee on a
flower"*.

## Usage

Image input needs the [ipsupport-llc/mlx-lm](https://github.com/ipsupport-llc/mlx-lm)
fork (LFM2-VL vision tower, projector and a torch-free tiling processor);
stock `mlx-lm` does not load `lfm2_vl` checkpoints. Liquid AI's settings:
temperature **0.2**, top_k 50, repetition penalty 1.0.

```bash
pip install "git+https://github.com/ipsupport-llc/mlx-lm.git"
mlx_lm.server --model roman220220/LFM2.5-VL-3B-gptq-mlx-jang --port 8080
```

Then send images as OpenAI chat `image_url` parts (data URLs). Large images
are cut into 512-pixel tiles (up to 10) plus a thumbnail, as in HF.

Speed was not measured on a Mac for this release.

Read the [base model card](https://huggingface.co/LiquidAI/LFM2.5-VL-3B) for
the model's scope and limitations, which apply here unchanged.

## Method and code

Pipeline, scripts and measurements:
[rromenskyi/quant-ternary](https://github.com/rromenskyi/quant-ternary),
`lfm2-quant/` (`poc/lfm2_mlx_pipeline.sh`, `poc/gptq_lfm2.py`,
`poc/convert_lfm2.py`, `poc/check_lfm2.py`, `poc/check_vl_lfm2.py`, `docs/FINDINGS.md`).

## The IPSupport local-AI stack

Local-first AI tools for macOS by [IPSupport](https://www.ipsupport.us) — nothing
leaves your Mac.

- **[LLMTray](https://www.ipsupport.us/llmtray/)** — your local AI
  workstation for macOS: chat with local LLMs, generate and edit images, make
  music, run agents, and serve an OpenAI-compatible API. Downloads models from
  Hugging Face in-app, with per-model profiles.
- **[IPSupport Code](https://ipsupport-llc.github.io/ipsupport-code/)** — your
  AI coding agent for real repositories: analyze, fix, test, report.

Runs in LLMTray as a chat model, with vision.

## Disclaimer

This repository contains a quantized conversion of a third-party model,
provided **"as is", without warranty of any kind**, express or implied.
IPSupport LLC did not train the model and is **not responsible for its
content, outputs or behavior, or for any damage, loss or liability** arising
from its use. You are responsible for evaluating the model and for how you
use it, including compliance with applicable laws and with the base model's
license and terms.

## License

Licensed under the **LFM Open License v1.0**, the same license as the base
model — see [`LICENSE`](LICENSE).

Modified from [LiquidAI/LFM2.5-VL-3B](https://huggingface.co/LiquidAI/LFM2.5-VL-3B): the decoder quantized with GPTQ to 8-bit attention, 6-bit convolution and 4-bit MLP weights, the embeddings, vision tower and projector to 8-bit, and
converted to MLX. The weights and configuration files in this repo are
therefore modified versions of the original, not the original files.
