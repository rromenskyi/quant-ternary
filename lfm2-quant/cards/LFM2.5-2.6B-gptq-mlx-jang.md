---
license: other
license_name: lfm1.0
license_link: LICENSE
base_model: LiquidAI/LFM2.5-2.6B
pipeline_tag: text-generation
library_name: mlx
tags:
  - mlx
  - quantized
  - gptq
  - jang
  - lfm2
  - liquid
---

<p align="center">
  <img src="llmtray-banner.png" alt="LLMTray" width="100%">
</p>

# LFM2.5-2.6B, GPTQ JANG 8/6/4 (MLX)

> ### ▶ Run it locally in [LLMTray](https://www.ipsupport.us/llmtray/)
> A free, native macOS app for local AI on Apple Silicon — chat, images,
> music, agents and an OpenAI-compatible API. Point it at this model; nothing
> leaves your Mac.
>
> [![Download LLMTray](https://img.shields.io/badge/Download-LLMTray%20for%20Mac-2f7d4f?style=for-the-badge&logo=apple&logoColor=white)](https://github.com/ipsupport-llc/llmtray/releases/latest/download/LLMTray-Full.dmg)
> [![GitHub stars](https://img.shields.io/github/stars/ipsupport-llc/llmtray?style=for-the-badge&logo=github)](https://github.com/ipsupport-llc/llmtray)

[LiquidAI/LFM2.5-2.6B](https://huggingface.co/LiquidAI/LFM2.5-2.6B) — Liquid AI's 2.6B on-device reasoning model (hybrid: gated short convolutions plus 8 attention layers) — quantized for MLX with
**GPTQ and a per-component bit recipe**: **1.78 GB instead of
5.39 GB, −0.5 % perplexity on text, +1.7 % on code**. Liquid AI publishes no MLX build of this model.

## The recipe

Bits by component, GPTQ-calibrated (Hessian error compensation, layer by
layer on the outputs of the already-quantized layers) on MLX's exact affine
grid, group 64: the integer codes GPTQ chose are written into the MLX files
as they are, not re-rounded by `mlx_lm.convert`. Calibration: 128 × 512
tokens, wikitext-2 train mixed with Python source code, on one L40S.

| Component | Bits (group 64) | Parameters |
|---|---|---|
| Full attention (q, k, v, out; 8 of 30 layers) | 8 | 0.08 B |
| Gated short convolution (in_proj, out_proj; 22 layers) | 6 | 0.37 B |
| MLP (w1, w3, w2; 30 layers) | 4 | 1.98 B |
| Embeddings (tied with the output head; round-to-nearest) | 8 | 0.26 B |
| Conv kernels (3 taps), norms | bf16 | < 0.001 B |

2.70 B parameters in all.

## Measurements

Perplexity on wikitext-2 test and on held-out Python code (standard
library), 40 × 512 tokens each; every row measured the same way, in MLX, so
the ratios compare like with like. LFM2.5-2.6B is a chat model, so its raw
perplexity on plain wikitext is high (we checked the bf16 figure against HF
transformers); what matters is the change against bf16.

| | Size | PPL text | vs bf16 | PPL code | vs bf16 |
|---|---|---|---|---|---|
| bf16 (original) | 5.39 GB | 51.77 | — | 5.007 | — |
| **this model, 8/6/4** | **1.78 GB** | **51.51** | **−0.5 %** | **5.091** | **+1.7 %** |

The text figure is within measurement noise of bf16 (slightly below it).

## Usage

Works with stock `mlx-lm` (it supports LFM2); Liquid AI's settings:
temperature **0.1**, top_k 50, repetition penalty 1.1. It always reasons before answering (`<think>…</think>`; the chat template opens the tag).

```bash
pip install mlx-lm
mlx_lm.generate --model roman220220/LFM2.5-2.6B-gptq-mlx-jang \
  --prompt "Explain what a hash map is in three sentences." --temp 0.1 --top-k 50 --max-tokens 1024
```

As an OpenAI-compatible server:

```bash
mlx_lm.server --model roman220220/LFM2.5-2.6B-gptq-mlx-jang --port 8080
```

Speed was not measured on a Mac for this release; the base model card has
Liquid AI's figures for the architecture.

Read the [base model card](https://huggingface.co/LiquidAI/LFM2.5-2.6B) for the model's
scope and limitations, which apply here unchanged.

## Method and code

Pipeline, scripts and measurements:
[rromenskyi/quant-ternary](https://github.com/rromenskyi/quant-ternary),
`lfm2-quant/` (`poc/lfm2_mlx_pipeline.sh`, `poc/gptq_lfm2.py`,
`poc/convert_lfm2.py`, `poc/check_lfm2.py`, `docs/FINDINGS.md`).

## The IPSupport local-AI stack

Local-first AI tools for macOS by [IPSupport](https://www.ipsupport.us) — nothing
leaves your Mac.

- **[LLMTray](https://www.ipsupport.us/llmtray/)** — your local AI
  workstation for macOS: chat with local LLMs, generate and edit images, make
  music, run agents, and serve an OpenAI-compatible API. Downloads models from
  Hugging Face in-app, with per-model profiles.
- **[IPSupport Code](https://ipsupport-llc.github.io/ipsupport-code/)** — your
  AI coding agent for real repositories: analyze, fix, test, report.

Runs in LLMTray as a local chat LLM.

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

Modified from [LiquidAI/LFM2.5-2.6B](https://huggingface.co/LiquidAI/LFM2.5-2.6B): the decoder quantized with GPTQ to 8-bit attention, 6-bit convolution and 4-bit MLP weights, the embeddings to 8-bit, and
converted to MLX. The weights and configuration files in this repo are
therefore modified versions of the original, not the original files.
