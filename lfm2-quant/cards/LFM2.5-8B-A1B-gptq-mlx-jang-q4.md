---
license: other
license_name: lfm1.0
license_link: LICENSE
base_model: LiquidAI/LFM2.5-8B-A1B
pipeline_tag: text-generation
library_name: mlx
tags:
  - mlx
  - quantized
  - gptq
  - jang
  - lfm2
  - lfm2_moe
  - moe
  - liquid
---

<p align="center">
  <img src="llmtray-banner.png" alt="LLMTray" width="100%">
</p>

# LFM2.5-8B-A1B, GPTQ JANG 8/6/6, experts 4-bit (MLX)

> ### ▶ Run it locally in [LLMTray](https://www.ipsupport.us/llmtray/)
> A free, native macOS app for local AI on Apple Silicon — chat, images,
> music, agents and an OpenAI-compatible API. Point it at this model; nothing
> leaves your Mac.
>
> [![Download LLMTray](https://img.shields.io/badge/Download-LLMTray%20for%20Mac-2f7d4f?style=for-the-badge&logo=apple&logoColor=white)](https://github.com/ipsupport-llc/llmtray/releases/latest/download/LLMTray-Full.dmg)
> [![GitHub stars](https://img.shields.io/github/stars/ipsupport-llc/llmtray?style=for-the-badge&logo=github)](https://github.com/ipsupport-llc/llmtray)

[LiquidAI/LFM2.5-8B-A1B](https://huggingface.co/LiquidAI/LFM2.5-8B-A1B) —
Liquid AI's on-device mixture-of-experts model, 8.5 B parameters with about
1.5 B active per token (32 experts, top-4, on a hybrid of gated short
convolutions and 6 attention layers) — quantized for MLX with **GPTQ and a
per-component bit recipe**: **5.03 GB instead of 16.94 GB, +2.2 % perplexity on text, +2.6 % on code** — a quarter (text) to a third (code) of the loss of Liquid AI's official 4-bit build, at 0.19 GB more. Sibling: [3-bit experts (4.06 GB)](https://huggingface.co/roman220220/LFM2.5-8B-A1B-gptq-mlx-jang).

## The recipe

Bits by component, GPTQ-calibrated (Hessian error compensation, layer by
layer on the outputs of the already-quantized layers) on MLX's exact affine
grid, group 64: the integer codes GPTQ chose are written into the MLX files
as they are, not re-rounded by `mlx_lm.convert`. Calibration: 128 × 512
tokens, wikitext-2 train mixed with Python source code, on one L40S.

| Component | Bits (group 64) | Parameters |
|---|---|---|
| Experts gate / up (w1, w3; 32 experts × 22 MoE layers) | 4 | 5.17 B |
| Experts down (w2) | 4 | 2.58 B |
| Full attention (q, k, v, out; 6 of 24 layers) | 8 | 0.06 B |
| Gated short convolution (in_proj, out_proj; 18 layers) | 6 | 0.30 B |
| Dense MLP (first 2 layers) | 6 | 0.09 B |
| Embeddings (tied with the output head; round-to-nearest) | 8 | 0.26 B |
| Router (gate, expert bias), conv kernels, norms | bf16 | 0.001 B |

The experts are 92 % of the weights, so their bits set the size; everything
the token passes through on every step (attention, convolutions, dense MLP,
embeddings) is kept at 6–8 bits. The experts are calibrated each on the
tokens the router actually sends it (up to 8,192 rows per expert).

## Measurements

Perplexity on wikitext-2 test and on held-out Python code (standard
library), 40 × 512 tokens each; every row measured the same way, in MLX, so
the ratios compare like with like. LFM2.5-8B-A1B is a chat model, so its raw
perplexity on plain wikitext is high (we checked the bf16 figure against HF
transformers); what matters is the change against bf16.

| | Size | PPL text | vs bf16 | PPL code | vs bf16 |
|---|---|---|---|---|---|
| bf16 (original) | 16.94 GB | 91.76 | — | 13.128 | — |
| [LiquidAI/LFM2.5-8B-A1B-MLX-4bit](https://huggingface.co/LiquidAI/LFM2.5-8B-A1B-MLX-4bit) (Liquid AI) | 4.83 GB | 100.25 | +9.2 % | 14.004 | +6.7 % |
| experts 3-bit ([sibling](https://huggingface.co/roman220220/LFM2.5-8B-A1B-gptq-mlx-jang)) | 4.06 GB | 100.00 | +9.0 % | 14.149 | +7.8 % |
| **experts 4-bit (this model)** | **5.03 GB** | **93.77** | **+2.2 %** | **13.468** | **+2.6 %** |

## Usage

Works with stock `mlx-lm` (it supports LFM2); Liquid AI's settings:
temperature **0.2**, top_k 80, repetition penalty 1.05. It reasons before answering (`<think>…</think>`).

```bash
pip install mlx-lm
mlx_lm.generate --model roman220220/LFM2.5-8B-A1B-gptq-mlx-jang-q4 \
  --prompt "Explain what a hash map is in three sentences." --temp 0.2 --top-k 80 --max-tokens 1024
```

As an OpenAI-compatible server:

```bash
mlx_lm.server --model roman220220/LFM2.5-8B-A1B-gptq-mlx-jang-q4 --port 8080
```

Speed was not measured on a Mac for this release; the base model card has
Liquid AI's figures for the architecture.

Read the [base model card](https://huggingface.co/LiquidAI/LFM2.5-8B-A1B) for
the model's scope and limitations, which apply here unchanged.

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

Modified from [LiquidAI/LFM2.5-8B-A1B](https://huggingface.co/LiquidAI/LFM2.5-8B-A1B): the experts quantized with GPTQ to 4-bit, attention to 8-bit, the convolutions and dense MLP to 6-bit, the embeddings to 8-bit (router kept bf16), and
converted to MLX. The weights and configuration files in this repo are
therefore modified versions of the original, not the original files.
