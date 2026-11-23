---
license: apache-2.0
base_model: thinkingmachines/Inkling-2-Mini
pipeline_tag: image-text-to-text
library_name: mlx
tags:
  - mlx
  - quantized
  - gptq
  - jang
  - inkling2
  - moe
  - vision
  - speculative-decoding
---

<p align="center">
  <img src="llmtray-banner.png" alt="LLMTray" width="100%">
</p>

# Inkling 2 Mini, GPTQ JANG 8/2/3 — extreme quant (MLX)

> ### ▶ Run it locally in [LLMTray](https://www.ipsupport.us/llmtray/)
> A free, native macOS app for local AI on Apple Silicon — chat, images,
> music, agents and an OpenAI-compatible API. Point it at this model; nothing
> leaves your Mac.
>
> [![Download LLMTray](https://img.shields.io/badge/Download-LLMTray%20for%20Mac-2f7d4f?style=for-the-badge&logo=apple&logoColor=white)](https://github.com/ipsupport-llc/llmtray/releases/latest/download/LLMTray-Full.dmg)
> [![GitHub stars](https://img.shields.io/github/stars/ipsupport-llc/llmtray?style=for-the-badge&logo=github)](https://github.com/ipsupport-llc/llmtray)

[thinkingmachines/Inkling-2-Mini](https://huggingface.co/thinkingmachines/Inkling-2-Mini)
— a 26.4 B-parameter mixture-of-experts model (3.4 B active per token) from
Thinking Machines Lab, with a looped middle (layers 6–17 run three times
with shared weights) and a vision tower — quantized for MLX with **GPTQ
and a per-component bit recipe** (attention 8 / routed experts' gate and up
2, down 3 bits): **11.1 GB instead of 19.5 GB (the NVFP4 release), +12.1 %
perplexity on text, +9.3 % on code**. This is an extreme quantization: it
loses noticeably more than the
[3-bit build](https://huggingface.co/roman220220/Inkling-2-Mini-gptq-mlx-jang)
(13.1 GB, +3.4 % / +2.4 %); take it when those 2 GB matter. The vision tower is kept in bf16, and the release's
DSpark draft ships too, for speculative decoding.

## The recipe

Bits by component, the same in every layer, GPTQ-calibrated (Hessian error
compensation on MLX's affine grid), starting from the release's weights
(its NVFP4 experts dequantized exactly). Each routed expert is calibrated on
the tokens the router sends to it, and the MLX files carry GPTQ's own codes,
scales and biases.

| Component | Bits (group 64) | Parameters |
|---|---|---|
| Attention (q, k, v, output, output gate, relative-position projection) | 8 | 0.65 B |
| Routed experts' gate and up (128 per layer, 8 active) | 2 | 16.3 B |
| Routed experts' down | 3 | 8.2 B |
| Router, attention sinks, dynamic-convolution weights, loop message gates | bf16 / fp32 | 0.03 B |
| Embeddings and output head (round-to-nearest) | 8 | 0.93 B |
| Vision tower | bf16 | 0.3 B |
| DSpark draft (`model-mtp.safetensors`; its recurrence bf16) | 8 | 0.32 B |

The looped layers have one weight for three passes: they are calibrated on
the first pass's inputs, then passes 2 and 3 run on the quantized layers and
the layers after the loop are calibrated on their output.
Calibration: 128 chunks of 512 tokens, half wikitext-2 train and half
Python source code, layer by layer, on one A100 (12 minutes).

## Measurements

Perplexity, 40 × 512 tokens, on English text (wikitext-2 test) and on
Python code (the Python standard library, which the calibration didn't use),
against the release (NVFP4 experts, everything else bf16) in HF transformers:

| | Size | PPL text | vs release | PPL code | vs release |
|---|---|---|---|---|---|
| release (HF transformers) | 19.5 GB | 8.985 | — | 2.466 | — |
| [3-bit build](https://huggingface.co/roman220220/Inkling-2-Mini-gptq-mlx-jang) | 13.1 GB | 9.292 | +3.4 % | 2.526 | +2.4 % |
| **this model (MLX)** | **11.1 GB** | **10.068** | **+12.1 %** | **2.696** | **+9.3 %** |

The routed experts hold 24.5 B of the 26.4 B parameters, so they set the
size: here 8.7 GB of the 11.1. MLX's own re-quantization of the calibrated
weights would have changed 11.9 % of the packed code words; the files carry
GPTQ's codes instead.

Speed and memory on a Mac are not measured yet for this model.

These are language-modelling numbers. The benchmarks on the base model's
card are Thinking Machines', for the release; they weren't re-run on this
quantization.

## Speculative decoding (DSpark)

The release ships a DSpark draft (`mtp.safetensors`): three small dense
layers that propose seven tokens in one pass from the model's own hidden
states, plus a recurrent head that chains them. It is here as
`model-mtp.safetensors` (342 MB, 8-bit; the recurrence bf16; the same file
as in the 3-bit build). With the
[ipsupport-llc/mlx-lm](https://github.com/ipsupport-llc/mlx-lm) fork the model
checks the drafts in one pass and keeps the longest correct prefix: every
token is still the model's own sample, at any temperature. Stock `mlx-lm`
has no Inkling 2 at all.

Tokens per verification step (greedy, all seven drafts tried, this repo's
weights): **1.92** on wikitext-2 test, **3.44** on the Python standard
library. The fork picks 0–7 drafts per step by what is fastest at the
moment. In bf16 a verification step over several tokens rounds slightly
unlike single steps, so on a near tie greedy output can differ from plain
decoding.

## Vision

On a test image the MLX vision tower's features match HF transformers' at
cosine 0.9988 (mean over image tokens), and the model describes the image
correctly ("A blue background with a yellow horizontal rectangle on the left
and a red circle overlapping its lower-right edge.").

## Usage

```bash
pip install "git+https://github.com/ipsupport-llc/mlx-lm.git"
mlx_lm.generate --model roman220220/Inkling-2-Mini-gptq-mlx-jang-small \
  --prompt "Write a Python function that parses ISO-8601 dates." --max-tokens 4096
mlx_lm.server --model roman220220/Inkling-2-Mini-gptq-mlx-jang-small --port 8080
```

It needs about 12 GB of GPU memory on Apple Silicon, plus the prompt cache.
Read the [base model card](https://huggingface.co/thinkingmachines/Inkling-2-Mini)
for its intended use and limitations, which apply here unchanged.

## Method and code

Pipeline, scripts and measurements:
[rromenskyi/quant-ternary](https://github.com/rromenskyi/quant-ternary),
`inkling2-quant/` (`poc/inkling2_mlx_pipeline.sh` and the scripts it runs,
`docs/FINDINGS.md`).

## The IPSupport local-AI stack

Local-first AI tools for macOS by [IPSupport](https://www.ipsupport.us) — nothing
leaves your Mac.

- **[LLMTray](https://www.ipsupport.us/llmtray/)** — your local AI
  workstation for macOS: chat with local LLMs, generate and edit images, make
  music, run agents, and serve an OpenAI-compatible API. Downloads models from
  Hugging Face in-app, with per-model profiles.
- **[IPSupport Code](https://ipsupport-llc.github.io/ipsupport-code/)** — your
  AI coding agent for real repositories: analyze, fix, test, report.

## Disclaimer

This repository contains a quantized conversion of a third-party model,
provided **"as is", without warranty of any kind**, express or implied.
IPSupport LLC did not train the model and is **not responsible for its
content, outputs or behavior, or for any damage, loss or liability** arising
from its use. You are responsible for evaluating the model and for how you
use it, including compliance with applicable laws and with the base model's
license and terms.

## License

Licensed under the **Apache License 2.0**, the same license as the base
model — see [`LICENSE`](LICENSE). The base repository declares Apache 2.0
in its metadata but ships no license file, so the standard text is included
here.

Modified from [thinkingmachines/Inkling-2-Mini](https://huggingface.co/thinkingmachines/Inkling-2-Mini):
the experts dequantized from NVFP4, quantized with GPTQ (attention 8,
routed experts' gate / up 2 and down 3 bits; embeddings and output head 8-bit) and converted to
MLX; the DSpark draft quantized to 8-bit (`model-mtp.safetensors`). The
weights and configuration files in this repo are therefore modified
versions of the original, not the original files.
