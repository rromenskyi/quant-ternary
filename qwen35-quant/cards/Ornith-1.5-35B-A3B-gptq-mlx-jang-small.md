---
license: mit
base_model: ornith-ai/Ornith-1.5-35B-A3B
pipeline_tag: image-text-to-text
library_name: mlx
tags:
  - mlx
  - quantized
  - gptq
  - jang
  - qwen3_5_moe
  - moe
  - vision
---

<p align="center">
  <img src="llmtray-banner.png" alt="LLMTray" width="100%">
</p>

# Ornith-1.5-35B-A3B, GPTQ JANG 8/6/6/2-3 — extreme quant (MLX)

> ### ▶ Run it locally in [LLMTray](https://www.ipsupport.us/llmtray/)
> A free, native macOS app for local AI on Apple Silicon — chat, images,
> music, agents and an OpenAI-compatible API. Point it at this model; nothing
> leaves your Mac.
>
> [![Download LLMTray](https://img.shields.io/badge/Download-LLMTray%20for%20Mac-2f7d4f?style=for-the-badge&logo=apple&logoColor=white)](https://github.com/ipsupport-llc/llmtray/releases/latest/download/LLMTray-Full.dmg)
> [![GitHub stars](https://img.shields.io/github/stars/ipsupport-llc/llmtray?style=for-the-badge&logo=github)](https://github.com/ipsupport-llc/llmtray)

> [!WARNING]
> **Extreme quantization — expect noticeably worse answers.** The routed
> experts' gate and up projections, two thirds of the model's weights, are at
> **2 bits**. Perplexity against bf16: **+21.1 % on text and +44.1 % on
> Python code**, where the 3-bit
> [Ornith-1.5-35B-A3B-gptq-mlx-jang](https://huggingface.co/roman220220/Ornith-1.5-35B-A3B-gptq-mlx-jang)
> (17.0 GB) loses 6.8 % and 15.7 %. Use it only if that one doesn't fit your Mac.

[ornith-ai/Ornith-1.5-35B-A3B](https://huggingface.co/ornith-ai/Ornith-1.5-35B-A3B)
— a 35 B-parameter mixture-of-experts model (about 3 B active per token)
from Ornith AI, built on the Qwen3.5-MoE architecture and post-trained with
reinforcement learning for coding and agentic tasks — quantized for MLX with
**GPTQ and a per-component bit recipe** (attention 8 / linear attention 6 /
shared expert 6 / routed experts' gate and up 2, down 3 bits): **14.4 GB instead
of 72 GB, +21.1 % perplexity on text, +44.1 % on code**. The vision tower is kept, at 8 bits.

## The recipe

Bits by component, the same in every layer, GPTQ-calibrated (Hessian error
compensation on MLX's affine grid). Each routed expert is calibrated on the
tokens the router sends to it, and the MLX files carry GPTQ's own codes,
scales and biases.

| Component | Bits (group 64) | Parameters |
|---|---|---|
| Full attention (q, k, v, o; 10 of 40 layers) | 8 | 0.27 B |
| Gated DeltaNet linear attention (in_proj_qkv, in_proj_z, out_proj; 30 layers) | 6 | 1.0 B |
| Routed experts' gate and up (256 per layer, 8 active) | **2** | 21.5 B |
| Routed experts' down | 3 | 10.7 B |
| Shared expert (gate, up, down) | 6 | 0.13 B |
| Router, shared-expert gate, delta-rule gates | bf16 | 0.03 B |
| Embeddings and output head (untied; round-to-nearest) | 8 | 1.02 B |
| Vision tower (position embedding bf16) | 8 | 0.4 B |
| MTP head (fc, one attention + MoE layer; `model-mtp.safetensors`) | 4 | 0.85 B |

Calibration: 128 chunks of 512 tokens, half wikitext-2 train and half
Python source code, layer by layer (each layer on the outputs of the
already-quantized ones), on one A100.

## Measurements

Perplexity, 40 × 512 tokens, on English text (wikitext-2 test) and on
Python code (the Python standard library, which the calibration didn't use):

| | Size | PPL text | vs bf16 | PPL code | vs bf16 |
|---|---|---|---|---|---|
| bf16 (HF transformers) | 72 GB | 9.711 | — | 2.166 | — |
| 3-bit experts ([Ornith-1.5-35B-A3B-gptq-mlx-jang](https://huggingface.co/roman220220/Ornith-1.5-35B-A3B-gptq-mlx-jang)) | 17.0 GB | 10.368 | +6.8 % | 2.507 | +15.7 % |
| **this model, gate / up at 2 bits (MLX)** | **14.4 GB** | **11.761** | **+21.1 %** | **3.122** | **+44.1 %** |

On code this build loses much more than on prose: for programming, prefer
the 3-bit build whenever it fits.

The routed experts hold 32.2 B of the 35 B parameters, so they set the
size. The down projections stay at 3 bits; gate and up go to 2. The files
carry GPTQ's own codes, scales and biases (MLX's re-derived min / max grid
would differ).

These are language-modelling numbers. The coding and agentic benchmarks on
the base model's card are Ornith AI's, for the bf16 model; they weren't
re-run on this quantization.

## Speculative decoding (MTP head)

The base model's multi-token-prediction head (one MoE layer, 256 experts),
GPTQ 4-bit, is in its own file `model-mtp.safetensors` (476 MB; the same
file in this repo and in the 8/6/6/3 build: one base model). With the
[ipsupport-llc/mlx-lm](https://github.com/ipsupport-llc/mlx-lm) fork the
head drafts tokens and the model checks them in one pass: the same output
(each token is the model's own sample, at any temperature). How much faster
that decodes on a Mac isn't measured yet for this model; the fork picks 0–3
drafts per step by what is fastest at the moment. Stock `mlx-lm` drops the head on load.

First draft accepted (greedy, this repo's weights): **85.4 %** on
wikitext-2 test, **82.7 %** on the Python standard library.
It adds 0.48 GB to the 14.4 GB, which a 26 GB Mac has room for.

## Vision

The vision tower works as in the base model: on a test image the MLX
tower's features match HF transformers' at cosine 0.994 (mean over
tokens; the vision tower is the same as the 3-bit build's), and the model
describes a test image correctly ("A blue background with a yellow
rectangle on the left and a red circle on the right, where the circle
partially overlaps the rectangle"). The tower is 8-bit except its
position embedding, which stays bf16 (Qwen3-VL interpolates it in the
weight's dtype, which quantized is an integer).

Image input needs the [ipsupport-llc/mlx-lm](https://github.com/ipsupport-llc/mlx-lm)
fork (Qwen3.5 / Qwen3.5-MoE vision tower, interleaved mRoPE, image
preprocessing); stock `mlx-lm` loads this model text-only.

## Usage

```bash
pip install "git+https://github.com/ipsupport-llc/mlx-lm.git"
mlx_lm.generate --model roman220220/Ornith-1.5-35B-A3B-gptq-mlx-jang-small \
  --prompt "Write a Python function that parses ISO-8601 dates." --max-tokens 4096
mlx_lm.server --model roman220220/Ornith-1.5-35B-A3B-gptq-mlx-jang-small --port 8080
```

It needs about 16 GB of GPU memory on Apple Silicon with a 2K-token context
(estimated from the 17 GB build's measured 18.8 GB). Read the
[base model card](https://huggingface.co/ornith-ai/Ornith-1.5-35B-A3B) for its
intended use and limitations, which apply here unchanged.

## Method and code

Pipeline, scripts and measurements:
[rromenskyi/quant-ternary](https://github.com/rromenskyi/quant-ternary),
`qwen35-quant/` (`poc/qwen35_mlx_pipeline.sh` and the scripts it runs,
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

Licensed under the **MIT License**, the same license as the base model — see
[`LICENSE`](LICENSE). The base repository declares MIT in its metadata but
ships no license file, so the standard MIT text is included here.

Modified from [ornith-ai/Ornith-1.5-35B-A3B](https://huggingface.co/ornith-ai/Ornith-1.5-35B-A3B):
quantized with GPTQ (8/6/6/2-3 bits by component; embeddings, output
head and vision tower 8-bit) and converted to MLX; the multi-token-prediction
head quantized to 4-bit (`model-mtp.safetensors`). The weights and
configuration files in this repo are therefore modified versions of the original, not the original files.
