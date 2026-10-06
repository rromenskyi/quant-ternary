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

# Ornith-1.5-35B-A3B, GPTQ JANG 8/6/6/3 (MLX)

> ### ▶ Run it locally in [LLMTray](https://www.ipsupport.us/llmtray/)
> A free, native macOS app for local AI on Apple Silicon — chat, images,
> music, agents and an OpenAI-compatible API. Point it at this model; nothing
> leaves your Mac.
>
> [![Download LLMTray](https://img.shields.io/badge/Download-LLMTray%20for%20Mac-2f7d4f?style=for-the-badge&logo=apple&logoColor=white)](https://github.com/ipsupport-llc/llmtray/releases/latest/download/LLMTray-Full.dmg)
> [![GitHub stars](https://img.shields.io/github/stars/ipsupport-llc/llmtray?style=for-the-badge&logo=github)](https://github.com/ipsupport-llc/llmtray)

[ornith-ai/Ornith-1.5-35B-A3B](https://huggingface.co/ornith-ai/Ornith-1.5-35B-A3B)
— a 35 B-parameter mixture-of-experts model (about 3 B active per token)
from Ornith AI, built on the Qwen3.5-MoE architecture and post-trained with
reinforcement learning for coding and agentic tasks — quantized for MLX with
**GPTQ and a per-component bit recipe** (attention 8 / linear attention 6 /
shared expert 6 / routed experts 3 bits): **17.0 GB instead of 72 GB,
+6.8 % perplexity on text, +15.7 % on code**. The vision tower is kept, at 8 bits.

## The recipe

Bits by component, the same in every layer, GPTQ-calibrated (Hessian error
compensation on MLX's affine grid). Each routed expert is calibrated on the
tokens the router sends to it, and the MLX files carry GPTQ's own codes,
scales and biases.

| Component | Bits (group 64) | Parameters |
|---|---|---|
| Full attention (q, k, v, o; 10 of 40 layers) | 8 | 0.27 B |
| Gated DeltaNet linear attention (in_proj_qkv, in_proj_z, out_proj; 30 layers) | 6 | 1.0 B |
| Routed experts (256 per layer, 8 active; gate, up, down) | 3 | 32.2 B |
| Shared expert (gate, up, down) | 6 | 0.13 B |
| Router, shared-expert gate, delta-rule gates | bf16 | 0.03 B |
| Embeddings and output head (untied; round-to-nearest) | 8 | 1.02 B |
| Vision tower (position embedding bf16) | 8 | 0.4 B |

Calibration: 128 chunks of 512 tokens, half wikitext-2 train and half
Python source code, layer by layer (each layer on the outputs of the
already-quantized ones), on one A100.

## Measurements

Perplexity, 40 × 512 tokens, on English text (wikitext-2 test) and on
Python code (the Python standard library, which the calibration didn't use):

| | Size | PPL text | vs bf16 | PPL code | vs bf16 |
|---|---|---|---|---|---|
| bf16 (HF transformers) | 72 GB | 9.711 | — | 2.166 | — |
| **this model (MLX)** | **17.0 GB** | **10.368** | **+6.8 %** | **2.507** | **+15.7 %** |
| [2-bit gate / up build](https://huggingface.co/roman220220/Ornith-1.5-35B-A3B-gptq-mlx-jang-small) | 14.4 GB | 11.761 | +21.1 % | 3.122 | +44.1 % |

Code is more predictable than prose (bf16 perplexity 2.2 against 9.7), so
the same damage to the weights shows as a larger relative loss there; for a
coding model it is the number to watch.

The routed experts hold 32.2 B of the 35 B parameters, so they set the
size: at 3 bits they are 14 GB of the 17. MLX's own re-quantization of the
calibrated weights would have changed 8.3 % of the packed code words
(GPTQ's error feedback leaves group extremes unused, and MLX re-derives the
grid from min / max); the files carry GPTQ's codes instead.

Speed and memory (MLX, MacBook Air M5, 26 GB): **43 tokens/s** decoding,
788 tokens/s prefill on a 2K-token prompt; peak memory 17.2 GB on a short
prompt and 18.8 GB with 2K tokens of context. That is close to macOS's
default GPU memory limit on a 26 GB Mac (about 19 GB): for long contexts use
a Mac with 32 GB or more, or raise the limit (`sudo sysctl
iogpu.wired_limit_mb=…`).

These are language-modelling numbers. The coding and agentic benchmarks on
the base model's card are Ornith AI's, for the bf16 model; they weren't
re-run on this quantization.

## Vision

The vision tower works as in the base model: on a test image the MLX
tower's features match HF transformers' at cosine 0.994 (mean over
tokens), and the model describes a test image correctly ("A yellow
rectangle on the left partially overlaps a red circle on the right, both set
against a solid blue background"). The tower is 8-bit except its
position embedding, which stays bf16 (Qwen3-VL interpolates it in the
weight's dtype, which quantized is an integer).

Image input needs the [ipsupport-llc/mlx-lm](https://github.com/ipsupport-llc/mlx-lm)
fork (Qwen3.5 / Qwen3.5-MoE vision tower, interleaved mRoPE, image
preprocessing); stock `mlx-lm` loads this model text-only. The base model's
multi-token-prediction head is not carried.

## Usage

```bash
pip install "git+https://github.com/ipsupport-llc/mlx-lm.git"
mlx_lm.generate --model roman220220/Ornith-1.5-35B-A3B-gptq-mlx-jang \
  --prompt "Write a Python function that parses ISO-8601 dates." --max-tokens 4096
mlx_lm.server --model roman220220/Ornith-1.5-35B-A3B-gptq-mlx-jang --port 8080
```

It needs about 18 GB of GPU memory on Apple Silicon (see above). Read the
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
quantized with GPTQ (8/6/6/3 bits by component; embeddings, output
head and vision tower 8-bit) and converted to MLX; the multi-token-prediction
head was removed. The weights and configuration files in this repo are
therefore modified versions of the original, not the original files.
