---
license: other
license_name: lfm1.0
license_link: LICENSE
base_model: LiquidAI/LFM2.5-1.2B-Instruct
pipeline_tag: text-generation
tags:
  - gguf
  - llama.cpp
  - lfm2
  - liquid
  - imatrix
  - jang
  - quantized
---

<p align="center">
  <img src="llmtray-banner.png" alt="LLMTray" width="100%">
</p>

# LFM2.5-1.2B-Instruct — GGUF, imatrix + JANG-mixed

> ### ▶ GGUF build for ollama / llama.cpp
> This is the **GGUF** build (ollama / llama.cpp, Arc, CPU). If you use
> [LLMTray](https://www.ipsupport.us/llmtray/) — IPSupport's local AI app for
> Apple Silicon, which runs MLX — use the MLX sibling
> [roman220220/LFM2.5-1.2B-Instruct-gptq-mlx-jang](https://huggingface.co/roman220220/LFM2.5-1.2B-Instruct-gptq-mlx-jang) instead.
>
> [![Download LLMTray](https://img.shields.io/badge/Download-LLMTray%20for%20Mac-2f7d4f?style=for-the-badge&logo=apple&logoColor=white)](https://github.com/ipsupport-llc/llmtray/releases/latest/download/LLMTray-Full.dmg)
> [![GitHub stars](https://img.shields.io/github/stars/ipsupport-llc/llmtray?style=for-the-badge&logo=github)](https://github.com/ipsupport-llc/llmtray)

A GGUF quantization of
[LiquidAI/LFM2.5-1.2B-Instruct](https://huggingface.co/LiquidAI/LFM2.5-1.2B-Instruct)
— Liquid AI's 1.2B on-device chat model (hybrid: gated short convolutions in
10 of 16 layers, full GQA attention with QK-layernorm in the other 6) —
with two deliberate improvements over a stock `Q4_K_M`:

1. **imatrix computed on a broad real corpus** (bartowski's
   `calibration_datav3` — code + prose + facts + multilingual).
2. **JANG-mixed precision**: the 6 full-attention layers and the tied
   embeddings kept at `Q8_0`, the 10 gated-short-convolution layers'
   in/out projections at `Q6_K`, the feed-forward at the `Q4_K_M` base
   preset. Same "spend bits where they matter" principle as this account's
   MLX GPTQ release of the same model.

Result: **0.84 GB** instead of 2.34 GB F16 (the bf16 original is 2.34 GB).

## The recipe

| Component | Tensors | Type |
|---|---|---|
| Full attention (q, k, v, out; 6 of 16 layers) | `attn_q/k/v/output` | `Q8_0` |
| Token embeddings (tied with the output head) | `token_embd` | `Q8_0` |
| Gated short convolution in/out projections (10 layers) | `shortconv.in_proj/out_proj` | `Q6_K` |
| Feed-forward (16 layers) | `ffn_gate/up/down` | `Q4_K_M` base preset |
| Conv kernels (3 taps), norms | `shortconv.conv`, `*_norm` | F32 (untouched) |

Importance matrix: `llama-imatrix`, 200 chunks on bartowski's
`calibration_datav3.txt`, on one RTX 4090.

## Measurements

Perplexity on wikitext-2 test (40 x 512-token chunks, `llama-perplexity`),
F16 vs this quant, same GGUF toolchain both sides:

| | Size | PPL (wikitext) |
|---|---|---|
| F16 | 2.34 GB | 17.87 ± 0.55 |
| **this model, Q4_K_M jang** | **0.84 GB** | **18.07 ± 0.55 (+1.1%)** |

## Usage

**Ollama** — the included `Modelfile` sets Liquid AI's recommended sampling
(temperature 0.1, top_k 50, top_p 0.1, repetition penalty 1.05) and pulls
the weights from this repo:

```bash
curl -L -o Modelfile https://huggingface.co/roman220220/LFM2.5-1.2B-Instruct-GGUF-jang-imatrix/resolve/main/Modelfile
ollama create lfm2.5-1.2b-instruct-jang -f Modelfile
ollama run lfm2.5-1.2b-instruct-jang
```

**llama.cpp** — launch with `--jinja` so the model's own chat template
applies:

```bash
llama-server -m lfm2.5-1.2b-instruct-jang.gguf -ngl 99 --jinja --port 8080
```

Read the
[base model card](https://huggingface.co/LiquidAI/LFM2.5-1.2B-Instruct) for
the model's scope and limitations, which apply here unchanged.

## Honest notes

- No dedicated LFM2 `RENDERER`/`PARSER` is known to ship in Ollama (unlike
  this account's gemma4 GGUF releases). The GGUF carries LFM2's jinja chat
  template via `convert_hf_to_gguf.py`, which recent Ollama applies
  directly; if tool-call tokens ever leak as raw text, fall back to
  llama.cpp `--jinja`.
- This is imatrix + JANG bit-allocation, **not** this account's GPTQ:
  GPTQ's Hessian correction is calibrated against a specific quant grid
  that doesn't transfer to llama.cpp's k-quant grid. The GPTQ/MLX version
  of this model is at
  [roman220220/LFM2.5-1.2B-Instruct-gptq-mlx-jang](https://huggingface.co/roman220220/LFM2.5-1.2B-Instruct-gptq-mlx-jang).

## Method and code

Pipeline: [rromenskyi/quant-ternary](https://github.com/rromenskyi/quant-ternary),
`lfm2-quant/poc/lfm2_gguf_pipeline.sh` (`VARIANT=lfm2-1.2b-instruct`),
`docs/FINDINGS.md`.

## The IPSupport local-AI stack

Local-first AI tools for macOS by [IPSupport](https://www.ipsupport.us) — nothing
leaves your Mac.

- **[LLMTray](https://www.ipsupport.us/llmtray/)** — your local AI
  workstation for macOS: chat with local LLMs, generate and edit images, make
  music, run agents, and serve an OpenAI-compatible API. Downloads models from
  Hugging Face in-app, with per-model profiles.
- **[IPSupport Code](https://ipsupport-llc.github.io/ipsupport-code/)** — your
  AI coding agent for real repositories: analyze, fix, test, report.

This is the GGUF build (ollama / llama.cpp). For the MLX build LLMTray runs, see [roman220220/LFM2.5-1.2B-Instruct-gptq-mlx-jang](https://huggingface.co/roman220220/LFM2.5-1.2B-Instruct-gptq-mlx-jang).

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

Modified from [LiquidAI/LFM2.5-1.2B-Instruct](https://huggingface.co/LiquidAI/LFM2.5-1.2B-Instruct): converted to GGUF and quantized with an importance matrix and per-tensor types (full attention and embeddings `Q8_0`, gated short convolutions `Q6_K`, feed-forward `Q4_K_M`). The weights and configuration files in this repo are therefore modified versions of the original, not the original files.
