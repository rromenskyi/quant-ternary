---
license: apache-2.0
base_model: thinkingmachines/Inkling-2-Mini
pipeline_tag: image-text-to-text
tags:
  - gguf
  - llama.cpp
  - inkling2
  - moe
  - jang
  - imatrix
  - quantized
  - vision
---

<p align="center">
  <img src="llmtray-banner.png" alt="LLMTray" width="100%">
</p>

# Inkling 2 Mini — GGUF, imatrix + JANG-mixed (text + vision)

> ### ▶ GGUF build for llama.cpp
> This is the **GGUF** build. If you use
> [LLMTray](https://www.ipsupport.us/llmtray/) — IPSupport's local AI app for
> Apple Silicon, which runs MLX — use the MLX sibling
> [roman220220/Inkling-2-Mini-gptq-mlx-jang](https://huggingface.co/roman220220/Inkling-2-Mini-gptq-mlx-jang)
> instead (13.1 GB, and it keeps the DSpark speculative-decoding draft,
> which llama.cpp does not run yet).
>
> [![Download LLMTray](https://img.shields.io/badge/Download-LLMTray%20for%20Mac-2f7d4f?style=for-the-badge&logo=apple&logoColor=white)](https://github.com/ipsupport-llc/llmtray/releases/latest/download/LLMTray-Full.dmg)
> [![GitHub stars](https://img.shields.io/github/stars/ipsupport-llc/llmtray?style=for-the-badge&logo=github)](https://github.com/ipsupport-llc/llmtray)

A GGUF quantization of
[thinkingmachines/Inkling-2-Mini](https://huggingface.co/thinkingmachines/Inkling-2-Mini)
— Thinking Machines' 26.4 B-parameter mixture-of-experts model (3.4 B
active per token, layers 6–17 looped three times) with a vision tower —
with an importance matrix from a broad corpus and higher precision where
the model is small and sensitive: **17.8 GB, +0.24 % perplexity on text
and +0.06 % on code against F16**. Vision is a separate mmproj file.

## The recipe

| Component | Tensors | Type |
|---|---|---|
| Attention (q, k, v, output, output gate, relative-position projection) | `attn_q/k/v/output/gate/rel` | `Q6_K` |
| Token embeddings and output head | `token_embd`, `output` | `Q6_K` |
| Routed experts (128 per layer, 8 active) | `ffn_gate/up/down_exps` | `Q4_K_M` base preset |
| Loop message gates, attention sinks, convolutions, norms, router | — | as llama.cpp's converter writes them |
| Vision tower | `mmproj-inkling-2-mini-f16.gguf` | F16 |

The release stores the experts in NVFP4; llama.cpp's converter dequantizes
them exactly, so the F16 GGUF is the release's own weights. Importance
matrix: `llama-imatrix`, 200 chunks of bartowski's `calibration_datav3`.
(`Q6_K` rather than `Q8_0` on attention: `Q8_0` decodes about 3x slower on
Vulkan cards without an int8 dot product.)

## Measurements

`llama-perplexity`, 40 chunks each, F16 against this quant, same build:

| | Size | PPL (wikitext-2 test) | PPL (Python stdlib) |
|---|---|---|---|
| F16 | 52.2 GB | 6.768 | 2.061 |
| **this model** | **17.8 GB** | **6.785 (+0.24 %)** | **2.062 (+0.06 %)** |

These are not comparable with the MLX build's numbers (different context
and windows). They are language-modelling numbers; the benchmarks on the
base model's card are Thinking Machines', not re-run here.

## Usage

Inkling 2 Mini needs a recent llama.cpp: Inkling 2 is in master since
#32217 and the Mini's loop since #32694 (November 2026); this build was
made and tested with master at `7778de0c3`. Launch with `--jinja` so the
model's own chat template applies.

```bash
llama-server -m inkling-2-mini-jang.gguf --mmproj mmproj-inkling-2-mini-f16.gguf \
  -ngl 99 --jinja --port 8080
```

Read the [base model card](https://huggingface.co/thinkingmachines/Inkling-2-Mini)
for its intended use and limitations, which apply here unchanged.

## Vision

```bash
llama-mtmd-cli -m inkling-2-mini-jang.gguf --mmproj mmproj-inkling-2-mini-f16.gguf \
  --image photo.png -p "Describe this image in one sentence." -ngl 99 --jinja
```

Tested on a synthetic image: "A yellow rectangle sits on a blue background
with a large red circle overlapping its lower-right corner." — correct.

## Honest notes

- The release's DSpark draft (`mtp.safetensors`) is not in this repo:
  llama.cpp's support for it is still a pull request (#32218). The MLX
  build has it.
- Open llama.cpp issues as of this build: perplexity NaN for the Mini on
  some CUDA setups (#32710; not seen here, A100) and prompts over ~96
  tokens decoding to `?` (#32778; our smoke test sends a longer prompt and
  got a correct answer). If you see either, update llama.cpp.
- Tested on one NVIDIA A100 with CUDA: perplexity, two chat answers through
  `llama-server --jinja` (no control tokens leaked), one image.

## Method and code

Pipeline, scripts and measurements:
[rromenskyi/quant-ternary](https://github.com/rromenskyi/quant-ternary),
`inkling2-quant/` (`poc/inkling2_gguf_pipeline.sh`, `docs/FINDINGS.md`).

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
converted to GGUF (experts dequantized from NVFP4) and quantized with
llama.cpp (`Q4_K_M` base, attention / embeddings / head `Q6_K`, importance
matrix); the vision tower converted to an F16 mmproj. The files in this
repo are therefore modified versions of the original, not the original files.
