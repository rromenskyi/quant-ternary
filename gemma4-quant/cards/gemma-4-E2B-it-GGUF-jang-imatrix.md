---
license: apache-2.0
base_model: google/gemma-4-E2B-it
pipeline_tag: image-text-to-text
tags:
  - gguf
  - llama.cpp
  - gemma4
  - imatrix
  - jang
  - quantized
  - vision
---

<p align="center">
  <img src="llmtray-banner.png" alt="LLMTray" width="100%">
</p>

# Gemma 4 E2B — GGUF, imatrix + JANG-mixed, text + vision

> ### ▶ GGUF build for ollama / llama.cpp
> This is the **GGUF** build (ollama / llama.cpp). If you use
> [LLMTray](https://www.ipsupport.us/llmtray/) — IPSupport's local AI app for Apple Silicon, which runs
> MLX — use the MLX sibling
> [roman220220/gemma-4-E2B-it-gptq-mlx-jang](https://huggingface.co/roman220220/gemma-4-E2B-it-gptq-mlx-jang) instead.
>
> [![Download LLMTray](https://img.shields.io/badge/Download-LLMTray%20for%20Mac-2f7d4f?style=for-the-badge&logo=apple&logoColor=white)](https://github.com/ipsupport-llc/llmtray/releases/latest/download/LLMTray-Full.dmg)
> [![GitHub stars](https://img.shields.io/github/stars/ipsupport-llc/llmtray?style=for-the-badge&logo=github)](https://github.com/ipsupport-llc/llmtray)

A GGUF quantization of
[google/gemma-4-E2B-it](https://huggingface.co/google/gemma-4-E2B-it) (the
dense E2B multimodal variant) for **llama.cpp / ollama / any GGUF runtime**,
built for **maximum quality at minimum size**, with two improvements over a
stock quantize:

1. **imatrix computed on a broad real corpus** (bartowski's
   `calibration_datav3` — code + prose + facts + multilingual) — the
   GGUF-world equivalent of calibrating on real data.
2. **JANG-mixed per-tensor precision** — attention kept high, feed-forward
   pushed lower, via `llama-quantize` per-tensor overrides.

**Why E2B quantizes cleaner than the bigger Gemma 4 GGUFs:** E2B's hidden
size (1536 = 6×256) and intermediate size (6144 = 24×256) are both divisible
by 256, so llama.cpp's K-quants and IQ2/IQ3 formats are eligible on **every**
tensor. The 26B (2688) and 4B (3136) variants aren't 256-aligned on the
hidden dimension, so many of their tensors fall back to legacy formats — E2B
does not, so it can go smaller at the same quality.

## Recipe & size

| component | type | note |
|---|---|---|
| attention (`attn_q/k/v/output`) | **Q6_K** | small, error-sensitive — kept high |
| feed-forward + rest | **Q4_K_M** base | the bulk of the weights |
| imatrix | bartowski `calibration_datav3`, 200 chunks | importance-weighted |

**Text model: 3.26 GB** (from 8.67 GB f16). Vision projector
`mmproj-gemma4-e2b-f16.gguf`: 0.92 GB (F16 — see note below).

## Quality (wikitext-2-raw perplexity)

Perplexity, same llama.cpp binary, same wiki.test.raw, `-c 512`, 20 chunks:

| model (of `google/gemma-4-E2B-it`) | size | PPL |
|---|---|---|
| ggml-org **official** Q8_0 (reference) | ~2.9 GB | 233.4 |
| **this build** — Q4_K_M + imatrix + JANG | **3.26 GB** | **215.4** |

Our 3.26 GB imatrix build **matches (slightly beats) the official ggml-org Q8_0** of the same base model — i.e. the quantization is lossless-competitive at this size.

**Read the number carefully.** Raw-wikitext PPL runs very high (~230) for this model in llama.cpp — that is a property of the metric on an instruct/multimodal model tuned for chat, **not** a defect: the official ggml-org GGUF lands in the same range. (Google's separate QAT `q4_0` GGUF scores ~69, but that is a different, quantization-aware-**trained** checkpoint, not a post-hoc quant of `-it`, so it isn't a like-for-like comparison.) Judge this build by generation, not by the absolute number.

## Files

- `gemma4-e2b-jang.gguf` — the text model (imatrix + JANG), 3.26 GB.
- `mmproj-gemma4-e2b-f16.gguf` — the vision projector (F16), for image input.
- `Modelfile`, `params` — ollama packaging (`RENDERER gemma4` / `PARSER gemma4`).

> **mmproj is F16, not Q8_0:** quantizing the E2B vision projector to Q8_0
> hits a llama.cpp `GGML_ASSERT` (size mismatch on an audio conv1d tensor), so
> the projector ships as F16 (0.92 GB), which converts and serves correctly.
>
> **Audio:** Gemma 4's audio tower is not converted here — llama.cpp's GGUF
> path covers text + vision (mmproj) only. For audio use the
> [MLX build](https://huggingface.co/roman220220/gemma-4-E2B-it-gptq-mlx-jang).

## Usage

**llama.cpp** (always pass `--jinja`, or chat/thinking breaks):

```bash
llama-server -m gemma4-e2b-jang.gguf --mmproj mmproj-gemma4-e2b-f16.gguf --jinja
```

**ollama:**

```bash
curl -L -o Modelfile https://huggingface.co/roman220220/gemma-4-E2B-it-GGUF-jang-imatrix/resolve/main/Modelfile
ollama create gemma4-e2b-jang -f Modelfile && ollama run gemma4-e2b-jang
```

## Fixes baked in (needed for this architecture to chat correctly)

- **Tokenizer:** transformers' list-form `extra_special_tokens` is converted
  to dict form before conversion (deleting it silently breaks chat).
- **Control tokens → CONTROL:** `convert_hf_to_gguf.py` tags Gemma 4's
  control tokens (`<|turn>`, `<|channel>`, `<|think|>`, `<|tool*>`, …) as
  NORMAL, so they leak into chat as literal text; an in-place `token_type`
  patch flips them to CONTROL.
- **ollama `RENDERER`/`PARSER gemma4`:** a HF repo alone can't set ollama's
  native Gemma 4 renderer/parser, so the shipped `Modelfile` does.

## Method / code

Built with [gemma4-quant](https://github.com/rromenskyi/quant-ternary/tree/main/gemma4-quant)'s
`gemma4_e2b_gguf_pipeline.sh` on a CUDA pod (current llama.cpp from source).

## The IPSupport local-AI stack

Local-first AI tools for macOS by [IPSupport](https://www.ipsupport.us) — nothing
leaves your Mac.

- **[LLMTray](https://www.ipsupport.us/llmtray/)** — your local AI
  workstation for macOS: chat with local LLMs, generate and edit images, make
  music, run agents, and serve an OpenAI-compatible API. Downloads models from
  Hugging Face in-app, with per-model profiles.
- **[IPSupport Code](https://ipsupport-llc.github.io/ipsupport-code/)** — your
  AI coding agent for real repositories: analyze, fix, test, report.

This is the GGUF build (ollama / llama.cpp). For the MLX build LLMTray runs, see [roman220220/gemma-4-E2B-it-gptq-mlx-jang](https://huggingface.co/roman220220/gemma-4-E2B-it-gptq-mlx-jang).

## License

Licensed under the **Apache License 2.0**, the same license as the base model — see [`LICENSE`](LICENSE).

Modified from [google/gemma-4-E2B-it](https://huggingface.co/google/gemma-4-E2B-it): converted to GGUF and quantized with an importance matrix and JANG per-tensor types; the vision tower was converted to a separate Q8_0 mmproj GGUF. The weights and configuration files in this repo are therefore modified versions of the original, not the original files.

Gemma 4 is released by Google under Apache 2.0 ([Gemma 4 license terms](https://ai.google.dev/gemma/docs/gemma_4_license)).
