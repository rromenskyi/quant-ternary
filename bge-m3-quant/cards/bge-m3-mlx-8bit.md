---
license: mit
base_model: BAAI/bge-m3
pipeline_tag: feature-extraction
library_name: mlx
tags:
  - mlx
  - embeddings
  - sentence-similarity
  - xlm-roberta
  - multilingual
  - quantized
---

<p align="center">
  <img src="llmtray-banner.png" alt="LLMTray" width="100%">
</p>

# BGE-M3 — MLX, 8-bit, group 64

> ### ▶ The embedder of [LLMTray](https://www.ipsupport.us/llmtray/)
> LLMTray — IPSupport's local AI app for Apple Silicon — uses BGE-M3 to
> search project files and memory. This is an MLX build of it, made for and
> measured through LLMTray's own embedder runtime.
>
> [![Download LLMTray](https://img.shields.io/badge/Download-LLMTray%20for%20Mac-2f7d4f?style=for-the-badge&logo=apple&logoColor=white)](https://github.com/ipsupport-llc/llmtray/releases/latest/download/LLMTray-Full.dmg)
> [![GitHub stars](https://img.shields.io/github/stars/ipsupport-llc/llmtray?style=for-the-badge&logo=github)](https://github.com/ipsupport-llc/llmtray)

[BAAI/bge-m3](https://huggingface.co/BAAI/bge-m3) (revision `5617a9f`) — a
568 M-parameter multilingual embedding model (XLM-RoBERTa, 100+ languages,
8192-token context, 1024-dimensional vectors) — for MLX, 610 MB. Indistinguishable from the original in retrieval at about half the size of fp16 — the build to use when memory matters.

**Dense embeddings only**: the CLS token's last hidden state, L2-normalized
(BGE-M3's dense vector). The sparse and ColBERT heads are not included.

## The MLX builds, measured

| Build | Size | Cosine to the original, mean / min: sentences · code · long texts | Translation retrieval | Code retrieval | Top-10 overlap |
|---|---|---|---|---|---|
| original (PyTorch, fp32) | 2.27 GB | — | 95.83 % | 49.7 % | — |
| [`bge-m3-mlx-fp16`](https://huggingface.co/roman220220/bge-m3-mlx-fp16) | 1133 MB | 1.00000 / 1.0000 · 1.00000 / 0.9995 · 0.99998 / 0.9999 | 95.84 % | 49.7 % | 99.8 % |
| **[`bge-m3-mlx-8bit`](https://huggingface.co/roman220220/bge-m3-mlx-8bit)** | **610 MB** | **0.99986 / 0.9993 · 0.99975 / 0.9990 · 0.99921 / 0.9983** | **95.85 %** | **49.8 %** | **98.5 %** |
| [`bge-m3-mlx-6bit`](https://huggingface.co/roman220220/bge-m3-mlx-6bit) | 471 MB | 0.99792 / 0.969 · 0.99642 / 0.9933 · 0.99257 / 0.9894 | 95.84 % | 50.0 % | 94.4 % |

Each build was run through LLMTray's embedder runtime
([`runtime/llmtray_embed`](https://github.com/ipsupport-llc/llmtray/tree/main/runtime/llmtray_embed):
its loader, tokenizer, batching, CLS pooling and normalization; fp16
compute, on an Apple M5) against the original in transformers (fp32), on
12,039 texts: the
[Tatoeba](https://huggingface.co/datasets/mteb/tatoeba-bitext-mining) test
pairs for Russian, Ukrainian, Chinese, Japanese and German against English
(translation retrieval: top-1 of the right translation among 1,000, both
directions, averaged), 1,000 Python standard-library functions (code
retrieval: a docstring finds its function body among 1,000) and 39 whole
standard-library modules of 5–6.5k tokens. Top-10 overlap: the share of
each query's ten nearest results that are the original model's.

The quantized builds quantize the linear layers and the word embeddings
(250k × 1024, about half the model) at group size 64; the position and
token-type embeddings and the layer norms stay fp16. GGUF builds for
llama.cpp: [roman220220/bge-m3-GGUF](https://huggingface.co/roman220220/bge-m3-GGUF)
(the MLX 8-bit is closer to the original than the GGUF Q8_0: 98.5 % against
97.1 % top-10 overlap).

## Files and loading

`model.safetensors` in the Hugging Face layout (`embeddings.*`,
`encoder.layer.N.*`, no pooler head), `config.json` (BAAI's, plus
`"quantization": {"group_size": 64, "bits": 8}`),
`tokenizer.json`. The modules stored quantized are the ones with `.scales`
and `.biases`: quantize exactly those (as LLMTray's loader does) before
loading. Other MLX embedding loaders were not tested.

The tokenizer is the one that doesn't add a token for trailing whitespace
(BAAI's `tokenizer.json` turns a trailing newline into an extra token before
`</s>`): strip your texts and every implementation agrees.

## Method and code

Scripts and the full measurements:
[rromenskyi/quant-ternary](https://github.com/rromenskyi/quant-ternary),
`bge-m3-quant/` (`poc/mlx_build.py`, `poc/eval_mlx.py`, `docs/FINDINGS.md`).

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

This repository contains a converted and quantized copy of a third-party model,
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

Modified from [BAAI/bge-m3](https://huggingface.co/BAAI/bge-m3): converted to
MLX (8-bit, group 64), linear layers and word embeddings quantized; the pooler, sparse and ColBERT
heads are not included. The files in this repo are therefore modified
versions of the original, not the original files.
