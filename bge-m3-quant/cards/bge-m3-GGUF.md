---
license: mit
base_model: BAAI/bge-m3
pipeline_tag: feature-extraction
tags:
  - gguf
  - llama.cpp
  - embeddings
  - sentence-similarity
  - xlm-roberta
  - multilingual
  - quantized
---

<p align="center">
  <img src="llmtray-banner.png" alt="LLMTray" width="100%">
</p>

# BGE-M3 — GGUF (F16, Q8_0, Q6_K, Q5_K_M, Q4_K_M)

> ### ▶ The embedder of [LLMTray](https://www.ipsupport.us/llmtray/)
> LLMTray — IPSupport's local AI app — uses BGE-M3 to search project files
> and memory. These are GGUF builds for llama.cpp (LLMTray for Windows, or
> any app that serves embeddings through `llama-server`).
>
> [![Download LLMTray](https://img.shields.io/badge/Download-LLMTray%20for%20Mac-2f7d4f?style=for-the-badge&logo=apple&logoColor=white)](https://github.com/ipsupport-llc/llmtray/releases/latest/download/LLMTray-Full.dmg)
> [![GitHub stars](https://img.shields.io/github/stars/ipsupport-llc/llmtray?style=for-the-badge&logo=github)](https://github.com/ipsupport-llc/llmtray)

[BAAI/bge-m3](https://huggingface.co/BAAI/bge-m3) (revision `5617a9f`) — a
568 M-parameter multilingual embedding model (XLM-RoBERTa, 100+ languages,
8192-token context, 1024-dimensional vectors) — converted to GGUF with
llama.cpp `b12751` and quantized, each build measured against the original
model in PyTorch.

**Dense embeddings only.** BGE-M3 also has sparse (lexical) and
multi-vector (ColBERT) outputs; their heads are not in these files. The
dense vector is the CLS token's, L2-normalized — llama.cpp reads the CLS
pooling from the file.

## Which file

| File | Size | Cosine to the original (mean / min) | Translation retrieval | Code retrieval | Top-10 overlap |
|---|---|---|---|---|---|
| original (PyTorch, fp32) | 2.27 GB | — | 95.83 % | 74.2 % | — |
| `bge-m3-f16.gguf` | 1158 MB | 0.99999 / 0.9996 | 95.83 % | 74.2 % | 99.7 % |
| **`bge-m3-q8_0.gguf`** | **635 MB** | **0.99958 / 0.998** | **95.87 %** | **74.2 %** | **97.2 %** |
| `bge-m3-q6_k.gguf` | 499 MB | 0.9965 / 0.984 | 95.79 % | 72.5 % | 92.8 % |
| `bge-m3-q5_k_m.gguf` | 468 MB | 0.9937 / 0.974 | 95.79 % | 74.2 % | 90.2 % |
| `bge-m3-q4_k_m.gguf` | 438 MB | 0.9826 / 0.928 | 95.88 % | 75.8 % | 84.4 % |

**Q8_0 is the one to use**: indistinguishable from the original in
retrieval, half the size of F16. Q6_K is a fair trade if 136 MB matter.
Q5_K_M and Q4_K_M keep the retrieval accuracy of this easy test but
reorder results: a sixth of the top-10 changes at Q4_K_M, and some vectors
move far (cosine 0.93) — they are here for very tight memory budgets.

Our Q8_0 has the same tensors as
[ggml-org/bge-m3-Q8_0-GGUF](https://huggingface.co/ggml-org/bge-m3-Q8_0-GGUF)
(cosine 1.000000 between their vectors; the file differs by one tokenizer
metadata flag), so an index built with either works with the other.

How it was measured: every build served by `llama-server --embedding`
(`/v1/embeddings`, as apps use it), against the original in transformers
(fp32, CLS, L2-normalized), on 10,240 texts: the
[Tatoeba](https://huggingface.co/datasets/mteb/tatoeba-bitext-mining) test
pairs for Russian, Ukrainian, Chinese, Japanese and German against English
(translation retrieval: top-1 of the right translation among 1,000, both
directions, averaged) and Python standard-library functions (code
retrieval: a docstring finds its function body). Top-10 overlap: the share
of each query's ten nearest results that are the original model's. The
tokenizer was checked too: llama.cpp's token ids equal transformers' on
519 of the texts (every 25th, and every long one), no difference. Long inputs (5–8k tokens) were still being
measured when this was published; they will be added here.

## Usage

```bash
llama-server -m bge-m3-q8_0.gguf --embedding --port 8080 -c 8192 -b 8192 -ub 8192
curl http://localhost:8080/v1/embeddings -H "Content-Type: application/json" \
  -d '{"input": ["Где ближайшая аптека?", "Where is the nearest pharmacy?"]}'
```

`-ub` must be at least the longest input in tokens (an embedding is
computed in one batch); llama-server refuses longer inputs rather than
truncating them. No query or document prefixes: BGE-M3 doesn't use them.

Strip trailing whitespace before embedding: BAAI's `tokenizer.json` (and
llama.cpp, which follows it) turns a trailing newline into an extra token
before `</s>`, which some other tokenizer copies don't — the vectors then
differ slightly between implementations.

## Method and code

Pipeline, scripts and the full measurements:
[rromenskyi/quant-ternary](https://github.com/rromenskyi/quant-ternary),
`bge-m3-quant/` (`poc/bge_m3_gguf_pipeline.sh`, `docs/FINDINGS.md`).

## The IPSupport local-AI stack

Local-first AI tools by [IPSupport](https://www.ipsupport.us) — nothing
leaves your machine.

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

Licensed under the **MIT License**, the same license as the base model — see
[`LICENSE`](LICENSE). The base repository declares MIT in its metadata but
ships no license file, so the standard MIT text is included here.

Modified from [BAAI/bge-m3](https://huggingface.co/BAAI/bge-m3): converted
to GGUF (F16) and quantized with llama.cpp (Q8_0, Q6_K, Q5_K_M, Q4_K_M);
the sparse and ColBERT heads are not included. The files in this repo are
therefore modified versions of the original, not the original files.
