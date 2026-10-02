---
license: apache-2.0
base_model: google/gemma-4-E2B-it-qat-q4_0-unquantized
pipeline_tag: image-text-to-text
tags:
  - mlx
  - gemma4
  - quantized
  - qat
  - multimodal
  - vision
  - audio
  - mobile
---

<p align="center">
  <img src="llmtray-banner.png" alt="LLMTray" width="100%">
</p>

# Gemma 4 E2B — the small QAT build, 3.1 GB, MLX (text + vision + audio)

> ### ▶ Run it locally in [LLMTray](https://www.ipsupport.us/llmtray/)
> A free, native macOS app for local AI on Apple Silicon — chat, images,
> music, agents and an OpenAI-compatible API. Point it at this model; nothing
> leaves your Mac.
>
> [![Download LLMTray](https://img.shields.io/badge/Download-LLMTray%20for%20Mac-2f7d4f?style=for-the-badge&logo=apple&logoColor=white)](https://github.com/ipsupport-llc/llmtray/releases/latest/download/LLMTray-Full.dmg)
> [![GitHub stars](https://img.shields.io/github/stars/ipsupport-llc/llmtray?style=for-the-badge&logo=github)](https://github.com/ipsupport-llc/llmtray)

The smallest build of Gemma 4 E2B we could make without making it dumber:
**3.10 GB, almost a gigabyte under**
[our q4_0-grid build](https://huggingface.co/roman220220/gemma-4-E2B-it-qat-mlx)
(4.04 GB), for KL +0.0016. Made for small machines: 8 GB Macs, iPhones and
iPads with MLX, anything where a gigabyte matters. It has the vision and audio
towers.

A GGUF build of the same recipe, for llama.cpp / ollama / phone apps:
[roman220220/gemma-4-E2B-it-qat-phone-GGUF](https://huggingface.co/roman220220/gemma-4-E2B-it-qat-phone-GGUF).

## How it's made

It starts from [our q4_0-grid build](https://huggingface.co/roman220220/gemma-4-E2B-it-qat-mlx)
and squeezes only what the measurements said was free:

- **The text decoder stays on Google's QAT grid.** Its 4-bit Linears carry
  the very codes of Google's own
  [q4_0 GGUF](https://huggingface.co/google/gemma-4-E2B-it-qat-q4_0-gguf),
  block for block (only the scales are rounded from fp16 to bf16): the grid
  the quantization-aware training trained for.
- **The 126 Linears that lose most at 4-bit stay at 8-bit** (+100 MB, chosen
  by measured KL per MB). Dropping them doubles the KL; they stay.
- **The per-layer embeddings go from 6 to 4 bits.** Half of E2B is its
  per-layer embedding table (1.9 GB at 6-bit). At 4-bit it costs KL +0.0015
  and saves ~0.6 GB. This is most of the saving.
- **The vision and audio towers go from 8 to 4 bits.**
- **What we did not do**, because it measurably costs: token embeddings
  below 6-bit (they're also the output head), per-layer embeddings at 3-bit.

**Why not Google's own mobile QAT?**
[google/gemma-4-E2B-it-qat-mobile-transformers](https://huggingface.co/google/gemma-4-E2B-it-qat-mobile-transformers)
is smaller (2-bit MLPs and embeddings), and we converted it to MLX exactly
(2.57 GB). But it was trained for an NPU that also runs activations in int8.
Without them, as MLX and llama.cpp run it, its perplexity is 65 instead of 41,
and its audio tower hears nothing. Both are worse than this build (36.4).

## Measured

Perplexity on wikitext-2 (test), 128 windows of 512 tokens, each in Gemma's
chat framing (a user turn, then the model's turn with thinking off). KL
divergence and top-1 agreement are to the bf16 QAT master weights.

| model | size | PPL | KL to QAT master | top-1 agree |
|---|---|---|---|---|
| QAT master weights, bf16 ([google/gemma-4-E2B-it-qat-q4_0-unquantized](https://huggingface.co/google/gemma-4-E2B-it-qat-q4_0-unquantized)) | — | 36.04 | — | — |
| [roman220220/gemma-4-E2B-it-qat-mlx](https://huggingface.co/roman220220/gemma-4-E2B-it-qat-mlx) | 4.04 GB | 36.38 | 0.0226 | 92.73% |
| **this model** | **3.10 GB** | **36.44** | **0.0242** | **92.48%** |

### Speed and memory (MacBook Air M5, mlx-lm, decode)

| build | tokens/s | peak memory |
|---|---|---|
| [roman220220/gemma-4-E2B-it-qat-mlx](https://huggingface.co/roman220220/gemma-4-E2B-it-qat-mlx) | 70.3 | 3.80 GB |
| **this model** | **70.2** | **3.26 GB** |

The same speed: the per-layer embeddings are looked up, not multiplied, so
fewer bits there save memory, not time.

### Checked

Before upload (`gemma4_smoke_test.py`):
- it answers in chat: three facts about the Roman Empire;
- vision, through the 4-bit tower: "A red fox is in this picture.";
- audio, through the 4-bit tower: a spoken clip transcribed word for word
  ("the quick brown fox jumps over the lazy dog");
- no large weight is left unquantized; peak 4.2 GB with an image and audio;
- it loads in LLMTray's runtime ([ipsupport-llc/mlx-lm](https://github.com/ipsupport-llc/mlx-lm)).

## Usage

```bash
pip install "mlx-lm @ git+https://github.com/ipsupport-llc/mlx-lm.git"
```

```python
from mlx_lm import load, generate

model, tokenizer = load("roman220220/gemma-4-E2B-it-qat-phone-mlx")
messages = [{"role": "user", "content": "What is the capital of France?"}]
prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
print(generate(model, tokenizer, prompt=prompt, max_tokens=200))
```

Images and audio go through `mlx_lm.multimodal` in the same fork, which is
what LLMTray uses.

## Method / code

[rromenskyi/quant-ternary/gemma4-quant](https://github.com/rromenskyi/quant-ternary/tree/main/gemma4-quant)
has the code and the full lab notes (`docs/FINDINGS.md`, "E2B for phones"):
- `e2b_phone.sh`: this build, in MLX and GGUF, each scored against its master;
- `mobile_sweep.sh`: the step-by-step sweep behind the recipe;
- `qat_mobile_convert.py`: Google's mobile QAT to MLX;
- `qat_aligned_convert.py`, `qat_sensitivity.py`, `qat_eval.py`.

## The IPSupport local-AI stack

Local-first AI tools for macOS by [IPSupport](https://www.ipsupport.us) — nothing
leaves your Mac.

- **[LLMTray](https://www.ipsupport.us/llmtray/)** — your local AI
  workstation for macOS: chat with local LLMs, generate and edit images, make
  music, run agents, and serve an OpenAI-compatible API. Downloads models from
  Hugging Face in-app, with per-model profiles.
- **[IPSupport Code](https://ipsupport-llc.github.io/ipsupport-code/)** — your
  AI coding agent for real repositories: analyze, fix, test, report.

Runs in LLMTray as a chat model, with vision and audio.

## License

Licensed under the **Apache License 2.0**, the same license as the base model — see [`LICENSE`](LICENSE).

Modified from [google/gemma-4-E2B-it-qat-q4_0-unquantized](https://huggingface.co/google/gemma-4-E2B-it-qat-q4_0-unquantized) (the QAT release of [google/gemma-4-E2B-it](https://huggingface.co/google/gemma-4-E2B-it)): quantized to MLX. The text decoder's Linears are on Google's q4_0 grid; 126 of them are at 8-bit, the token embeddings at 6-bit, the per-layer embeddings and the vision/audio towers at 4-bit. The weights and configuration files in this repo are therefore modified versions of the original, not the original files.
