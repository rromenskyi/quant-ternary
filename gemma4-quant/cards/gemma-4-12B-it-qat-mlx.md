---
license: apache-2.0
base_model: google/gemma-4-12B-it-qat-q4_0-unquantized
pipeline_tag: image-text-to-text
tags:
  - mlx
  - gemma4
  - quantized
  - qat
  - multimodal
  - vision
  - audio
---

<p align="center">
  <img src="llmtray-banner.png" alt="LLMTray" width="100%">
</p>

# Gemma 4 12B — Google's QAT on its exact q4_0 grid, MLX (text + vision + audio)

> ### ▶ Run it locally in [LLMTray](https://www.ipsupport.us/llmtray/)
> A free, native macOS app for local AI on Apple Silicon — chat, images,
> music, agents and an OpenAI-compatible API. Point it at this model; nothing
> leaves your Mac.
>
> [![Download LLMTray](https://img.shields.io/badge/Download-LLMTray%20for%20Mac-2f7d4f?style=for-the-badge&logo=apple&logoColor=white)](https://github.com/ipsupport-llc/llmtray/releases/latest/download/LLMTray-Full.dmg)
> [![GitHub stars](https://img.shields.io/github/stars/ipsupport-llc/llmtray?style=for-the-badge&logo=github)](https://github.com/ipsupport-llc/llmtray)

Google trained Gemma 4 12B with quantization-aware training (QAT) for
llama.cpp's **q4_0**. This is that model in MLX, **on the very grid the QAT
trained for**. The text decoder's 4-bit weights are identical, bit for bit,
to Google's own [q4_0 GGUF](https://huggingface.co/google/gemma-4-12B-it-qat-q4_0-gguf).
On top of that, the 44 Linears that lose the most at 4-bit are kept
at 8-bit, which costs 200 MB. Images and audio are included.

The 12B has no vision or audio encoder: images go in as raw 48×48 pixel
patches and audio as raw waveform frames, each through one small projection
into the text model. Mainline mlx-lm loads it text-only; LLMTray's runtime
([ipsupport-llc/mlx-lm](https://github.com/ipsupport-llc/mlx-lm), from
`17ab9af`) runs its images and audio.

## How it's made

- **The QAT grid, exactly.** Google's GGUF is q4_0 of the
  `-qat-q4_0-unquantized` master weights: every block's scale and every code
  matches. MLX's affine 4-bit with group 32 stores `scale·q + bias`; with
  `scale = d` and `bias = -8d` that is q4_0's grid itself. 284 Linears are
  written this way, and each is checked against the GGUF at conversion. The
  one difference is that the fp16 scale is rounded to bf16, MLX's scale dtype:
  under 1/64 of a step per weight.
- **Why not the usual conversions:**
  - `mlx_lm.convert -q` refits every group's min/max;
  - mlx-community's qat builds use group 64, which spans two q4_0 blocks,
    and keep every MLP at 8-bit.

  Both move the weights off the grid the QAT trained for.
- **A few Linears at 8-bit, chosen by measurement.** Each of the text Linears
  was raised to 8-bit on its own and scored by the KL divergence it removes,
  against the bf16 QAT master weights, per MB it adds. The best
  44 fit in +200 MB. Almost all attention projections (17 V, 15 K, 10 O, 1 Q); one MLP Linear.
- **Everything else follows Google's own split:**
  - embeddings at 6-bit, as the GGUF's Q6_K;
  - the image patch projection and the image and audio projections into
    the text model, which were never QAT-trained, at 8-bit;
  - norms as they are.

## Measured

Wikitext-2 (test), 128 windows of 512 tokens, scored as the model's chat
reply (the chat template, the thinking channel closed). KL divergence and
top-1 agreement are measured to the bf16 QAT master weights, the model this
one reproduces.

| model | size | PPL | KL to QAT master | top-1 agree |
|---|---|---|---|---|
| QAT master weights, bf16 ([google/gemma-4-12B-it-qat-q4_0-unquantized](https://huggingface.co/google/gemma-4-12B-it-qat-q4_0-unquantized)) | — | 22.26 | — | — |
| **this model** | **7.89 GB** | **22.77** | **0.0253** | **93.44**% |
| [mlx-community/gemma-4-12B-it-qat-4bit](https://huggingface.co/mlx-community/gemma-4-12B-it-qat-4bit) | 10.99 GB | 22.82 | 0.0259 | 93.35% |
| [mlx-community/gemma-4-12B-it-4bit](https://huggingface.co/mlx-community/gemma-4-12B-it-4bit) (no QAT) | 6.74 GB | 26.70 | 0.340 | 78.78% |

How to read it:
- **KL and top-1 measure faithfulness;** a lower KL is closer.
- Why as a chat reply: on raw text this checkpoint, Google's own master and
  q4_0 GGUF included, scores in the thousands -- it reads text outside a chat
  turn as its own reasoning. Framed as its reply, it's the model it is.
- PPL compares this build with its master, not with other models: instruct
  models' confidence differs by size.

### Speed (MacBook Air M5, mlx-lm, decode)

| build | tokens/s | peak memory |
|---|---|---|
| **this model** | **14.8** | **8.1 GB** |
| mlx-community/gemma-4-12B-it-qat-4bit | 9.8 | 11.2 GB |

About 1.5× faster (a fanless Mac throttles; a second pair measured 9.2 vs
6.4): mlx-community's build keeps every MLP at 8-bit. 3 GB smaller, it fits
a 16 GB Mac with room for context.

### Checked

- It answers in chat.
- It names the animal in a photo: "There is a red fox in this picture."
- It hears speech: given a spoken "The quick brown fox jumps over the lazy
  dog", it recognizes the pangram (asked to transcribe a well-known
  sentence, it tends to discuss it instead).
- No large weight is left unquantized.
- It loads in LLMTray's runtime ([ipsupport-llc/mlx-lm](https://github.com/ipsupport-llc/mlx-lm)).

## Usage

```bash
pip install "mlx-lm @ git+https://github.com/ipsupport-llc/mlx-lm.git"
```

```python
from mlx_lm import load, generate

model, tokenizer = load("roman220220/gemma-4-12B-it-qat-mlx")
messages = [{"role": "user", "content": "What is the capital of France?"}]
prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
print(generate(model, tokenizer, prompt=prompt, max_tokens=200))
```

Images and audio go through `mlx_lm.multimodal` in the same fork, which is
what LLMTray uses.

## Method / code

[rromenskyi/quant-ternary/gemma4-quant](https://github.com/rromenskyi/quant-ternary/tree/main/gemma4-quant)
has the code and the full lab notes (`docs/FINDINGS.md`, "Gemma 4 QAT in MLX on the q4_0 grid"):
- `qat_mlx_pipeline.sh`: one pipeline for every Gemma 4 size;
- `qat_aligned_convert.py`;
- `qat_sensitivity.py`;
- `qat_eval.py`.

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

Modified from [google/gemma-4-12B-it-qat-q4_0-unquantized](https://huggingface.co/google/gemma-4-12B-it-qat-q4_0-unquantized) (the QAT release of [google/gemma-4-12B-it](https://huggingface.co/google/gemma-4-12B-it)): quantized to MLX. The text decoder's Linears are on Google's q4_0 grid; 44 of them, the embeddings and the image/audio projections are at 6/8-bit. The weights and configuration files in this repo are therefore modified versions of the original, not the original files.
