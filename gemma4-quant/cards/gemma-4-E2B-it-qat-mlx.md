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
---

<p align="center">
  <img src="llmtray-banner.png" alt="LLMTray" width="100%">
</p>

# Gemma 4 E2B — Google's QAT on its exact q4_0 grid, MLX (text + vision + audio)

> ### ▶ Run it locally in [LLMTray](https://www.ipsupport.us/llmtray/)
> A free, native macOS app for local AI on Apple Silicon — chat, images,
> music, agents and an OpenAI-compatible API. Point it at this model; nothing
> leaves your Mac.
>
> [![Download LLMTray](https://img.shields.io/badge/Download-LLMTray%20for%20Mac-2f7d4f?style=for-the-badge&logo=apple&logoColor=white)](https://github.com/ipsupport-llc/llmtray/releases/latest/download/LLMTray-Full.dmg)
> [![GitHub stars](https://img.shields.io/github/stars/ipsupport-llc/llmtray?style=for-the-badge&logo=github)](https://github.com/ipsupport-llc/llmtray)

Google trained Gemma 4 E2B with quantization-aware training (QAT) for
llama.cpp's **q4_0**. This is that model in MLX, **on the very grid the QAT
trained for**. The text decoder's 4-bit codes are the very ones in Google's own
[q4_0 GGUF](https://huggingface.co/google/gemma-4-E2B-it-qat-q4_0-gguf), block for block; only their scales are rounded
from fp16 to bf16 (MLX's scale type).
On top of that, the 126 Linears that lose the most at 4-bit are kept
at 8-bit, which costs 100 MB. The vision and audio towers are
included.

## How it's made

- **The QAT grid, exactly.** Google's GGUF is q4_0 of the
  `-qat-q4_0-unquantized` master weights: every block's scale and every code
  matches. MLX's affine 4-bit with group 32 stores `scale·q + bias`; with
  `scale = d` and `bias = -8d` that is q4_0's grid itself. 149 Linears are
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
  126 fit in +100 MB. Mostly the small per-layer input gates and projections (63) and attention projections (54); only 9 MLP Linears.
- **Everything else follows Google's own split:**
  - embeddings at 6-bit, as the GGUF's Q6_K;
  - vision and audio towers, which were never QAT-trained, at 8-bit;
  - norms as they are.

## Measured

Raw-text perplexity on wikitext-2 (test), 128 windows of 512 tokens, BOS at
the start of each. KL divergence and top-1 agreement are measured to the bf16
QAT master weights, the model this one reproduces.

| model | size | PPL | KL to QAT master | top-1 agree |
|---|---|---|---|---|
| QAT master weights, bf16 ([google/gemma-4-E2B-it-qat-q4_0-unquantized](https://huggingface.co/google/gemma-4-E2B-it-qat-q4_0-unquantized)) | — | 66.1 | — | — |
| **this model** | **4.04 GB** | **64.5** | **0.030** | **91.8**% |
| [mlx-community/gemma-4-E2B-it-qat-4bit](https://huggingface.co/mlx-community/gemma-4-E2B-it-qat-4bit) | 4.33 GB | 66.2 | 0.067 | 87.6% |
| [mlx-community/gemma-4-e2b-it-4bit](https://huggingface.co/mlx-community/gemma-4-e2b-it-4bit) | 3.6 GB | 239.7 | 0.853 | 66.3% |

How to read it:
- **KL and top-1 measure faithfulness;** a lower KL is closer.
- PPL a little *below* the master's is within the noise of a raw-text test.
- It also reflects how QAT works: the network was trained through the 4-bit
  weights, so the q4_0 model is the one that was trained, and the bf16 master
  is its shadow copy.

### Speed (MacBook Air M5, mlx-lm, decode)

| build | tokens/s |
|---|---|
| **this model** | **67.5** |
| mlx-community/gemma-4-E2B-it-qat-4bit | 54.7 |
| mlx-community/gemma-4-e2b-it-4bit (no QAT) | 83.7 |

Faster than mlx-community's QAT build, which keeps every MLP at 8-bit. Slower than the plain 4-bit build, the cost of group 32 and the raised Linears, at a quarter of its perplexity.

### Checked

- It answers in chat.
- It names the animal in a photo.
- It transcribes speech through the audio tower.
- No large weight is left unquantized.
- It loads in LLMTray's runtime ([ipsupport-llc/mlx-lm](https://github.com/ipsupport-llc/mlx-lm)).

## Usage

```bash
pip install "mlx-lm @ git+https://github.com/ipsupport-llc/mlx-lm.git"
```

```python
from mlx_lm import load, generate

model, tokenizer = load("roman220220/gemma-4-E2B-it-qat-mlx")
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

Modified from [google/gemma-4-E2B-it-qat-q4_0-unquantized](https://huggingface.co/google/gemma-4-E2B-it-qat-q4_0-unquantized) (the QAT release of [google/gemma-4-E2B-it](https://huggingface.co/google/gemma-4-E2B-it)): quantized to MLX. The text decoder's Linears are on Google's q4_0 grid; 126 of them, the embeddings and the vision/audio towers are at 6/8-bit. The weights and configuration files in this repo are therefore modified versions of the original, not the original files.
