---
license: apache-2.0
base_model: google/gemma-4-31B-it-qat-q4_0-unquantized
pipeline_tag: image-text-to-text
tags:
  - mlx
  - gemma4
  - quantized
  - qat
  - multimodal
  - vision

---

<p align="center">
  <img src="llmtray-banner.png" alt="LLMTray" width="100%">
</p>

# Gemma 4 31B — Google's QAT on its exact q4_0 grid, MLX (text + vision)

> ### ▶ Run it locally in [LLMTray](https://www.ipsupport.us/llmtray/)
> A free, native macOS app for local AI on Apple Silicon — chat, images,
> music, agents and an OpenAI-compatible API. Point it at this model; nothing
> leaves your Mac.
>
> [![Download LLMTray](https://img.shields.io/badge/Download-LLMTray%20for%20Mac-2f7d4f?style=for-the-badge&logo=apple&logoColor=white)](https://github.com/ipsupport-llc/llmtray/releases/latest/download/LLMTray-Full.dmg)
> [![GitHub stars](https://img.shields.io/github/stars/ipsupport-llc/llmtray?style=for-the-badge&logo=github)](https://github.com/ipsupport-llc/llmtray)

Google trained Gemma 4 31B with quantization-aware training (QAT) for
llama.cpp's **q4_0**. This is that model in MLX, **on the very grid the QAT
trained for**. The text decoder's 4-bit codes are the very ones in Google's own
[q4_0 GGUF](https://huggingface.co/google/gemma-4-31B-it-qat-q4_0-gguf), block for block; only their scales are rounded
from fp16 to bf16 (MLX's scale type).
On top of that, the 35 Linears that lose the most at 4-bit are kept
at 8-bit, which costs 400 MB. The vision tower is
included.

About 20.6 GB of weights: for a 48 GB Mac or larger (on a 32 GB Mac the GPU limit leaves no room for the KV cache). The vision tower's MLP down projections (width 4304, not a multiple of MLX's smallest group of 32) stay in bf16, about 270 MB.

## How it's made

- **The QAT grid, exactly.** Google's GGUF is q4_0 of the
  `-qat-q4_0-unquantized` master weights: every block's scale and every code
  matches. MLX's affine 4-bit with group 32 stores `scale·q + bias`; with
  `scale = d` and `bias = -8d` that is q4_0's grid itself. 375 Linears are
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
  35 fit in +400 MB. On the 31B they are almost all attention K/V projections, and the whole budget curve stays within KL 0.019-0.021: Google's q4_0 grid is already close to the master.
- **Everything else follows Google's own split:**
  - embeddings at 6-bit, as the GGUF's Q6_K;
  - the vision tower, which was never QAT-trained, at 8-bit;
  - norms as they are.

## Measured

Wikitext-2 (test), 128 windows of 512 tokens, scored as the model's chat
reply (the chat template, the thinking channel closed). KL divergence and
top-1 agreement are measured to the bf16 QAT master weights, the model this
one reproduces.

| model | size | PPL | KL to QAT master | top-1 agree |
|---|---|---|---|---|
| QAT master weights, bf16 ([google/gemma-4-31B-it-qat-q4_0-unquantized](https://huggingface.co/google/gemma-4-31B-it-qat-q4_0-unquantized)) | — | 26.94 | — | — |
| **this model** | **20.6 GB** | **27.82** | **0.0206** | **94.93**% |

How to read it:
- **KL and top-1 measure faithfulness;** a lower KL is closer.
- Why as a chat reply: on raw text this checkpoint, Google's own master and
  q4_0 GGUF included, scores in the thousands -- it reads text outside a chat
  turn as its own reasoning. Framed as its reply, it's the model it is.
- PPL compares this build with its master, not with other models: instruct
  models' confidence differs by size.

### Checked

- It answers in chat.
- It names the animal in a photo.
- No large weight is left unquantized but the vision tower's MLP down projections (above).
- It loads in LLMTray's runtime ([ipsupport-llc/mlx-lm](https://github.com/ipsupport-llc/mlx-lm)).

## Usage

```bash
pip install "mlx-lm @ git+https://github.com/ipsupport-llc/mlx-lm.git"
```

```python
from mlx_lm import load, generate

model, tokenizer = load("roman220220/gemma-4-31B-it-qat-mlx")
messages = [{"role": "user", "content": "What is the capital of France?"}]
prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
print(generate(model, tokenizer, prompt=prompt, max_tokens=200))
```

Images go through `mlx_lm.multimodal` in the same fork, which is
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

Runs in LLMTray as a chat model, with vision.

## License

Licensed under the **Apache License 2.0**, the same license as the base model — see [`LICENSE`](LICENSE).

Modified from [google/gemma-4-31B-it-qat-q4_0-unquantized](https://huggingface.co/google/gemma-4-31B-it-qat-q4_0-unquantized) (the QAT release of [google/gemma-4-31B-it](https://huggingface.co/google/gemma-4-31B-it)): quantized to MLX. The text decoder's Linears are on Google's q4_0 grid; 35 of them, the embeddings and the vision tower are at 6/8-bit. The weights and configuration files in this repo are therefore modified versions of the original, not the original files.
