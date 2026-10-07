---
license: apache-2.0
base_model: microsoft/FrogNano-4B-2609
pipeline_tag: image-text-to-text
library_name: mlx
tags:
  - mlx
  - quantized
  - gptq
  - jang
  - qwen3_5
  - vision
---

<p align="center">
  <img src="llmtray-banner.png" alt="LLMTray" width="100%">
</p>

# FrogNano-4B-2609, GPTQ JANG 8/6/4 (MLX)

> ### ▶ Run it locally in [LLMTray](https://www.ipsupport.us/llmtray/)
> A free, native macOS app for local AI on Apple Silicon — chat, images,
> music, agents and an OpenAI-compatible API. Point it at this model; nothing
> leaves your Mac.
>
> [![Download LLMTray](https://img.shields.io/badge/Download-LLMTray%20for%20Mac-2f7d4f?style=for-the-badge&logo=apple&logoColor=white)](https://github.com/ipsupport-llc/llmtray/releases/latest/download/LLMTray-Full.dmg)
> [![GitHub stars](https://img.shields.io/github/stars/ipsupport-llc/llmtray?style=for-the-badge&logo=github)](https://github.com/ipsupport-llc/llmtray)

[microsoft/FrogNano-4B-2609](https://huggingface.co/microsoft/FrogNano-4B-2609)
— Microsoft's compact repository-level coding agent, Qwen3.5-4B post-trained
with reinforcement learning on ~1,500 synthetic software-engineering tasks
(SWE-bench Verified 61.5 % Avg@3 in Microsoft's Leaf harness) — quantized for
MLX with **GPTQ and a per-component bit recipe**: **3.2 GB instead of 9.3 GB,
+1.5 % perplexity**. The vision tower inherited from Qwen3.5-4B is kept, at
8 bits.

## The recipe

Bits by component, the same in every layer, GPTQ-calibrated (Hessian error
compensation, MLX's exact affine grid, so `mlx_lm.convert` reproduces the
calibrated codes instead of re-rounding):

| Component | Bits (group 64) | Parameters |
|---|---|---|
| Full attention (q, k, v, o; 8 of 32 layers) | 8 | 0.29 B |
| Gated DeltaNet linear attention (in_proj_qkv, in_proj_z, out_proj; 24 layers) | 6 | 1.0 B |
| MLP (gate, up, down; 32 layers) | 4 | 2.26 B |
| Delta-rule gates (in_proj_a, in_proj_b) | bf16 | 0.005 B |
| Embeddings (tied with the output head; round-to-nearest) | 8 | 0.64 B |
| Vision tower (position embedding bf16) | 8 | 0.33 B |
| MTP head (fc, one attention + MLP layer; `model-mtp.safetensors`) | 4 | 0.12 B |

Calibration: 64 × 512 tokens of wikitext-2 train, layer by layer (each layer
on the outputs of the already-quantized ones), on one L40S.

## Measurements

Perplexity on wikitext-2 test, 40 × 512 tokens:

| | Size | PPL | vs bf16 |
|---|---|---|---|
| bf16 (HF transformers) | 9.3 GB | 12.366 | — |
| **this model, 8/6/4 (MLX)** | **3.2 GB** | **12.551** | **+1.5 %** |
| same weights in HF transformers (decoder on-grid, embeddings bf16) | — | 12.521 | +1.3 % |
| 8/6/3 (MLP at 3 bits; not released) | ≈2.9 GB | 13.377 (HF) | +8.2 % |

The MLP holds most of this model's weights, so 3 bits there cost too much;
at 4 bits the whole model loses 1.5 %.

Speed (MLX, MacBook Air M5, 26 GB): **38 tokens/s** decoding, 3.9 GB peak
memory (another model was loaded in LLMTray at the time).

## Speculative decoding (MTP head)

Qwen3.5 ships a multi-token-prediction head; this repo keeps it, GPTQ
4-bit like the rest, in its own file `model-mtp.safetensors` (68 MB), so a
copy downloaded before it was added can get just that file. With the
[ipsupport-llc/mlx-lm](https://github.com/ipsupport-llc/mlx-lm) fork the
head drafts tokens and the model checks them in one pass: the same output
as without it (each token is the model's own sample, at any temperature),
faster. Stock `mlx-lm` drops the head on load.

| | |
|---|---|
| First draft accepted (greedy; wikitext-2 test / Python stdlib) | 88.5 % / 91.3 % |
| Decoding, MacBook Air M5 (code / English / Russian, greedy) | ×1.37–1.54 / ×1.21 / ×1.17 |
| Decoding at temperature 1.0 | ×1.03–1.32 |
| Memory at a 16K-token prompt | +0.1 GB |

The fork picks 0–3 drafts per step by what is fastest at the moment, so
the head never makes decoding slower. LLMTray uses it when "Speculative
decoding (MTP)" is on in the model's profile.

These are language-modelling numbers. The coding-agent benchmarks above are
Microsoft's, for the bf16 model in their harness; they weren't re-run on
this quantization.

## Vision

Microsoft didn't post-train or evaluate the image/video components and
doesn't support them for FrogNano; they are Qwen3.5-4B's. They are kept here,
and they work as Qwen3.5-4B's do: on a test image the MLX vision tower's
features match HF transformers' at cosine 0.991 (mean over tokens; 0.994 with
the tower in bf16), and it describes a test image the way the HF bf16 model
does. The tower is 8-bit except its position embedding, which stays bf16:
Qwen3-VL interpolates it in the weight's dtype, and quantized that dtype is
an integer (the tower's features fell to cosine 0.76).

Image input needs the [ipsupport-llc/mlx-lm](https://github.com/ipsupport-llc/mlx-lm)
fork's Qwen3.5 vision support (Qwen3-VL vision tower, interleaved mRoPE,
image preprocessing); stock `mlx-lm` loads this model text-only.

## Usage

Microsoft's settings: temperature **0.6**, repetition penalty 1.0, up to
8,192 tokens per turn, context ~131K. It reasons before answering
(`<think>…</think>`).

```bash
pip install mlx-lm
mlx_lm.generate --model roman220220/FrogNano-4B-2609-gptq-mlx-jang \
  --prompt "Write a Python function that parses ISO-8601 dates." --temp 0.6 --max-tokens 4096
```

As an OpenAI-compatible server (image input with the fork):

```bash
pip install "git+https://github.com/ipsupport-llc/mlx-lm.git"
mlx_lm.server --model roman220220/FrogNano-4B-2609-gptq-mlx-jang --port 8080
```

Microsoft's intended use is a sandboxed, human-reviewed coding agent (their
Leaf harness); read the
[base model card](https://huggingface.co/microsoft/FrogNano-4B-2609) for its
scope and limitations, which apply here unchanged.

## Method and code

Pipeline, scripts and measurements:
[rromenskyi/quant-ternary](https://github.com/rromenskyi/quant-ternary),
`qwen35-quant/` (formerly `frognano-quant/`; `poc/gptq_qwen35.py`, `poc/convert_mlx.py`,
`poc/check_mlx.py`, `docs/FINDINGS.md`).

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

Licensed under the **Apache License 2.0**, as stated in the base model's
card (FrogNano is derived from Qwen/Qwen3.5-4B, Apache 2.0; the base
repository's metadata says MIT, which Apache 2.0 also satisfies) — see
[`LICENSE`](LICENSE).

Modified from [microsoft/FrogNano-4B-2609](https://huggingface.co/microsoft/FrogNano-4B-2609):
quantized with GPTQ to 8-bit attention, 6-bit linear attention and 4-bit MLP
weights (8-bit embeddings and vision tower), the MTP head to 4-bit, and
converted to MLX. The weights and configuration files in this repo are
therefore modified versions of the original, not the original files.
