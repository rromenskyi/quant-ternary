---
license: other
license_name: openmdw-1.1
license_link: LICENSE
base_model: nvidia/NVIDIA-Nemotron-Labs-Dory1.0-4B-BF16
pipeline_tag: text-generation
tags:
  - mlx
  - quantized
  - dory
  - gptq
  - jang
---

<p align="center">
  <img src="llmtray-banner.png" alt="LLMTray" width="100%">
</p>

# Dory 1.0 4B — MLX, GPTQ JANG

> ### ▶ Run it locally in [LLMTray](https://www.ipsupport.us/llmtray/)
> A free, native macOS app for local AI on Apple Silicon — chat, images,
> music, agents and an OpenAI-compatible API. Point it at this model; nothing
> leaves your Mac.
>
> [![Download LLMTray](https://img.shields.io/badge/Download-LLMTray%20for%20Mac-2f7d4f?style=for-the-badge&logo=apple&logoColor=white)](https://github.com/ipsupport-llc/llmtray/releases/latest/download/LLMTray-Full.dmg)
> [![GitHub stars](https://img.shields.io/github/stars/ipsupport-llc/llmtray?style=for-the-badge&logo=github)](https://github.com/ipsupport-llc/llmtray)

NVIDIA's [Dory 1.0 4B](https://huggingface.co/nvidia/NVIDIA-Nemotron-Labs-Dory1.0-4B-BF16)
is a looped, nGPT-style dense model: 16 input layers feed a shared 40-layer
recurrent block, run twice per forward pass, into 16 output layers (72 raw
layers total, alternating pure-attention and pure-MLP, never both in one
layer). This is a GPTQ quantization to MLX, 8-bit attention / 6-bit MLP
("JANG": per-component bits, uniform across layers), with GPTQ's own
integer codes written back exactly rather than left to MLX's own
re-derivation from the rounded weights.

Dory has no `transformers` modeling file (only vLLM and llama.cpp implement
it); this quantization runs on an original PyTorch reference implementation
we wrote against our own MLX port
([`mlx_lm.models.dory`](https://github.com/ipsupport-llc/mlx-lm), merged
upstream work from [ml-explore/mlx-lm#1988](https://github.com/ml-explore/mlx-lm/pull/1988)),
cross-checked numerically against it before calibration. The recurrent
block's weights are shared across both loops of a forward pass, so the
usual "layer *i* on layer *i-1*'s quantized output" GPTQ chain doesn't
apply there directly — see [FINDINGS.md](https://github.com/ipsupport-llc/quant-ternary/blob/main/dory-quant/docs/FINDINGS.md)
for how this build handles it.

## Measurements

| | bf16 | this build |
|---|---|---|
| wikitext-2 perplexity | 14.03 | 14.09 (+0.41 %) |
| held-out Python code perplexity | 3.739 | 3.748 (+0.24 %) |
| size | 7.95 GB | 3.5 GB |

Generation sample (greedy, "The capital of France is"): *"Paris. The
capital of Germany is Berlin. The capital of Italy is Rome. The capital of
Spain is Madrid. The capital of the United Kingdom is"*.

Recipe: `attn=8,mlp=6`, group size 64. Embedding and LM head at 8 bits
(RTN). The nGPT scale vectors (`sqk`, `s_gate`, `suv_gate`, `suv_up`,
`attn_alpha`, `mlp_alpha`, `logit_scale`) are not quantized — they're tiny
trainable vectors, not weight matrices, same as every other model's
gate/router scalars in this pipeline.

## The IPSupport local-AI stack

Local-first AI tools for macOS by [IPSupport](https://www.ipsupport.us) — nothing
leaves your Mac.

- **[LLMTray](https://www.ipsupport.us/llmtray/)** — your local AI
  workstation for macOS: chat with local LLMs, generate and edit images, make
  music, run agents, and serve an OpenAI-compatible API. Downloads models from
  Hugging Face in-app, with per-model profiles.
- **[IPSupport Code](https://ipsupport-llc.github.io/ipsupport-code/)** — your
  AI coding agent for real repositories: analyze, fix, test, report.

Runs in LLMTray as a local chat LLM.

## Disclaimer

This repository contains a quantized conversion of a third-party model,
provided **"as is", without warranty of any kind**, express or implied.
IPSupport LLC did not train the model and is **not responsible for its
content, outputs or behavior, or for any damage, loss or liability** arising
from its use. You are responsible for evaluating the model and for how you
use it, including compliance with applicable laws and with the base model's
license and terms.

## License

Licensed under the **OpenMDW License Agreement, version 1.1**, the same
license as the base model — see [`LICENSE`](LICENSE).

Modified from [nvidia/NVIDIA-Nemotron-Labs-Dory1.0-4B-BF16](https://huggingface.co/nvidia/NVIDIA-Nemotron-Labs-Dory1.0-4B-BF16):
quantized to 8-bit attention / 6-bit MLP (GPTQ, MLX affine grid, exact
codes) and converted to MLX. The weights and configuration files in this
repo are therefore modified versions of the original, not the original
files.
