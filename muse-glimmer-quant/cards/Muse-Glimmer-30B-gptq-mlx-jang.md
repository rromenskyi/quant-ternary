---
license: apache-2.0
base_model: meta-models/Muse-Glimmer-30B
pipeline_tag: text-generation
library_name: mlx
tags:
  - mlx
  - quantized
  - gptq
  - jang
  - muse-glimmer
  - meta
---

<p align="center">
  <img src="llmtray-banner.png" alt="LLMTray" width="100%">
</p>

# Muse Glimmer 30B, GPTQ JANG 8/4 (MLX, text)

> ### ▶ Run it locally in [LLMTray](https://www.ipsupport.us/llmtray/)
> A free, native macOS app for local AI on Apple Silicon — chat, images,
> music, agents and an OpenAI-compatible API. Point it at this model; nothing
> leaves your Mac.
>
> [![Download LLMTray](https://img.shields.io/badge/Download-LLMTray%20for%20Mac-2f7d4f?style=for-the-badge&logo=apple&logoColor=white)](https://github.com/ipsupport-llc/llmtray/releases/latest/download/LLMTray-Full.dmg)
> [![GitHub stars](https://img.shields.io/github/stars/ipsupport-llc/llmtray?style=for-the-badge&logo=github)](https://github.com/ipsupport-llc/llmtray)

[meta-models/Muse-Glimmer-30B](https://huggingface.co/meta-models/Muse-Glimmer-30B)
— Meta Superintelligence Labs' 30B dense agentic model (Apache 2.0,
released 2026-08-10) — quantized for MLX with **GPTQ and a per-component
bit recipe**: **19.23 GB instead of 55.71 GB bf16, +1.8% perplexity on
text, +2.2% on code — smaller and more accurate than Meta's own reference
4-bit MLX build** (`mlx-community/Muse-Glimmer-30B-4bit`, 19.42 GB,
+3.2% / +3.8%).

This is the **text tower only**. Glimmer is natively multimodal, but
upstream `mlx-lm` (Apple) ports only the language model — see "Honest
notes" below.

## The recipe

Unlike this account's hybrid-decoder releases (LFM2, Nemotron), Glimmer is
a plain dense transformer: full attention (with an extra sigmoid output
gate) in every one of 52 layers, alternating sliding-window / full /
NoPE by a fixed `[local, local, local, global]` pattern. No conv layers,
no MoE, nothing to split a recipe by except attention vs. feed-forward:

| Component | Bits (group 64) | Parameters |
|---|---|---|
| Attention (q, k, v, o, gate; all 52 layers) | 8 | 4.4 B |
| Feed-forward (gate, up, down; all 52 layers) | 4 | 20.7 B |
| Embeddings (`embed_tokens`, round-to-nearest) | 8 | 1.3 B |
| Output head (`lm_head`, **not** tied to the embeddings; round-to-nearest) | 8 | 1.3 B |
| Norms (4 "centered" RMSNorms per layer), scaleless QK-norm | bf16 | < 0.02 B |

~29.6 B parameters in all. GPTQ-calibrated (Hessian error compensation,
layer by layer on the outputs of the already-quantized layers) on MLX's
exact affine grid, group 64: the integer codes GPTQ chose are written into
the MLX files as they are, not re-rounded by `mlx_lm.convert`. Calibration:
128 x 512 tokens, wikitext-2 train mixed with Python source code, on one
RTX PRO 6000 (96 GB) — the whole calibration run took under 10 minutes
(layer-by-layer: only one layer is ever on the GPU at a time, so this
fits a modest single card despite the 30B size).

## Measurements

Perplexity on wikitext-2 test and held-out Python code, 40 x 512 tokens
each, in MLX, so every row is measured the same way and the ratios compare
like with like. Glimmer is a reasoning/agent model that always opens a
`<|message|>` channel before answering, so raw perplexity on plain
wikitext is not directly comparable across model families — what matters
is the change against this model's own bf16.

| | Size | PPL text | vs bf16 | PPL code | vs bf16 |
|---|---|---|---|---|---|
| bf16 (original) | 55.71 GB | 8.074 | — | 2.641 | — |
| Meta/mlx-community reference 4-bit (uniform) | 19.42 GB | 8.330 | +3.17% | 2.742 | +3.83% |
| **this model, 8/4 JANG** | **19.23 GB** | **8.217** | **+1.77%** | **2.699** | **+2.18%** |

Smaller than the reference 4-bit build (not re-deriving the grid from
already-quantized-and-corrected weights adds up) and roughly half its
quality loss, from spending the saved bits on attention instead of
distributing them uniformly.

## Usage

Needs upstream `mlx-lm` (Apple's, not this account's fork) — Muse Glimmer
support landed there, not in `ipsupport-llc/mlx-lm` (which this account
otherwise runs for LLMTray):

```bash
pip install "mlx-lm==0.32.2"
mlx_lm.generate --model roman220220/Muse-Glimmer-30B-gptq-mlx-jang \
  --prompt "Explain what a hash map is in three sentences." --max-tokens 1024
```

As an OpenAI-compatible server:

```bash
mlx_lm.server --model roman220220/Muse-Glimmer-30B-gptq-mlx-jang --port 8080
```

Read the [base model card](https://huggingface.co/meta-models/Muse-Glimmer-30B)
for the model's scope, agentic/tool-calling format and limitations, which
apply here unchanged.

## Honest notes

- **Text only.** Glimmer ships with a ~1.8B ViT-G/14 perception encoder;
  `mlx-lm`'s port doesn't include it (its own source comment: "this
  implements the language model only"), so there is nothing to quantize
  or ship for vision yet. Porting the vision tower into
  `ipsupport-llc/mlx-lm` (the way this account did for LFM2.5-VL-3B) is a
  planned follow-up, from the same checkpoint — no re-calibration of the
  text decoder needed when it lands.
- **A patched `mlx-lm` was needed to convert this at all.** Its
  `muse_glimmer.py` `sanitize()` drops vision weights by checking an
  un-nested `"vision_tower"` / `"vision_adapter"` / `"vision_projection"`
  prefix, but this checkpoint nests them under `"model."` — the check
  never matches, and `mlx_lm.load()` / `.convert()` both fail on
  "unexpected" vision weights. The pipeline patches the installed
  package's one `startswith(...)` call to also match the `"model."`-nested
  names (see `poc/muse_glimmer_mlx_pipeline.sh`'s `patch_mlx_lm` step); it
  isn't upstreamed yet.
- This is GPTQ on MLX's grid, not the GGUF/imatrix style this account also
  ships for some models — the two aren't the same correction and don't
  transfer between formats (see `docs/GPTQ_EXACT_CODES.md` in
  `qwen35-quant/`).

## Method and code

Pipeline, scripts and measurements:
[rromenskyi/quant-ternary](https://github.com/rromenskyi/quant-ternary),
`muse-glimmer-quant/` (`poc/muse_glimmer_mlx_pipeline.sh`,
`poc/gptq_muse_glimmer.py`, `poc/convert_muse_glimmer.py`).

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
model — see [`LICENSE`](LICENSE).

Modified from [meta-models/Muse-Glimmer-30B](https://huggingface.co/meta-models/Muse-Glimmer-30B):
the decoder quantized with GPTQ to 8-bit attention and 4-bit feed-forward
weights, the embeddings and output head to 8-bit, and converted to MLX.
The vision tower is not included. The weights and configuration files in
this repo are therefore modified versions of the original, not the
original files.
