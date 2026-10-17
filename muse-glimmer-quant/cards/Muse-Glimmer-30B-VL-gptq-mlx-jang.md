---
license: apache-2.0
base_model: meta-models/Muse-Glimmer-30B
pipeline_tag: image-text-to-text
library_name: mlx
tags:
  - mlx
  - quantized
  - gptq
  - jang
  - muse-glimmer
  - meta
  - vision
---

<p align="center">
  <img src="llmtray-banner.png" alt="LLMTray" width="100%">
</p>

# Muse Glimmer 30B, GPTQ JANG 8/4 + vision (MLX)

> ### ▶ Run it locally in [LLMTray](https://www.ipsupport.us/llmtray/)
> A free, native macOS app for local AI on Apple Silicon — chat, images,
> music, agents and an OpenAI-compatible API. Point it at this model; nothing
> leaves your Mac.
>
> [![Download LLMTray](https://img.shields.io/badge/Download-LLMTray%20for%20Mac-2f7d4f?style=for-the-badge&logo=apple&logoColor=white)](https://github.com/ipsupport-llc/llmtray/releases/latest/download/LLMTray-Full.dmg)
> [![GitHub stars](https://img.shields.io/github/stars/ipsupport-llc/llmtray?style=for-the-badge&logo=github)](https://github.com/ipsupport-llc/llmtray)

The text-plus-vision sibling of
[roman220220/Muse-Glimmer-30B-gptq-mlx-jang](https://huggingface.co/roman220220/Muse-Glimmer-30B-gptq-mlx-jang):
the same GPTQ JANG 8/4 text decoder, with
[meta-models/Muse-Glimmer-30B](https://huggingface.co/meta-models/Muse-Glimmer-30B)'s
perception encoder (a windowed-attention ViT, Qwen2.5-VL style) added on top,
RTN-quantized to 8-bit. **No text recalibration was needed** — the already
GPTQ-calibrated decoder is reused as-is; only the vision tower, adapter and
projection are new.

This requires `mlx-lm` support for Glimmer's vision tower, which didn't
exist anywhere (upstream's port is text-only by its own admission: "Glimmer
is multimodal; this implements the language model only") until this
release's pipeline added it to
[ipsupport-llc/mlx-lm](https://github.com/ipsupport-llc/mlx-lm) — see
"Honest notes".

## The recipe

| Component | Bits | Notes |
|---|---|---|
| Text decoder (attention, feed-forward, embeddings, output head) | 8/4 GPTQ | identical to the text-only release — see its card |
| Vision tower (50-layer windowed/full attention ViT) | 8-bit RTN, group 64 | `patch_embedding` kept bf16 (its input dim, 1176, isn't a multiple of 64) |
| Vision adapter (2-layer GELU projector) | 8-bit RTN | |
| Vision-to-text projection | 8-bit RTN | |

## Verification

No ground truth exists for "is this vision port correct" beyond comparing
against transformers' own reference implementation, so that's what was
done — on a real image, loading only the vision tensors via
`safetensors.safe_open` (never materializing the 30B text decoder):

- Patch embedder (position-embedding interpolation): **cosine 1.000000** vs
  HF fp32.
- Window index, position ids, pixel-shuffle index: **exact match** against
  `transformers.vision_utils`'s reference functions.
- Full vision tower output (50 layers): **389/391 merged tokens ≥ 0.99
  cosine, mean 0.9996**, worst 0.951 — traced layer by layer; the gap
  accumulates gradually from around layer 30 onward at one specific token,
  consistent with ordinary fp32 rounding-error accumulation between
  MLX/CUDA and PyTorch/CPU matmuls (confirmed unaffected by
  `MLX_ENABLE_TF32`), not a logic bug.
- End to end: asked a real photo "what animals are in this picture" —
  correctly answered cats.

## Usage

Needs the `ipsupport-llc/mlx-lm` fork (vision support for Glimmer isn't
upstream yet):

```bash
pip install git+https://github.com/ipsupport-llc/mlx-lm.git@main
```

```python
from mlx_lm import load
model, tokenizer = load("roman220220/Muse-Glimmer-30B-VL-gptq-mlx-jang")
# image_features / embed_with_images(input_ids, pixel_values, grid_thw)
# build pixel_values / grid_thw with transformers' MuseGlimmerImageProcessor
```

Read the [base model card](https://huggingface.co/meta-models/Muse-Glimmer-30B)
for the model's scope, agentic/tool-calling format and limitations.

## Honest notes

- **Needs this account's mlx-lm fork, not upstream.** Upstream `mlx-lm`
  has no vision module for `muse_glimmer` at all; loading this repo's
  weights with vanilla `mlx-lm` will fail on "unexpected" vision keys. A
  PR against `ipsupport-llc/mlx-lm` adds it; upstreaming to
  `ml-explore/mlx-lm` is a possible follow-up, not done yet.
- **Vision tower is RTN, not GPTQ.** Only the text decoder got this
  account's full GPTQ Hessian calibration; the vision tower is simple
  round-to-nearest, same approach as this account's other VL releases
  (LFM2.5-VL-3B) — calibrating a vision encoder's Hessian needs an image
  corpus and hasn't been built for this architecture.
- **No server-side image upload wiring yet.** The model loads and answers
  image questions when given `pixel_values`/`grid_thw` directly (as this
  card's verification did); LLMTray/`mlx_lm.server`'s multimodal request
  handling for `muse_glimmer` specifically (the `multimodal.py` dispatch
  other VL models use) is a follow-up, not part of this release.

## Method and code

Pipeline: [rromenskyi/quant-ternary](https://github.com/rromenskyi/quant-ternary),
`muse-glimmer-quant/poc/build_vl.py`, `verify_vision.py`. Vision-tower
code: [ipsupport-llc/mlx-lm#45](https://github.com/ipsupport-llc/mlx-lm/pull/45).

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
the decoder quantized with GPTQ (8-bit attention, 4-bit feed-forward,
8-bit embeddings/output head), the vision tower, adapter and projection
quantized with 8-bit round-to-nearest, converted to MLX. The weights and
configuration files in this repo are therefore modified versions of the
original, not the original files.
