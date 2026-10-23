---
license: other
license_name: openmdw-1.1
license_link: LICENSE
base_model: nvidia/NVIDIA-Nemotron-3.5-Bolt-E4B-BF16
pipeline_tag: image-text-to-text
tags:
  - mlx
  - gemma4
  - gptq
  - quantized
  - multimodal
  - vision
  - audio
  - jang
  - nvidia
  - nemotron
---

<p align="center">
  <img src="llmtray-banner.png" alt="LLMTray" width="100%">
</p>

# Nemotron-3.5-Bolt-E4B — GPTQ, JANG-mixed precision, text + vision + audio, MLX

> ### ▶ Run it locally in [LLMTray](https://www.ipsupport.us/llmtray/)
> A free, native macOS app for local AI on Apple Silicon — chat, images,
> music, agents and an OpenAI-compatible API. Point it at this model; nothing
> leaves your Mac.
>
> [![Download LLMTray](https://img.shields.io/badge/Download-LLMTray%20for%20Mac-2f7d4f?style=for-the-badge&logo=apple&logoColor=white)](https://github.com/ipsupport-llc/llmtray/releases/latest/download/LLMTray-Full.dmg)
> [![GitHub stars](https://img.shields.io/github/stars/ipsupport-llc/llmtray?style=for-the-badge&logo=github)](https://github.com/ipsupport-llc/llmtray)

A JANG-style mixed-precision MLX quantization of NVIDIA's
[Nemotron-3.5-Bolt-E4B](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Bolt-E4B-BF16)
— a small language model (8B total / 4B effective parameters) tuned by
NVIDIA for fast local voice/device assistants: low memory, low
time-to-first-token, strong multi-turn tool use and instruction
following. NVIDIA's own release targets a 4GB-class deployment (their
official GGUF is named accordingly); this is the MLX sibling for
LLMTray on Apple Silicon.

**Text, vision *and* audio — not just text.** NVIDIA's own card lists
this model as text-only in its summary table, and every other
quantization of it found publicly (NVIDIA's own GGUF+imatrix release,
several independent MLX conversions) only covers the text decoder. The
underlying checkpoint still carries a full vision tower and audio
encoder, inherited from its base architecture and never stripped out
by NVIDIA's post-training. This release GPTQ-quantizes and verifies
**all three** — as far as checked, the first public build of this
model where image and speech input actually work.

- **6.3 GB** (down from ~16 GB bf16).
- Verified end to end on the real quantized checkpoint (see "Does it
  work?" below) — not just that it loads, that it produces correct
  output for text, a real photo, and real speech.

## Recipe: JANG-mixed, by role

| Role | Bits | Notes |
|---|---|---|
| Attention (`q/k/v/o_proj`, all three towers) | 8-bit GPTQ | Small, error-sensitive |
| Feed-forward (`gate/up/down_proj`, all three towers) | 4-bit GPTQ | Dense model — no MoE here |
| Embeddings / per-layer projections | 8-bit RTN | No shared-input matmul for GPTQ to correct |
| `vision_tower.patch_embedder.input_proj` | **float, untouched** | Quantizing it corrupts the pixel-to-embedding cast itself (measured: no discernible subjects in a test photo) |

Calibrated with `--group-size 64` on real data: text prompts for the
language model, real photographs for the vision tower, real speech for
the audio tower.

## Architecture notes

- Base architecture is `Gemma4ForConditionalGeneration` (NVIDIA
  post-trained `google/gemma-4-E4B-it`) — 42 dense decoder layers,
  hidden size 2560, hybrid sliding/full attention, per-layer input
  embeddings. No MoE.
- **Checkpoint layout isn't the usual single-file one.** NVIDIA's
  release ships `model-00001-of-00001.safetensors` + an index.json,
  plus a *separate* `model-frozen-shared-kv.safetensors` holding the
  full-attention layers' `k_proj`/`v_proj`/`k_norm` weights that later
  sliding-attention layers reuse — real tensors, not duplicates. The
  quantization pipeline (originally written for the vanilla single-file
  checkpoint) needed a fix to read both files; see the pipeline repo's
  commit history if reusing this on another third-party checkpoint of
  the same architecture family.
- Vision tower: 16 layers, hidden size 768, patch size 16 (own
  `gemma4_vision` config, not the 26B sibling's).
- Audio tower: 12 layers, hidden size 1024, Conformer-style
  convolution + attention.

## Quality: wikitext-2 PPL and KL vs bf16

64 windows of 512 tokens, this account's standard `qat_eval.py` harness
(same token ids as the bf16 reference's tokenizer, reference logits
computed once per window):

| | PPL | KL(ref ‖ model) | top-1 agreement |
|---|---|---|---|
| bf16 (reference) | 173.95 | — | — |
| **this quant** | 167.25 | **0.214** | **82.06%** |

(PPL on wikitext-2-raw is noisy at this window count and model size —
read the KL/agreement numbers as the more reliable signal that the
quantized model's output distribution tracks the original closely, not
the PPL gap as "quantization improved the model.")

## Does it work? Real end-to-end validation

All three run through this actual GPTQ-quantized MLX checkpoint — not the bf16 original.

**Text** (factual/reasoning prompt):
> The Colosseum was a massive amphitheater used for gladiatorial contests.
> Roman Law heavily influenced legal systems across the Western world.
> The Western Roman Empire traditionally fell in 476 AD.

**Vision** (real photo, "What animals are in this picture?"):
> There are two cats in this picture.

**Audio** (real 16kHz speech clip, "The quick brown fox jumps over the lazy dog", transcribed):
> A quick brown fox jumps over the lazy dog.

Peak memory during the combined smoke test: 7.8 GB.

## Usage

```bash
pip install git+https://github.com/ipsupport-llc/mlx-lm.git@main
```

```python
from mlx_lm import load, generate
model, tok = load("roman220220/Nemotron-3.5-Bolt-E4B-gptq-mlx-jang")
messages = [{"role": "user", "content": "What is the capital of France?"}]
print(generate(model, tok, prompt=tok.apply_chat_template(messages, add_generation_prompt=True)))
```

Image/audio input needs a manual generation loop (`mlx_lm.generate()`
has no multimodal plumbing for this fork's Gemma4 models yet) — full
example in the
[gemma4-quant pipeline repo](https://github.com/rromenskyi/quant-ternary/tree/main/gemma4-quant)
(`gemma4_smoke_test.py`).

## Method / code

- Model code: [ipsupport-llc/mlx-lm](https://github.com/ipsupport-llc/mlx-lm) (`gemma4.py`, `gemma4_text.py`, `gemma4_vision.py`, `gemma4_audio.py`)
- Quantization pipeline: [rromenskyi/quant-ternary/gemma4-quant](https://github.com/rromenskyi/quant-ternary/tree/main/gemma4-quant) (`gemma4_gptq_calibrate.py`, `gemma4_gptq_splice.py`)
- Same `gptq_nbit`/`gptq_nbit_batched` implementation as this account's other quantization work (`nemotron-extreme-quant`, `zimage-quant`), unmodified.

## The IPSupport local-AI stack

Local-first AI tools for macOS by [IPSupport](https://www.ipsupport.us) — nothing
leaves your Mac.

- **[LLMTray](https://www.ipsupport.us/llmtray/)** — your local AI
  workstation for macOS: chat with local LLMs, generate and edit images, make
  music, run agents, and serve an OpenAI-compatible API. Downloads models from
  Hugging Face in-app, with per-model profiles.
- **[IPSupport Code](https://ipsupport-llc.github.io/ipsupport-code/)** — your
  AI coding agent for real repositories: analyze, fix, test, report.

Runs in LLMTray as a chat model, with vision and audio input.

## Disclaimer

This repository contains a quantized conversion of a third-party model,
provided **"as is", without warranty of any kind**, express or implied.
IPSupport LLC did not train the model and is **not responsible for its
content, outputs or behavior, or for any damage, loss or liability** arising
from its use. You are responsible for evaluating the model and for how you
use it, including compliance with applicable laws and with the base model's
license and terms.

## License

Licensed under the **OpenMDW License, version 1.1**, the same license as
NVIDIA's release — see [`LICENSE`](LICENSE). NVIDIA's model was itself
post-trained from `google/gemma-4-E4B-it`, which remains governed by the
[Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0); both
terms apply, per NVIDIA's own card.

Modified from [nvidia/NVIDIA-Nemotron-3.5-Bolt-E4B-BF16](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Bolt-E4B-BF16):
the text decoder, vision tower and audio tower were GPTQ-quantized
(attention 8-bit, feed-forward 4-bit, group size 64), embeddings were
quantized to 8-bit RTN, `patch_embedder.input_proj` was left
unquantized, and the model was converted to MLX. The weights and
configuration files in this repo are therefore modified versions of
the original, not the original files.
