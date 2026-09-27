---
license: apache-2.0
base_model: google/gemma-4-E4B-it
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
---

# Gemma 4 E4B — GPTQ, JANG-mixed precision, full multimodal (text + vision + audio), MLX

> ### ▶ Run it locally in [LLMTray](https://www.ipsupport.us/llmtray/)
> A free, native macOS app for local AI on Apple Silicon — chat, images,
> music, agents and an OpenAI-compatible API. Point it at this model; nothing
> leaves your Mac.
>
> [![Download LLMTray](https://img.shields.io/badge/Download-LLMTray%20for%20Mac-2f7d4f?style=for-the-badge&logo=apple&logoColor=white)](https://github.com/ipsupport-llc/llmtray/releases/latest/download/LLMTray-Full.dmg)
> [![GitHub stars](https://img.shields.io/github/stars/ipsupport-llc/llmtray?style=for-the-badge&logo=github)](https://github.com/ipsupport-llc/llmtray)


A JANG-style mixed-precision MLX quantization of
[google/gemma-4-E4B-it](https://huggingface.co/google/gemma-4-E4B-it) using
**GPTQ Hessian-based error correction** across all three real components
(text decoder, vision tower, audio tower) — attention projections at 8-bit,
feed-forward projections at 4-bit, and every embedding table quantized to
8-bit RTN. **Nothing in this checkpoint is left in bf16.**

## This replaces an earlier, buggy 8-bit release

An earlier upload from this account (`gemma-4-E4B-it-gptq-mlx-8bit`) shipped
with two real bugs found and fixed while building this one:

1. **The vision tower was never actually quantized.** The splice script's
   raw-checkpoint key prefix for vision was missing `.encoder.` (the real
   checkpoint nests vision layers under
   `vision_tower.encoder.layers.N...`, not `vision_tower.layers.N...`), so
   all 112 vision GPTQ corrections silently failed to match any real key
   and the entire vision tower shipped as plain, uncorrected bf16.
2. **~7.2GB of embeddings were never quantized at all**, dominated by
   `embed_tokens_per_layer` (the per-layer-embedding table, 5.6GB by
   itself) plus `embed_tokens` and several small per-layer projection
   `Linear`s — none of these were in the GPTQ calibration's target module
   list, so they stayed in bf16 and inflated that release to ~11GB despite
   being labeled "8-bit".

Both are fixed here: vision now genuinely receives GPTQ correction (correct
checkpoint paths, verified by directly checking which corrected tensors
match real keys before shipping), and every embedding/per-layer-projection
table is quantized to 8-bit (RTN — GPTQ's Hessian correction doesn't apply
to a pure lookup table the same way it does to a `Linear` reused across
many inputs, so these use plain round-to-nearest instead, called out
honestly rather than silently mixed in as GPTQ).

## Recipe: JANG-mixed, by component and by role

Uniform bit-width wastes precision on the wrong things and starves the
wrong things too. Each of the three towers (text, vision, audio) is
corrected independently, and *within* each tower:

| Role | Bits | Why |
|---|---|---|
| Attention (`q/k/v/o_proj`, audio's `lconv1d`) | 8-bit | Small relative to FFN weights, disproportionately sensitive to error (feeds every downstream token's representation) |
| Feed-forward (`gate/up/down_proj`, audio's `feed_forward1/2`) | 4-bit | The overwhelming majority of parameters; tolerates more aggressive compression much better than attention |
| Embeddings (`embed_tokens`, `embed_tokens_per_layer`, per-layer projections) | 8-bit RTN | No shared-calibration-input matmul to run GPTQ against; kept at 8-bit for safety since these define the entire vocabulary's representation |
| Router (MoE variants only, not present in E4B) | full precision | Not applicable to E4B (no MoE layers) |

Calibrated with `--group-size 64` on real data: 8 diverse text prompts for
the language model, 6 real COCO photographs for the vision tower, 6 real
LibriSpeech clips for the audio tower.

## Does it actually work? Real end-to-end validation, all three modalities

**Text**:
> The capital of Japan is Tokyo.

**Vision** (real COCO photo of two cats, asked "What animal is in this
picture?"):
> There are two cats in this picture.

**Audio** (real LibriSpeech clip, asked to transcribe):
> Ground truth: *"mister quilter is the apostle of the middle classes and we are glad to welcome his gospel"*
> Model output: *"Mr. Quilter is the apostle of the middle classes and we are glad to welcome his gospel."*

All three run through this actual GPTQ-quantized checkpoint via a real
autoregressive generation loop (not a single forward pass).

## A related server-side fix that shipped alongside this

Gemma 4's KV-shared layers (`num_kv_shared_layers`, layers that reuse an
earlier layer's keys/values instead of computing their own) crashed when
combined with `mlx_lm`'s quantized KV-cache feature
(`--kv-bits`/`--quantized-kv-start`) — the shared layer always passed
`cache=None` to the attention-dispatch check that decides whether to use
quantized attention, so once the source layer's cache actually quantized,
the shared layer handed a quantized tuple to the *unquantized* attention
path and crashed. Fixed in
[ipsupport-llc/mlx-lm#1](https://github.com/ipsupport-llc/mlx-lm/pull/1)
with a regression test — needed for this checkpoint to be usable with
KV-cache quantization enabled (the default in some clients, including
LLMTray).

## Usage

```bash
pip install git+https://github.com/ipsupport-llc/mlx-lm.git@fix-gemma4-quantized-kv-shared
```

```python
import mlx.core as mx
from transformers import AutoProcessor
from mlx_lm import load, generate

model, tokenizer = load("roman220220/gemma-4-E4B-it-gptq-mlx-jang")
proc = AutoProcessor.from_pretrained("google/gemma-4-E4B-it")

messages = [{"role": "user", "content": "What is the capital of France?"}]
print(generate(model, tokenizer, prompt=tokenizer.apply_chat_template(messages, add_generation_prompt=True)))
```

Image/audio inputs need a manual generation loop (`mlx_lm.generate()` has
no image/audio plumbing yet for this or any other vision model in this
fork) — full working examples in the
[gemma4-quant pipeline repo](https://github.com/rromenskyi/quant-ternary/tree/main/gemma4-quant).

## Method / code

- Model code: [ipsupport-llc/mlx-lm](https://github.com/ipsupport-llc/mlx-lm) (`gemma4.py`, `gemma4_vision.py`, `gemma4_audio.py`, `gemma4_text.py`)
- Quantization pipeline: [rromenskyi/quant-ternary/gemma4-quant](https://github.com/rromenskyi/quant-ternary/tree/main/gemma4-quant)
- Same `gptq_nbit` implementation as this account's other quantization work (`nemotron-extreme-quant`, `zimage-quant`), unmodified.

## Size

~6.3GB (down from 16GB bf16, and down from the earlier, buggily-larger
~11GB "8-bit" release that had an unquantized vision tower and ~7.2GB of
unquantized embeddings).

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

Modified from [google/gemma-4-E4B-it](https://huggingface.co/google/gemma-4-E4B-it): the text decoder, vision tower and audio tower were GPTQ-quantized (attention 8-bit, feed-forward 4-bit, group size 64), all embedding and per-layer-projection tables were quantized to 8-bit RTN, and the model was converted to MLX. The weights and configuration files in this repo are therefore modified versions of the original, not the original files.

Gemma 4 is released by Google under Apache 2.0 ([Gemma 4 license terms](https://ai.google.dev/gemma/docs/gemma_4_license)).
