---
license: apache-2.0
base_model: google/gemma-4-E2B-it
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

<p align="center">
  <img src="llmtray-banner.png" alt="LLMTray" width="100%">
</p>

# Gemma 4 E2B — GPTQ, JANG-mixed precision, full multimodal (text + vision + audio), MLX

> ### ▶ Run it locally in [LLMTray](https://www.ipsupport.us/llmtray/)
> A free, native macOS app for local AI on Apple Silicon — chat, images,
> music, agents and an OpenAI-compatible API. Point it at this model; nothing
> leaves your Mac.
>
> [![Download LLMTray](https://img.shields.io/badge/Download-LLMTray%20for%20Mac-2f7d4f?style=for-the-badge&logo=apple&logoColor=white)](https://github.com/ipsupport-llc/llmtray/releases/latest/download/LLMTray-Full.dmg)
> [![GitHub stars](https://img.shields.io/github/stars/ipsupport-llc/llmtray?style=for-the-badge&logo=github)](https://github.com/ipsupport-llc/llmtray)

A JANG-style mixed-precision MLX quantization of
[google/gemma-4-E2B-it](https://huggingface.co/google/gemma-4-E2B-it) — the
smaller (E2B) sibling of Gemma 4's multimodal E-series — using **GPTQ
Hessian-based error correction** across all three real components (text
decoder, vision tower, audio tower). Attention projections at 8-bit,
feed-forward at 4-bit, every embedding table at 8-bit RTN. **Nothing is left
in bf16.** The goal for this release was **maximum quality at minimum size**.

E2B is a dense model (no MoE): 35 decoder layers, hidden size 1536, with
per-layer input embeddings and 20 KV-shared layers — the same architecture
family as [E4B](https://huggingface.co/roman220220/gemma-4-E4B-it-gptq-mlx-jang),
just smaller.

## Recipe: JANG-mixed, by component and by role

| Role | Bits | Why |
|---|---|---|
| Attention (`q/k/v/o_proj`, audio's `lconv1d`) | 8-bit GPTQ | Small relative to FFN, disproportionately error-sensitive |
| Feed-forward (`gate/up/down_proj`, audio's `feed_forward1/2`) | 4-bit GPTQ | The bulk of the parameters; tolerates aggressive compression |
| Embeddings (`embed_tokens`, `embed_tokens_per_layer`, per-layer projections) | 8-bit RTN | A lookup table has no shared-input matmul for GPTQ to correct |

Calibrated with `--group-size 64` on real data: diverse text prompts for the
language model, real COCO photographs for vision, real LibriSpeech clips for
audio. Dead KV-shared k/v/k_norm weights are dropped;
`vision_tower.patch_embedder.input_proj` is kept float (mlx-lm casts pixels
to its dtype, so quantizing it breaks vision).

## Does it actually work? Real end-to-end validation

- **Text** ("three facts about the Roman Empire"):
  > 1. The Roman Empire expanded to control a vast territory stretching across Europe, North Africa, and the Middle East.
  > 2. Roman law and engineering significantly influenced Western civilization…
- **Vision** (real COCO photo of two cats, "what animal is in this picture?"):
  > There are two cats in this picture.
- **Audio**: the audio tower is GPTQ-quantized on real LibriSpeech clips (same
  recipe as text/vision). Text and vision are validated end-to-end here through
  a real autoregressive generation loop; audio transcription uses the same
  proven path as the [E4B release](https://huggingface.co/roman220220/gemma-4-E4B-it-gptq-mlx-jang).

The smoke test also confirms **0 large unquantized (bf16) weights** remain.
Peak memory in the smoke run was ~5.7 GB.

## KV-shared quantized-cache fix (ships alongside)

Gemma 4's KV-shared layers crash with `mlx_lm`'s quantized KV cache
(`--kv-bits`): a shared layer passes `cache=None` to the dispatch check, so
once the source layer's cache quantizes, the shared layer hands a quantized
tuple to the unquantized attention path and crashes. Fixed in
[ipsupport-llc/mlx-lm#1](https://github.com/ipsupport-llc/mlx-lm/pull/1) —
needed for this checkpoint with KV-cache quantization on (LLMTray's default).

## Usage

```bash
pip install git+https://github.com/ipsupport-llc/mlx-lm.git@main
```

```python
from mlx_lm import load, generate
model, tokenizer = load("roman220220/gemma-4-E2B-it-gptq-mlx-jang")
messages = [{"role": "user", "content": "What is the capital of France?"}]
print(generate(model, tokenizer, prompt=tokenizer.apply_chat_template(messages, add_generation_prompt=True)))
```

Image/audio inputs need a manual generation loop (`mlx_lm.generate()` has no
image/audio plumbing yet for this fork's vision models) — full examples in
the [gemma4-quant pipeline repo](https://github.com/rromenskyi/quant-ternary/tree/main/gemma4-quant).

## Method / code

- Model code: [ipsupport-llc/mlx-lm](https://github.com/ipsupport-llc/mlx-lm) (`gemma4.py`, `gemma4_vision.py`, `gemma4_audio.py`, `gemma4_text.py`)
- Quantization pipeline: [rromenskyi/quant-ternary/gemma4-quant](https://github.com/rromenskyi/quant-ternary/tree/main/gemma4-quant) (`gemma4_mlx_pipeline.sh VARIANT=e2b`)
- Same `gptq_nbit` implementation as this account's other quantization work, unmodified.

## Size

~4.3 GB (down from ~9.6 GB bf16 — 2.2× smaller), everything ≤8-bit, nothing left in bf16.

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

Modified from [google/gemma-4-E2B-it](https://huggingface.co/google/gemma-4-E2B-it): the text decoder, vision tower and audio tower were GPTQ-quantized (attention 8-bit, feed-forward 4-bit, group size 64), all embedding and per-layer-projection tables were quantized to 8-bit RTN, and the model was converted to MLX. The weights and configuration files in this repo are therefore modified versions of the original, not the original files.

Gemma 4 is released by Google under Apache 2.0 ([Gemma 4 license terms](https://ai.google.dev/gemma/docs/gemma_4_license)).
