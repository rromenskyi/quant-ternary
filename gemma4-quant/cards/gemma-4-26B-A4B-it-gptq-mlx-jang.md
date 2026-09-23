---
license: apache-2.0
base_model: google/gemma-4-26B-A4B-it
pipeline_tag: image-text-to-text
tags:
  - mlx
  - gemma4
  - gptq
  - quantized
  - moe
  - multimodal
  - vision
  - jang
---

# Gemma 4 26B-A4B (MoE) — GPTQ, JANG-mixed precision, text + vision, MLX

A JANG-style mixed-precision MLX quantization of
[google/gemma-4-26B-A4B-it](https://huggingface.co/google/gemma-4-26B-A4B-it)
— the 26.5B-parameter, 128-expert / top-8 MoE variant of Gemma 4 (≈4B
active parameters per token, hence "A4B") — using **GPTQ Hessian-based
error correction** across every quantizable Linear in both the text
decoder and the vision tower. Attention projections at 8-bit, feed-forward
(dense MLP **and** all 128 routed experts) at 4-bit, embeddings and every
uncalibrated Linear at 8-bit RTN. Kept in bf16 on purpose (~23MB total):
- the MoE routers, because routing precision is disproportionately
  sensitive;
- `vision_tower.patch_embedder.input_proj`, because mlx-lm casts pixels to
  its weight dtype, so quantizing it breaks vision.

- ~15GB (down from 51.6GB bf16 — 3.4× smaller).
- This variant has **no audio tower** (`audio_config` is null on the real
  checkpoint) — text + vision only, unlike the E4B sibling.

## Recipe: JANG-mixed, by role

| Role | Bits | Notes |
|---|---|---|
| Attention (`q/k/v/o_proj`, text + vision) | 8-bit GPTQ | Small, error-sensitive |
| Dense MLP (`gate/up/down_proj`) | 4-bit GPTQ | Every layer has a dense MLP *and* MoE experts (hybrid) |
| **MoE experts** (`experts.gate_up_proj`/`down_proj`, 128 per layer) | 4-bit GPTQ | The bulk of the parameters; calibrated per-expert on the tokens actually routed to each |
| Router (`router.*`) | untouched | Routing precision is disproportionately sensitive — never quantized |
| Embeddings (`embed_tokens`) | 8-bit RTN | No shared-input matmul for GPTQ to correct |

Calibrated with `--group-size 64` on real data: diverse text prompts for
the language model, real COCO photographs for the vision tower.

## Architecture notes (verified directly against the real checkpoint)

- **Hybrid dense+MoE layers**: every one of the 30 decoder layers computes
  BOTH a dense MLP path and a routed-experts path, combined additively
  (`hidden_states_1 + hidden_states_2`) — the router operates on the
  pre-dense-MLP residual. Both paths are quantized here.
- **Per-expert GPTQ calibration**: the `Gemma4TextExperts` module consumes
  raw stacked `[128, out, in]` `nn.Parameter` tensors inside a Python
  loop, not `nn.Linear` submodules — so calibration hooks the whole
  experts module, replicates its real `expert_mask`/`token_idx` gathering
  to bucket calibration rows per expert, and runs batched GPTQ across all
  128 experts at once. `down_proj`'s calibration input is recomputed as
  `act_fn(gate)*up` using the real (uncorrected) `gate_up_proj`, exactly
  as the real forward pass feeds it.
- **Vision intermediate_size padding**: the vision MLP's intermediate
  width (4304) isn't divisible by any group size `mx.quantize` supports
  (32/64/128 — 4304 = 2⁴·269). Rather than leave those tensors in bf16,
  `gate/up/down_proj` are zero-padded to 4352 (next multiple of 64) — this
  is mathematically exact (GELU(0)·0 = 0, and the padded activations hit
  correspondingly-zero-weighted `down_proj` columns), declared in
  `config.json`'s `vision_config.intermediate_size` so the model is built
  at the padded width with no inference-code changes.

## Does it work? Real end-to-end validation

**Text**: correctly answers factual/reasoning prompts (Gemma 4's
thinking-mode chain included).

**Vision** (real COCO photo of two cats, "What animal is in this
picture?"):
> There are two cats in this picture.

Both run through this actual GPTQ-quantized checkpoint (MoE routing + vision
tower + fusion all exercised).

## Usage

```bash
pip install git+https://github.com/ipsupport-llc/mlx-lm.git@main
```

```python
from mlx_lm import load, generate
model, tok = load("roman220220/gemma-4-26B-A4B-it-gptq-mlx-jang")
messages = [{"role": "user", "content": "What is the capital of France?"}]
print(generate(model, tok, prompt=tok.apply_chat_template(messages, add_generation_prompt=True)))
```

Image input needs a manual generation loop (`mlx_lm.generate()` has no
image plumbing for this fork's vision models yet) — full example in the
[gemma4-quant pipeline repo](https://github.com/rromenskyi/quant-ternary/tree/main/gemma4-quant).

## Method / code

- Model code: [ipsupport-llc/mlx-lm](https://github.com/ipsupport-llc/mlx-lm) (`gemma4.py`, `gemma4_text.py`, `gemma4_vision.py`)
- Quantization pipeline: [rromenskyi/quant-ternary/gemma4-quant](https://github.com/rromenskyi/quant-ternary/tree/main/gemma4-quant) (`gemma4_26b_gptq_calibrate.py`, `gemma4_26b_gptq_splice.py`)
- Same `gptq_nbit`/`gptq_nbit_batched` implementation as this account's other quantization work (`nemotron-extreme-quant`, `zimage-quant`), unmodified.
