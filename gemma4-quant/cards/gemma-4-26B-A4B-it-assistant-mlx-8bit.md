---
library_name: mlx
license: apache-2.0
license_link: https://ai.google.dev/gemma/docs/gemma_4_license
pipeline_tag: text-generation
base_model: google/gemma-4-26B-A4B-it-assistant
tags:
- mlx
- gemma4
- speculative-decoding
- mtp
---

# Gemma 4 26B-A4B MTP drafter — MLX 8-bit

Google's Multi-Token-Prediction drafter for Gemma 4 26B-A4B
([google/gemma-4-26B-A4B-it-assistant](https://huggingface.co/google/gemma-4-26B-A4B-it-assistant)),
quantized to 8-bit (group 64) for MLX: **446MB** (bf16: 840MB).

It is not a standalone model. It's a 4-layer head that reads the main
model's final hidden state and its KV cache (last full- and last
sliding-attention layer), guesses a few tokens ahead, and the main model
verifies them all in one forward pass. Every emitted token is still the main
model's own pick, so output quality is unchanged and decoding gets faster.

## Usage

Needs the `gemma4_assistant` support from
[ipsupport-llc/mlx-lm](https://github.com/ipsupport-llc/mlx-lm) (branch
`gemma4-mtp`; stock mlx-lm doesn't have it yet):

```bash
mlx_lm.server \
  --model roman220220/gemma-4-26B-A4B-it-gptq-mlx-jang \
  --draft-model roman220220/gemma-4-26B-A4B-it-assistant-mlx-8bit \
  --num-draft-tokens 3
```

Works with any Gemma 4 26B-A4B-it MLX checkpoint (it only shares the
tokenizer and hidden size with the main model).

## Measured

Main model [roman220220/gemma-4-26B-A4B-it-gptq-mlx-jang](https://huggingface.co/roman220220/gemma-4-26B-A4B-it-gptq-mlx-jang)
(4-bit JANG), Apple Silicon Mac with 26GB, greedy, `num_draft_tokens=3`:

| Prompt | plain | with drafter |
|---|---|---|
| short (37 tokens), 256 out | 35.8 tok/s | **52.9 tok/s** (+48%) |
| 3.9k-token context, 300 out | 32.7 tok/s | **41.7 tok/s** (+28%, k=2) |

About 70% of emitted tokens come from the drafter. Acceptance with the 8-bit
drafter is the same as with bf16. The drafter was trained against the bf16
26B model, but acceptance on the 4-bit JANG quant is still high.

Image requests currently decode without the drafter (same output, no
speed-up).
