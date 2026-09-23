---
license: apache-2.0
base_model: google/gemma-4-26B-A4B-it
pipeline_tag: image-text-to-text
tags:
  - gguf
  - llama.cpp
  - gemma4
  - moe
  - imatrix
  - jang
  - quantized
  - vision
---

# Gemma 4 26B-A4B (MoE) — GGUF, imatrix + JANG-mixed, text + vision

A GGUF quantization of
[google/gemma-4-26B-A4B-it](https://huggingface.co/google/gemma-4-26B-A4B-it)
(the 128-expert / top-8 MoE variant, ~4B active params) for **llama.cpp /
Arc / CPU / any GGUF runtime** — with two deliberate improvements over a
stock `Q3_K_M`:

1. **imatrix computed on a broad real corpus** (bartowski's
   `calibration_datav3` — code + prose + facts + multilingual). This is the
   GGUF-world equivalent of calibrating on real data, and lets a smaller
   quant match the quality of a larger stock one.
2. **JANG-mixed precision** (this account's "spend bits where they matter"
   recipe): attention kept HIGH (`Q5_K`), the 128 routed experts — which are
   ~90% of the weights — pushed LOW (`IQ3_S`). Same principle as the MLX
   JANG releases, expressed via `llama-quantize --tensor-type` overrides.

Result: **12.5GB** text model (smaller than a stock ~13GB Q3_K_M) **plus a
quantized vision projector** — the stock text-only q3km has no vision.

## Files

| File | Size | Contents |
|---|---|---|
| `gemma4-26b-a4b-jang-iq3s.gguf` | 12.5GB | Text decoder + MoE. Attention `Q5_K`, dense MLP `Q3_K_M`, experts `IQ3_S`, all imatrix-guided |
| `mmproj-gemma4-26b-q8.gguf` | 0.77GB | Vision tower + projector, `Q8_0` (27 of its `ffn_down` tensors fall back to F16 — their 4304-wide dim isn't divisible by any GGUF block size) |
| `gemma4-26b-assistant-q8_0.gguf` | 0.46GB | Optional MTP drafter for speculative decoding (Google's `gemma-4-26B-A4B-it-assistant`, `Q8_0`, same as the official ollama `gemma4:26b` draft layer) -- see the Ollama section |

## Verified working (text + vision)

**Text** (chat):
> Here are three facts about the Roman Empire: 1. It reached its greatest territorial extent under Emperor Trajan (117 AD)... *Mare Nostrum*... 2. over 400,000 km of roads...

**Vision** (real COCO photo of two cats):
> The picture shows two cats sleeping on a couch.

Both run through this actual quantized checkpoint, ~130 tok/s generation on
an A100.

## Usage (Ollama) — use the included Modelfile

Plain `ollama run hf.co/roman220220/gemma-4-26B-A4B-it-GGUF-jang-imatrix`
**loads and runs, but thinking/tool tokens leak into the chat as raw text**
(`<|channel>thought … <channel|>`). A Hugging Face repo can only give Ollama a
Go template / params / system prompt — it cannot set Ollama's native Gemma 4
`RENDERER`/`PARSER`, which is what the official `gemma4` model uses to hide
the thinking block and parse Gemma 4's native tool-call syntax. The included
`Modelfile` sets both and pulls the weights straight from this repo:

```bash
curl -L -o Modelfile https://huggingface.co/roman220220/gemma-4-26B-A4B-it-GGUF-jang-imatrix/resolve/main/Modelfile
curl -L -o gemma4-26b-assistant-q8_0.gguf https://huggingface.co/roman220220/gemma-4-26B-A4B-it-GGUF-jang-imatrix/resolve/main/gemma4-26b-assistant-q8_0.gguf
ollama create gemma4-26b-jang -f Modelfile
ollama run gemma4-26b-jang
```

**Speculative decoding (MTP drafter).** `gemma4-26b-assistant-q8_0.gguf`
(462MB) is Google's Multi-Token-Prediction drafter for Gemma 4 26B-A4B
([google/gemma-4-26B-A4B-it-assistant](https://huggingface.co/google/gemma-4-26B-A4B-it-assistant)),
byte-identical to the draft layer of the official ollama `gemma4:26b`
(sha256 `6326fb9f…`). It's a 4-layer head that reads the main model's hidden
state and KV cache and guesses a few tokens ahead; the main model verifies
them in one pass, so output is identical and decode is faster. Ollama's
`DRAFT` only takes a local file, hence the second download. The drafter was
trained against the bf16 model, so acceptance on this quantized one may be a
bit lower than Google's "up to 3x" — measure tok/s with and without it. If
memory is tight, delete the two draft lines from the Modelfile.

Requires a recent Ollama (Gemma 4 support; the official model declares
`requires: 0.30.0`).

## Usage (llama.cpp) — `--jinja` is REQUIRED

Gemma 4's chat template uses control tokens (`<|turn>`, `<|channel>`,
`<|think|>`); without `--jinja`, llama.cpp applies a simplified template and
chat/thinking mode breaks (the model emits raw `<thought` and stops). Always
pass `--jinja`.

Text:
```bash
llama-cli -m gemma4-26b-a4b-jang-iq3s.gguf -ngl 99 --jinja \
  -p "List three facts about the Roman Empire."
```

Vision:
```bash
llama-mtmd-cli -m gemma4-26b-a4b-jang-iq3s.gguf \
  --mmproj mmproj-gemma4-26b-q8.gguf -ngl 99 --jinja \
  --image photo.jpg -p "What is in this picture?"
```

## Honest notes

- **This is imatrix + JANG bit-allocation, NOT this account's GPTQ.** GPTQ's
  Hessian weight-correction is calibrated against a specific quant grid;
  llama.cpp's k-quant super-block grid differs, so the correction doesn't
  transfer and gets re-quantized away. imatrix + JANG are the parts that DO
  carry over to GGUF. (The GPTQ version of this model lives at
  [roman220220/gemma-4-26B-A4B-it-gptq-mlx-jang](https://huggingface.co/roman220220/gemma-4-26B-A4B-it-gptq-mlx-jang), for MLX / Apple Silicon.)
- **Perplexity as a metric is currently broken for Gemma 4 MoE in llama.cpp**
  — both this quant and stock reference GGUFs report absurd PPL (~25k–33k on
  wikitext) while generating perfectly. Quality here was verified by
  generation, not PPL.
- Requires a recent llama.cpp built from source (Gemma 4 MoE + vision
  support landed only in the `conversion/gemma.py` refactor; prebuilt release
  binaries don't have it yet).

**MLX / Apple Silicon:** an 8-bit MLX build of the same drafter is at
[roman220220/gemma-4-26B-A4B-it-assistant-mlx-8bit](https://huggingface.co/roman220220/gemma-4-26B-A4B-it-assistant-mlx-8bit).

## Method / code

- Pipeline: [rromenskyi/quant-ternary/gemma4-quant](https://github.com/rromenskyi/quant-ternary/tree/main/gemma4-quant) (`gemma4_26b_gguf_pipeline.sh`, `gemma4_26b_fix_tokenizer.py`)
