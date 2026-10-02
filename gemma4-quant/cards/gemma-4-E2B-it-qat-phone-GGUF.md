---
license: apache-2.0
base_model: google/gemma-4-E2B-it-qat-q4_0-unquantized
pipeline_tag: image-text-to-text
tags:
  - gguf
  - llama.cpp
  - gemma4
  - quantized
  - qat
  - vision
  - audio
  - mobile
---

<p align="center">
  <img src="llmtray-banner.png" alt="LLMTray" width="100%">
</p>

# Gemma 4 E2B — the small QAT build, 2.86 GB GGUF, smaller and closer than Google's q4_0

> ### ▶ GGUF build for llama.cpp / ollama / phone apps
> This is the **GGUF** build. If you use
> [LLMTray](https://www.ipsupport.us/llmtray/) — IPSupport's local AI app for Apple Silicon, which runs
> MLX — use the MLX sibling
> [roman220220/gemma-4-E2B-it-qat-phone-mlx](https://huggingface.co/roman220220/gemma-4-E2B-it-qat-phone-mlx) instead.
>
> [![Download LLMTray](https://img.shields.io/badge/Download-LLMTray%20for%20Mac-2f7d4f?style=for-the-badge&logo=apple&logoColor=white)](https://github.com/ipsupport-llc/llmtray/releases/latest/download/LLMTray-Full.dmg)
> [![GitHub stars](https://img.shields.io/github/stars/ipsupport-llc/llmtray?style=for-the-badge&logo=github)](https://github.com/ipsupport-llc/llmtray)

Google trained Gemma 4 E2B with quantization-aware training (QAT) for
llama.cpp's `Q4_0` and shipped it as
[google/gemma-4-E2B-it-qat-q4_0-gguf](https://huggingface.co/google/gemma-4-E2B-it-qat-q4_0-gguf)
(3.35 GB). This build is **half a gigabyte smaller and closer to the
unquantized model**:

| GGUF | size | PPL | PPL vs bf16 |
|---|---|---|---|
| bf16 of the QAT master weights | 9.27 GB | 42.84 | — |
| Google's `gemma-4-E2B_q4_0-it.gguf` | 3.35 GB | 46.47 | +8.5% |
| **this build** | **2.86 GB** | **42.24** | **−1.4%** |

llama.cpp `llama-perplexity --kl-divergence` against the bf16 GGUF, raw
wikitext-2 (test), 32 chunks of 512 tokens; all three PPLs in that one
scoring (`llama-perplexity` on its own puts the bf16 at 43.63).
A PPL slightly under the bf16 one is within the noise of this test: the QAT
model was trained through its 4-bit weights.

## Files

| File | Size | Contents |
|---|---|---|
| `gemma-4-E2B-it-qat-phone-Q4_0.gguf` | 2.86 GB | Text decoder |
| `gemma-4-E2B-it-mmproj.gguf` | 0.99 GB | Vision and audio towers + projectors: Google's own file from [google/gemma-4-E2B-it-qat-q4_0-gguf](https://huggingface.co/google/gemma-4-E2B-it-qat-q4_0-gguf), unchanged |

## How it's made

- **The same QAT grid.** The 149 text Linears at `Q4_0` are **byte-identical
  to Google's** GGUF: the grid the QAT trained for, unchanged.
- **126 Linears at `Q8_0`.** The ones that lose most at 4-bit, chosen by
  measured KL per MB on the MLX build (mostly the small per-layer gates and
  projections and attention projections). This is where the quality gain
  over Google's GGUF comes from.
- **Per-layer embeddings at `Q4_K`.** Half of E2B is its per-layer embedding
  table; Google keeps it bigger. At 4 bits it costs almost nothing and is
  most of the size saving.
- **Token embeddings at `Q6_K`**, as in Google's GGUF.

Made with `llama-quantize --tensor-type` overrides from the bf16 GGUF of
[google/gemma-4-E2B-it-qat-q4_0-unquantized](https://huggingface.co/google/gemma-4-E2B-it-qat-q4_0-unquantized).

## Checked

Before upload, with llama.cpp (`llama-cli`, `llama-mtmd-cli`, `--jinja`):
- text: three facts about the Roman Empire;
- vision, with the mmproj: a photo of a fox, described as "a small mammal
  with reddish-orange fur";
- audio, with the mmproj: a spoken clip transcribed word for word ("the
  quick brown fox jumps over the lazy dog").

## Usage (llama.cpp) — `--jinja` is required

Gemma 4's chat template uses control tokens (`<|turn>`, `<|channel>`); without
`--jinja` llama.cpp applies a simplified template and chat breaks.

```bash
llama-cli -m gemma-4-E2B-it-qat-phone-Q4_0.gguf --jinja -p "What is the capital of France?"
```

Images and audio:
```bash
llama-mtmd-cli -m gemma-4-E2B-it-qat-phone-Q4_0.gguf --mmproj gemma-4-E2B-it-mmproj.gguf --jinja \
  --image photo.jpg -p "What is in this picture?"
llama-mtmd-cli -m gemma-4-E2B-it-qat-phone-Q4_0.gguf --mmproj gemma-4-E2B-it-mmproj.gguf --jinja \
  --audio speech.wav -p "Transcribe this audio."
```

## Method / code

[rromenskyi/quant-ternary/gemma4-quant](https://github.com/rromenskyi/quant-ternary/tree/main/gemma4-quant)
has the code and the full lab notes (`docs/FINDINGS.md`, "E2B for phones"):
`e2b_phone.sh` builds this GGUF and its MLX sibling and scores both;
`gguf_types.py` maps the MLX sensitivity scan to `--tensor-type` overrides.

## The IPSupport local-AI stack

Local-first AI tools for macOS by [IPSupport](https://www.ipsupport.us) — nothing
leaves your Mac.

- **[LLMTray](https://www.ipsupport.us/llmtray/)** — your local AI
  workstation for macOS: chat with local LLMs, generate and edit images, make
  music, run agents, and serve an OpenAI-compatible API. Downloads models from
  Hugging Face in-app, with per-model profiles.
- **[IPSupport Code](https://ipsupport-llc.github.io/ipsupport-code/)** — your
  AI coding agent for real repositories: analyze, fix, test, report.

This is the GGUF build (llama.cpp / ollama). For the MLX build LLMTray runs, see [roman220220/gemma-4-E2B-it-qat-phone-mlx](https://huggingface.co/roman220220/gemma-4-E2B-it-qat-phone-mlx).

## License

Licensed under the **Apache License 2.0**, the same license as the base model — see [`LICENSE`](LICENSE).

Modified from [google/gemma-4-E2B-it-qat-q4_0-unquantized](https://huggingface.co/google/gemma-4-E2B-it-qat-q4_0-unquantized) (the QAT release of [google/gemma-4-E2B-it](https://huggingface.co/google/gemma-4-E2B-it)): converted to GGUF and quantized with per-tensor types (text Linears `Q4_0`, 126 of them `Q8_0`, per-layer embeddings `Q4_K`, token embeddings `Q6_K`). The weights in this repo are therefore modified versions of the original, not the original files.

`gemma-4-E2B-it-mmproj.gguf` is Google's file from [google/gemma-4-E2B-it-qat-q4_0-gguf](https://huggingface.co/google/gemma-4-E2B-it-qat-q4_0-gguf) (Apache 2.0), unchanged.

Gemma 4 is released by Google under Apache 2.0 ([Gemma 4 license terms](https://ai.google.dev/gemma/docs/gemma_4_license)).
