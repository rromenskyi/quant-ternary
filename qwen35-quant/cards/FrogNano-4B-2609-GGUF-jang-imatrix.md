---
license: apache-2.0
base_model: microsoft/FrogNano-4B-2609
pipeline_tag: text-generation
tags:
  - gguf
  - llama.cpp
  - qwen3_5
  - jang
  - imatrix
  - quantized
---

<p align="center">
  <img src="llmtray-banner.png" alt="LLMTray" width="100%">
</p>

# FrogNano-4B-2609 — GGUF, imatrix + JANG-mixed (text + vision)

> ### ▶ GGUF build for ollama / llama.cpp
> This is the **GGUF** build (ollama / llama.cpp, Arc, CPU). Includes a
> vision mmproj (see "Vision" below) — load both files together for
> image understanding. If you use
> [LLMTray](https://www.ipsupport.us/llmtray/) — IPSupport's local AI app for
> Apple Silicon, which runs MLX — use the MLX sibling
> [roman220220/FrogNano-4B-2609-gptq-mlx-jang](https://huggingface.co/roman220220/FrogNano-4B-2609-gptq-mlx-jang) instead (it also keeps the MTP speculative-decoding head, which GGUF carries as inert tensors — see "Honest notes").
>
> [![Download LLMTray](https://img.shields.io/badge/Download-LLMTray%20for%20Mac-2f7d4f?style=for-the-badge&logo=apple&logoColor=white)](https://github.com/ipsupport-llc/llmtray/releases/latest/download/LLMTray-Full.dmg)
> [![GitHub stars](https://img.shields.io/github/stars/ipsupport-llc/llmtray?style=for-the-badge&logo=github)](https://github.com/ipsupport-llc/llmtray)

A GGUF quantization of
[microsoft/FrogNano-4B-2609](https://huggingface.co/microsoft/FrogNano-4B-2609)
— Microsoft's compact repository-level coding agent, Qwen3.5-4B post-trained
with reinforcement learning on ~1,500 synthetic software-engineering tasks
(SWE-bench Verified 61.5% Avg@3 in Microsoft's Leaf harness) — with two
deliberate improvements over a stock `Q4_K_M`:

1. **imatrix computed on a broad real corpus** (bartowski's
   `calibration_datav3` — code + prose + facts + multilingual).
2. **JANG-mixed precision**: full-attention layers, embeddings and the
   gated-DeltaNet linear-attention layers (most of the network) kept at
   `Q6_K`, feed-forward at the `Q4_K_M` base preset. (Not `Q8_0` on
   attention — see "Honest notes": `Q8_0` is ~3x slower than `Q6_K` on
   Vulkan hardware without a hardware int8 dot product, for no quality
   gain worth the cost.)

Result: **3.06 GB** instead of 8.67 GB F16.

## The recipe

Qwen3.5's hybrid decoder, as llama.cpp represents it: full attention
(`attn_q/k/v/output`, plus an output gate, `attn_gate`) in a few layers,
gated-DeltaNet linear attention (`ssm_a/conv1d/dt/norm/alpha/beta/out`) in
the rest, feed-forward (`ffn_gate/up/down`) everywhere.

| Component | Tensors | Type |
|---|---|---|
| Full attention + gate (8 of 32 layers) | `attn_q/k/v/output/gate` | `Q6_K` |
| Token embeddings (tied with the output head) | `token_embd` | `Q6_K` |
| Gated-DeltaNet linear attention (24 layers) | `ssm_out` | `Q6_K` |
| `ssm_alpha`/`ssm_beta` (the delta rule's input gates) | — | F32 (not quantized — see Honest notes) |
| Feed-forward (32 layers) | `ffn_gate/up/down` | `Q4_K_M` base preset |
| Other norms, NextN/MTP tensors (preserved, unused by plain inference) | — | F32 / base preset |

Importance matrix: `llama-imatrix`, 200 chunks on bartowski's
`calibration_datav3.txt`, on one RTX 4090.

## Measurements

Perplexity on wikitext-2 test and held-out Python stdlib source (40 x
512-token chunks each, `llama-perplexity`), F16 vs this quant, same GGUF
toolchain both sides — the stdlib number matters most here: this is a
coding-agent model.

| | Size | PPL (wikitext) | PPL (Python) |
|---|---|---|---|
| F16 | 8.67 GB | 8.623 | 2.915 |
| **this model, Q4_K_M jang** | **3.06 GB** | **8.723 (+1.15%)** | **2.947 (+1.10%)** |

## Usage

Microsoft's settings: temperature **0.6**, repetition penalty 1.0, up to
8,192 tokens per turn. It reasons before answering (`<think>…</think>`).

**Ollama:**

```bash
ollama run hf.co/roman220220/FrogNano-4B-2609-GGUF-jang-imatrix
```

**llama.cpp** — launch with `--jinja` so the model's own chat template
applies:

```bash
llama-server -m frognano-4b-jang.gguf -ngl 99 --jinja --port 8080
```

Microsoft's intended use is a sandboxed, human-reviewed coding agent
(their Leaf harness); read the
[base model card](https://huggingface.co/microsoft/FrogNano-4B-2609) for
its scope and limitations, which apply here unchanged. These are
language-modelling numbers — the coding-agent benchmarks above are
Microsoft's, for the bf16 model in their harness, not re-run on this
quantization.

## Vision (`mmproj-frognano-4b-f16.gguf`)

Qwen3.5's vision tower, exported standalone (`convert_hf_to_gguf.py
--mmproj`, F16, 672 MB) via upstream `llama.cpp`'s own `qwen3vl`
conversion path -- earlier releases of this card said the GGUF
architecture couldn't carry vision at all; that was true when this card
was first written and stopped being true once upstream added
`Qwen3_5ForConditionalGeneration` support, found by re-checking rather
than assuming the old conclusion still held. Not quantized further:
vision towers are a small fraction of total size already, and this
account's own `Q8_0`-on-Vulkan finding (see below) made a further
quantization pass not worth the risk without re-testing it specifically
for the vision encode path.

```bash
llama-mtmd-cli -m frognano-4b-jang-q6k.gguf \
  --mmproj mmproj-frognano-4b-f16.gguf \
  --image photo.jpg -p "Describe this image." -ngl 99
```

Verified working end to end: correctly identified a real photo of a
black Labrador puppy (breed, pose, fur/eye/nose detail, floor material)
on the same AMD D700 test hardware used throughout this card. Two
non-fatal warnings appear at load: `Qwen-VL models require at minimum
1024 image tokens` (add `--image-min-tokens 1024` if grounding/spatial
accuracy on small images matters) and a `find_slot: non-consecutive
token position` warning during image-token batching — neither affected
the output in this test, but neither was investigated further either.

## MTP speculative-decoding draft (`mtp-frognano-4b-q4km.gguf`)

Qwen3.5's own multi-token-prediction (MTP) head, exported standalone
(`convert_hf_to_gguf.py --mtp`) and quantized `Q4_K_M` (607 MB). Use it
as the draft model in classic draft+target speculative decoding:

```bash
llama-speculative-simple -m frognano-4b-jang-q6k.gguf \
  -md mtp-frognano-4b-q4km.gguf --spec-type draft-mtp \
  -ngl 99 -ngld 99 -p "..."
```

**Honest note: whether this helps depends entirely on your GPU, not on
this model.** Live-measured on the same AMD FirePro D700 (Tahiti, GCN1,
Vulkan, `int dot: 0`) from the note above: the draft head itself is
good — 85-93% accept rate on code continuations — but speculative
decoding still came out **~2x slower** than plain generation (best
tuned case: 12.3 tok/s vs 24.4 tok/s plain). Each draft+verify round is
an extra GPU dispatch round-trip, and on this GPU/driver, dispatch
latency — not compute — is already the bottleneck (see the `Q8_0` note
below); speculative decoding adds more of exactly what's expensive
here. Tuning (`--spec-draft-n-max`, `--poll`) recovered about +30% over
defaults but never closed the gap. On a GPU where per-dispatch overhead
is cheap (most current NVIDIA/AMD RDNA cards), the same 85-93% accept
rate should translate into a real speedup instead — not verified on
such hardware by this account, only reasoned from the mechanism.

## Honest notes

- **`Q8_0` on attention tested ~3x slower than `Q6_K` on old GPUs,
  live-measured.** The first release of this card used `Q8_0` for
  attention + embeddings (matching this account's usual GGUF convention
  for "keep this precise"). A user reported 7-9 tok/s on dual AMD FirePro
  D700 (Tahiti, GCN1) via Vulkan vs ~26 tok/s for a reference `Q4_K_M`
  build at the same nominal quant level. `GGML_SCHED_DEBUG=2` showed
  every op correctly scheduled to the GPU (no CPU fallback, not a
  multi-GPU sync issue) — the Vulkan device reports `int dot: 0` (no
  hardware int8 dot product), and `llama-bench` confirmed re-quantizing
  just the `Q8_0` tensors to `Q6_K` (nothing else changed) recovered the
  speed (9.2 -> 25.0 tok/s on one GPU), matching the reference build.
  `Q8_0`'s Vulkan kernel apparently leans on that instruction and falls
  back to a slow generic path without it; K-quants (`Q4_K`/`Q5_K`/`Q6_K`)
  aren't built around it and don't pay the penalty. `ssm_out` was `Q6_K`
  in both the slow and fast builds — it was never the cause.
- `ssm_alpha`/`ssm_beta` (the delta rule's input gates, `(hidden, 32)`
  matrices) are left unquantized: forcing them into a quant type cost
  ~12.5 MB (0.36% of the file) for no measurable benefit, and matches
  bartowski's own `Q4_K_M` build (confirmed from its published
  `layouts/*.tensor-types.txt`) and this account's MLX pipeline's
  `DEFAULT_KEEP_FLOAT` (`linear_attn.in_proj_[ab]$`), which already knew
  these need to stay float.
- **Text only.** Qwen3.5-4B's (and this model's) vision tower and MTP
  speculative-decoding head aren't part of the `QWEN35` GGUF architecture
  llama.cpp converts to — both exist only in the MLX sibling (via the
  `ipsupport-llc/mlx-lm` fork). Microsoft doesn't post-train or evaluate
  FrogNano's vision components regardless.
- This is imatrix + JANG bit-allocation, **not** this account's GPTQ:
  GPTQ's Hessian correction is calibrated against a specific quant grid
  that doesn't transfer to llama.cpp's k-quant grid. The GPTQ/MLX version
  of this model is at
  [roman220220/FrogNano-4B-2609-gptq-mlx-jang](https://huggingface.co/roman220220/FrogNano-4B-2609-gptq-mlx-jang).

## Method and code

Pipeline: [rromenskyi/quant-ternary](https://github.com/rromenskyi/quant-ternary),
`qwen35-quant/poc/frognano_gguf_pipeline.sh`.

## The IPSupport local-AI stack

Local-first AI tools for macOS by [IPSupport](https://www.ipsupport.us) — nothing
leaves your Mac.

- **[LLMTray](https://www.ipsupport.us/llmtray/)** — your local AI
  workstation for macOS: chat with local LLMs, generate and edit images, make
  music, run agents, and serve an OpenAI-compatible API. Downloads models from
  Hugging Face in-app, with per-model profiles.
- **[IPSupport Code](https://ipsupport-llc.github.io/ipsupport-code/)** — your
  AI coding agent for real repositories: analyze, fix, test, report.

This is the GGUF build (ollama / llama.cpp). For the MLX build LLMTray runs (with vision and the MTP speculative-decoding head), see [roman220220/FrogNano-4B-2609-gptq-mlx-jang](https://huggingface.co/roman220220/FrogNano-4B-2609-gptq-mlx-jang).

## Disclaimer

This repository contains a quantized conversion of a third-party model,
provided **"as is", without warranty of any kind**, express or implied.
IPSupport LLC did not train the model and is **not responsible for its
content, outputs or behavior, or for any damage, loss or liability** arising
from its use. You are responsible for evaluating the model and for how you
use it, including compliance with applicable laws and with the base model's
license and terms.

## License

Licensed under the **Apache License 2.0**, as stated in the base model's
card (FrogNano is derived from Qwen/Qwen3.5-4B, Apache 2.0; the base
repository's metadata says MIT, which Apache 2.0 also satisfies) — see
[`LICENSE`](LICENSE).

Modified from [microsoft/FrogNano-4B-2609](https://huggingface.co/microsoft/FrogNano-4B-2609):
converted to GGUF and quantized with an importance matrix and per-tensor
types (full attention, embeddings and gated-DeltaNet linear attention
`Q6_K`, feed-forward `Q4_K_M`). The weights and configuration files in
this repo are therefore modified versions of the original, not the
original files.
