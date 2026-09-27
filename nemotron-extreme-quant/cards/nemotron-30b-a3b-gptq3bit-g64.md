---
base_model: nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16
license: other
license_name: openmdw-1.1
license_link: LICENSE
tags:
  - mlx
  - quantized
  - gptq
  - nemotron
---

<p align="center">
  <img src="llmtray-banner.png" alt="LLMTray" width="100%">
</p>

# Nemotron-3.5-Lightning-30B-A3B — GPTQ uniform 3-bit (MLX)

> ### ▶ Run it locally in [LLMTray](https://www.ipsupport.us/llmtray/)
> A free, native macOS app for local AI on Apple Silicon — chat, images,
> music, agents and an OpenAI-compatible API. Point it at this model; nothing
> leaves your Mac.
>
> [![Download LLMTray](https://img.shields.io/badge/Download-LLMTray%20for%20Mac-2f7d4f?style=for-the-badge&logo=apple&logoColor=white)](https://github.com/ipsupport-llc/llmtray/releases/latest/download/LLMTray-Full.dmg)
> [![GitHub stars](https://img.shields.io/github/stars/ipsupport-llc/llmtray?style=for-the-badge&logo=github)](https://github.com/ipsupport-llc/llmtray)

GPTQ-calibrated (Hessian-based error compensation, Frantar et al. 2022)
uniform 3-bit quantization of NVIDIA's Nemotron-3.5-Lightning-30B-A3B
(NemotronH hybrid Mamba2 + Attention + MoE, 52 layers), packed with stock
`mlx_lm.convert` (no custom kernel, no custom model-loading code — runs with
plain `mlx_lm`/LM Studio like any other MLX model).

- **Bits**: 3, uniform across every linear layer (mamba in/out proj,
  attention q/k/v/o, MoE routed + shared experts).
- **Group size**: 64.
- **Size on disk**: ~13.8 GB.
- **Method**: weights are GPTQ-calibrated against 24×512-token wikitext-2
  chunks, snapped to the *exact* grid MLX's own affine quantization kernel
  will independently re-derive (verified bit-exact against
  `mx.quantize(mode="affine")`), then packed with stock `mlx_lm.convert -q
  --q-bits 3 --q-group-size 64` — so the calibration is preserved rather
  than silently discarded by a second, uncoordinated quantization pass.
  Uniform bit-width means this release is NOT affected by the
  switch_mlp.fc1/fc2 naming bug found in this project's mixed-precision
  releases (a uniform `-q --q-bits N` conversion applies the same bit-width
  everywhere regardless of tensor name).

## Perplexity (wikitext-2-raw, 20×512-token chunks, quarter-in offset)

| model | PPL |
|---|---|
| bf16 (full precision, reference) | 5.11 |
| JANG_2L-CRACK (3rd party, mixed-precision) | 5.43 |
| **this model** | **6.24** |
| naive RTN uniform 3-bit (no calibration) | 6.54 |

GPTQ's Hessian correction measurably beats naive round-to-nearest at the
same bit-width and size (6.24 vs 6.54).

## The IPSupport local-AI stack

Local-first AI tools for macOS by [IPSupport](https://www.ipsupport.us) — nothing
leaves your Mac.

- **[LLMTray](https://www.ipsupport.us/llmtray/)** — your local AI
  workstation for macOS: chat with local LLMs, generate and edit images, make
  music, run agents, and serve an OpenAI-compatible API. Downloads models from
  Hugging Face in-app, with per-model profiles.
- **[IPSupport Code](https://ipsupport-llc.github.io/ipsupport-code/)** — your
  AI coding agent for real repositories: analyze, fix, test, report.

Runs in LLMTray as a local chat LLM.

## License

Distributed under NVIDIA's **OpenMDW License Agreement v1.1** (same license
as the base model) — see `LICENSE` in this repo.

## Usage

```
pip install mlx-lm
python -m mlx_lm.generate --model roman220220/nemotron-30b-a3b-gptq3bit-g64 --prompt "..."
```

Produced with [this project's GPTQ pipeline](https://github.com/rromenskyi/quant-ternary)
(`poc/gptq_stock_convert.py`, `poc/run_pipeline.sh`) — see
`docs/session_findings_2026-09-11.md` section 7p for the full methodology
and formula-verification writeup.
