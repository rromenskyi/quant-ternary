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
  - mixed-precision
---

# Nemotron-3.5-Lightning-30B-A3B — GPTQ, JANG-style component-type bit allocation (MLX)

> ### ▶ Run it locally in [LLMTray](https://www.ipsupport.us/llmtray/)
> A free, native macOS app for local AI on Apple Silicon — chat, images,
> music, agents and an OpenAI-compatible API. Point it at this model; nothing
> leaves your Mac.
>
> [![Download LLMTray](https://img.shields.io/badge/Download-LLMTray%20for%20Mac-2f7d4f?style=for-the-badge&logo=apple&logoColor=white)](https://github.com/ipsupport-llc/llmtray/releases/latest/download/LLMTray-Full.dmg)
> [![GitHub stars](https://img.shields.io/github/stars/ipsupport-llc/llmtray?style=for-the-badge&logo=github)](https://github.com/ipsupport-llc/llmtray)

GPTQ-calibrated (Hessian-based error compensation) mixed-precision
quantization of NVIDIA's Nemotron-3.5-Lightning-30B-A3B (NemotronH hybrid
Mamba2 + Attention + MoE, 52 layers), using a **component-type bit
allocation reverse-engineered from JANG_2L-CRACK's** published
`config.json` (a third-party MLX release of this same base model), applied
through this project's own GPTQ Hessian calibration instead of JANG's
apparent naive RTN (their README describes their quantization only as
"3.73-bit affine (MLX)" with no mention of calibration).

- **Method**: bits assigned purely by COMPONENT TYPE, uniform across every
  layer — no layer position or per-layer score involved:

  | component | bits |
  |---|---|
  | attention q/k/v/o_proj | 8 |
  | mamba in_proj/out_proj | 6 |
  | MoE shared_experts (up+down) | 8 |
  | MoE routed up (`switch_mlp.fc1`) | 4 |
  | MoE routed down (`switch_mlp.fc2`) | 3 |
  | embeddings | 6 |
  | lm_head | 8 |

  The strategy: generous precision where a component is cheap in total
  parameter count (attention, shared experts — only a handful of these vs.
  128 routed experts per MoE block), and the least precision on the single
  largest pool of parameters (routed experts, ~93% of the model) — with an
  asymmetric up(4)/down(3) split within routed experts.

- **Group size**: 64. **Average**: 4.237 bits/weight.
- **Size on disk**: ~16 GB.

## Perplexity (wikitext-2-raw, 20×512-token chunks, quarter-in offset)

| model | PPL |
|---|---|
| bf16 (full precision, reference) | 5.11 |
| **this model (JANG bit-allocation + our GPTQ calibration)** | **5.24** |
| JANG_2L-CRACK (3rd party, same bit-allocation strategy, presumed naive RTN) | 5.43 |
| this project's `mixed_3_6` positional recipe | 5.81 |
| this project's `smart_3_6` sensitivity recipe | 5.90 |
| this project's uniform 3-bit GPTQ | 6.24 |
| naive RTN uniform 3-bit (no calibration) | 6.54 |

**This is this project's best result of the session**, and it beats the
third-party JANG_2L-CRACK release it's modeled after — direct evidence
that GPTQ Hessian-based calibration adds real value ON TOP OF a good bit
allocation strategy, not just as a substitute for one. The bit allocation
(which layer/component gets how many bits) and the calibration method (how
values are chosen at a fixed bit-width) are independent levers; this
result combines a strong allocation (borrowed) with a validated
calibration method (this project's own, verified bit-exact against MLX's
affine kernel) rather than treating them as alternatives.

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
python -m mlx_lm.generate --model roman220220/nemotron-30b-a3b-gptq-jang-component --prompt "..."
```

Produced with [this project's GPTQ pipeline](https://github.com/rromenskyi/quant-ternary)
(`poc/gptq_stock_convert.py --quant-recipe-mode component --component-recipe jang`,
`poc/mlx_convert_recipe.py --mode component`) — see
`docs/session_findings_2026-09-11.md` section 7q for the full methodology,
including how JANG's bit allocation was reverse-engineered from its public
`config.json` and the real `switch_mlp.fc1`/`fc2` naming bug this project
found and fixed along the way.
