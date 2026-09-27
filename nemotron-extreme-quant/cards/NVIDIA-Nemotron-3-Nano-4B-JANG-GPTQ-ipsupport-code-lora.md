---
base_model: nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16
license: other
license_name: nvidia-nemotron-open-model-license
license_link: LICENSE
tags:
  - mlx
  - quantized
  - nemotron
  - lora
  - tool-calling
  - coding-agent
---

<p align="center">
  <img src="ipsupport-code-banner.png" alt="IPSupport Code" width="100%">
</p>

# NVIDIA-Nemotron-3-Nano-4B, ipsupport-code LoRA (MLX)

> ### ▶ A coding-agent model — built for [IPSupport Code](https://ipsupport-llc.github.io/ipsupport-code/)
> This model is fine-tuned for **IPSupport Code**, a local AI coding agent for
> real repositories (analyze · fix · test · report). It works best paired with
> that agent; tool-calling is trained in (see "Tool calling is trained in"
> below). You can also run it locally in [LLMTray](https://www.ipsupport.us/llmtray/).
>
> [![Download LLMTray](https://img.shields.io/badge/Download-LLMTray%20for%20Mac-2f7d4f?style=for-the-badge&logo=apple&logoColor=white)](https://github.com/ipsupport-llc/llmtray/releases/latest/download/LLMTray-Full.dmg)
> [![GitHub stars](https://img.shields.io/github/stars/ipsupport-llc/llmtray?style=for-the-badge&logo=github)](https://github.com/ipsupport-llc/llmtray)


`nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16` fine-tuned for
**[ipsupport-code](https://github.com/ipsupport-llc/ipsupport-code)**
(a local terminal coding agent), then quantized to 8-bit for MLX. This
release's exact recipe was arrived at through an extensive investigation
that overturned several of this project's own initial assumptions --
full details in this project's
[`docs/session_findings_2026-09-11.md`](https://github.com/rromenskyi/quant-ternary),
sections 7u-7v. Summary below.

## What changed from the first release, and why

**1. The `<think>` block wasn't closing for plain replies.** The
original 110-conversation training set had zero non-tool-call examples,
so the model never saw a demonstration of "no tool needed, just answer
directly." Fixed by adding ~125 synthetic examples covering greetings,
clarifying questions, and plain factual answers -- each with a short,
real `reasoning_content` (not empty), since the chat template only
renders a genuine `<think>...</think>` block when `reasoning_content`
is non-empty; without it, training data collapses to `<think></think>`
glued directly to the reply, a different token sequence than what
inference forces (`<think>\n`, expecting the model to close it).

**2. LoRA targeting BOTH attention and MLP modules together
catastrophically broke general code competency** (verified via
compile-and-run tests, not just PPL): C++ went from working
correctly (matching the un-fine-tuned base model) to failing to
compile on almost every sample -- missing `#include`s, mismatched
braces, invented library functions. Neither attention-only nor
MLP-only LoRA (tested via post-hoc ablation of a jointly-trained
adapter) reproduced this on their own; only training both together did,
regardless of rank or epoch count tested. This release's adapter
targets **only `q_proj/k_proj/v_proj/o_proj`** -- dropping
`up_proj/down_proj` entirely restores C++ competency to base-model
levels in bf16.

**3. This project's own GPTQ pipeline underperforms plain RTN
quantization for this model.** Across every bit-width tested (3, 6,
and 8-bit MLP, and near-uniform 8-bit everywhere), this project's
Hessian-calibrated GPTQ quantizer landed at the same ceiling
(~65% of unquantized C++ compile-pass rate) no matter how many bits
were given to any component, and no matter what calibration corpus was
used (wikitext alone vs wikitext+SFT-tool-calling text). Directly
comparing against `mlx_lm.convert`'s stock **naive round-to-nearest**
quantizer at the *same* bit-width showed RTN performing as well or
slightly better. Working theory: GPTQ's Hessian-based correction
optimizes weight reconstruction fidelity *for the calibration corpus's
activation distribution* -- since neither wikitext nor a tool-calling
dataset contains any C++, that correction has no signal to preserve
C++-specific representations, and may distort them more than
calibration-agnostic RTN would. **This release uses stock
`mlx_lm.convert -q --q-bits 8 --q-group-size 64` (no custom GPTQ)**
instead of this project's usual component-recipe pipeline.

## Recipe

- Base: `nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16`
- LoRA: `lora_r=32`, `lora_alpha=64`, targets `q_proj/k_proj/v_proj/o_proj`
  only (no MLP), same ipsupport-code tool-calling dataset (augmented
  with ~125 non-tool-call examples, see above)
- Quantization: stock `mlx_lm.convert`, uniform 8-bit, group_size=64
  (8.503 bits/weight, ~4.0GB from ~7.5GB bf16)

## Verified behavior

Casual replies close `<think>` correctly and answer directly (no more
empty-`content` responses):

```
[casual] 'hi'          -> closes </think>, answers directly
[casual] 'привет'      -> closes </think>, answers directly
[casual] 'thanks, bye' -> closes </think>, answers directly
```

Code generation (compile-and-run tested, 3 seeds x 3 temperatures,
C++ + Python): comparable to the un-fine-tuned base model's own
quantization-independent ceiling -- see the session findings doc for
the full comparison table across every recipe tried.

## Tool calling is trained in — intended behaviour, not a glitch

This model was fine-tuned for **coding agents** ([ipsupport-code](https://github.com/ipsupport-llc/ipsupport-code)) on tool-call-heavy data, so it has a strong habit of acting through tools. That is the point of the fine-tune: in a coding agent, "read the file / run the command" is the right default.

Given tools in a general chat UI, it may still call them for small talk. The 30B sibling, measured, calls an image tool on "hello" 10 of 20 times; this 4B model wasn't measured separately.

- A system rule ("only call a tool when the user explicitly asks for that action") and a lower temperature reduce it.
- For plain chat, use a model without this LoRA.

See [FINDINGS §2.5](https://github.com/rromenskyi/quant-ternary/blob/main/nemotron-extreme-quant/docs/FINDINGS.md).

## The IPSupport local-AI stack

Local-first AI tools for macOS by [IPSupport](https://www.ipsupport.us) — nothing
leaves your Mac.

- **[LLMTray](https://www.ipsupport.us/llmtray/)** — your local AI
  workstation for macOS: chat with local LLMs, generate and edit images, make
  music, run agents, and serve an OpenAI-compatible API. Downloads models from
  Hugging Face in-app, with per-model profiles.
- **[IPSupport Code](https://ipsupport-llc.github.io/ipsupport-code/)** — your
  AI coding agent for real repositories: analyze, fix, test, report.

Run this model in LLMTray, or wire it into IPSupport Code as the agent's backend.

## License

Governed by the **NVIDIA Nemotron Open Model License**, the same license as the base model — see [`LICENSE`](LICENSE) (official text: [nvidia.com](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-nemotron-open-model-license/)).

Licensed by NVIDIA Corporation under the NVIDIA Nemotron Model License.

Modified from [nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16): fine-tuned with a LoRA (`lora_r=32`, `lora_alpha=64`, `q_proj/k_proj/v_proj/o_proj` only) on ipsupport-code tool-calling data, merged, and quantized to 8-bit (group size 64) with stock `mlx_lm.convert`.

`configuration_nemotron_h.py` and `modeling_nemotron_h.py` are unmodified copies from the base repo and are licensed under the Apache License 2.0 per their file headers — see [`LICENSE-APACHE-2.0.txt`](LICENSE-APACHE-2.0.txt).

## Usage

```
pip install mlx-lm
python -m mlx_lm.generate --model roman220220/NVIDIA-Nemotron-3-Nano-4B-JANG-GPTQ-ipsupport-code-lora --prompt "..."
```
