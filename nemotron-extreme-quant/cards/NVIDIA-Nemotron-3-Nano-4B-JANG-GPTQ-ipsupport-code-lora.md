---
base_model: nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16
license: other
license_name: openmdw-1.1
license_link: LICENSE
tags:
  - mlx
  - quantized
  - nemotron
  - lora
  - tool-calling
  - coding-agent
---

# NVIDIA-Nemotron-3-Nano-4B, ipsupport-code LoRA (MLX)

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

## License

Distributed under NVIDIA's **OpenMDW License Agreement v1.1** (same
license as the base model) -- see `LICENSE` in this repo.

## Usage

```
pip install mlx-lm
python -m mlx_lm.generate --model roman220220/NVIDIA-Nemotron-3-Nano-4B-JANG-GPTQ-ipsupport-code-lora --prompt "..."
```
