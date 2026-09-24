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
  - jang
  - lora
  - tool-calling
  - coding-agent
---

# Nemotron-3.5-Lightning-30B-A3B, JANG bit-allocation + GPTQ calibration + ipsupport-code coding-agent LoRA (MLX)

NVIDIA's **Nemotron-3.5-Lightning-30B-A3B**, quantized to MLX using the
**JANG component-type bit allocation** (reverse-engineered from
[dealignai/Nemotron-3.5-Lightning-30B-A3B-JANG_2L-CRACK](https://huggingface.co/dealignai/Nemotron-3.5-Lightning-30B-A3B-JANG_2L-CRACK)'s
published `config.json`) applied through this project's own **GPTQ
Hessian calibration** (sequential, block-by-block) instead of JANG's
presumed naive RTN — and merged with a **LoRA fine-tune for
[ipsupport-code](https://github.com/ipsupport-llc/ipsupport-code)**, a
local terminal coding agent, trained to fix real tool-calling reliability
failures observed in production usage.

## Bit allocation (JANG recipe)

| component | bits |
|---|---|
| attention q/k/v/o_proj | 8 |
| mamba in_proj/out_proj | 6 |
| MoE shared_experts (up+down) | 8 |
| MoE routed up (`switch_mlp.fc1`) | 4 |
| MoE routed down (`switch_mlp.fc2`) | 3 |
| embeddings | 6 |
| lm_head | 8 |

**Group size**: 64. **Average**: 4.237 bits/weight, ~16.4GB on disk.

## What's different about THIS release vs. this project's earlier JANG-recipe release

- **Calibration corpus**: mixed wikitext-2-raw + real ipsupport-code
  tool-calling transcripts (not wikitext-only), so quantization error is
  compensated for the actual agentic/tool-calling text distribution this
  model is meant to run, not just generic prose.
- **Sequential calibration**: each block's activations are captured
  against the model with all prior blocks already quantized in place
  (not one-shot against the pristine model).
- **Built on the LoRA-merged model**: the base model was first fine-tuned
  (QLoRA, `lora_r=32`, targeting attention q/k/v/o + MoE expert
  up/down_proj) on real + synthetic ipsupport-code tool-call examples,
  merged back into full precision, and THEN GPTQ-quantized -- so the
  quantization is calibrated against the actual post-fine-tune model, not
  the vanilla base.

## Perplexity

| | wikitext-2-raw | ipsupport-code tool-call data |
|---|---|---|
| bf16 (reference) | 5.11 | -- |
| vanilla JANG-recipe, one-shot, wikitext-only calib (this project's earlier release) | 5.24 | 2.464 |
| **this release** (LoRA-merged + sequential + mixed calib) | 5.28 | **2.379** |
| JANG_2L-CRACK (3rd party, presumed naive RTN) | 5.43 | -- |

Slightly worse on generic wikitext prose (+0.8%), meaningfully better on
the actual target domain (-3.5%) -- the intended trade for a model meant
to run as a coding-agent backend, not a general chatbot.

## Inference speed (M-series Mac, single-request decode)

Component-type mixed-precision recipes (this release, the earlier
wikitext-only release, and JANG_2L-CRACK) all run at essentially the
same speed (~42-43 tok/s) regardless of calibration method or claimed
average bpw, because runtime speed is governed by the bit-allocation
STRUCTURE (which components get 8/6/4/3 bits), not by the specific
quantized values. For comparison, a uniform 3-bit release of this same
base model runs at ~65-68 tok/s. See this project's
[session findings, §7r](https://github.com/rromenskyi/quant-ternary/blob/main/nemotron-extreme-quant/docs/session_findings_2026-09-11.md)
for the full root-cause writeup.

## Tool calling is trained in — intended behaviour, not a glitch

This model was fine-tuned for **coding agents** ([ipsupport-code](https://github.com/ipsupport-llc/ipsupport-code)): the training data is tool-call-heavy, so the model has a strong habit of acting through tools. That habit shows up when you give it tools in a general chat. Measured on the -MTP release (same fine-tune as the non-MTP one), with an image-generation tool in the prompt (temperature 1.0, 20 samples each):

| prompt | calls the tool | stock Nemotron-3.5 / this quant without the LoRA |
|---|---|---|
| "привет" (hello) | 10/20 | 0/20 |
| "как дела? что умеешь?" (how are you, what can you do) | 14/20 | — |
| "нарисуй кота в космосе" (draw a cat in space, should call) | 20/20 | — |

Occasionally (3/20) it also closes its `<think>` block twice and repeats the answer.

**This is what the fine-tune is for.** In a coding agent, "reach for a tool" is the right default: read the file, run the command, edit the code. It is the wrong default for small talk in a general chat UI.

What helps if you use it as a chat model with tools attached:

- **A system rule** such as *"Only call a tool when the user's latest message explicitly asks you to perform that action; for greetings, small talk and questions, answer in text"* plus a tool description that says the same. This cut tool calls on questions from 14/20 to 4–6/20. Greetings stay at 7–11/20.
  [LLMTray](https://github.com/ipsupport-llc/llmtray) sends this rule by default.
- **Lower temperature**: 0.6 cut greetings to 1/10 in one run.
- **Don't attach tools the conversation doesn't need.**
- **For plain chat without tools**, the base quant without the LoRA doesn't have this habit.

Details and methodology: [FINDINGS §2.5](https://github.com/rromenskyi/quant-ternary/blob/main/nemotron-extreme-quant/docs/FINDINGS.md).

## License

Distributed under NVIDIA's **OpenMDW License Agreement v1.1** (same
license as the base model) — see `LICENSE` in this repo.

## Usage

```
pip install mlx-lm
python -m mlx_lm.generate --model roman220220/Nemotron-3.5-Lightning-30B-A3B-JANG-GPTQ-ipsupport-code-lora --prompt "..."
```

Produced with [this project's GPTQ pipeline](https://github.com/rromenskyi/quant-ternary)
(`poc/gptq_stock_convert.py --quant-recipe-mode component --component-recipe jang --sequential --extra-calib-file <flattened tool-call data>`,
`poc/mlx_convert_recipe.py --mode component`). The LoRA adapter used to
produce the fine-tuned base is at
[roman220220/ipsupport-code-nemotron-lora](https://huggingface.co/roman220220/ipsupport-code-nemotron-lora).
