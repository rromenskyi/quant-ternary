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
  - mtp
  - speculative-decoding
---

# Nemotron-3.5-Lightning-30B-A3B, JANG bit-allocation + GPTQ + ipsupport-code LoRA + MTP (MLX)

> ### 🚀 Easiest way to run this model: [**LLMTray**](https://github.com/ipsupport-llc/llmtray)
>
> A free, native macOS menu bar app for running local LLMs on Apple
> Silicon. Point it at this model, flip on **Settings → Advanced →
> Experimental → "Use MTP-enabled mlx-lm"**, and self-speculative
> decoding (see below) just works — no terminal, no `pip install`.
>
> [![Download LLMTray](https://img.shields.io/badge/Download-LLMTray-0A84FF?style=for-the-badge&logo=apple&logoColor=white)](https://github.com/ipsupport-llc/llmtray/releases/latest)
> [![GitHub stars](https://img.shields.io/github/stars/ipsupport-llc/llmtray?style=for-the-badge&color=yellow)](https://github.com/ipsupport-llc/llmtray)

Same lineage as
[roman220220/Nemotron-3.5-Lightning-30B-A3B-JANG-GPTQ-ipsupport-code-lora](https://huggingface.co/roman220220/Nemotron-3.5-Lightning-30B-A3B-JANG-GPTQ-ipsupport-code-lora)
(NVIDIA's **Nemotron-3.5-Lightning-30B-A3B**, **JANG** component-type bit
allocation, GPTQ Hessian calibration, merged with the
[ipsupport-code](https://github.com/ipsupport-llc/ipsupport-code) coding-agent
LoRA) — **plus the source checkpoint's Multi-Token-Prediction (MTP) head**,
which every prior release of this model (this project's and third-party
mlx-community/JANG_2L-CRACK releases alike) silently lost.

## Why prior releases have no MTP head

`nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16` ships a DeepSeek-style
MTP head (~1.34B params, ~4% of total size) for self-speculative decoding.
HF transformers' `NemotronHForCausalLM` has
`_keys_to_ignore_on_load_unexpected = [r"mtp.*"]`, so **any**
`transformers.AutoModelForCausalLM.from_pretrained()` call — the LoRA merge
step and GPTQ calibration, both PyTorch-based — silently drops these
weights on load, before quantization or MLX conversion ever see them.
Confirmed directly: the earlier `...-ipsupport-code-lora` release has 0 of
729 keys starting with `mtp.`. This is a `transformers`/pipeline-plumbing
issue, not something specific to the JANG recipe or this project.

Fixed here by extracting `mtp.*` from the untouched bf16 source before the
LoRA merge/GPTQ ever run, and splicing it back into the finished MLX model
afterward with mlx_lm's own model classes (no transformers involved at
that point) — see
[ipsupport-llc/mlx-lm@nemotron-h-mtp](https://github.com/ipsupport-llc/mlx-lm/tree/nemotron-h-mtp)'s
`extract_mtp_weights.py`/`inject_mtp_weights.py`, documented in this
project's [docs/FINDINGS.md §4](https://github.com/rromenskyi/quant-ternary/blob/main/nemotron-extreme-quant/docs/FINDINGS.md).

## Bit allocation (JANG recipe, backbone + MTP tiers)

| component | bits |
|---|---|
| attention q/k/v/o_proj | 8 |
| mamba in_proj/out_proj | 6 |
| MoE shared_experts (up+down) | 8 |
| MoE routed up (`switch_mlp.fc1`) | 4 |
| MoE routed down (`switch_mlp.fc2`) | 3 |
| embeddings | 6 |
| lm_head | 8 |
| **MTP attention / shared / fusion** | **8** |
| **MTP routed up/down** | **6** |

Group size: 64. Backbone average: 4.237 bits/weight. MTP head average:
6.589 bits/weight (plain round-to-nearest, not GPTQ-calibrated — see
caveats below). ~17.4GB backbone + ~1.1GB MTP head on disk.

The MTP head deliberately gets a higher bit-width tier than the backbone's
most aggressive one (`moe_routed_up`/`down` at 4/3-bit): a degraded draft
head only costs *speculative-decoding accept rate*, never correctness (see
below), but a badly-quantized head would defeat the entire point of having
it. ~4% of total size buys meaningfully better accept rate.

## Calibration

One-shot, wikitext-2-raw only (not the ipsupport-code-mixed, sequential
calibration the `...-ipsupport-code-lora` release above used) — this
release's focus is verifying the MTP-preservation pipeline end-to-end, not
re-optimizing the backbone's calibration. Perplexity has not been
re-measured for this specific release; see the linked release above for
that methodology and numbers on the same recipe without MTP.

## MTP / self-speculative decoding: what's included and what isn't

**Included and verified**: the MTP head loads (`model.mtp is not None`),
and a reference generation driver
(`nemotron_h_mtp_generate_step` in
[ipsupport-llc/mlx-lm@nemotron-h-mtp](https://github.com/ipsupport-llc/mlx-lm/tree/nemotron-h-mtp)'s
`mlx_lm/generate.py`) produces **bit-exact output vs. plain greedy
decoding** — verified against both synthetic weights and a real
`nemotron_h` MTP checkpoint, at this exact quantization (group_size=64,
component recipe). A degraded/wrong draft is only ever rejected by the
verify pass; it cannot produce a wrong final token.

**Also included**: `mlx_lm.generate`/`mlx_lm.server`/[LLMTray](https://github.com/ipsupport-llc/llmtray)
now auto-dispatch to `nemotron_h_mtp_generate_step` for free — `stream_generate`
routes there automatically for a single-sequence, greedy (temp=0) request with
no logits processors / KV quantization, and falls back to plain decoding
otherwise, so nothing breaks for requests outside that subset. Requires
installing `ipsupport-llc/mlx-lm@nemotron-h-mtp` (stock `mlx-lm` still loads
this model fine for ordinary generation; the MTP weights just sit unused).

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

Ordinary generation (MTP head present but unused, stock `mlx-lm` is fine):

```
pip install mlx-lm
python -m mlx_lm.generate --model roman220220/Nemotron-3.5-Lightning-30B-A3B-JANG-GPTQ-ipsupport-code-lora-MTP --prompt "..."
```

Self-speculative decoding (needs the patched fork):

```
pip install "git+https://github.com/ipsupport-llc/mlx-lm.git@nemotron-h-mtp"
```
```python
from mlx_lm.utils import load
from mlx_lm.generate import nemotron_h_mtp_generate_step
model, tokenizer = load("roman220220/Nemotron-3.5-Lightning-30B-A3B-JANG-GPTQ-ipsupport-code-lora-MTP")
for token, logprobs, from_draft in nemotron_h_mtp_generate_step(prompt_tokens, model, max_tokens=256):
    ...
```

Produced with [this project's GPTQ pipeline](https://github.com/rromenskyi/quant-ternary)
(`poc/run_pipeline.sh --quant-recipe-mode component --component-recipe jang --group-size 64 --lora-adapter roman220220/ipsupport-code-nemotron-lora --mlx-lm-git "git+https://github.com/ipsupport-llc/mlx-lm.git@nemotron-h-mtp"`).
The LoRA adapter is at
[roman220220/ipsupport-code-nemotron-lora](https://huggingface.co/roman220220/ipsupport-code-nemotron-lora).
