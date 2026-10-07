---
license: apache-2.0
base_model:
  - microsoft/FrogNano-4B-2609
  - roman220220/FrogNano-4B-2609-gptq-mlx-jang
pipeline_tag: image-text-to-text
library_name: mlx
tags:
  - mlx
  - quantized
  - gptq
  - jang
  - qwen3_5
  - lora
  - tool-calling
  - agent
  - ipsupport-code
---

<p align="center">
  <img src="llmtray-banner.png" alt="LLMTray" width="100%">
</p>

# FrogNano-4B-2609 for IPSupport Code, GPTQ JANG 8/6/4 (MLX)

> ### ▶ Run it locally in [LLMTray](https://www.ipsupport.us/llmtray/)
> A free, native macOS app for local AI on Apple Silicon — chat, images,
> music, agents and an OpenAI-compatible API. Point it at this model; nothing
> leaves your Mac.
>
> [![Download LLMTray](https://img.shields.io/badge/Download-LLMTray%20for%20Mac-2f7d4f?style=for-the-badge&logo=apple&logoColor=white)](https://github.com/ipsupport-llc/llmtray/releases/latest/download/LLMTray-Full.dmg)
> [![GitHub stars](https://img.shields.io/github/stars/ipsupport-llc/llmtray?style=for-the-badge&logo=github)](https://github.com/ipsupport-llc/llmtray)

[microsoft/FrogNano-4B-2609](https://huggingface.co/microsoft/FrogNano-4B-2609)
— Microsoft's compact coding agent (Qwen3.5-4B, RL post-trained on
software-engineering tasks) — fine-tuned with a LoRA for
**[IPSupport Code](https://ipsupport-llc.github.io/ipsupport-code/)**, our
coding agent, merged, and quantized for MLX with the same GPTQ 8/6/4 recipe
as [FrogNano-4B-2609-gptq-mlx-jang](https://huggingface.co/roman220220/FrogNano-4B-2609-gptq-mlx-jang):
**3.5 GB**. The LoRA teaches the agent's tools and their call format
(`{"action", "params"}` per tool) and when *not* to call one.

## The fine-tune

- **Data:** IPSupport Code's own sessions (one goal = one conversation, with
  its system prompt and its 8 tools: `file`, `run`, `git`, `web`, `help`,
  `calc`, `done`, `agent`), malformed calls cut out and the corrected call
  kept; 225 of them are plain replies with the tools available. Plus 243
  synthetic conversations (Russian / English) for the tools and actions real
  sessions rarely use. 729 conversations for training, 25 held out. The
  dataset is private (real sessions).
- **LoRA:** attention only (full attention q/k/v/o and the Gated DeltaNet
  in_proj_qkv / in_proj_z / out_proj; the MLP untouched, it holds the coding
  skill), r 16, alpha 32, lr 1e-4, 2 epochs, sequences up to 16K tokens;
  12.4 M trainable parameters. Loss on the assistant turns only, after their
  think block: the model still reasons before it acts.
- **Merged** into the bf16 weights, then GPTQ-calibrated and converted like
  the base release.

## Measurements

| | Held-out agent loss | PPL wikitext-2 (bf16) | PPL code (bf16) |
|---|---|---|---|
| FrogNano | 0.805 | 12.369 | 3.166 |
| **+ this LoRA** | **0.599** | **12.166** | **3.171 (+0.15 %)** |

Coding perplexity held; the agent's own turns got much more likely.

This MLX build against the merged bf16 model: wikitext-2 PPL 12.308
(+1.2 %), code 3.224 (+1.7 %).

**Behaviour in the agent's setup** (its system prompt and tools, sampling at
temperature 1.0 / top-p 0.95, both models as MLX 8/6/4 builds): the first
move on 25 held-out goals, 3 samples each —

| | Valid first step (known tool + action, params an object, or a plain reply) | Malformed calls |
|---|---|---|
| FrogNano 8/6/4 | 64 / 75 | 0 |
| **this model** | **74 / 75** | 0 |

On greetings and small talk with the tools available ("привет", "thanks",
"what can you do?", "ок, понял" … 8 messages × 10 samples), a plain reply is
right and a tool call is not:

| | Tool calls | of them `done` (ends the task) | other tools |
|---|---|---|---|
| FrogNano 8/6/4 | 4 / 80 | 3 | 1 |
| this model | 7 / 80 | 2 | **5** |

**Known weakness:** the fine-tune made the model a little more eager to act:
on a bare "ок, понял" or "как дела?" with no task it sometimes starts
working (lists or reads files). Behind an agent that sends a real goal this
rarely matters; in a plain chat with tools on, expect it now and then.

## Speculative decoding (MTP head)

The Qwen3.5 multi-token-prediction head is kept, GPTQ 4-bit, in
`model-mtp.safetensors` (68 MB). The LoRA moved the hidden states the head
reads, so the head was **retrained on the merged model** (backbone frozen;
the agent's conversations plus wikitext and code). With the
[ipsupport-llc/mlx-lm](https://github.com/ipsupport-llc/mlx-lm) fork it drafts
tokens and the model checks them: the same output as without it, faster.

| First draft accepted (greedy) | wikitext-2 test | Python stdlib |
|---|---|---|
| this model's head | 77.6 % | 85.0 % |

Stock `mlx-lm` drops the head on load. LLMTray uses it when "Speculative
decoding (MTP)" is on in the model's profile.

## Usage

Made for IPSupport Code: point the agent at an OpenAI-compatible server
running this model (LLMTray, or `mlx_lm.server`). Microsoft's settings:
temperature **0.6**, repetition penalty 1.0, up to 8,192 tokens per turn,
context ~131K.

```bash
pip install "git+https://github.com/ipsupport-llc/mlx-lm.git"
mlx_lm.server --model roman220220/FrogNano-4B-2609-gptq-mlx-jang-ipsupport-code-lora --port 8080
```

Vision is inherited from Qwen3.5-4B as in the base release (needs the fork);
it wasn't part of the fine-tune. Read the
[base model card](https://huggingface.co/microsoft/FrogNano-4B-2609) for its
scope and limitations, which apply here unchanged.

## Method and code

Pipeline, scripts and measurements:
[rromenskyi/quant-ternary](https://github.com/rromenskyi/quant-ternary),
`qwen35-quant/lora/` (`lora_pipeline.sh`, `train_lora.py`, `merge_lora.py`,
`train_mtp.py`, `eval_lora.py`) and `qwen35-quant/docs/FINDINGS.md` §4b.

## The IPSupport local-AI stack

Local-first AI tools for macOS by [IPSupport](https://www.ipsupport.us) — nothing
leaves your Mac.

- **[LLMTray](https://www.ipsupport.us/llmtray/)** — your local AI
  workstation for macOS: chat with local LLMs, generate and edit images, make
  music, run agents, and serve an OpenAI-compatible API. Downloads models from
  Hugging Face in-app, with per-model profiles.
- **[IPSupport Code](https://ipsupport-llc.github.io/ipsupport-code/)** — your
  AI coding agent for real repositories: analyze, fix, test, report.

## License

Licensed under the **Apache License 2.0**, as stated in the base model's
card (FrogNano is derived from Qwen/Qwen3.5-4B, Apache 2.0; the base
repository's metadata says MIT, which Apache 2.0 also satisfies) — see
[`LICENSE`](LICENSE).

Modified from [microsoft/FrogNano-4B-2609](https://huggingface.co/microsoft/FrogNano-4B-2609):
fine-tuned with a LoRA (merged), the MTP head retrained, quantized with GPTQ
to 8-bit attention, 6-bit linear attention and 4-bit MLP weights (8-bit
embeddings and vision tower), the MTP head to 4-bit, and converted to MLX.
The weights and configuration files in this repo are therefore modified
versions of the original, not the original files.
