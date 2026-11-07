---
license: mit
base_model: ResembleAI/chatterbox
pipeline_tag: text-to-speech
language:
  - ar
  - da
  - de
  - el
  - en
  - es
  - fi
  - fr
  - he
  - hi
  - it
  - ja
  - ko
  - ms
  - nl
  - "no"
  - pl
  - pt
  - ru
  - sv
  - sw
  - tr
  - zh
tags:
  - mlx
  - mlx-audio
  - chatterbox
  - text-to-speech
  - voice-cloning
  - apple-silicon
  - quantized
---

<p align="center">
  <img src="llmtray-banner.png" alt="LLMTray" width="100%">
</p>

# Chatterbox Multilingual V3 for MLX: 8-bit T3

> ### ▶ Run it locally in [LLMTray](https://www.ipsupport.us/llmtray/)
> A free, native macOS app for local AI on Apple Silicon — chat, images,
> music, agents and an OpenAI-compatible API. Point it at this model; nothing
> leaves your Mac.
>
> [![Download LLMTray](https://img.shields.io/badge/Download-LLMTray%20for%20Mac-2f7d4f?style=for-the-badge&logo=apple&logoColor=white)](https://github.com/ipsupport-llc/llmtray/releases/latest/download/LLMTray-Full.dmg)
> [![GitHub stars](https://img.shields.io/github/stars/ipsupport-llc/llmtray?style=for-the-badge&logo=github)](https://github.com/ipsupport-llc/llmtray)

[Chatterbox Multilingual V3](https://huggingface.co/ResembleAI/chatterbox) (Resemble AI, MIT) speaks 23 languages and
clones a voice from a short clip. This repo is its MLX conversion with **T3** (the 0.5B Llama that turns text into
speech tokens) quantized to **8 bits**. The rest is unchanged: S3Gen (speech tokens to audio) and the voice encoder
stay fp32. It also adds the **built-in voice** (`conds.safetensors`), which the multilingual MLX conversion lacks, and an optional
**fast decoder**: Chatterbox Turbo's distilled S3Gen (`meanflow/`), 4x faster than the 10-step one with no audible
difference in our blind test.

## Why 8-bit T3

Measured on a base M5 Mac (26 GB) against the fp32 conversion, one 3.5–4 s sentence. The Mac was under memory
pressure (swap in use), so single timings varied up to 2x; the ranges are three runs each, and the gap between
the two held in every run.

| | T3 tokens/s | T3 | S3Gen | total | peak memory |
|---|---|---|---|---|---|
| fp32 (mlx-community/chatterbox-multilingual-v3) | 14–26 | 3.9–7.1 s | 3.3–4.0 s | 8.0–10.9 s | 4.1 GB |
| **this repo** | **38–51** | **1.8–2.4 s** | 3.2–3.6 s | **5.0–6.0 s** | **3.5 GB** |

- **T3 at 8 bits**: 2–3x faster. The published weights give the same audio as quantizing on the fly (equal within
  1e-4 per sample in our check); by ear the 8-bit and fp32 T3 were left to compare.
- **4 bits** was no faster than 8.
- **S3Gen stays fp32**: in bf16 it was both slower and audibly different.
- **Download**: 1.32 GB for this repo instead of 2.71 GB (the speech tokenizer, 0.49 GB, comes separately in both
  cases).

## The fast decoder (`meanflow/`)

S3Gen, which turns speech tokens into audio, was the slow half: 10 flow-matching steps with CFG, about 0.6 s of compute
per 1 s of speech. `meanflow/s3gen_meanflow.safetensors` is Chatterbox Turbo's distilled S3Gen (2 steps, no CFG). It
takes the same speech tokens:

| decoder | 3.7 s phrase | blind test (2 ru + 2 en phrases) |
|---|---|---|
| 10 steps (standard) | 2.1 s | no difference heard |
| **Turbo meanflow, 2 steps** | **0.5 s** | no difference heard |

With it, and a T3 loop that skips a CPU-GPU sync per token, a whole sentence takes 0.24–0.31 s per second of speech
on a base M5 (fp32: 2–3). mlx-audio's standard loader ignores the subfolder, so the model loads as usual without it.
LLMTray uses it like this:

```python
from mlx_audio.tts.models.chatterbox_turbo.models.s3gen import S3Gen
fast = S3Gen(meanflow=True)
fast.load_weights(list(mx.load("meanflow/s3gen_meanflow.safetensors").items()), strict=False)
wav = fast.inference(speech_tokens, ref_dict, n_cfm_timesteps=2)[0]   # tokens and ref_dict from the multilingual model
```

The code is in LLMTray's `runtime/llmtray_tts_runner.py`.

## Use with mlx-audio

Needs an mlx-audio with Chatterbox multilingual V3 (`--lang_code`): mlx-community converted the base with mlx-audio
0.4.7; this repo was built and tested with the [ipsupport-llc/mlx-audio](https://github.com/ipsupport-llc/mlx-audio)
fork (commit 67f89de).

```bash
pip install -U mlx-audio
mlx_audio.tts.generate --model roman220220/chatterbox-multilingual-v3-mlx-8bit \
    --text "Привет! Чем могу помочь?" --lang_code ru
```

- **Voice**: with no `--ref_audio`, the built-in voice is used (that is what our tests ran). Pass a clip to clone
  a voice; Resemble suggests about 10 s. Use only your own voice or one you have permission to use.
- **Speech tokenizer**: mlx-audio fetches it separately at load: `mlx-community/S3TokenizerV2`, Apache-2.0, converted
  from CosyVoice2-0.5B.
- **Russian**: Chatterbox reads Russian best from stressed text (U+0301 after the stressed vowel). LLMTray adds the
  marks itself with [ruaccent](https://github.com/Den4ikAI/ruaccent) (MIT).

## How it was made

- **Source**: `mlx-community/chatterbox-multilingual-v3`, loaded with mlx-audio.
- **Quantization**: `nn.quantize(model.t3.tfmr, group_size=64, bits=8)`, saved with `"quantization"` in
  `config.json`, so mlx-audio's own loader reads it.
- **Built-in voice**: `conds.safetensors` from `mlx-community/chatterbox-fp16`, which is Resemble's `conds.pt`; their
  multilingual model loads the same file.
- **Fast decoder**: `scripts/extract_meanflow.py` takes `mlx-community/chatterbox-turbo-fp16`'s S3Gen without its
  speech tokenizer and renames it to the module's names. It matches the full Turbo model's S3Gen parameter for
  parameter.
- **Script and measurements**: in [quant-ternary/chatterbox-quant](https://github.com/ipsupport-llc/quant-ternary).

## The IPSupport local-AI stack

Local-first AI tools for macOS by [IPSupport](https://www.ipsupport.us) — nothing
leaves your Mac.

- **[LLMTray](https://www.ipsupport.us/llmtray/)** — your local AI
  workstation for macOS: chat with local LLMs, generate and edit images, make
  music, run agents, and serve an OpenAI-compatible API. Downloads models from
  Hugging Face in-app, with per-model profiles.
- **[IPSupport Code](https://ipsupport-llc.github.io/ipsupport-code/)** — your
  AI coding agent for real repositories: analyze, fix, test, report.

LLMTray uses this model to read chat answers aloud.

## Disclaimer

This repository contains a quantized conversion of a third-party model,
provided **"as is", without warranty of any kind**, express or implied.
IPSupport LLC did not train the model and is **not responsible for its
content, outputs or behavior, or for any damage, loss or liability** arising
from its use. You are responsible for evaluating the model and for how you
use it, including compliance with applicable laws and with the base model's
license and terms.

## License

Licensed under the **MIT License**, the same license as the base model — see [`LICENSE`](LICENSE).

Modified from [ResembleAI/chatterbox](https://huggingface.co/ResembleAI/chatterbox) (Multilingual V3, `t3_mtl23ls_v3`),
through [mlx-community/chatterbox-multilingual-v3](https://huggingface.co/mlx-community/chatterbox-multilingual-v3): T3
quantized to 8 bits (group size 64) for MLX, and the built-in voice conditionals added from
[mlx-community/chatterbox-fp16](https://huggingface.co/mlx-community/chatterbox-fp16). The weights and configuration files
in this repo are therefore modified versions of the original, not the original files.

`meanflow/s3gen_meanflow.safetensors` is from [ResembleAI/chatterbox-turbo](https://huggingface.co/ResembleAI/chatterbox-turbo)
(MIT), through [mlx-community/chatterbox-turbo-fp16](https://huggingface.co/mlx-community/chatterbox-turbo-fp16), whose card
lists Apache-2.0: the speech tokenizer was removed and the weights were renamed. Use it under the terms of both.
