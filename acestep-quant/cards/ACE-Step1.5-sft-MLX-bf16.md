---
license: mit
base_model: ACE-Step/acestep-v15-sft
pipeline_tag: text-to-audio
tags:
  - mlx
  - mlx-audio
  - ace-step
  - music-generation
  - text-to-music
  - apple-silicon
  - quantized
---

# ACE-Step 1.5 sft for MLX: bfloat16 (reference)

> ### ▶ Run it locally in [LLMTray](https://www.ipsupport.us/llmtray/)
> A free, native macOS app for local AI on Apple Silicon — chat, images,
> music, agents and an OpenAI-compatible API. Point it at this model; nothing
> leaves your Mac.
>
> [![Download LLMTray](https://img.shields.io/badge/Download-LLMTray%20for%20Mac-2f7d4f?style=for-the-badge&logo=apple&logoColor=white)](https://github.com/ipsupport-llc/llmtray/releases/latest/download/LLMTray-Full.dmg)
> [![GitHub stars](https://img.shields.io/github/stars/ipsupport-llc/llmtray?style=for-the-badge&logo=github)](https://github.com/ipsupport-llc/llmtray)

[ACE-Step 1.5](https://huggingface.co/ACE-Step/Ace-Step1.5) **sft** (ACE-Step team, MIT, trained on licensed and royalty-free data) is
converted for Apple Silicon here: songs with **sung, intelligible lyrics** from a style description, 48 kHz stereo.

The DiT as published by ACE-Step, converted to mlx-audio's layout, no quantization.

| | this repo |
|---|---|
| DiT | 4.5 GB |
| total download (DiT + Qwen3-Embedding text encoder + VAE) | ~7.4 GB |
| speed, M5 | ~36 s for 30 s of music (50 steps, CFG) |
| peak memory | ~10.6 GB |

## sft vs turbo

ACE-Step 1.5 ships two DiTs. We compared them on 3 songs × 4 seeds: pop, rock and a ballad, 30 s each. The score is Whisper large-v3-turbo's
word error rate against the lyrics that were asked for (lower = the lyrics come through more clearly).

| | mean WER | tracks with WER < 0.3 | time / 30 s |
|---|---|---|---|
| turbo (mlx-community 4-bit + LM 1.7B) | 0.56 | 3 / 12 | ~22 s |
| **sft (bf16, this family)** | **0.22** (pop 0.02) | 6 / 12 | ~36 s |
| official PyTorch/MLX pipeline, sft | 0.17 | 9 / 12 | ~114 s |

By ear: turbo's mix sounds fuller and more like a finished song, while sft puts the voice upfront and gets the words across. On electronic music
sft sounded best.

## Quantization

Teacher-forced error against bf16: every decoder call of the bf16 run, fed to each variant.

| DiT | velocity error | condition-encoder error | DiT size |
|---|---|---|---|
| 8-bit (RTN) | 2.1% | 0.5% | 2.7 GB |
| 4-bit RTN (plain round-to-nearest) | 13.9% | 5.0% | 1.65 GB |
| **4-bit GPTQ** | **10.4% (−25%)** | **2.7% (−47%)** | 1.65 GB |

Lyrics intelligibility, the same 3 songs × 4 seeds (Whisper WER, lower is better; 12 tracks, so differences of a few points are noise):

| DiT | mean WER | pop | rock | ballad | tracks with WER < 0.3 | time / 30 s | peak memory |
|---|---|---|---|---|---|---|---|
| bf16 | 0.22 | 0.02 | 0.22 | 0.43 | 6 / 12 | 36 s | 10.6 GB |
| 8-bit | 0.24 | 0.02 | 0.26 | 0.45 | 6 / 12 | 49 s | 8.6 GB |
| GPTQ 4-bit | 0.29 | 0.06 | 0.35 | 0.45 | 5 / 12 | 36 s | 7.5 GB |

Pick bf16 or 8-bit if you have the memory, GPTQ 4-bit on a 16 GB Mac.

## ⚠️ How to run it: the stock mlx-audio settings sing gibberish

Pulled through mlx-audio's defaults (branch [`pc/add-ace`](https://github.com/Blaizzy/mlx-audio/tree/pc/add-ace), commit `1e8264a`), the sft DiT
produces cacophony. Two causes, both found and measured
([findings](https://github.com/rromenskyi/quant-ternary/blob/main/acestep-quant/docs/FINDINGS.md)):

1. **The 5 Hz LM hints.** mlx-audio feeds its planner's audio codes into the DiT, and that breaks the sft model. sft doesn't need a planner: run it
   with `use_lm=False`. It is also ~15 s faster.
2. **The unconditional CFG branch.** mlx-audio encodes all-zero text there. The official pipeline uses the trained `null_condition_emb`. The patch
   below uses it.

```python
import mlx.core as mx, mlx.nn as nn
import mlx_audio.utils as u
from mlx_audio.tts import load

# a local folder doesn't tell mlx-audio which model class it is
pick = u.get_model_class
u.get_model_class = lambda model_type, model_name, category, model_remapping: pick("ace_step", None, category, model_remapping)
model = load("roman220220/ACE-Step1.5-sft-MLX-bf16")

class NullAwareEncoder(nn.Module):          # CFG's unconditional branch, as the official pipeline does it
    def __init__(self, inner, null):
        super().__init__(); self.inner, self._null = inner, null
    def __call__(self, text_hidden_states=None, lyric_hidden_states=None, **kw):
        out, mask = self.inner(text_hidden_states=text_hidden_states, lyric_hidden_states=lyric_hidden_states, **kw)
        if not mx.any(text_hidden_states).item() and not mx.any(lyric_hidden_states).item():
            out = mx.broadcast_to(self._null.astype(out.dtype), out.shape)
        return out, mask
model.encoder = NullAwareEncoder(model.encoder, model.null_condition_emb)

result = list(model.generate(
    text="upbeat pop song with female vocals, bright synths, driving beat",
    lyrics="[Verse]\nCity lights are calling out my name\n...",
    duration=30.0, seed=1, vocal_language="en",
    use_lm=False, num_steps=50, guidance_scale=7.0, shift=1.0, guidance_interval=1.0, cfg_type="apg",
))[-1]   # result.audio: [2, samples] at 48 kHz
```

## The IPSupport local-AI stack

Local-first AI tools for macOS by [IPSupport](https://www.ipsupport.us) — nothing
leaves your Mac.

- **[LLMTray](https://www.ipsupport.us/llmtray/)** — your local AI
  workstation for macOS: chat with local LLMs, generate and edit images, make
  music, run agents, and serve an OpenAI-compatible API. Downloads models from
  Hugging Face in-app, with per-model profiles.
- **[IPSupport Code](https://ipsupport-llc.github.io/ipsupport-code/)** — your
  AI coding agent for real repositories: analyze, fix, test, report.

LLMTray uses this model for in-chat music generation.

## License

Licensed under the **MIT License**, the same license as the ACE-Step 1.5 model. See [`LICENSE`](LICENSE).

This is a modified version of [ACE-Step/acestep-v15-sft](https://huggingface.co/ACE-Step/acestep-v15-sft), together with the ACE-Step 1.5
VAE and its Qwen3-Embedding-0.6B text encoder:

- converted to mlx-audio's MLX layout: the decoder's Conv1d weights transposed and the rotary caches dropped;
- the DiT stored in bfloat16.
