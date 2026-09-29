---
license: other
license_name: openmdw-1.1
license_link: LICENSE
base_model: nvidia/NVIDIA-NemotronLabs-VoiceChat-11B
library_name: mlx
pipeline_tag: audio-to-audio
language:
  - en
tags:
  - mlx
  - quantized
  - gptq
  - nemotron
  - audio
  - speech-to-speech
  - full-duplex
  - apple-silicon
---

<p align="center">
  <img src="llmtray-banner.png" alt="LLMTray" width="100%">
</p>

# NemotronLabs VoiceChat 11B: GPTQ 3-bit LLM + 8-bit speech, MLX

> ### ▶ Run it locally in [LLMTray](https://www.ipsupport.us/llmtray/)
> LLMTray is a free, native macOS app for local AI on Apple Silicon. It does
> chat, images, music, agents, voice and an OpenAI-compatible API. Point it at
> this model; nothing leaves your Mac.
>
> [![Download LLMTray](https://img.shields.io/badge/Download-LLMTray%20for%20Mac-2f7d4f?style=for-the-badge&logo=apple&logoColor=white)](https://github.com/ipsupport-llc/llmtray/releases/latest/download/LLMTray-Full.dmg)
> [![GitHub stars](https://img.shields.io/github/stars/ipsupport-llc/llmtray?style=for-the-badge&logo=github)](https://github.com/ipsupport-llc/llmtray)

This is an MLX build of NVIDIA's full-duplex speech-to-speech model,
[nvidia/NVIDIA-NemotronLabs-VoiceChat-11B](https://huggingface.co/nvidia/NVIDIA-NemotronLabs-VoiceChat-11B).
The model listens, transcribes, answers with text and speaks on one
continuous 80 ms timeline.

The language model inside is **Nemotron-Nano-9B-v2**, a hybrid of Mamba-2 and
attention with 56 layers. It is **GPTQ-quantized to 3 bits (group size 64)**,
calibrated on real duplex conversations. The FastConformer encoder, the TTS
backbone and the TTS mixture head are quantized to 8 bits; the codec stays
in bf16. It is the successor of
[NemotronLabs-VoiceChat-11B-gptq-mlx-3bit](https://huggingface.co/roman220220/NemotronLabs-VoiceChat-11B-gptq-mlx-3bit),
which keeps all of the speech parts in bf16.

The goal was an 80 ms frame in close to 80 ms on a base M5 MacBook, for
LLMTray's Voice Lab.

- **6.4 GB of weights**; 8.0 GB MLX peak memory in a session (9.1 GB for the all-bf16-speech 3-bit build, 10.6 GB for the 4-bit build).
- Every answer is correct on our 20 spoken test questions.
- Each step reads its weights in full and is limited by memory bandwidth, so
  fewer bits mean a faster step. The LLM step drops from 62 ms to about 40 ms
  compared with the 4-bit build. The TTS drops from 24 to 19 ms compared with
  the bf16-speech 3-bit build.

## Recipe

| Part | Precision | Notes |
|---|---|---|
| LLM: Mamba `in_proj`/`out_proj`, MLP `up_proj`/`down_proj`, attention `q/k/v/o_proj` (all 56 layers) | **3-bit GPTQ**, g64 | Layer-by-layer sequential GPTQ in MLX-CUDA, directly onto MLX's affine grid; bf16 scales/biases |
| `lm_head`, `function_head`, `embed_tokens` | 4-bit, g64 | |
| LLM norms, Mamba `A`/`D`/conv, other small tensors | bf16 | |
| Perception (FastConformer encoder) Linears | 8-bit RTN, g64 | Its convolutions stay bf16 |
| TTS Gemma3 backbone Linears, MoG head | 8-bit RTN, g64 | Every TTS step reads them in full, and the step is limited by memory bandwidth |
| RNNT decoder/joint, TTS subword encoder, NeMo codec | bf16 | Unchanged from the source conversion |

**Calibration.** The calibration set is 256 real duplex conversations. Each
spoken clip was streamed through the **bf16 model's own streaming session**:
system-prompt prefill, then the user's speech, then silence until the reply
ended. The LLM's fused per-frame input was captured at every step. GPTQ
therefore sees the same inputs as the model does at inference: speech
embeddings, user-transcript tokens and the model's own replies.

## Measurements (base M5, 26 GB, fanless, 2026-09-28)

The test set is 20 spoken questions, each with keywords the answer must
contain. The model's spoken reply is transcribed by an independent Whisper.
Timings are milliseconds per 80 ms duplex frame, averaged over every frame of
every question.

| Build | Perception | LLM | TTS | Codec | **Total** | RTF | User WER | Keyword accuracy | Reply WER |
|---|---|---|---|---|---|---|---|---|---|
| `mlx-community/…-4bit`, upstream code | 62.7 | 62.3 | 47.8 | 9.5 | **185.2** | 2.32 | 0.020 | 0.95 | 0.053 |
| [`…-gptq-mlx-3bit`](https://huggingface.co/roman220220/NemotronLabs-VoiceChat-11B-gptq-mlx-3bit) + fork | 16.9 | 40.4 | 24.2 | 6.4 | 89.4 | 1.12 | 0.020 | 1.00 | 0.043 |
| **this model** + [ipsupport-llc/mlx-audio `llmtray`](https://github.com/ipsupport-llc/mlx-audio/tree/llmtray) | 15.9 | 40.7 | **18.8** | 6.5 | **83.3** | 1.04 | 0.020 | **1.00** | 0.049 |

Two things together give the 185 → 83 ms:

- **This checkpoint** accounts for the LLM column.
- **Our mlx-audio fork** accounts for the other columns:
  - perception runs in bf16, with the conformer compiled for its steady state;
  - the TTS mixture head computes only the sampled mixture;
  - TTS code generation and the codec step are compiled.

On a base M5 the model is **just short of real time**: about 1.04× (the two
builds were measured back to back, each after a 5-minute cool-down). The
base M5 is fanless. Minutes of full load throttle its GPU, and a 3-bit run
measured 84 ms cold and 117 ms hot, so long conversations run slower than
these figures. M5 Pro and Max chips have more memory bandwidth and fans;
we haven't measured them.

## Usage

**LLMTray:** open **Voice Lab**, which is opt-in in Settings, and pick this
model. LLMTray unloads the chat model while you talk.

**Python:** use our mlx-audio fork, which has the speedups above:

```bash
pip install "mlx-audio[stt] @ git+https://github.com/ipsupport-llc/mlx-audio.git@llmtray"
```

```python
from mlx_audio.sts import load

model = load("roman220220/NemotronLabs-VoiceChat-11B-gptq-mlx-mixed")

# Offline: a WAV in, text + speech out
output = model.generate("input.wav")
print(output.text)

# Full duplex: keep one session and feed 16 kHz mono PCM as it arrives
session = model.create_duplex_session()
for chunk in microphone_chunks:
    for event in session.push_audio(chunk, sample_rate=16_000):
        if event.kind == "assistant_text_delta":
            print(event.delta, end="", flush=True)
        elif event.kind == "audio":
            play(event.samples, event.sample_rate)  # 22.05 kHz
session.flush()
```

Input is 16 kHz mono. Output is 22,050 Hz mono in the model's built-in voice.
The model is English only. The checkpoint uses the same config layout as the
mlx-community conversions (`mlx_runtime_config_version: 2`) and ships its
tokenizer, so a downloaded copy loads with `HF_HUB_OFFLINE=1`.

## Method / code

- Quantization pipeline, calibration capture, evaluation and the full results:
  [rromenskyi/quant-ternary/voicechat-quant](https://github.com/rromenskyi/quant-ternary/tree/main/voicechat-quant)
  (`calib_capture.py`, `gptq_llm.py`, `splice_llm.py`, `vc_eval.py`,
  `docs/FINDINGS.md`).
- Model code: [ipsupport-llc/mlx-audio](https://github.com/ipsupport-llc/mlx-audio/tree/llmtray)
  (`mlx_audio/sts/models/nemotron_voicechat`).
- Source conversion: `mlx-community/NemotronLabs-VoiceChat-11B-bf16` and
  `-4bit`, built from NVIDIA's checkpoint at source revision
  `c5d3b70183b6bb9d7553590e111b05685049751c`.

## Intended use and limitations

This model inherits NVIDIA's intended use, limitations and responsible-AI
notes. See the base model card and the notices shipped in this repo
(`bias.md`, `explainability.md`, `privacy.md`, `safety.md`).

Quantizing to 3 bits changes the weights. We checked quality only on the
English test set described above. Validate it for your own use case.

## The IPSupport local-AI stack

Local-first AI tools for macOS by [IPSupport](https://www.ipsupport.us) — nothing
leaves your Mac.

- **[LLMTray](https://www.ipsupport.us/llmtray/)** is your local AI
  workstation for macOS. Chat with local LLMs, generate and edit images, make
  music, run agents, and serve an OpenAI-compatible API. It downloads models
  from Hugging Face in-app, with per-model profiles.
- **[IPSupport Code](https://ipsupport-llc.github.io/ipsupport-code/)** is
  your AI coding agent for real repositories: analyze, fix, test, report.

LLMTray runs this model in Voice Lab for full-duplex local voice conversation.

## License

This model is licensed under the **OpenMDW License Agreement, version 1.1**,
the same license as the base model. See [`LICENSE`](LICENSE), which keeps
NVIDIA's copyright notice.

Modified from [nvidia/NVIDIA-NemotronLabs-VoiceChat-11B](https://huggingface.co/nvidia/NVIDIA-NemotronLabs-VoiceChat-11B):

- converted to MLX;
- the language model (Nemotron-Nano-9B-v2) was GPTQ-quantized to 3 bits at
  group size 64;
- the LM head, function head and embeddings were quantized to 4 bits;
- the perception encoder's, TTS backbone's and TTS mixture head's Linear layers were quantized to 8 bits (round-to-nearest, group size 64);
- the codec and the remaining parts were left in bf16.

The weights and configuration files in this repo are therefore modified
versions of the original, not the original files.

NemotronLabs VoiceChat is © 2026 NVIDIA CORPORATION & AFFILIATES and is
released under OpenMDW 1.1. NVIDIA's model-card notices (`bias.md`,
`explainability.md`, `privacy.md`, `safety.md`) are included unchanged.
