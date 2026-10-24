---
license: other
license_name: openmdw-1.1
license_link: LICENSE
base_model: nvidia/NVIDIA-Nemotron-3.5-Bolt-E4B-BF16
pipeline_tag: image-text-to-text
tags:
  - gguf
  - llama.cpp
  - gemma4
  - multimodal
  - vision
  - audio
  - nvidia
  - nemotron
---

<p align="center">
  <img src="llmtray-banner.png" alt="LLMTray" width="100%">
</p>

# Nemotron-3.5-Bolt-E4B — vision + audio mmproj for llama.cpp

> ### ▶ For LLMTray/Mac, use the MLX sibling instead
> This is the **GGUF mmproj** (llama.cpp / ollama / Arc / CPU). If you use
> [LLMTray](https://www.ipsupport.us/llmtray/) — IPSupport's local AI app for
> Apple Silicon, which runs MLX — use
> [roman220220/Nemotron-3.5-Bolt-E4B-gptq-mlx-jang](https://huggingface.co/roman220220/Nemotron-3.5-Bolt-E4B-gptq-mlx-jang) instead.
>
> [![Download LLMTray](https://img.shields.io/badge/Download-LLMTray%20for%20Mac-2f7d4f?style=for-the-badge&logo=apple&logoColor=white)](https://github.com/ipsupport-llc/llmtray/releases/latest/download/LLMTray-Full.dmg)
> [![GitHub stars](https://img.shields.io/github/stars/ipsupport-llc/llmtray?style=for-the-badge&logo=github)](https://github.com/ipsupport-llc/llmtray)

The vision + audio multimodal projector (mmproj) for NVIDIA's
[Nemotron-3.5-Bolt-E4B](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Bolt-E4B-BF16),
exported for `llama.cpp`'s `mtmd` runtime. Pair it with NVIDIA's own
official text GGUF
([NVIDIA-Nemotron-3.5-Bolt-E4B-4GB](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Bolt-E4B-4GB))
— no need to re-quantize the text model, NVIDIA already did that well.

**Text, vision *and* audio — not just text.** NVIDIA's own release
actively disables multimodal input in its recommended vLLM serving
command (`--limit-mm-per-prompt '{"image":0,"video":0,"audio":0}'`) and
states its post-training data modality as "Text" only — and every
other public GGUF/quant of this model found covers text only too. The
checkpoint still carries a full vision tower and Conformer-style audio
encoder, inherited from the Gemma 4 base architecture and never
stripped by NVIDIA's post-training. As far as checked, this is the
first public build where image and speech input actually work with
this model via `llama.cpp`.

990 MB, **F16, not quantized** — see "Honest notes."

## Usage

```bash
llama-mtmd-cli \
  -m Nemotron-3.5-Bolt-E4B-Q4_K_M.gguf \
  --mmproj mmproj-nemotron-bolt-e4b-f16.gguf \
  --image photo.jpg -p "Describe this image." --jinja

llama-mtmd-cli \
  -m Nemotron-3.5-Bolt-E4B-Q4_K_M.gguf \
  --mmproj mmproj-nemotron-bolt-e4b-f16.gguf \
  --audio speech.wav -p "Transcribe this audio exactly." --jinja
```

(`--jinja` is required — the model's chat template isn't one of
llama.cpp's built-in presets.)

## Does it work? Real end-to-end validation

Both run through the actual quantized text GGUF (NVIDIA's official
`Q4_K_M`) + this mmproj:

**Vision** (real photo, "Describe this image in one sentence."):
> A sleek black puppy lies on rustic wooden planks, looking directly at the viewer.

**Audio** (real 16kHz speech clip, "The quick brown fox jumps over the lazy dog", "Transcribe this audio exactly."):
> A quick brown fox jumped over the lazy dog

## Honest notes

- **Not quantized — stays F16.** Tried `Q8_0`: `llama-quantize` crashes
  on the audio tower's two small depthwise-conv weight tensors
  (`a.conv1d.0.weight`, `a.conv1d.1.weight`, an unusual 4D shape) with
  a size-mismatch assert — a real llama.cpp rough edge: the tensor is
  registered in the output GGUF with its original type *before* the
  quantization pass decides its final type, and for this tensor's
  shape the two don't agree on size. Worked around the crash with
  `--tensor-type 'a.conv1d.*.weight=f32'` — but even past the crash,
  `Q8_0` **corrupts the audio tower's actual output** (transcription
  degenerated into repeated meaningless syllables, not just a quality
  dip). Vision held up reasonably at `Q8_0`; audio did not, at all.
  Not worth shipping a quantized mmproj that works for one modality
  and breaks the other — F16 only.
- Vision encode: ~0.8s per image. Audio encode: ~0.4s for a 2.8s clip
  (both on an AMD FirePro D700, GCN1, via Vulkan — numbers will differ
  on other hardware).

## Method / code

Exported with `convert_hf_to_gguf.py --mmproj` from
[ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp) (no code
changes needed — upstream already maps `Gemma4ForConditionalGeneration`
to its `Gemma4VisionAudioModel` mmproj converter). Verified with
`llama-mtmd-cli` on real test hardware.

## The IPSupport local-AI stack

Local-first AI tools for macOS by [IPSupport](https://www.ipsupport.us) — nothing
leaves your Mac.

- **[LLMTray](https://www.ipsupport.us/llmtray/)** — your local AI
  workstation for macOS: chat with local LLMs, generate and edit images, make
  music, run agents, and serve an OpenAI-compatible API. Downloads models from
  Hugging Face in-app, with per-model profiles.
- **[IPSupport Code](https://ipsupport-llc.github.io/ipsupport-code/)** — your
  AI coding agent for real repositories: analyze, fix, test, report.

## Disclaimer

This repository contains a derived artifact of a third-party model,
provided **"as is", without warranty of any kind**, express or implied.
IPSupport LLC did not train the model and is **not responsible for its
content, outputs or behavior, or for any damage, loss or liability** arising
from its use. You are responsible for evaluating the model and for how you
use it, including compliance with applicable laws and with the base model's
license and terms.

## License

Licensed under the **OpenMDW License, version 1.1**, the same license as
NVIDIA's release — see [`LICENSE`](LICENSE). NVIDIA's model was itself
post-trained from `google/gemma-4-E4B-it`, which remains governed by the
[Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0); both
terms apply, per NVIDIA's own card.

Derived from [nvidia/NVIDIA-Nemotron-3.5-Bolt-E4B-BF16](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Bolt-E4B-BF16):
the vision and audio towers were converted to GGUF format (F16, no
quantization). The files in this repo are therefore a modified/derived
form of the original, not the original files.
