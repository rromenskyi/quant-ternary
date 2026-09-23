# flux2-quant

Quantization of [FLUX.2 klein 4B](https://huggingface.co/black-forest-labs/FLUX.2-klein-4B)
(Apache 2.0; text-to-image + editing) for [mflux](https://github.com/filipstrand/mflux)
on Apple Silicon, for LLMTray's image generation / editing tools. Sibling of
[`zimage-quant`](../zimage-quant) (same GPTQ method, planned) and
[`gemma4-quant`](../gemma4-quant).

Status: **research**. Anatomy, memory, and first quantization mixes are
measured in [docs/FINDINGS.md](docs/FINDINGS.md). No release yet.

- `poc/klein_stage_memory.py`: per-stage peak memory and time.
- `poc/klein_quant_mix.py`: per-component quantization mixes vs bf16.
