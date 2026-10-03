# quant-ternary

IPSupport LLC's quantization and fine-tuning research: the pipelines, findings and model cards behind our open releases on Hugging Face.

**Whitepaper: [Any Model. Your Hardware. Your Data. (PDF)](IPSupport-quantization-whitepaper.pdf)**: how we compress and fine-tune open models to run on less hardware, faster, and on the client's own tasks, across all 23 public releases. Sources and build: [`docs/whitepaper/`](docs/whitepaper/).

## Projects

| Directory | Model |
|---|---|
| [`gemma4-quant/`](gemma4-quant/) | Gemma 4 (E2B, E4B, 12B, 26B-A4B, 31B): MLX and GGUF, QAT transport, MTP drafter |
| [`nemotron-extreme-quant/`](nemotron-extreme-quant/) | Nemotron 3.5 Lightning 30B-A3B and Nemotron 3 Nano 4B: JANG + GPTQ, coding-agent LoRA |
| [`voicechat-quant/`](voicechat-quant/) | NemotronLabs VoiceChat 11B: full-duplex voice on MLX |
| [`flux2-quant/`](flux2-quant/) | FLUX.2 klein 4B image model |
| [`zimage-quant/`](zimage-quant/) | Z-Image-Turbo image model |
| [`acestep-quant/`](acestep-quant/) | ACE-Step 1.5 music model |

Each project keeps its measurements in `docs/FINDINGS.md`. [`HF_CARD_TEMPLATE.md`](HF_CARD_TEMPLATE.md) is the template for our Hugging Face model cards.

IPSupport LLC · [ipsupport.us](https://ipsupport.us) · hello@ipsupport.us
