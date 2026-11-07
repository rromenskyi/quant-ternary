# chatterbox-quant

Chatterbox Multilingual V3 (Resemble AI, MIT) on MLX for LLMTray's spoken
answers: T3 quantized to 8 bits, the built-in voice added.

- `scripts/quantize_t3.py`: builds the published folder from
  mlx-community/chatterbox-multilingual-v3.
- `scripts/extract_meanflow.py`: the fast decoder (Turbo's meanflow S3Gen) as
  one file for `meanflow/`.
- `docs/FINDINGS.md`: speed per variant, what didn't work (bf16), the honest
  timing trap, Russian stress.
- `cards/`: the Hugging Face card and LICENSE.

Published: [roman220220/chatterbox-multilingual-v3-mlx-8bit](https://huggingface.co/roman220220/chatterbox-multilingual-v3-mlx-8bit).
