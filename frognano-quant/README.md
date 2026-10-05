# frognano-quant

[microsoft/FrogNano-4B-2609](https://huggingface.co/microsoft/FrogNano-4B-2609)
(Qwen3.5-4B architecture) for MLX: GPTQ with a per-component bit recipe, the
vision tower kept.

Release: [roman220220/FrogNano-4B-2609-gptq-mlx-jang](https://huggingface.co/roman220220/FrogNano-4B-2609-gptq-mlx-jang)
— 3.5 GB, +1.5 % wikitext-2 perplexity. Measurements: [`docs/FINDINGS.md`](docs/FINDINGS.md).

## Pipeline

1. On a GPU pod (PyTorch, transformers >= 5, torch >= 2.5):
   `poc/gptq_qwen35.py --recipe attn=8,linear=6,mlp=4` writes a copy of the
   source checkpoint with the decoder on MLX's affine grid, plus
   `quant_recipe.json`. `poc/ppl_hf.py` measures it.
2. On the Mac, with the [ipsupport-llc/mlx-lm](https://github.com/ipsupport-llc/mlx-lm)
   fork (qwen3_5 vision): `poc/convert_mlx.py` converts with the same bits.
3. `poc/check_mlx.py`: vision features vs HF's (`ref_vision_feats.npy`, saved
   on the pod), an image question through the server's path, perplexity.

Card: [`cards/FrogNano-4B-2609-gptq-mlx-jang.md`](cards/FrogNano-4B-2609-gptq-mlx-jang.md).
