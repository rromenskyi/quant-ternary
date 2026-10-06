# qwen35-quant

GPTQ -> MLX (JANG: bits by component) for Qwen3.5-family checkpoints, dense
(`qwen3_5`) or mixture-of-experts (`qwen3_5_moe`), with the vision tower
kept. One resumable pipeline, run on a CUDA pod:

    VARIANT=ornith-35b SETUP=1 ./poc/qwen35_mlx_pipeline.sh
    VARIANT=ornith-35b PUBLISH=1 ./poc/qwen35_mlx_pipeline.sh   # + HF upload (HF_TOKEN)

Steps: deps, data (wikitext-2, a Python code corpus for calibration, the
Python standard library as held-out code), HF snapshot, HF bf16 reference
(text and code perplexity, vision features), GPTQ layer by layer
(`gptq_qwen35.py`), the calibrated checkpoint (`assemble_checkpoint.py`),
MLX with GPTQ's exact codes (`convert_mlx.py`), checks (`check_mlx.py`:
vision vs HF, an image question, both perplexities), card, upload. Presets
(`VARIANT`) set the model, the recipe and the repo; everything else is a
variable.

Releases:

| Model | Repo | Size | PPL text / code vs bf16 |
|---|---|---|---|
| FrogNano-4B-2609, 8/6/4 | [roman220220/FrogNano-4B-2609-gptq-mlx-jang](https://huggingface.co/roman220220/FrogNano-4B-2609-gptq-mlx-jang) | 3.2 GB | +1.5 % / +3.5 % |
| Ornith-1.5-35B-A3B, 8/6/6/3 | [roman220220/Ornith-1.5-35B-A3B-gptq-mlx-jang](https://huggingface.co/roman220220/Ornith-1.5-35B-A3B-gptq-mlx-jang) | 17.05 GB | +6.8 % / +15.7 % |
| Ornith-1.5-35B-A3B, experts 2/2/3 | [roman220220/Ornith-1.5-35B-A3B-gptq-mlx-jang-small](https://huggingface.co/roman220220/Ornith-1.5-35B-A3B-gptq-mlx-jang-small) | 14.36 GB | +21.1 % / +44.1 % |

How to run it: [`docs/RUNBOOK.md`](docs/RUNBOOK.md). Measurements and lessons:
[`docs/FINDINGS.md`](docs/FINDINGS.md). Cards:
`cards/`.
