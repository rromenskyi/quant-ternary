# lfm2-quant

GPTQ -> MLX (JANG: bits by component) for Liquid AI's LFM2 / LFM2.5
checkpoints, dense (`lfm2`) or mixture-of-experts (`lfm2_moe`). One
resumable pipeline, run on a CUDA pod:

    VARIANT=lfm2-8b SETUP=1 ./poc/lfm2_mlx_pipeline.sh
    VARIANT=lfm2-8b PUBLISH=1 ./poc/lfm2_mlx_pipeline.sh   # + HF upload (HF_TOKEN)

Steps: deps, data (`../qwen35-quant/poc/make_data.py`: wikitext-2, a Python
code corpus for calibration, the Python standard library as held-out code),
HF snapshot, bf16 and vendor-MLX reference perplexities, GPTQ layer by layer
(`gptq_lfm2.py`), the calibrated checkpoint (`assemble_lfm2.py`), MLX with
GPTQ's exact codes (`convert_lfm2.py`), check (`check_lfm2.py`: text and code
perplexity, one chat answer), card, upload. The GPTQ helpers are
`qwen35-quant/poc/gptq_qwen35.py`'s.

Releases (2026-10-11; perplexity in MLX, 40 x 512 tokens, vs bf16):

| Model | Repo | Recipe attn/conv/mlp/experts | Size | PPL text / code | Liquid AI MLX-4bit |
|---|---|---|---|---|---|
| LFM2.5-8B-A1B | [roman220220/LFM2.5-8B-A1B-gptq-mlx-jang-q4](https://huggingface.co/roman220220/LFM2.5-8B-A1B-gptq-mlx-jang-q4) | 8/6/6/4 | 5.03 GB | +2.2 % / +2.6 % | 4.83 GB, +9.2 % / +6.7 % |
| LFM2.5-8B-A1B | [roman220220/LFM2.5-8B-A1B-gptq-mlx-jang](https://huggingface.co/roman220220/LFM2.5-8B-A1B-gptq-mlx-jang) | 8/6/6/3 | 4.06 GB | +9.0 % / +7.8 % | 〃 |
| LFM2.5-2.6B | [roman220220/LFM2.5-2.6B-gptq-mlx-jang](https://huggingface.co/roman220220/LFM2.5-2.6B-gptq-mlx-jang) | 8/6/4 | 1.78 GB | −0.5 % / +1.7 % | none |
| LFM2.5-1.2B-Instruct | [roman220220/LFM2.5-1.2B-Instruct-gptq-mlx-jang](https://huggingface.co/roman220220/LFM2.5-1.2B-Instruct-gptq-mlx-jang) | 8/6/4 | 0.80 GB | +4.5 % / +5.7 % | 0.66 GB, +18.9 % / +16.3 % |
| LFM2.5-1.2B-Thinking | [roman220220/LFM2.5-1.2B-Thinking-gptq-mlx-jang](https://huggingface.co/roman220220/LFM2.5-1.2B-Thinking-gptq-mlx-jang) | 8/6/4 | 0.80 GB | +3.8 % / +2.5 % | 0.66 GB, +14.8 % / +11.6 % |

Embeddings 8-bit everywhere; the MoE router stays bf16. Mirrored on Kaggle
(`romanromenskyi/lfm2-5-*-ipsupport`). Raw results: `results/`. How to run
it: [`docs/RUNBOOK.md`](docs/RUNBOOK.md). Measurements and lessons:
[`docs/FINDINGS.md`](docs/FINDINGS.md). Cards: `cards/`.
