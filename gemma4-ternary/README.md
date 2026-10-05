# gemma4-ternary

Ternary distillation of Gemma 4 12B onto MLX's stock 2-bit grid: can the
QAT master be re-trained to work with weights in {-s, 0, +s}, and how close
does it get to the 4-bit builds?

The ternary grid fits MLX's 2-bit affine format exactly (bias = -s,
scale = s, codes {0, 1, 2}), so a model that trains well runs in stock MLX
and LLMTray with no fork or custom kernel: 2.25 bits/weight at group 128,
about 3.6 GB for the 12B. That differs from PrismML's Bonsai 2, which needs
its own llama.cpp/MLX forks.

## Pipeline (`poc/`)

| Step | Script | Resumes by |
|---|---|---|
| master | `snapshot_download` of `google/gemma-4-12B-it-qat-q4_0-unquantized` | HF's own resume |
| gen_eval, gen (optional, `GEN_TOKENS` > 0) | `gen_teacher_data.py` in the vLLM venv: the master answers smoltalk prompts in the exact inference format, both thinking modes | per shard |
| data | `prepare_data.py`: packs the sources (`gen:<dir>:<w>` teacher replies, or HF datasets) into 2048-token rows; eval from held-out prompts | deterministic stream, missing shards only |
| teacher | `teacher_topk.py`: the master's top-32 log-probs per token, ahead of training | per shard |
| train | `train_ternary.py`: reference evals, the ternary start, then KL distillation with a straight-through estimator | checkpoints (weights, SRAdamW moments, data cursor, RNG) |

```bash
# on the GPU host (DGX Spark), inside tmux:
WORK=~/ternary TOKENS=50000000 bash poc/ternary_pipeline.sh
bash poc/stop.sh          # graceful stop: training saves, then exits
# rerun the pipeline command to continue where it stopped

# on the Mac:
python poc/ternary_dashboard.py --host dgx --work '~/ternary'   # http://localhost:8422
```

Run 3 (teacher-generated data, starting from run 2's weights):

```bash
GEN_TOKENS=16000000 TOKENS=21000000 DATA_DIR=~/ternary/data_v3 TEACHER_DIR=~/ternary/teacher_v3 \
RUN=~/ternary/run_v3_gen_init901 LR=1e-4 \
SOURCES="gen:$HOME/ternary/gen:0.8 HuggingFaceFW/fineweb-edu:sample-10BT:train:text:0.2" \
EVAL_SOURCES="gen:$HOME/ternary/gen_eval:1" \
TRAIN_EXTRA="--init-ckpt ~/ternary/run_qw150_lr1e-4/ckpt/step_000901 --eval-every 50" \
bash poc/ternary_pipeline.sh
```

The vLLM venv (`$WORK/.venv-vllm`, `uv pip install vllm --torch-backend=cu130 datasets ninja`) is
separate from the training venv; vLLM serves the text-only view
`masters/gemma-4-12B-text` (see FINDINGS).

Every knob is an environment variable of `ternary_pipeline.sh` (`MASTER_ID`,
`TOKENS`, `GROUP`, `LR`, `BATCH`, `ACCUM`, `TOPK`, `SOURCES`, `EVAL_SOURCES`,
`TRAIN_EXTRA`, `DATA_DIR`, `TEACHER_DIR`, `RUN`, `GEN_TOKENS`, `VLLM_PY`, `TEXT_MASTER`).
Export a checkpoint for the Mac with `export_mlx.py` (on the host) and
`splice_mlx.py` (on the Mac).

Findings: [`docs/FINDINGS.md`](docs/FINDINGS.md).
