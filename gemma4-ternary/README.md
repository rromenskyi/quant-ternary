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
| data | `prepare_data.py`: 30% fineweb-edu, 70% smoltalk chat (train split); eval from smoltalk's test split | deterministic stream, missing shards only |
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

Every knob is an environment variable of `ternary_pipeline.sh` (`MASTER_ID`,
`TOKENS`, `GROUP`, `LR`, `BATCH`, `ACCUM`, `TOPK`, `SOURCES`, `EVAL_SOURCES`,
`TRAIN_EXTRA`).

Findings: [`docs/FINDINGS.md`](docs/FINDINGS.md).
