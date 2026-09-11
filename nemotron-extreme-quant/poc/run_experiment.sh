#!/usr/bin/env bash
# Run one Stage A experiment end to end: sequential quantize N blocks with
# salient-weight pinning, then evaluate quality against the original model.
#
# Usage:
#   poc/run_experiment.sh <num_blocks> [salient_fraction] [device]
#
# Examples:
#   poc/run_experiment.sh 5            # 5 blocks, 3% salient (default), cpu
#   poc/run_experiment.sh 20 0.05       # 20 blocks, 5% salient
#   poc/run_experiment.sh 10 0.03 mps   # 10 blocks, 3% salient, MPS
#
# Appends to poc/experiments_log.csv and writes
# poc/quality_report_<checkpoint-name>.md — nothing here overwrites a
# previous run's results.

set -euo pipefail

NUM_BLOCKS="${1:?usage: run_experiment.sh <num_blocks> [salient_fraction] [device]}"
SALIENT_FRACTION="${2:-0.03}"
DEVICE="${3:-cpu}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

MODEL_DIR="../cache/models/Nemotron-3-Nano-4B-BF16"
CHECKPOINT_NAME="Nemotron-3-Nano-4B-BF16-sequential-salient${SALIENT_FRACTION}-${NUM_BLOCKS}"
CHECKPOINT_DIR="../cache/models/${CHECKPOINT_NAME}"

echo "=== [$(date)] quantizing: ${NUM_BLOCKS} blocks, salient=${SALIENT_FRACTION}, device=${DEVICE} ==="
rm -rf "$CHECKPOINT_DIR"
../.venv/bin/python quantize_sequential.py \
    --model "$MODEL_DIR" \
    --output "$CHECKPOINT_DIR" \
    --num-blocks "$NUM_BLOCKS" \
    --salient-fraction "$SALIENT_FRACTION" \
    --device "$DEVICE"

echo "=== [$(date)] evaluating quality ==="
../.venv/bin/python eval_quality.py \
    --original "$MODEL_DIR" \
    --quantized "$CHECKPOINT_DIR"

echo "=== [$(date)] done: ${CHECKPOINT_NAME} — see poc/experiments_log.csv ==="
