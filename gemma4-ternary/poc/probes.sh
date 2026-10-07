#!/usr/bin/env bash
# Short ablation runs forked from a finished checkpoint, one after another:
# which change lowers the eval KL fastest? Every probe trains on the same
# data with the same schedule; "control" changes nothing, so the others are
# read against it (all of them see that data for a second time).
#
#   INIT=~/ternary/run_v3_gen_init901/ckpt/step_001281 bash probes.sh
#
# Waits for any running train_ternary.py first. Rerun to continue: a probe
# resumes from its own checkpoints, a finished one exits at once. stop.sh
# stops the current probe and the queue.
set -u
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
: "${WORK:=$HOME/ternary}"
: "${PY:=$WORK/.venv/bin/python}"
: "${MASTER:=$WORK/masters/gemma-4-12B-qat-unq}"
: "${DATA_DIR:=$WORK/data_v3}"
: "${TEACHER_DIR:=$WORK/teacher_v3}"
: "${PROBE_TOKENS:=6000000}"
: "${INIT:?set INIT to the checkpoint the probes start from}"

# name | extra train_ternary.py arguments
PROBES=(
  "control|--lr 2e-5"
  "lr5e-5|--lr 5e-5"
  "attn4|--lr 2e-5 --affine-pattern self_attn --affine-bits 4 --affine-group 64"
  "g64|--lr 2e-5 --group 64"
)

while pgrep -f '[t]rain_ternary.py' >/dev/null; do sleep 60; done

for p in "${PROBES[@]}"; do
  name="${p%%|*}"; extra="${p#*|}"
  out="$WORK/run_probe_$name"
  echo "=== probe $name: $extra ($(date))"
  # shellcheck disable=SC2086
  "$PY" "$SCRIPT_DIR/train_ternary.py" --master "$MASTER" --data "$DATA_DIR" --teacher "$TEACHER_DIR" \
    --out "$out" --tokens "$PROBE_TOKENS" --init-ckpt "$INIT" --refs= --eval-every 50 --warmup 20 \
    $extra >> "$WORK/logs/probe_$name.log" 2>&1
  phase=$("$PY" -c "import json,sys; print(json.load(open(sys.argv[1]))['phase'])" "$out/status.json" 2>/dev/null)
  echo "=== probe $name: $phase ($(date))"
  [ "$phase" = "done" ] || { echo "queue stopped"; exit 1; }
done
echo "all probes done"
