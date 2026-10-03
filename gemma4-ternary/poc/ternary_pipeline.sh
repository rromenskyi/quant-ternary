#!/usr/bin/env bash
# Gemma 4 ternary distillation, end to end and resumable:
#   master -> data -> teacher -> train (reference evals + ternary start + training)
#
#   WORK=~/ternary TOKENS=50000000 bash ternary_pipeline.sh
#
# Rerun the same command after any stop: finished steps skip, the teacher
# continues shard by shard, training continues from its last checkpoint.
# Stop training gracefully with stop.sh (saves, then exits). HF_TOKEN is
# needed only while the master is still downloading; pass it through the
# environment, never a file.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
: "${WORK:=$HOME/ternary}"
: "${LOG_DIR:=$WORK/logs}"
# shellcheck source=/dev/null
source "$SCRIPT_DIR/pipeline_lib.sh" 2>/dev/null || source "$SCRIPT_DIR/../../gemma4-quant/poc/pipeline_lib.sh"
pipeline_init ternary "$@"

: "${PY:=$WORK/.venv/bin/python}"
: "${MASTER_ID:=google/gemma-4-12B-it-qat-q4_0-unquantized}"
: "${MASTER:=$WORK/masters/gemma-4-12B-qat-unq}"
: "${TOKENS:=50000000}"
: "${SEQ:=2048}"
: "${TOPK:=32}"
: "${BATCH:=2}"
: "${ACCUM:=4}"
: "${LR:=3e-5}"
: "${GROUP:=128}"   # MLX 2-bit group: 128 = 2.25 bits/weight, 64 = 2.5
: "${RUN:=$WORK/run}"
: "${SOURCES:=HuggingFaceFW/fineweb-edu:sample-10BT:train:text:0.3 HuggingFaceTB/smoltalk:all:train:messages:0.7}"
: "${EVAL_SOURCES:=HuggingFaceTB/smoltalk:all:test:messages:1}"
: "${TRAIN_EXTRA:=}"

master_complete () {
  "$PY" - "$MASTER" <<'PYEOF'
import json, sys, pathlib
d = pathlib.Path(sys.argv[1]); idx = d / "model.safetensors.index.json"
if idx.exists():
    ok = all((d / f).exists() for f in set(json.loads(idx.read_text())["weight_map"].values()))
else:  # single-file checkpoint; snapshot_download renames it into place only when complete
    ok = (d / "model.safetensors").exists()
sys.exit(0 if ok else 1)
PYEOF
}

if master_complete; then skip_step master "present"; else
  run_step master "download $MASTER_ID" "$PY" -c "
from huggingface_hub import snapshot_download
snapshot_download('$MASTER_ID', local_dir='$MASTER')"
fi

src_args=(); for s in $SOURCES; do src_args+=(--source "$s"); done
for s in $EVAL_SOURCES; do src_args+=(--eval-source "$s"); done
run_step data "tokenize $TOKENS tokens" "$PY" "$SCRIPT_DIR/prepare_data.py" \
  --master "$MASTER" --out "$WORK/data" --tokens "$TOKENS" --seq "$SEQ" "${src_args[@]}"

run_step teacher "teacher top-$TOPK" "$PY" "$SCRIPT_DIR/teacher_topk.py" \
  --master "$MASTER" --data "$WORK/data" --out "$WORK/teacher" --k "$TOPK"

# shellcheck disable=SC2086
run_step train "ternary distillation" "$PY" "$SCRIPT_DIR/train_ternary.py" \
  --master "$MASTER" --data "$WORK/data" --teacher "$WORK/teacher" --out "$RUN" \
  --tokens "$TOKENS" --batch "$BATCH" --accum "$ACCUM" --lr "$LR" --group "$GROUP" $TRAIN_EXTRA
