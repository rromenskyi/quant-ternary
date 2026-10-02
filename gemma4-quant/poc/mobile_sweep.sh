#!/bin/bash
# E2B for phones: the QAT q4_0 text decoder kept, the rest squeezed step by
# step; every build scored against the bf16 master (chat-framed windows).
#   bash mobile_sweep.sh   (on the pod; QAT=/workspace/qat)
set -euo pipefail
QAT="${QAT:-/workspace/qat}"
PY="${PY:-/workspace/venv/bin/python}"
HERE="$(cd "$(dirname "$0")" && pwd)"
SRC="$QAT/E2B-unq"
GGUF="$(ls "$QAT"/E2B-gguf/*q4_0*.gguf | grep -v mmproj | head -1)"
convert() {  # name ple embed other
  local out="$QAT/E2B-mob-$1"
  [ -f "$out/config.json" ] || "$PY" "$HERE/qat_aligned_convert.py" --hf-checkpoint-dir "$SRC" --output-dir "$out" --gguf "$GGUF" \
      --ple-bits "$2" --embed-bits "$3" --other-bits "$4" | tail -2
  echo "$1: $(du -s --block-size=1M "$out" | cut -f1) MB"
}
convert r0     6 6 8
convert p4     4 6 8
convert p4e4   4 4 8
convert p4e4o4 4 4 4
convert p3e4o4 3 4 4
"$PY" "$HERE/qat_eval.py" --text "$QAT/wiki.test.raw" --ref "$SRC" --windows 128 --chat --json "$QAT/E2B-mobile.json" \
    "$QAT/E2B-qat-mlx-r100" "$QAT"/E2B-mob-r0 "$QAT"/E2B-mob-p4 "$QAT"/E2B-mob-p4e4 "$QAT"/E2B-mob-p4e4o4 "$QAT"/E2B-mob-p3e4o4
echo MOBILE_SWEEP_DONE
