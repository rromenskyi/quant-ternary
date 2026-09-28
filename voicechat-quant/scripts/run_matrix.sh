#!/usr/bin/env bash
# Run vc_eval.py for every variant line in a matrix file, one fresh process per
# variant (so peak memory and caches are per-variant), then score them all.
#
#   scripts/run_matrix.sh eval/matrix_rtn.txt results/rtn [extra vc_eval args...]
#
# Matrix file: one variant per line, "<name> <vc_eval flags...>", '#' comments.
# Refuses to start while LLMTray's LLM server or music runner is using the Mac.
set -euo pipefail
HERE="$(cd "$(dirname "$0")/.." && pwd)"
MATRIX="$1"; OUT="$2"; shift 2
PY="${PY:-/Users/roman220/gh/mlx-audio-fork/.venv/bin/python}"
export HF_HUB_OFFLINE=1

busy() { pgrep -f mlx_lm.server >/dev/null || pgrep -f llmtray_music_runner >/dev/null; }

mkdir -p "$OUT"
while read -r name flags; do
  [[ -z "$name" || "$name" == \#* ]] && continue
  if [[ -f "$OUT/$name/run.json" ]]; then echo "skip $name (done)"; continue; fi
  if busy; then echo "LLMTray server/music runner is running -- stopping matrix before $name"; exit 3; fi
  echo "=== $name: $flags"
  # shellcheck disable=SC2086
  "$PY" "$HERE/scripts/vc_eval.py" --out "$OUT/$name" $flags "$@" </dev/null 2>&1 \
    | grep --line-buffered -v -E "Warning|You are using a model of type" | tee "$OUT/$name.log" | tail -3 || echo "FAILED $name"
done < "$MATRIX"

if [[ "${SCORE:-1}" == 1 ]]; then
  "$PY" "$HERE/scripts/score.py" "$OUT"/*/ --table "$OUT/table.md"
fi
