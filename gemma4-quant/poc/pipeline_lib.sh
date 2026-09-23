# Shared helpers for the gemma4-quant pipelines. Source it, don't run it:
#   source "$(dirname "$0")/pipeline_lib.sh"; pipeline_init mlx
#
# Every step writes one machine-readable line to $LOG_DIR/pipeline.log:
#   PIPE <unix-ts> <pipeline> <START|DONE|SKIP|FAIL> <step> [detail...]
# pipeline_dashboard.py reads those lines (plus the per-step logs and the
# calibration progress JSONs) to show where every pipeline is. Step output
# goes to $LOG_DIR/<pipeline>_<step>.log, tee'd to the terminal.

: "${WORK:=/workspace}"
: "${LOG_DIR:=$WORK/logs}"
PIPELINE_LOG="$LOG_DIR/pipeline.log"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

pipeline_init () {
  PIPELINE_NAME="$1"
  set -eE -o pipefail  # -E: the ERR trap must also fire inside functions
  mkdir -p "$LOG_DIR"
  _pipe_mark START "_pipeline" "$*"
  trap '_pipe_fail $?' ERR
  trap '[ -n "${_PIPE_CUR:-}" ] || _pipe_mark DONE _pipeline' EXIT
}

_pipe_mark () {
  local status="$1" step="$2"; shift 2
  echo "PIPE $(date +%s) $PIPELINE_NAME $status $step $*" >> "$PIPELINE_LOG"
}

_pipe_fail () {
  local rc="$1"
  # Inside $(...) the ERR trap is inherited (set -E): just propagate, the
  # parent shell's trap records the failure once.
  if [ "$BASH_SUBSHELL" -gt 0 ]; then exit "$rc"; fi
  if [ -n "${_PIPE_CUR:-}" ]; then
    _pipe_mark FAIL "$_PIPE_CUR" "exit=$rc"
    echo "!!! step $_PIPE_CUR failed (exit $rc) -- see $LOG_DIR/${PIPELINE_NAME}_${_PIPE_CUR}.log" >&2
  fi
  _pipe_mark FAIL _pipeline "exit=$rc"
  _PIPE_CUR=failed
  exit "$rc"
}

# run_step <step> <description> <command...>
# Runs the command with output tee'd to the step log. The command itself is
# responsible for being resumable (every script here is).
run_step () {
  local step="$1" desc="$2"; shift 2
  local log="$LOG_DIR/${PIPELINE_NAME}_${step}.log"
  _PIPE_CUR="$step"
  echo "=== [$PIPELINE_NAME] $step: $desc ==="
  _pipe_mark START "$step" "$desc"
  set -o pipefail
  "$@" 2>&1 | tee -a "$log"
  _pipe_mark DONE "$step"
  _PIPE_CUR=""
}

# skip_step <step> <reason>
skip_step () {
  echo "=== [$PIPELINE_NAME] $1: skip ($2) ==="
  _pipe_mark SKIP "$1" "$2"
}

# Resolve an HF repo to its local snapshot dir (downloads if needed).
hf_snapshot () {
  python3 -c "from huggingface_hub import snapshot_download; print(snapshot_download('$1'))"
}
