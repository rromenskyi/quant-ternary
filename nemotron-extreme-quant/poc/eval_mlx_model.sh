#!/usr/bin/env bash
# Run test_model_mlx.py (coherence + coding-ability smoke test, no code
# execution -- see that file's docstring for why) against one or two MLX
# models. With two models, runs both and prints them one after another for
# an easy side-by-side read (e.g. our new GPTQ 3-bit vs the naive stock
# 3-bit baseline).
#
# Usage:
#   ./eval_mlx_model.sh <model-path> [--trust-remote-code] [--max-tokens N]
#   ./eval_mlx_model.sh <model-path> --compare-to <other-model-path>
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${MLX_SERVER_VENV:-$SCRIPT_DIR/.mlx_server_venv}"

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <model-path> [--compare-to <other-model-path>] [--trust-remote-code] [--max-tokens N]" >&2
  exit 1
fi

MODEL="$1"; shift
COMPARE_TO=""
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --compare-to) COMPARE_TO="$2"; shift 2 ;;
    *) EXTRA_ARGS+=("$1"); shift 1 ;;
  esac
done

if [[ ! -d "$VENV_DIR" ]]; then
  echo "--- creating venv at $VENV_DIR ---"
  python3 -m venv "$VENV_DIR"
  "$VENV_DIR/bin/pip" install --quiet --upgrade pip
fi
"$VENV_DIR/bin/pip" install --quiet mlx-lm

echo "=========================================="
echo "=== model 1: ${MODEL} ==="
echo "=========================================="
"$VENV_DIR/bin/python" "$SCRIPT_DIR/test_model_mlx.py" --model "$MODEL" "${EXTRA_ARGS[@]}"

if [[ -n "$COMPARE_TO" ]]; then
  echo
  echo "=========================================="
  echo "=== model 2 (comparison): ${COMPARE_TO} ==="
  echo "=========================================="
  "$VENV_DIR/bin/python" "$SCRIPT_DIR/test_model_mlx.py" --model "$COMPARE_TO" "${EXTRA_ARGS[@]}"
fi
