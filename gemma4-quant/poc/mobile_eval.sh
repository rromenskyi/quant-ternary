#!/bin/bash
# Google's mobile QAT E2B: convert to MLX, check the conversion against
# transformers, score it against the bf16 QAT master next to our q4_0 build,
# smoke-test vision and audio.
set -uo pipefail
QAT="${QAT:-/workspace/qat}"; PY="${PY:-/workspace/venv/bin/python}"; HF="${HF:-/workspace/venv/bin/hf}"
HERE="$(cd "$(dirname "$0")" && pwd)"; cd "$HERE"
[ -f "$QAT/E2B-mobile-hf/config.json" ] || "$HF" download google/gemma-4-E2B-it-qat-mobile-transformers --local-dir "$QAT/E2B-mobile-hf" > /dev/null
[ -f "$QAT/E2B-mobile-mlx/config.json" ] || "$PY" qat_mobile_convert.py --hf-checkpoint-dir "$QAT/E2B-mobile-hf" --output-dir "$QAT/E2B-mobile-mlx" | tail -2
echo "=== conversion vs transformers"
"$PY" mobile_ref.py --hf "$QAT/E2B-mobile-hf" --mlx "$QAT/E2B-mobile-mlx" --text "$QAT/wiki.test.raw" --windows 8 --json "$QAT/E2B-mobile-ref.json"
echo "=== vs the QAT master (128 chat windows)"
"$PY" qat_eval.py --text "$QAT/wiki.test.raw" --ref "$QAT/E2B-unq" --windows 128 --chat --json "$QAT/E2B-mobile-eval.json" \
    "$QAT/E2B-qat-mlx-r100" "$QAT/E2B-mobile-mlx"
echo "=== smoke"
"$PY" gemma4_smoke_test.py --model "$QAT/E2B-mobile-mlx" --image "$QAT/z.png" --expect-in-image-answer fox \
    --audio "$QAT/speech-fox.wav" --expect-in-audio-answer fox
echo MOBILE_EVAL_DONE
