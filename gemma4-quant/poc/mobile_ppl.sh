#!/bin/bash
QAT=/workspace/qat; PY=/workspace/venv/bin/python; cd "$(dirname "$0")"
"$PY" mobile_ref.py --hf "$QAT/E2B-mobile-hf" --mlx "$QAT/E2B-mobile-mlx" --text "$QAT/wiki.test.raw" --windows 32
echo MOBILE_PPL_DONE
