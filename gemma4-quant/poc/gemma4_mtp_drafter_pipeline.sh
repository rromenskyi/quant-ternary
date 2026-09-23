#!/usr/bin/env bash
# Gemma 4 26B-A4B MTP drafter (speculative decoding) -> MLX 8-bit.
# Needs mlx (Apple Silicon, or mlx[cuda] on a pod) with ipsupport-llc/mlx-lm
# (models/gemma4_assistant.py -- stock mlx-lm has no support). The GGUF
# drafter for ollama is handled by gemma4_26b_gguf_pipeline.sh (step
# `drafter`), not here.
#
#   ./gemma4_mtp_drafter_pipeline.sh
#   PY_TORCH=.venv/bin/python PY_MLX=/path/to/mlx-venv/bin/python3 ./gemma4_mtp_drafter_pipeline.sh
#   MAIN_MODEL=roman220220/gemma-4-26B-A4B-it-gptq-mlx-jang ... # + speed/acceptance benchmark
#   PUBLISH=1 ...                                               # + hf upload
#
# Steps (-> $LOG_DIR/mtp_<step>.log):
#   download   google drafter (bf16, 840MB)
#   parity     mlx gemma4_assistant vs transformers, fp32, real weights
#              (needs PY_TORCH with transformers >= 5.x; skipped if absent)
#   convert    mlx_lm.convert -q 8-bit group 64 (446MB; acceptance equals bf16)
#   bench      plain vs MTP on MAIN_MODEL, fresh AND cached-prefix prompt
#              cache (the server's case); fails below MIN_SPEEDUP
#   card       model card LAST (convert writes a stub README)
#   publish    hf upload (PUBLISH=1 only)
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/pipeline_lib.sh"

DRAFTER_ID="${DRAFTER_ID:-google/gemma-4-26B-A4B-it-assistant}"
PY_TORCH="${PY_TORCH:-python3}"
PY_MLX="${PY_MLX:-python3}"
Q_BITS="${Q_BITS:-8}"
OUT_DIR="${OUT_DIR:-$WORK/gemma-4-26B-A4B-it-assistant-mlx-${Q_BITS}bit}"
HF_REPO="${HF_REPO:-roman220220/gemma-4-26B-A4B-it-assistant-mlx-8bit}"
CARD="$HERE/../cards/${HF_REPO#*/}.md"
MIN_SPEEDUP="${MIN_SPEEDUP:-1.15}"

pipeline_init mtp "DRAFTER_ID=$DRAFTER_ID OUT_DIR=$OUT_DIR"

_PIPE_CUR=download; _pipe_mark START download "$DRAFTER_ID"
SNAPSHOT="$("$PY_MLX" -c "from huggingface_hub import snapshot_download; print(snapshot_download('$DRAFTER_ID'))")"
_pipe_mark DONE download "$SNAPSHOT"; _PIPE_CUR=""

if "$PY_TORCH" -c "import torch, transformers.models.gemma4_assistant" 2>/dev/null; then
  run_step parity "mlx vs transformers, fp32, real weights" bash -c "
    '$PY_TORCH' '$HERE/gemma4_mtp_drafter_parity.py' dump --drafter '$SNAPSHOT' --out '$WORK/mtp_parity_ref.npz' &&
    '$PY_MLX' '$HERE/gemma4_mtp_drafter_parity.py' check --drafter '$SNAPSHOT' --ref '$WORK/mtp_parity_ref.npz'"
else
  skip_step parity "PY_TORCH has no torch + transformers gemma4_assistant"
fi

if [ -f "$OUT_DIR/model.safetensors" ]; then
  skip_step convert "$OUT_DIR exists"
else
  run_step convert "mlx_lm.convert -q $Q_BITS-bit group 64" \
    "$PY_MLX" -m mlx_lm convert --hf-path "$SNAPSHOT" --mlx-path "$OUT_DIR" -q --q-bits "$Q_BITS" --q-group-size 64
fi

if [ -n "${MAIN_MODEL:-}" ]; then
  run_step bench "plain vs MTP on $MAIN_MODEL (fresh + cached-prefix)" \
    "$PY_MLX" "$HERE/gemma4_mtp_bench.py" --model "$MAIN_MODEL" --drafter "$OUT_DIR" --min-speedup "$MIN_SPEEDUP"
else
  skip_step bench "MAIN_MODEL not set"
fi

run_step card "copy $(basename "$CARD") -> README.md" cp "$CARD" "$OUT_DIR/README.md"

if [ "${PUBLISH:-0}" = 1 ]; then
  run_step publish "hf upload $HF_REPO" hf upload "$HF_REPO" "$OUT_DIR" . --commit-message "gemma4_mtp_drafter_pipeline.sh"
else
  skip_step publish "PUBLISH!=1"
fi

echo "GEMMA4_MTP_DRAFTER_PIPELINE_DONE $OUT_DIR"
