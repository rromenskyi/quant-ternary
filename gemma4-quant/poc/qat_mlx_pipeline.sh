#!/usr/bin/env bash
# Google's Gemma 4 QAT -> MLX on the q4_0 grid, with the Linears worth 8 bits
# raised: one pipeline for every size (FINDINGS "Gemma 4 QAT in MLX on the
# q4_0 grid"). Runs on a CUDA pod with mlx[cuda]; every step is resumable
# (a finished step's output is kept and the step skipped).
#
#   MODEL=E2B ./qat_mlx_pipeline.sh                 # build + measure
#   MODEL=31B BUDGET_MB=400 ./qat_mlx_pipeline.sh   # another size budget
#   MODEL=E4B BASELINES="mlx-community/gemma-4-E4B-it-qat-4bit" ./qat_mlx_pipeline.sh
#   MODEL=E2B PUBLISH=1 HF_REPO=roman220220/... ./qat_mlx_pipeline.sh
#
# Everything model-specific comes from MODEL: Google's repos are
# google/gemma-4-$MODEL-it-qat-q4_0-{unquantized,gguf}.
#
# Steps (-> $LOG_DIR/qat_<step>.log, status lines in $LOG_DIR/pipeline.log):
#   download   the QAT master weights and Google's q4_0 GGUF (the check)
#   scan       qat_sensitivity.py: KL gain per MB of each Linear at 8-bit,
#              and the measured size/quality curve at $BUDGETS_MB
#   convert    qat_aligned_convert.py: q4_0 grid (checked against the GGUF),
#              the $BUDGET_MB point's Linears at 8-bit
#   eval       qat_eval.py against the QAT master: ours (from disk) and
#              $BASELINES
#   card       $CARD copied in last (convert copies Google's)
#   publish    hf upload (PUBLISH=1 only)
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/pipeline_lib.sh"

: "${MODEL:?MODEL=E2B|E4B|12B|31B}"
BUDGET_MB="${BUDGET_MB:-100}"
BUDGETS_MB="${BUDGETS_MB:-50 100 200 400 800}"
SCAN_WINDOWS="${SCAN_WINDOWS:-16}"
EVAL_WINDOWS="${EVAL_WINDOWS:-128}"
BASELINES="${BASELINES:-}"
QAT="${QAT:-$WORK/qat}"
TEXT="${TEXT:-$QAT/wiki.test.raw}"
CARD="${CARD:-$HERE/../cards/gemma-4-$MODEL-it-qat-mlx.md}"
PY="${PY:-python3}"
HF="${HF:-hf}"

SRC="$QAT/$MODEL-unq"
GGUF_DIR="$QAT/$MODEL-gguf"
SCAN_JSON="$QAT/$MODEL-sens.json"
OUT="$QAT/$MODEL-qat-mlx-r${BUDGET_MB%.*}"
pipeline_init "qat_$MODEL"

if [ -f "$SRC/config.json" ] && ls "$GGUF_DIR"/*q4_0*.gguf >/dev/null 2>&1; then
  skip_step download "already here"
else
  run_step download "google/gemma-4-$MODEL-it-qat-q4_0-{unquantized,gguf}" bash -c "
    $HF download google/gemma-4-$MODEL-it-qat-q4_0-unquantized --local-dir '$SRC' &&
    $HF download google/gemma-4-$MODEL-it-qat-q4_0-gguf --include '*q4_0*' --exclude '*mmproj*' --local-dir '$GGUF_DIR'"
fi
GGUF="$(ls "$GGUF_DIR"/*q4_0*.gguf | grep -v mmproj | head -1)"

if [ ! -f "$TEXT" ]; then
  run_step text "wikitext-2 test -> $TEXT" bash -c "
    $HF download Salesforce/wikitext --repo-type dataset --include 'wikitext-2-raw-v1/test-*' --local-dir '$QAT/wikitext' &&
    $PY -c \"import glob, pyarrow.parquet as pq; open('$TEXT', 'w').write(''.join(pq.read_table(glob.glob('$QAT/wikitext/wikitext-2-raw-v1/test-*.parquet')[0]).column('text').to_pylist()))\""
fi

if [ -f "$SCAN_JSON" ]; then
  skip_step scan "$SCAN_JSON exists"
else
  # shellcheck disable=SC2086  # BUDGETS_MB is a list
  run_step scan "8-bit sensitivity, budgets $BUDGETS_MB MB" \
    "$PY" "$HERE/qat_sensitivity.py" --master "$SRC" --text "$TEXT" --scan-windows "$SCAN_WINDOWS" \
      --windows "$EVAL_WINDOWS" --budgets-mb $BUDGETS_MB --json "$SCAN_JSON"
fi

if [ -f "$OUT/model.safetensors.index.json" ] && [ "$OUT/config.json" -nt "$SCAN_JSON" ]; then
  skip_step convert "$OUT is newer than the scan"
else
  run_step convert "q4_0 grid + the +$BUDGET_MB MB point -> $OUT" \
    "$PY" "$HERE/qat_aligned_convert.py" --hf-checkpoint-dir "$SRC" --output-dir "$OUT" --gguf "$GGUF" \
      --raise-json "$SCAN_JSON" --raise-budget-mb "$BUDGET_MB"
fi

BASE_DIRS=()
for repo in $BASELINES; do
  dir="$QAT/baseline-${repo//\//--}"
  [ -f "$dir/config.json" ] || $HF download "$repo" --local-dir "$dir" > /dev/null
  BASE_DIRS+=("$dir")
done
run_step eval "vs the QAT master, $EVAL_WINDOWS windows" \
  "$PY" "$HERE/qat_eval.py" --text "$TEXT" --ref "$SRC" --windows "$EVAL_WINDOWS" --json "$OUT/eval.json" "$OUT" "${BASE_DIRS[@]}"

if [ -f "$CARD" ]; then
  run_step card "$(basename "$CARD") -> README.md" cp "$CARD" "$OUT/README.md"
else
  skip_step card "no $CARD yet"
fi

if [ "${PUBLISH:-0}" = 1 ]; then
  : "${HF_REPO:?HF_REPO=org/name to publish}"
  run_step publish "hf upload $HF_REPO" "$HF" upload "$HF_REPO" "$OUT" . --commit-message "qat_mlx_pipeline.sh MODEL=$MODEL BUDGET_MB=$BUDGET_MB"
else
  skip_step publish "PUBLISH!=1"
fi
