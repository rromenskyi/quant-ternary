#!/usr/bin/env bash
# GPTQ -> MLX (JANG) pipeline for Liquid AI LFM2 / LFM2.5 checkpoints, dense
# (lfm2) or mixture-of-experts (lfm2_moe). Runs on a rented CUDA pod:
# calibration needs torch + CUDA, the MLX conversion and perplexity run on
# the same pod via mlx[cuda]. Resumable: re-run after an interruption and
# finished steps are skipped (calibration resumes at the next layer).
#
#   VARIANT=lfm2-8b SETUP=1 ./lfm2_mlx_pipeline.sh     # deps, build, check
#   VARIANT=lfm2-8b PUBLISH=1 ./lfm2_mlx_pipeline.sh   # ... and upload to HF
#
# Steps (-> $LOG_DIR/lfm2_<step>.log, status lines in $LOG_DIR/pipeline.log):
#   setup       pip deps (SETUP=1 only)
#   data        wikitext-2 (calibration + test) and Python code (calibration + held-out test)
#   download    HF snapshot of the bf16 checkpoint
#   reference   bf16 perplexity in MLX (check_lfm2.py), once per model
#   baseline    the same for $BASELINE_REPO (e.g. the vendor's MLX 4-bit), once per model
#   calibrate   GPTQ, layer by layer (gptq_lfm2.py)
#   assemble    the checkpoint with the calibrated weights swapped in
#   convert     MLX at the recipe's bits, then GPTQ's exact codes
#   check       perplexity vs bf16 (fails above MAX_PPL_RATIO)
#   card        model card copied in LAST
#   publish     hf upload (PUBLISH=1 only; HF_TOKEN in the environment)
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/../../gemma4-quant/poc/pipeline_lib.sh"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

VARIANT="${VARIANT:-lfm2-8b}"
GROUP_SIZE="${GROUP_SIZE:-64}"
EMBED_BITS="${EMBED_BITS:-8}"
CALIB_CHUNKS="${CALIB_CHUNKS:-128}"
MLX_LM_REF="${MLX_LM_REF:-main}"

case "$VARIANT" in
  lfm2-8b)
    MODEL_ID="${MODEL_ID:-LiquidAI/LFM2.5-8B-A1B}"
    RECIPE="${RECIPE:-attn=8,conv=6,mlp=6,experts=3}"
    HF_REPO="${HF_REPO:-roman220220/LFM2.5-8B-A1B-gptq-mlx-jang}"
    BASELINE_REPO="${BASELINE_REPO:-LiquidAI/LFM2.5-8B-A1B-MLX-4bit}"
    ;;
  lfm2-8b-small)
    # The experts' gate / up at 2 bits, down at 3: an extreme quant.
    MODEL_ID="${MODEL_ID:-LiquidAI/LFM2.5-8B-A1B}"
    RECIPE="${RECIPE:-attn=8,conv=6,mlp=6,experts_gate_up=2,experts_down=3}"
    HF_REPO="${HF_REPO:-roman220220/LFM2.5-8B-A1B-gptq-mlx-jang-small}"
    BASELINE_REPO="${BASELINE_REPO:-LiquidAI/LFM2.5-8B-A1B-MLX-4bit}"
    MAX_PPL_RATIO="${MAX_PPL_RATIO:-1.5}"
    ;;
  lfm2-8b-q4)
    # The experts at 4 bits: the vendor's 4-bit size, attention / conv kept higher.
    MODEL_ID="${MODEL_ID:-LiquidAI/LFM2.5-8B-A1B}"
    RECIPE="${RECIPE:-attn=8,conv=6,mlp=6,experts=4}"
    HF_REPO="${HF_REPO:-roman220220/LFM2.5-8B-A1B-gptq-mlx-jang-q4}"
    BASELINE_REPO="${BASELINE_REPO:-LiquidAI/LFM2.5-8B-A1B-MLX-4bit}"
    ;;
  lfm2-2.6b)
    MODEL_ID="${MODEL_ID:-LiquidAI/LFM2.5-2.6B}"
    RECIPE="${RECIPE:-attn=8,conv=6,mlp=4}"
    HF_REPO="${HF_REPO:-roman220220/LFM2.5-2.6B-gptq-mlx-jang}"
    BASELINE_REPO="${BASELINE_REPO:-}"   # no vendor MLX build
    ;;
  lfm2-1.2b-instruct)
    MODEL_ID="${MODEL_ID:-LiquidAI/LFM2.5-1.2B-Instruct}"
    RECIPE="${RECIPE:-attn=8,conv=6,mlp=4}"
    HF_REPO="${HF_REPO:-roman220220/LFM2.5-1.2B-Instruct-gptq-mlx-jang}"
    BASELINE_REPO="${BASELINE_REPO:-LiquidAI/LFM2.5-1.2B-Instruct-MLX-4bit}"
    ;;
  lfm2-1.2b-thinking)
    MODEL_ID="${MODEL_ID:-LiquidAI/LFM2.5-1.2B-Thinking}"
    RECIPE="${RECIPE:-attn=8,conv=6,mlp=4}"
    HF_REPO="${HF_REPO:-roman220220/LFM2.5-1.2B-Thinking-gptq-mlx-jang}"
    BASELINE_REPO="${BASELINE_REPO:-LiquidAI/LFM2.5-1.2B-Thinking-MLX-4bit}"
    ;;
  *) echo "VARIANT must be lfm2-8b, lfm2-8b-small, lfm2-8b-q4, lfm2-2.6b, lfm2-1.2b-instruct or lfm2-1.2b-thinking (or add a preset)" >&2; exit 2 ;;
esac
NAME="${HF_REPO#*/}"
CARD="${CARD:-$HERE/../cards/$NAME.md}"
DATA="$WORK/data"
MAX_PPL_RATIO="${MAX_PPL_RATIO:-1.10}"
RESULTS="$WORK/results-${MODEL_ID##*/}.json"   # per model: bf16, baseline and every variant
GPTQ="$WORK/gptq-$VARIANT"
ONGRID="$WORK/ongrid-$VARIANT"
OUT_DIR="${OUT_DIR:-$WORK/$NAME}"
export HF_HOME="${HF_HOME:-$WORK/hf}"
CHECK=(python3 "$HERE/check_lfm2.py" --out "$RESULTS" --wikitext "$DATA/wiki.test.raw" --code "$DATA/code.test.txt")
has_result () { [ -s "$RESULTS" ] && python3 -c "import json,sys; sys.exit(not json.load(open('$RESULTS')).get('$1', {}).get('passed'))"; }

pipeline_init lfm2 "VARIANT=$VARIANT MODEL_ID=$MODEL_ID RECIPE=$RECIPE OUT_DIR=$OUT_DIR"

# mlx[cuda]'s NCCL breaks the image's torch (undefined ncclCommWindowRegister)
# wherever transformers imports it: the MLX steps preload torch's NCCL.
TORCH_NCCL="$(python3 -c 'import importlib.util, pathlib; t = pathlib.Path(importlib.util.find_spec("torch").origin).parents[1]; print(next(iter(sorted((t / "nvidia" / "nccl" / "lib").glob("libnccl.so*"))), ""))' 2>/dev/null || true)"
mlx_env () { if [ -n "$TORCH_NCCL" ]; then env LD_PRELOAD="$TORCH_NCCL" "$@"; else "$@"; fi; }

# --- setup -------------------------------------------------------------------
if [ "${SETUP:-0}" = 1 ]; then
  run_step setup "pip deps" bash -c "
    pip install --quiet 'transformers>=5.8' accelerate safetensors huggingface_hub hf_transfer numpy pandas pyarrow \
      jinja2 protobuf sentencepiece &&
    pip install --quiet --upgrade 'mlx[cuda]' &&
    pip install --quiet --force-reinstall --no-deps 'git+https://github.com/ipsupport-llc/mlx-lm.git@$MLX_LM_REF'"
else
  skip_step setup "SETUP!=1"
fi

# --- data --------------------------------------------------------------------
# The same sets as qwen35-quant: wikitext-2, calibration code from the
# installed transformers / torch sources, held-out code from the stdlib.
if [ -s "$DATA/wiki.train.raw" ] && [ -s "$DATA/wiki.test.raw" ] && [ -s "$DATA/code.txt" ] && [ -s "$DATA/code.test.txt" ]; then
  skip_step data "already in $DATA"
else
  run_step data "wikitext-2 + Python code corpus" python3 "$HERE/../../qwen35-quant/poc/make_data.py" "$DATA"
fi

# --- download ----------------------------------------------------------------
_PIPE_CUR=download; _pipe_mark START download "$MODEL_ID"
SNAPSHOT="$(HF_HUB_ENABLE_HF_TRANSFER=1 hf_snapshot "$MODEL_ID")"
_pipe_mark DONE download "$SNAPSHOT"; _PIPE_CUR=""
echo "checkpoint: $SNAPSHOT"

# --- reference + baseline ------------------------------------------------------
if has_result bf16; then
  skip_step reference "bf16 in $RESULTS"
else
  run_step reference "bf16 perplexity (MLX)" mlx_env "${CHECK[@]}" --model "$SNAPSHOT" --name bf16
fi
if [ -z "${BASELINE_REPO:-}" ] || has_result baseline; then
  skip_step baseline "${BASELINE_REPO:-no BASELINE_REPO} (done or none)"
else
  run_step baseline "$BASELINE_REPO perplexity" \
    mlx_env "${CHECK[@]}" --model "$BASELINE_REPO" --name baseline --ref bf16 --max-ppl-ratio 100
fi

# --- calibrate -----------------------------------------------------------------
if [ -s "$GPTQ/progress.json" ] && python3 -c "import json,sys; p=json.load(open('$GPTQ/progress.json')); m=p.get('made_with',{}); sys.exit(not (p['done']+1==p['layers'] and m.get('recipe')=='$RECIPE' and m.get('group_size')==$GROUP_SIZE and m.get('calib_chunks')==$CALIB_CHUNKS))"; then
  skip_step calibrate "all layers in $GPTQ"
else
  run_step calibrate "GPTQ $RECIPE (group $GROUP_SIZE)" \
    python3 "$HERE/gptq_lfm2.py" --model "$SNAPSHOT" --work "$GPTQ" --recipe "$RECIPE" \
      --group-size "$GROUP_SIZE" --calib-chunks "$CALIB_CHUNKS" \
      --calib "wikitext:$DATA/wiki.train.raw" --calib "code:$DATA/code.txt"
fi

# --- assemble ------------------------------------------------------------------
if grep -q ASSEMBLE_DONE "$LOG_DIR/lfm2_assemble.log" 2>/dev/null && [ "$ONGRID/quant_recipe.json" -nt "$GPTQ/progress.json" ]; then
  skip_step assemble "$ONGRID newer than the calibration"
else
  run_step assemble "calibrated weights into $ONGRID" \
    python3 "$HERE/assemble_lfm2.py" --model "$SNAPSHOT" --work "$GPTQ" --out "$ONGRID"
fi

# --- convert -------------------------------------------------------------------
if grep -q CONVERT_DONE "$LOG_DIR/lfm2_convert.log" 2>/dev/null && [ "$OUT_DIR/quant_recipe.json" -nt "$ONGRID/quant_recipe.json" ]; then
  skip_step convert "$OUT_DIR newer than $ONGRID"
else
  rm -rf "$OUT_DIR"
  run_step convert "MLX at the recipe's bits + GPTQ codes" \
    mlx_env python3 "$HERE/convert_lfm2.py" --hf "$ONGRID" --out "$OUT_DIR" --gptq-work "$GPTQ" --embed-bits "$EMBED_BITS"
fi

# --- check ---------------------------------------------------------------------
if has_result "$VARIANT" && [ "$RESULTS" -nt "$OUT_DIR/quant_recipe.json" ]; then
  skip_step check "$VARIANT passed in $RESULTS"
else
  run_step check "perplexity (text, code) vs bf16" \
    mlx_env "${CHECK[@]}" --model "$OUT_DIR" --name "$VARIANT" --ref bf16 --max-ppl-ratio "$MAX_PPL_RATIO"
fi

# --- model card (LAST) -----------------------------------------------------------
if [ -f "$CARD" ]; then
  run_step card "copy $(basename "$CARD") -> README.md" bash -c '
    cp "$1" "$2/README.md"
    for img in $(grep -o "[a-z-]*banner\.png" "$1" | sort -u); do cp "$3/$img" "$2/$img"; done
    if [ -f "$4/LICENSE" ]; then cp "$4/LICENSE" "$2/LICENSE"; fi' _ "$CARD" "$OUT_DIR" "$HERE/../../cards_assets" "$SNAPSHOT"
else
  skip_step card "no $CARD yet"
fi

# --- publish -------------------------------------------------------------------
if [ "${PUBLISH:-0}" = 1 ]; then
  [ -f "$OUT_DIR/README.md" ] || { echo "no model card in $OUT_DIR" >&2; exit 1; }
  run_step publish "hf upload $HF_REPO" hf upload "$HF_REPO" "$OUT_DIR" . --commit-message "lfm2_mlx_pipeline.sh VARIANT=$VARIANT"
else
  skip_step publish "PUBLISH!=1"
fi

echo "LFM2_MLX_PIPELINE_DONE $OUT_DIR"
