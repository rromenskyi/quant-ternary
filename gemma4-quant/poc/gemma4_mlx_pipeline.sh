#!/usr/bin/env bash
# End-to-end GPTQ -> MLX (JANG-mixed) pipeline for Gemma 4, E4B or 26B-A4B.
# Runs entirely on a rented CUDA pod: calibration needs torch+CUDA, and the
# splice runs on the same pod via mlx[cuda] (FINDINGS: no 47GB round-trip
# through a Mac). Every step is resumable -- re-run the script after an
# interruption and finished steps are skipped (calibration resumes per layer
# from its gptq_progress_*.json).
#
#   VARIANT=26b ./gemma4_mlx_pipeline.sh              # build + smoke test
#   VARIANT=26b PUBLISH=1 ./gemma4_mlx_pipeline.sh    # ... and upload to HF
#   SETUP=1 ...                                       # pip-install deps first
#
# Watch it with: python pipeline_dashboard.py --local   (or --pod-host ...)
#
# Steps (-> $LOG_DIR/mlx_<step>.log, status lines in $LOG_DIR/pipeline.log):
#   setup           pip deps (SETUP=1 only)
#   download        HF snapshot of the bf16 checkpoint
#   calibrate_<c>   GPTQ correction per component (text / vision / audio)
#   splice          mx.quantize + re-shard + config quantization dict; also
#                   RTN for uncalibrated leftover Linears, drops dead
#                   KV-shared weights (see splice_common.py)
#   smoke           gemma4_smoke_test.py: text, vision, audio (E4B), no
#                   large float weights left
#   card            model card copied in LAST (a splice re-run would clobber
#                   it with Google's card, which lacks the mlx tag)
#   publish         hf upload (PUBLISH=1 only)
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/pipeline_lib.sh"

VARIANT="${VARIANT:-26b}"
ATTN_BITS="${ATTN_BITS:-8}"
FFN_BITS="${FFN_BITS:-4}"
GROUP_SIZE="${GROUP_SIZE:-64}"
EMBED_BITS="${EMBED_BITS:-8}"
LEFTOVER_BITS="${LEFTOVER_BITS:-8}"
MLX_LM_REF="${MLX_LM_REF:-main}"   # ipsupport-llc/mlx-lm ref (needs its gemma4 multimodal support)

# Weights that must stay float, passed to both splice and smoke test:
#  - patch_embedder.input_proj: gemma4_vision casts pixels to
#    input_proj.weight.dtype -- uint32 once quantized, vision then sees noise
#    (measured: "no discernible animals" on the cats photo).
#  - router.proj (26B): routing precision is disproportionately sensitive,
#    quantizing it wrecks expert selection (FINDINGS: JANG recipe).
KEEP_FLOAT=(--keep-float 'patch_embedder\.input_proj')

case "$VARIANT" in
  26b)
    MODEL_ID="google/gemma-4-26B-A4B-it"
    COMPONENTS=(text vision)
    CALIBRATE="$HERE/gemma4_26b_gptq_calibrate.py"
    SPLICE="$HERE/gemma4_26b_gptq_splice.py"
    HF_REPO="${HF_REPO:-roman220220/gemma-4-26B-A4B-it-gptq-mlx-jang}"
    KEEP_FLOAT+=(--keep-float 'router\.proj')
    CAL_EXTRA_vision=(--layers-per-batch 4)
    ;;
  e4b)
    MODEL_ID="google/gemma-4-E4B-it"
    COMPONENTS=(text vision audio)
    CALIBRATE="$HERE/gemma4_gptq_calibrate.py"
    SPLICE="$HERE/gemma4_gptq_splice.py"
    HF_REPO="${HF_REPO:-roman220220/gemma-4-E4B-it-gptq-mlx-jang}"
    ;;
  *) echo "VARIANT must be 26b or e4b" >&2; exit 2 ;;
esac
CARD="$HERE/../cards/${HF_REPO#*/}.md"
CORRECTED="${CORRECTED:-$WORK/gemma4-$VARIANT-corrected}"
OUT_DIR="${OUT_DIR:-$WORK/output-$VARIANT-jang}"
ASSETS="$WORK/smoke-assets"

pipeline_init mlx "VARIANT=$VARIANT MODEL_ID=$MODEL_ID CORRECTED=$CORRECTED OUT_DIR=$OUT_DIR"

# --- setup -------------------------------------------------------------------
if [ "${SETUP:-0}" = 1 ]; then
  run_step setup "pip deps" bash -c "
    pip install --quiet torch transformers accelerate safetensors huggingface_hub pillow librosa &&
    pip install --quiet --upgrade 'mlx[cuda]' &&
    pip install --quiet --force-reinstall --no-deps 'git+https://github.com/ipsupport-llc/mlx-lm.git@$MLX_LM_REF'"
else
  skip_step setup "SETUP!=1"
fi

# --- download ----------------------------------------------------------------
_PIPE_CUR=download; _pipe_mark START download "$MODEL_ID"
SNAPSHOT="$(hf_snapshot "$MODEL_ID")"
_pipe_mark DONE download "$SNAPSHOT"; _PIPE_CUR=""
echo "checkpoint: $SNAPSHOT"

# --- calibrate, per component --------------------------------------------------
# Complete when every layer is in the component's progress file(s); checked
# here because the calibrate scripts only notice "nothing left" AFTER loading
# the whole model (~50GB for the 26B).
component_done () {
  python3 "$HERE/pipeline_status.py" component-done \
    --snapshot "$SNAPSHOT" --corrected "$CORRECTED" --component "$1" --variant "$VARIANT"
}
for c in "${COMPONENTS[@]}"; do
  if component_done "$c"; then
    skip_step "calibrate_$c" "all layers already in progress file"
    continue
  fi
  extra_var="CAL_EXTRA_$c[@]"
  run_step "calibrate_$c" "GPTQ $c (attn $ATTN_BITS / ffn $FFN_BITS, group $GROUP_SIZE)" \
    python3 "$CALIBRATE" --model-dir "$SNAPSHOT" --output-dir "$CORRECTED" --component "$c" \
      --attn-bits "$ATTN_BITS" --ffn-bits "$FFN_BITS" --group-size "$GROUP_SIZE" "${!extra_var}"
done

# --- splice ------------------------------------------------------------------
if [ -f "$OUT_DIR/model.safetensors.index.json" ] && grep -q "_SPLICE_DONE" "$LOG_DIR/mlx_splice.log" 2>/dev/null \
   && [ "$OUT_DIR/model.safetensors.index.json" -nt "$CORRECTED" ]; then
  skip_step splice "output newer than corrected weights"
else
  run_step splice "quantize + re-shard into $OUT_DIR" \
    python3 "$SPLICE" --hf-checkpoint-dir "$SNAPSHOT" --corrected-dir "$CORRECTED" --output-dir "$OUT_DIR" \
      --group-size "$GROUP_SIZE" --embedding-bits "$EMBED_BITS" \
      --rtn-leftovers-bits "$LEFTOVER_BITS" --drop-kv-shared-dead "${KEEP_FLOAT[@]}"
fi

# --- smoke test ----------------------------------------------------------------
mkdir -p "$ASSETS"
[ -f "$ASSETS/cats.jpg" ] || curl -sfL -o "$ASSETS/cats.jpg" http://images.cocodataset.org/val2017/000000039769.jpg
SMOKE_ARGS=(--model "$OUT_DIR" "${KEEP_FLOAT[@]}" --image "$ASSETS/cats.jpg" --expect-in-image-answer cat)
if [ "$VARIANT" = e4b ]; then
  if [ ! -f "$ASSETS/speech.wav" ] && command -v say >/dev/null; then
    say -o "$ASSETS/speech.aiff" "The quick brown fox jumps over the lazy dog."
    afconvert -f WAVE -d LEI16@16000 -c 1 "$ASSETS/speech.aiff" "$ASSETS/speech.wav"
  fi
  # On a pod without `say`, drop any 16kHz mono speech WAV at $ASSETS/speech.wav.
  if [ -f "$ASSETS/speech.wav" ]; then
    SMOKE_ARGS+=(--audio "$ASSETS/speech.wav" --expect-in-audio-answer "${AUDIO_EXPECT:-quick brown fox}")
  else
    echo "  (no $ASSETS/speech.wav -- audio check skipped)"
  fi
fi
run_step smoke "text / vision / audio / no float leftovers" python3 "$HERE/gemma4_smoke_test.py" "${SMOKE_ARGS[@]}"

# --- model card (LAST) -----------------------------------------------------------
run_step card "copy $(basename "$CARD") -> README.md" cp "$CARD" "$OUT_DIR/README.md"

# --- publish -------------------------------------------------------------------
if [ "${PUBLISH:-0}" = 1 ]; then
  run_step publish "hf upload $HF_REPO" hf upload "$HF_REPO" "$OUT_DIR" . --commit-message "gemma4_mlx_pipeline.sh VARIANT=$VARIANT"
else
  skip_step publish "PUBLISH!=1"
fi

echo "GEMMA4_MLX_PIPELINE_DONE $OUT_DIR"
