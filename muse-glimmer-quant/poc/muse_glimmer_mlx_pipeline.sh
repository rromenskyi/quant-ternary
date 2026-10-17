#!/usr/bin/env bash
# GPTQ -> MLX (JANG) pipeline for Meta's Muse Glimmer 30B (meta-models/Muse-Glimmer-30B,
# dense, `muse_glimmer`, text tower only -- the perception encoder isn't
# ported to mlx-lm). Runs on a rented CUDA pod with a big GPU: the full
# bf16 checkpoint is ~60 GB (text + vision), so the bf16/baseline MLX
# reference passes need a card that actually holds it (96 GB card used for
# the initial run, not a 24 GB one).
#
#   SETUP=1 ./muse_glimmer_mlx_pipeline.sh       # deps, check
#   PUBLISH=1 ./muse_glimmer_mlx_pipeline.sh     # ... and upload to HF
#
# Steps (-> $LOG_DIR/glimmer_<step>.log, status lines in $LOG_DIR/pipeline.log):
#   setup       pip deps (SETUP=1 only)
#   data        wikitext-2 (calibration + test) and a Python code corpus
#   download    HF snapshot of the bf16 checkpoint (~60 GB, 2 shards)
#   reference   bf16 perplexity in MLX (check_muse_glimmer.py), once
#   baseline    the vendor-adjacent mlx-community 4-bit build, once
#   calibrate   GPTQ, layer by layer (gptq_muse_glimmer.py)
#   assemble    the checkpoint with the calibrated weights swapped in
#   convert     MLX at the recipe's bits, then GPTQ's exact codes
#   check       perplexity vs bf16 (fails above MAX_PPL_RATIO)
#   card        model card copied in LAST
#   publish     hf upload (PUBLISH=1 only; HF_TOKEN in the environment)
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/../../gemma4-quant/poc/pipeline_lib.sh"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

MODEL_ID="${MODEL_ID:-meta-models/Muse-Glimmer-30B}"
RECIPE="${RECIPE:-attn=8,mlp=4}"
HF_REPO="${HF_REPO:-roman220220/Muse-Glimmer-30B-gptq-mlx-jang}"
BASELINE_REPO="${BASELINE_REPO:-mlx-community/Muse-Glimmer-30B-4bit}"
GROUP_SIZE="${GROUP_SIZE:-64}"
EMBED_BITS="${EMBED_BITS:-8}"
HEAD_BITS="${HEAD_BITS:-8}"
CALIB_CHUNKS="${CALIB_CHUNKS:-128}"
MAX_PPL_RATIO="${MAX_PPL_RATIO:-1.10}"
MLX_LM_VERSION="${MLX_LM_VERSION:-0.32.2}"   # upstream ml-explore/mlx-lm: muse_glimmer landed here

NAME="${HF_REPO#*/}"
CARD="${CARD:-$HERE/../cards/$NAME.md}"
DATA="$WORK/data"
RESULTS="$WORK/results-glimmer.json"
GPTQ="$WORK/gptq-glimmer"
ONGRID="$WORK/ongrid-glimmer"
OUT_DIR="${OUT_DIR:-$WORK/$NAME}"
export HF_HOME="${HF_HOME:-$WORK/hf}"
CHECK=(python3 "$HERE/check_muse_glimmer.py" --out "$RESULTS" --wikitext "$DATA/wiki.test.raw" --code "$DATA/code.test.txt")
has_result () { [ -s "$RESULTS" ] && python3 -c "import json,sys; sys.exit(not json.load(open('$RESULTS')).get('$1', {}).get('passed'))"; }

pipeline_init glimmer "MODEL_ID=$MODEL_ID RECIPE=$RECIPE OUT_DIR=$OUT_DIR"

# mlx[cuda]'s NCCL breaks the image's torch (undefined ncclCommWindowRegister)
# wherever transformers imports it: the MLX steps preload torch's NCCL.
TORCH_NCCL="$(python3 -c 'import importlib.util, pathlib; t = pathlib.Path(importlib.util.find_spec("torch").origin).parents[1]; print(next(iter(sorted((t / "nvidia" / "nccl" / "lib").glob("libnccl.so*"))), ""))' 2>/dev/null || true)"
mlx_env () { if [ -n "$TORCH_NCCL" ]; then env LD_PRELOAD="$TORCH_NCCL" "$@"; else "$@"; fi; }

# --- setup -------------------------------------------------------------------
if [ "${SETUP:-0}" = 1 ]; then
  run_step setup "pip deps" bash -c "
    pip install --quiet --break-system-packages 'transformers>=5.15' accelerate safetensors huggingface_hub hf_transfer numpy pandas pyarrow &&
    pip install --quiet --break-system-packages --upgrade 'mlx[cuda]' &&
    pip install --quiet --break-system-packages 'mlx-lm==$MLX_LM_VERSION'"
else
  skip_step setup "SETUP!=1"
fi

# --- data --------------------------------------------------------------------
if [ -s "$DATA/wiki.train.raw" ] && [ -s "$DATA/wiki.test.raw" ] && [ -s "$DATA/code.txt" ] && [ -s "$DATA/code.test.txt" ]; then
  skip_step data "already in $DATA"
else
  run_step data "wikitext-2 + Python code corpus" python3 "$HERE/../../qwen35-quant/poc/make_data.py" "$DATA"
fi

# --- download ------------------------------------------------------------------
_PIPE_CUR=download; _pipe_mark START download "$MODEL_ID"
SNAPSHOT="$(HF_HUB_ENABLE_HF_TRANSFER=1 hf_snapshot "$MODEL_ID")"
_pipe_mark DONE download "$SNAPSHOT"; _PIPE_CUR=""
echo "checkpoint: $SNAPSHOT"

# --- patch mlx-lm's muse_glimmer sanitize() (upstream bug) ----------------------
# It drops vision weights by checking an un-nested "vision_tower" /
# "vision_adapter" / "vision_projection" prefix, but this checkpoint nests
# them under "model." -- the check never matches and mlx_lm.load()/convert()
# both fail on "unexpected" vision weights (it's a text-only port; the
# weights are just meant to be dropped, not used). Patching the installed
# package in place beats keeping a second ~56 GB checkpoint copy around:
# this is one `startswith` call, not a reason to recompress the model.
MLX_GLIMMER_PY="$(python3 -c 'import mlx_lm.models.muse_glimmer as m; print(m.__file__)')"
if grep -q 'model.vision_tower' "$MLX_GLIMMER_PY" 2>/dev/null; then
  skip_step patch_mlx_lm "$MLX_GLIMMER_PY already patched"
else
  run_step patch_mlx_lm "nest-aware vision-prefix drop in $MLX_GLIMMER_PY" sed -i \
    's/if k.startswith((\"vision_tower\", \"vision_adapter\", \"vision_projection\")):/if k.startswith((\"vision_tower\", \"vision_adapter\", \"vision_projection\", \"model.vision_tower\", \"model.vision_adapter\", \"model.vision_projection\", \"model.multi_modal_projector\")):/' \
    "$MLX_GLIMMER_PY"
fi

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
    python3 "$HERE/gptq_muse_glimmer.py" --model "$SNAPSHOT" --work "$GPTQ" --recipe "$RECIPE" \
      --group-size "$GROUP_SIZE" --calib-chunks "$CALIB_CHUNKS" \
      --calib "wikitext:$DATA/wiki.train.raw" --calib "code:$DATA/code.txt"
fi

# --- assemble ------------------------------------------------------------------
if grep -q ASSEMBLE_DONE "$LOG_DIR/glimmer_assemble.log" 2>/dev/null && [ "$ONGRID/quant_recipe.json" -nt "$GPTQ/progress.json" ]; then
  skip_step assemble "$ONGRID newer than the calibration"
else
  run_step assemble "calibrated weights into $ONGRID" \
    python3 "$HERE/assemble_checkpoint.py" --model "$SNAPSHOT" --work "$GPTQ" --out "$ONGRID"
fi

# --- convert -------------------------------------------------------------------
if grep -q CONVERT_DONE "$LOG_DIR/glimmer_convert.log" 2>/dev/null && [ "$OUT_DIR/quant_recipe.json" -nt "$ONGRID/quant_recipe.json" ]; then
  skip_step convert "$OUT_DIR newer than $ONGRID"
else
  rm -rf "$OUT_DIR"
  run_step convert "MLX at the recipe's bits + GPTQ codes" \
    mlx_env python3 "$HERE/convert_muse_glimmer.py" --hf "$ONGRID" --out "$OUT_DIR" --gptq-work "$GPTQ" \
      --embed-bits "$EMBED_BITS" --head-bits "$HEAD_BITS"
fi

# --- check ---------------------------------------------------------------------
if has_result ours && [ "$RESULTS" -nt "$OUT_DIR/quant_recipe.json" ]; then
  skip_step check "ours passed in $RESULTS"
else
  run_step check "perplexity (text, code) vs bf16" \
    mlx_env "${CHECK[@]}" --model "$OUT_DIR" --name ours --ref bf16 --max-ppl-ratio "$MAX_PPL_RATIO"
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
  run_step publish "hf upload $HF_REPO" hf upload "$HF_REPO" "$OUT_DIR" . --commit-message "muse_glimmer_mlx_pipeline.sh"
else
  skip_step publish "PUBLISH!=1"
fi

echo "MUSE_GLIMMER_MLX_PIPELINE_DONE $OUT_DIR"
