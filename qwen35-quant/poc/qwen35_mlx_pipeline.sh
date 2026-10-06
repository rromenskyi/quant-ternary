#!/usr/bin/env bash
# End-to-end GPTQ -> MLX (JANG) pipeline for Qwen3.5-family checkpoints,
# dense (qwen3_5) or mixture-of-experts (qwen3_5_moe), vision kept. Runs on
# a rented CUDA pod: calibration needs torch + CUDA, and the MLX conversion
# and checks run on the same pod via mlx[cuda] (no checkpoint round-trip
# through a Mac). Every step is resumable: re-run after an interruption and
# finished steps are skipped (calibration resumes at the next layer).
#
#   VARIANT=ornith-35b SETUP=1 ./qwen35_mlx_pipeline.sh   # deps, build, check
#   VARIANT=ornith-35b PUBLISH=1 ./qwen35_mlx_pipeline.sh # ... and upload to HF
#
# Steps (-> $LOG_DIR/qwen35_<step>.log, status lines in $LOG_DIR/pipeline.log,
# shown by gemma4-quant/poc/pipeline_dashboard.py):
#   setup       pip deps (SETUP=1 only)
#   data        wikitext-2 (calibration + test) and a Python code corpus
#   download    HF snapshot of the bf16 checkpoint
#   reference   HF bf16 perplexity + vision features (hf_reference.py)
#   calibrate   GPTQ, layer by layer (gptq_qwen35.py)
#   assemble    the source checkpoint with the calibrated decoder swapped in
#   convert     MLX at the recipe's bits, then GPTQ's exact codes (convert_mlx.py)
#   check       vision vs HF, an image question, perplexity (check_mlx.py)
#   card        model card copied in LAST
#   publish     hf upload (PUBLISH=1 only)
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/../../gemma4-quant/poc/pipeline_lib.sh"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

VARIANT="${VARIANT:-ornith-35b}"
GROUP_SIZE="${GROUP_SIZE:-64}"
EMBED_BITS="${EMBED_BITS:-8}"
HEAD_BITS="${HEAD_BITS:-8}"
VISION_BITS="${VISION_BITS:-8}"
CALIB_CHUNKS="${CALIB_CHUNKS:-128}"
MLX_LM_REF="${MLX_LM_REF:-main}"   # ipsupport-llc/mlx-lm ref (qwen3_5 / qwen3_5_moe vision)

case "$VARIANT" in
  ornith-35b)
    MODEL_ID="${MODEL_ID:-ornith-ai/Ornith-1.5-35B-A3B}"
    RECIPE="${RECIPE:-attn=8,linear=6,shared=6,experts=3}"
    HF_REPO="${HF_REPO:-roman220220/Ornith-1.5-35B-A3B-gptq-mlx-jang}"
    ;;
  frognano-4b)
    MODEL_ID="${MODEL_ID:-microsoft/FrogNano-4B-2609}"
    RECIPE="${RECIPE:-attn=8,linear=6,mlp=4}"
    HF_REPO="${HF_REPO:-roman220220/FrogNano-4B-2609-gptq-mlx-jang}"
    ;;
  *) echo "VARIANT must be ornith-35b or frognano-4b (or add a preset)" >&2; exit 2 ;;
esac
NAME="${HF_REPO#*/}"
CARD="${CARD:-$HERE/../cards/$NAME.md}"
DATA="$WORK/data"
REF="$WORK/ref-$VARIANT"
GPTQ="$WORK/gptq-$VARIANT"
ONGRID="$WORK/ongrid-$VARIANT"
OUT_DIR="${OUT_DIR:-$WORK/$NAME}"
export HF_HOME="${HF_HOME:-$WORK/hf}"

pipeline_init qwen35 "VARIANT=$VARIANT MODEL_ID=$MODEL_ID RECIPE=$RECIPE OUT_DIR=$OUT_DIR"

# --- setup -------------------------------------------------------------------
if [ "${SETUP:-0}" = 1 ]; then
  run_step setup "pip deps" bash -c "
    pip install --quiet 'transformers>=5.8' accelerate safetensors huggingface_hub hf_transfer pillow numpy &&
    pip install --quiet --upgrade 'mlx[cuda]' &&
    pip install --quiet --force-reinstall --no-deps 'git+https://github.com/ipsupport-llc/mlx-lm.git@$MLX_LM_REF' &&
    pip install --quiet jinja2 protobuf sentencepiece"
else
  skip_step setup "SETUP!=1"
fi

# --- data --------------------------------------------------------------------
if [ -s "$DATA/wiki.train.raw" ] && [ -s "$DATA/wiki.test.raw" ] && [ -s "$DATA/code.txt" ]; then
  skip_step data "already in $DATA"
else
  run_step data "wikitext-2 + Python code corpus" python3 - "$DATA" <<'PY'
import sys, pathlib, urllib.request, zipfile, io, transformers, torch
out = pathlib.Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
for split in ("train", "test"):
    url = f"https://huggingface.co/datasets/Salesforce/wikitext/resolve/main/wikitext-2-raw-v1/{split}-00000-of-00001.parquet"
    import pandas as pd
    df = pd.read_parquet(io.BytesIO(urllib.request.urlopen(url).read()))
    (out / f"wiki.{split}.raw").write_text("".join(df["text"]), encoding="utf-8")
# Code calibration: the installed libraries' own Python sources (a fixed,
# license-clean corpus that is on every pod), sorted for a stable order.
files = sorted(p for root in (transformers.__path__[0], torch.__path__[0]) for p in pathlib.Path(root).rglob("*.py"))
text, size = [], 0
for p in files:
    t = p.read_text(encoding="utf-8", errors="ignore")
    if len(t) > 2000:
        text.append(t); size += len(t)
    if size > 40_000_000:
        break
(out / "code.txt").write_text("\n\n".join(text), encoding="utf-8")
print({f.name: f.stat().st_size for f in out.iterdir()})
PY
fi

# --- download ----------------------------------------------------------------
_PIPE_CUR=download; _pipe_mark START download "$MODEL_ID"
SNAPSHOT="$(HF_HUB_ENABLE_HF_TRANSFER=1 hf_snapshot "$MODEL_ID")"
_pipe_mark DONE download "$SNAPSHOT"; _PIPE_CUR=""
echo "checkpoint: $SNAPSHOT"

# --- reference -----------------------------------------------------------------
if [ -s "$REF/ref.json" ]; then
  skip_step reference "$REF/ref.json exists"
else
  run_step reference "HF bf16 perplexity + vision features" \
    python3 "$HERE/hf_reference.py" --model "$SNAPSHOT" --out "$REF" --wikitext "$DATA/wiki.test.raw"
fi

# --- calibrate -----------------------------------------------------------------
if grep -q GPTQ_DONE "$LOG_DIR/qwen35_calibrate.log" 2>/dev/null && [ -s "$GPTQ/progress.json" ] \
   && python3 -c "import json,sys; p=json.load(open('$GPTQ/progress.json')); sys.exit(p['done']+1!=p['layers'])"; then
  skip_step calibrate "all layers in $GPTQ"
else
  run_step calibrate "GPTQ $RECIPE (group $GROUP_SIZE)" \
    python3 "$HERE/gptq_qwen35.py" --model "$SNAPSHOT" --work "$GPTQ" --recipe "$RECIPE" \
      --group-size "$GROUP_SIZE" --calib-chunks "$CALIB_CHUNKS" \
      --calib "wikitext:$DATA/wiki.train.raw" --calib "code:$DATA/code.txt"
fi

# --- assemble ------------------------------------------------------------------
if grep -q ASSEMBLE_DONE "$LOG_DIR/qwen35_assemble.log" 2>/dev/null && [ "$ONGRID/quant_recipe.json" -nt "$GPTQ/progress.json" ]; then
  skip_step assemble "$ONGRID newer than the calibration"
else
  run_step assemble "calibrated decoder into $ONGRID" \
    python3 "$HERE/assemble_checkpoint.py" --model "$SNAPSHOT" --work "$GPTQ" --out "$ONGRID"
fi

# --- convert -------------------------------------------------------------------
if grep -q CONVERT_DONE "$LOG_DIR/qwen35_convert.log" 2>/dev/null && [ "$OUT_DIR/quant_recipe.json" -nt "$ONGRID/quant_recipe.json" ]; then
  skip_step convert "$OUT_DIR newer than $ONGRID"
else
  rm -rf "$OUT_DIR"
  run_step convert "MLX at the recipe's bits" \
    python3 "$HERE/convert_mlx.py" --hf "$ONGRID" --out "$OUT_DIR" --gptq-work "$GPTQ" \
      --embed-bits "$EMBED_BITS" --head-bits "$HEAD_BITS" --vision-bits "$VISION_BITS"
fi

# --- check ---------------------------------------------------------------------
run_step check "vision vs HF, image question, perplexity" \
  python3 "$HERE/check_mlx.py" --model "$OUT_DIR" --ref "$REF" --wikitext "$DATA/wiki.test.raw"

# --- model card (LAST) -----------------------------------------------------------
if [ -f "$CARD" ]; then
  run_step card "copy $(basename "$CARD") -> README.md" cp "$CARD" "$OUT_DIR/README.md"
else
  skip_step card "no $CARD yet"
fi

# --- publish -------------------------------------------------------------------
if [ "${PUBLISH:-0}" = 1 ]; then
  [ -f "$OUT_DIR/README.md" ] || { echo "no model card in $OUT_DIR" >&2; exit 1; }
  run_step publish "hf upload $HF_REPO" hf upload "$HF_REPO" "$OUT_DIR" . --commit-message "qwen35_mlx_pipeline.sh VARIANT=$VARIANT"
else
  skip_step publish "PUBLISH!=1"
fi

echo "QWEN35_MLX_PIPELINE_DONE $OUT_DIR"
