#!/usr/bin/env bash
# End-to-end GPTQ -> MLX (JANG) pipeline for Thinking Machines' Inkling 2
# Mini (inkling2_loop: 24 layers, layers 6-17 looped 3 times, 128 routed
# experts, vision kept). Runs on a rented CUDA pod: calibration needs torch +
# CUDA, and the MLX conversion and checks run on the same pod via mlx[cuda].
# Every step is resumable: re-run after an interruption and finished steps
# are skipped (calibration resumes at the next layer).
#
#   VARIANT=mini SETUP=1 ./inkling2_mlx_pipeline.sh     # deps, build, check
#   VARIANT=mini PUBLISH=1 ./inkling2_mlx_pipeline.sh   # ... and upload to HF
#
# Steps (-> $LOG_DIR/inkling2_<step>.log, status lines in $LOG_DIR/pipeline.log,
# shown by gemma4-quant/poc/pipeline_dashboard.py):
#   setup       pip deps (SETUP=1 only)
#   data        wikitext-2 (calibration + test) and a Python code corpus
#   download    HF snapshot (NVFP4 experts; transformers dequantizes them on load)
#   reference   transformers perplexity + vision features (hf_reference.py)
#   calibrate   GPTQ, layer by layer, the loop on its first pass (gptq_inkling2.py)
#   assemble    the source checkpoint with the calibrated decoder swapped in, bf16 experts
#   convert     MLX at the recipe's bits, then GPTQ's exact codes (convert_mlx.py)
#   check       vision vs HF, image + text answers, perplexity, DSpark (check_mlx.py)
#   card        model card copied in LAST
#   publish     hf upload (PUBLISH=1 only)
#
# A CPU dry run of the same scripts on a tiny random checkpoint: smoke_test.sh.
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/../../gemma4-quant/poc/pipeline_lib.sh"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

VARIANT="${VARIANT:-mini}"
GROUP_SIZE="${GROUP_SIZE:-64}"
EMBED_BITS="${EMBED_BITS:-8}"
HEAD_BITS="${HEAD_BITS:-8}"
VISION_BITS="${VISION_BITS:-0}"   # bf16, as mlx-lm's inkling2 quant_predicate
MTP_BITS="${MTP_BITS:-8}"         # the DSpark draft (mtp.safetensors), RTN; its recurrence stays float
CALIB_CHUNKS="${CALIB_CHUNKS:-128}"
MLX_LM_REF="${MLX_LM_REF:-main}"   # ipsupport-llc/mlx-lm ref (mlx_lm.models.inkling2)

case "$VARIANT" in
  mini)
    MODEL_ID="${MODEL_ID:-thinkingmachines/Inkling-2-Mini}"
    RECIPE="${RECIPE:-attn=8,experts=3}"
    HF_REPO="${HF_REPO:-roman220220/Inkling-2-Mini-gptq-mlx-jang}"
    MAX_PPL_RATIO="${MAX_PPL_RATIO:-1.2}"   # 3-bit experts, as Ornith
    ;;
  mini-small)
    # Smaller: the routed experts' gate / up at 2 bits, down at 3.
    MODEL_ID="${MODEL_ID:-thinkingmachines/Inkling-2-Mini}"
    RECIPE="${RECIPE:-attn=8,experts_gate_up=2,experts_down=3}"
    HF_REPO="${HF_REPO:-roman220220/Inkling-2-Mini-gptq-mlx-jang-small}"
    MAX_PPL_RATIO="${MAX_PPL_RATIO:-1.5}"   # an extreme quant: the card says how much worse
    ;;
  *) echo "VARIANT must be mini or mini-small (or add a preset)" >&2; exit 2 ;;
esac
NAME="${HF_REPO#*/}"
CARD="${CARD:-$HERE/../cards/$NAME.md}"
DATA="$WORK/data"
CHECK_JSON="$WORK/check-$VARIANT.json"
REF="$WORK/ref-${MODEL_ID##*/}"   # per model: variants of one model share it
GPTQ="$WORK/gptq-$VARIANT"
ONGRID="$WORK/ongrid-$VARIANT"
OUT_DIR="${OUT_DIR:-$WORK/$NAME}"
export HF_HOME="${HF_HOME:-$WORK/hf}"

# Before any step: a missing token shows only at the upload, at the end
# (dory-quant/poc/dory_mlx_pipeline.sh has the story).
if [ "${PUBLISH:-0}" = 1 ] && [ ! -s "$HF_HOME/token" ]; then
  echo "no HF token at \$HF_HOME/token ($HF_HOME/token) -- push one from the operator's machine (stdin only)" >&2
  exit 1
fi

pipeline_init inkling2 "VARIANT=$VARIANT MODEL_ID=$MODEL_ID RECIPE=$RECIPE OUT_DIR=$OUT_DIR"

# mlx[cuda] brings its own NCCL into the venv, and the image's torch then
# fails to load (undefined ncclCommWindowRegister) wherever transformers
# imports it -- mlx_lm does. The MLX steps preload the NCCL next to torch.
TORCH_NCCL="$(python3 -c 'import importlib.util, pathlib; t = pathlib.Path(importlib.util.find_spec("torch").origin).parents[1]; print(next(iter(sorted((t / "nvidia" / "nccl" / "lib").glob("libnccl.so*"))), ""))' 2>/dev/null || true)"
mlx_env () { if [ -n "$TORCH_NCCL" ]; then env LD_PRELOAD="$TORCH_NCCL" "$@"; else "$@"; fi; }

# --- setup -------------------------------------------------------------------
# transformers >= 5.24: the first with inkling2_loop (NVFP4 experts dequantized on load).
if [ "${SETUP:-0}" = 1 ]; then
  run_step setup "pip deps" bash -c "
    pip install --quiet 'transformers>=5.24' accelerate safetensors huggingface_hub hf_transfer pillow numpy pandas pyarrow &&
    { python3 -c 'import torchvision' 2>/dev/null || pip install --quiet --no-deps \
        \"torchvision==\$(python3 -c 'import torch; m=int(torch.__version__.split(\".\")[1]); print(f\"0.{m + 15}.*\")')\" \
        --index-url https://download.pytorch.org/whl/cu128; } &&
    pip install --quiet --upgrade 'mlx[cuda]' &&
    pip install --quiet --force-reinstall --no-deps 'git+https://github.com/ipsupport-llc/mlx-lm.git@$MLX_LM_REF' &&
    pip install --quiet jinja2 protobuf sentencepiece tiktoken"
else
  skip_step setup "SETUP!=1"
fi

# --- data --------------------------------------------------------------------
if [ -s "$DATA/wiki.train.raw" ] && [ -s "$DATA/wiki.test.raw" ] && [ -s "$DATA/code.txt" ] && [ -s "$DATA/code.test.txt" ]; then
  skip_step data "already in $DATA"
else
  run_step data "wikitext-2 + Python code corpus" python3 - "$DATA" <<'PY'
import sys, pathlib, urllib.request, io, sysconfig, transformers, torch
out = pathlib.Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
# Held-out code: the Python standard library (no test suites), which the
# calibration corpus below doesn't contain.
std = pathlib.Path(sysconfig.get_paths()["stdlib"])
test = [p for p in sorted(std.rglob("*.py")) if not {"test", "tests", "idlelib", "site-packages", "dist-packages"} & set(p.parts)]
(out / "code.test.txt").write_text("\n\n".join(p.read_text(encoding="utf-8", errors="ignore") for p in test), encoding="utf-8")
for split in ("train", "test"):
    if (out / f"wiki.{split}.raw").exists():
        continue
    url = f"https://huggingface.co/datasets/Salesforce/wikitext/resolve/main/wikitext-2-raw-v1/{split}-00000-of-00001.parquet"
    import pandas as pd
    df = pd.read_parquet(io.BytesIO(urllib.request.urlopen(url).read()))
    (out / f"wiki.{split}.raw").write_text("".join(df["text"]), encoding="utf-8")
# Code calibration: the installed libraries' own Python sources, sorted for a stable order.
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
if [ -s "$REF/ref.json" ] && grep -q ppl_bf16_code "$REF/ref.json" && [ -s "$REF/ref_vision_feats.npy" ]; then
  skip_step reference "$REF has both perplexities and the vision features"
else
  run_step reference "transformers perplexity (text, code) + vision features" \
    python3 "$HERE/hf_reference.py" --model "$SNAPSHOT" --out "$REF" --wikitext "$DATA/wiki.test.raw" \
      --code "$DATA/code.test.txt"
fi

# --- calibrate -----------------------------------------------------------------
if [ -s "$GPTQ/progress.json" ] && python3 -c "
import json, sys
p = json.load(open('$GPTQ/progress.json')); m = p.get('made_with', {})
ok = p['done'] + 1 == p['layers'] and m.get('recipe') == '$RECIPE' and m.get('group_size') == $GROUP_SIZE and m.get('calib_chunks') == $CALIB_CHUNKS
sys.exit(not ok)"; then
  skip_step calibrate "all layers in $GPTQ"
else
  run_step calibrate "GPTQ $RECIPE (group $GROUP_SIZE)" \
    python3 "$HERE/gptq_inkling2.py" --model "$SNAPSHOT" --work "$GPTQ" --recipe "$RECIPE" \
      --group-size "$GROUP_SIZE" --calib-chunks "$CALIB_CHUNKS" \
      --calib "wikitext:$DATA/wiki.train.raw" --calib "code:$DATA/code.txt"
fi

# --- assemble ------------------------------------------------------------------
if grep -q ASSEMBLE_DONE "$LOG_DIR/inkling2_assemble.log" 2>/dev/null && [ "$ONGRID/quant_recipe.json" -nt "$GPTQ/progress.json" ]; then
  skip_step assemble "$ONGRID newer than the calibration"
else
  run_step assemble "calibrated decoder into $ONGRID" \
    python3 "$HERE/assemble_checkpoint.py" --model "$SNAPSHOT" --work "$GPTQ" --out "$ONGRID"
fi

# --- convert -------------------------------------------------------------------
if grep -q CONVERT_DONE "$LOG_DIR/inkling2_convert.log" 2>/dev/null && [ "$OUT_DIR/quant_recipe.json" -nt "$ONGRID/quant_recipe.json" ]; then
  skip_step convert "$OUT_DIR newer than $ONGRID"
else
  rm -rf "$OUT_DIR"
  run_step convert "MLX at the recipe's bits" \
    mlx_env python3 "$HERE/convert_mlx.py" --hf "$ONGRID" --out "$OUT_DIR" --gptq-work "$GPTQ" \
      --embed-bits "$EMBED_BITS" --head-bits "$HEAD_BITS" --vision-bits "$VISION_BITS" --mtp-bits "$MTP_BITS"
fi

# --- check ---------------------------------------------------------------------
if [ -s "$CHECK_JSON" ] && [ "$CHECK_JSON" -nt "$OUT_DIR/quant_recipe.json" ] \
   && python3 -c "import json,sys; c=json.load(open('$CHECK_JSON')); sys.exit(not (c.get('passed') and 'ppl_code' in c and c.get('max_ppl_ratio')==$MAX_PPL_RATIO))"; then
  skip_step check "$CHECK_JSON passed, newer than the model"
else
  run_step check "vision vs HF, image + text answers, perplexity (text, code)" \
    mlx_env python3 "$HERE/check_mlx.py" --model "$OUT_DIR" --ref "$REF" --wikitext "$DATA/wiki.test.raw" \
      --code "$DATA/code.test.txt" --out "$CHECK_JSON" --max-ppl-ratio "$MAX_PPL_RATIO" --expect-dspark
fi

# --- model card (LAST) -----------------------------------------------------------
if [ -f "$CARD" ]; then
  run_step card "copy $(basename "$CARD") -> README.md" bash -c '
    cp "$1" "$2/README.md"
    for img in $(grep -o "[a-z-]*banner\.png" "$1" | sort -u); do cp "$3/$img" "$2/$img"; done
    lic="$(dirname "$1")/licenses/$4-LICENSE"
    if [ -f "$lic" ]; then cp "$lic" "$2/LICENSE"; fi' _ "$CARD" "$OUT_DIR" "$HERE/../../cards_assets" "${MODEL_ID##*/}"
else
  skip_step card "no $CARD yet"
fi

# --- publish -------------------------------------------------------------------
if [ "${PUBLISH:-0}" = 1 ]; then
  [ -f "$OUT_DIR/README.md" ] || { echo "no model card in $OUT_DIR" >&2; exit 1; }
  run_step publish "hf upload $HF_REPO" hf upload "$HF_REPO" "$OUT_DIR" . --commit-message "inkling2_mlx_pipeline.sh VARIANT=$VARIANT"
else
  skip_step publish "PUBLISH!=1"
fi

echo "INKLING2_MLX_PIPELINE_DONE $OUT_DIR"
