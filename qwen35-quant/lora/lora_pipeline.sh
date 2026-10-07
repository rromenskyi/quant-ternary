#!/usr/bin/env bash
# LoRA fine-tune of a Qwen3.5 checkpoint for an agent (ipsupport-code), then
# the same GPTQ -> MLX pipeline as any release (../poc/qwen35_mlx_pipeline.sh).
# Resumable like it: finished steps are skipped on a re-run.
#
#   SETUP=1 ./lora_pipeline.sh                 # train, merge, quantize, evaluate
#   PUBLISH=1 HF_TOKEN=... ./lora_pipeline.sh  # ... and publish if the eval gate passes
#
# Steps (-> $LOG_DIR/lora_<step>.log, status in $LOG_DIR/pipeline.log):
#   data       the SFT dataset (build_dataset.py + synth.py output) from a
#              private HF dataset repo: sft.jsonl, synth.jsonl, sft.eval.jsonl
#   download   the base checkpoint (HF snapshot)
#   train      LoRA (train_lora.py)
#   merge      the adapter merged into the checkpoint's own shards (merge_lora.py)
#   mtp        the MTP head retrained on the merged backbone (train_mtp.py;
#              MTP_RETRAIN=0 keeps the base's head)
#   quant      qwen35_mlx_pipeline.sh with MODEL_ID=<merged> (QUANT_VARIANT)
#   eval       base MLX build vs the new one, the agent's way (eval_lora.py
#              --backend mlx): first steps on held-out goals, reflex on greetings
#   gate       the new build must call tools on greetings no more often than
#              the base, and get no fewer first steps valid (EVAL_TOLERANCE)
#   publish    PUBLISH=1 and the gate passed: the MLX build (quant pipeline's
#              publish) and the adapter (private repo)
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/../../gemma4-quant/poc/pipeline_lib.sh"
LORA="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"   # pipeline_lib sets HERE to its own folder

WORK="${WORK:-/workspace}"
BASE_MODEL="${BASE_MODEL:-microsoft/FrogNano-4B-2609}"
LORA_NAME="${LORA_NAME:-ipsupport-code}"
DATASET="${DATASET:-roman220220/ipsupport-code-sft}"
QUANT_VARIANT="${QUANT_VARIANT:-frognano-4b-ipsupport-code}"
BASE_MLX="${BASE_MLX:-roman220220/FrogNano-4B-2609-gptq-mlx-jang}"
ADAPTER_REPO="${ADAPTER_REPO:-roman220220/FrogNano-4B-2609-ipsupport-code-lora}"
LORA_TARGETS="${LORA_TARGETS:-attention}"
LORA_R="${LORA_R:-16}"
LORA_ALPHA="${LORA_ALPHA:-32}"
LORA_LR="${LORA_LR:-1e-4}"
LORA_EPOCHS="${LORA_EPOCHS:-2}"
LORA_MAX_LEN="${LORA_MAX_LEN:-16384}"
MTP_RETRAIN="${MTP_RETRAIN:-1}"
EVAL_SAMPLES="${EVAL_SAMPLES:-3}"
EVAL_TOLERANCE="${EVAL_TOLERANCE:-0.05}"
export HF_HOME="${HF_HOME:-$WORK/hf}"

LDIR="$WORK/lora-$LORA_NAME"
DATA="$LDIR/data"
ADAPTER="$LDIR/adapter"
MERGED="${MERGED:-$WORK/${BASE_MODEL##*/}-$LORA_NAME}"
EVAL_JSON="$LDIR/eval.json"
mkdir -p "$LDIR"
pipeline_init lora "BASE_MODEL=$BASE_MODEL LORA_NAME=$LORA_NAME MERGED=$MERGED"

TORCH_NCCL="$(python3 -c 'import importlib.util, pathlib; t = pathlib.Path(importlib.util.find_spec("torch").origin).parents[1]; print(next(iter(sorted((t / "nvidia" / "nccl" / "lib").glob("libnccl.so*"))), ""))' 2>/dev/null || true)"
mlx_env () { if [ -n "$TORCH_NCCL" ]; then env LD_PRELOAD="$TORCH_NCCL" "$@"; else "$@"; fi; }

if [ "${SETUP:-0}" = 1 ]; then
  run_step setup "peft, flash-linear-attention" pip install --quiet peft flash-linear-attention
else
  skip_step setup "SETUP!=1"
fi

# --- data ----------------------------------------------------------------------
if [ -s "$DATA/sft.jsonl" ] && [ -s "$DATA/sft.eval.jsonl" ]; then
  skip_step data "already in $DATA"
else
  run_step data "$DATASET (private)" python3 -c "
from huggingface_hub import snapshot_download
snapshot_download('$DATASET', repo_type='dataset', local_dir='$DATA', allow_patterns=['*.jsonl'])"
fi
TRAIN_ARGS=(--train "$DATA/sft.jsonl")
[ -s "$DATA/synth.jsonl" ] && TRAIN_ARGS+=(--train "$DATA/synth.jsonl")

# --- download --------------------------------------------------------------------
_PIPE_CUR=download; _pipe_mark START download "$BASE_MODEL"
SNAPSHOT="$(hf_snapshot "$BASE_MODEL")"
_pipe_mark DONE download "$SNAPSHOT"; _PIPE_CUR=""

# --- train -------------------------------------------------------------------------
WANT="targets=$LORA_TARGETS r=$LORA_R alpha=$LORA_ALPHA lr=$LORA_LR epochs=$LORA_EPOCHS max_len=$LORA_MAX_LEN"
if [ -s "$ADAPTER/adapter_model.safetensors" ] && [ "$(cat "$ADAPTER/made_with" 2>/dev/null)" = "$WANT" ]; then
  skip_step train "$ADAPTER ($WANT)"
else
  rm -rf "$ADAPTER"
  run_step train "LoRA $WANT" bash -c "python3 '$LORA/train_lora.py --model '$SNAPSHOT' $(printf "%q " "${TRAIN_ARGS[@]}") \
    --eval '$DATA/sft.eval.jsonl' --out '$ADAPTER' --targets '$LORA_TARGETS' --r $LORA_R --alpha $LORA_ALPHA \
    --lr $LORA_LR --epochs $LORA_EPOCHS --max-len $LORA_MAX_LEN && echo '$WANT' > '$ADAPTER/made_with'"
fi

# --- merge ---------------------------------------------------------------------------
if [ -f "$MERGED/.merged" ] && [ "$MERGED/.merged" -nt "$ADAPTER/made_with" ]; then
  skip_step merge "$MERGED newer than the adapter"
else
  rm -rf "$MERGED"
  run_step merge "adapter into $MERGED" bash -c "python3 '$LORA/merge_lora.py --model '$SNAPSHOT' --adapter '$ADAPTER' --out '$MERGED' && touch '$MERGED/.merged'"
fi

# --- mtp ----------------------------------------------------------------------------------
# Plain text keeps the head general; the quant pipeline's data step makes it
# (here when that ran before, e.g. a re-run).
TEXT_ARGS=()
[ -s "$WORK/data/wiki.train.raw" ] && TEXT_ARGS+=(--text "wikitext:$WORK/data/wiki.train.raw")
[ -s "$WORK/data/code.txt" ] && TEXT_ARGS+=(--text "code:$WORK/data/code.txt")
if [ "$MTP_RETRAIN" != 1 ]; then
  skip_step mtp "MTP_RETRAIN!=1 (the base's head)"
elif [ -f "$MERGED/.mtp-retrained" ] && [ "$MERGED/.mtp-retrained" -nt "$MERGED/.merged" ]; then
  skip_step mtp "already retrained"
elif ! python3 -c "import json,sys; sys.exit(not any(k.startswith('mtp.') for k in json.load(open('$MERGED/model.safetensors.index.json'))['weight_map']))"; then
  skip_step mtp "the checkpoint has no MTP head"
else
  run_step mtp "MTP head on the merged backbone" bash -c "python3 '$LORA/train_mtp.py --model '$MERGED' $(printf "%q " "${TRAIN_ARGS[@]}") \
    --eval '$DATA/sft.eval.jsonl' $(printf "%q " "${TEXT_ARGS[@]}") && touch '$MERGED/.mtp-retrained'"
fi

# --- quant ------------------------------------------------------------------------------------
# Its own resumable steps; a retrained head is calibrated again (its head file
# is older than the merged checkpoint's head).
GPTQ_HEAD="$WORK/gptq-$QUANT_VARIANT/layers/mtp.safetensors"
if [ -f "$GPTQ_HEAD" ] && [ -f "$MERGED/.mtp-retrained" ] && [ "$MERGED/.mtp-retrained" -nt "$GPTQ_HEAD" ]; then rm -f "$GPTQ_HEAD"; fi
run_step quant "qwen35_mlx_pipeline.sh VARIANT=$QUANT_VARIANT" \
  env VARIANT="$QUANT_VARIANT" MODEL_ID="$MERGED" WORK="$WORK" PUBLISH=0 "$LORA/../poc/qwen35_mlx_pipeline.sh"
OUT_DIR="$(sed -n 's/^QWEN35_MLX_PIPELINE_DONE //p' "$LOG_DIR/lora_quant.log" | tail -1)"

# --- eval -------------------------------------------------------------------------------------------
if [ -s "$EVAL_JSON" ] && [ "$EVAL_JSON" -nt "$OUT_DIR/quant_recipe.json" ]; then
  skip_step eval "$EVAL_JSON newer than the build"
else
  BASE_DIR="$(hf_snapshot "$BASE_MLX")"
  run_step eval "$BASE_MLX vs $OUT_DIR" mlx_env python3 "$LORA/eval_lora.py" --backend mlx --model "$BASE_DIR" \
    --compare "$OUT_DIR" --eval "$DATA/sft.eval.jsonl" --samples "$EVAL_SAMPLES" --out "$EVAL_JSON"
fi

# --- gate ---------------------------------------------------------------------------------------------
run_step gate "reflex no worse, first steps no worse than -$EVAL_TOLERANCE" python3 - "$EVAL_JSON" "$EVAL_TOLERANCE" <<'PY'
import json, sys
r, tol = json.load(open(sys.argv[1])), float(sys.argv[2])
def share(m, k):
    f = r[m]["first_step"]
    return f[k] / max(f["n"], 1)
reflex = {m: r[m]["reflex"]["calls"] / max(r[m]["reflex"]["n"], 1) for m in ("base", "lora")}
print({m: {"valid": share(m, "valid"), "same": share(m, "same"), "reflex_calls": reflex[m]} for m in ("base", "lora")})
bad = []
if reflex["lora"] > reflex["base"]:
    bad.append(f"tool calls on greetings: {reflex['lora']:.2f} > base {reflex['base']:.2f}")
if share("lora", "valid") < share("base", "valid") - tol:
    bad.append(f"valid first steps: {share('lora', 'valid'):.2f} < base {share('base', 'valid'):.2f} - {tol}")
if bad:
    sys.exit("; ".join(bad))
print("GATE_PASSED")
PY

# --- publish ------------------------------------------------------------------------------------------
if [ "${PUBLISH:-0}" = 1 ]; then
  run_step publish "MLX build (quant pipeline) + adapter -> $ADAPTER_REPO (private)" bash -c "
    env VARIANT='$QUANT_VARIANT' MODEL_ID='$MERGED' WORK='$WORK' PUBLISH=1 '$LORA/../poc/qwen35_mlx_pipeline.sh' &&
    python3 -c \"
from huggingface_hub import HfApi
api = HfApi()
api.create_repo('$ADAPTER_REPO', private=True, exist_ok=True)
api.upload_folder(repo_id='$ADAPTER_REPO', folder_path='$ADAPTER', commit_message='lora_pipeline.sh $LORA_NAME')\""
else
  skip_step publish "PUBLISH!=1"
fi
echo "LORA_PIPELINE_DONE $OUT_DIR"
