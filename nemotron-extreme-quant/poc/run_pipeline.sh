#!/usr/bin/env bash
# Reproducible end-to-end pipeline: GPTQ-calibrate a NemotronH model at a
# single uniform bit-width (no rotation, no salient overlay -- see
# gptq_stock_convert.py's docstring for why that's the right trade-off at
# 3+ bits), then pack it with STOCK mlx_lm.convert (no custom kernel, no
# custom loader), then push the result to HF and pull it to this Mac.
#
# Every run-specific parameter is a flag, not a hardcoded constant -- run
# names/output paths are derived from the parameters so different
# bits/group-size/calibration settings never collide or get confused with
# each other on disk or on HF.
#
# Usage:
#   ./run_pipeline.sh --bits 3 --group-size 64 \
#       --calib-chunks 24 --calib-chunk-tokens 512 --moe-subbatch 12
#
#   # Sequential calibration, custom run name, skip the HF upload:
#   ./run_pipeline.sh --bits 4 --group-size 64 --sequential \
#       --run-name my-4bit-test --no-upload
#
#   # Merge a LoRA adapter into the base before GPTQ, and swap in a patched
#   # mlx-lm fork/branch for the conversion step (e.g. one that preserves
#   # mtp.* weights instead of stock mlx-lm's silent strip). Both are
#   # optional and independent; using either auto-suffixes the run name
#   # (-lora / -mtp) so the result never collides with a plain run's repo:
#   ./run_pipeline.sh --bits 4 --group-size 64 \
#       --lora-adapter roman220220/ipsupport-code-nemotron-lora \
#       --mlx-lm-git "git+https://github.com/ipsupport-llc/mlx-lm.git@nemotron-h-mtp"
#
# Requires (set as environment variables, or edit the defaults below):
#   POD_HOST, POD_PORT, POD_SSH_KEY  -- this project's RunPod GPU pod.
#     RunPod reassigns host/port on every pod start; check the current
#     values with `runpodctl pod list` / the RunPod dashboard and export
#     them before running this script if they've changed:
#       export POD_HOST=1.2.3.4 POD_PORT=12345
#   HF_USER  -- your Hugging Face username/org, for the upload step.
#   MODEL_SRC_DIR  -- path to the source bf16 HF checkpoint *on the pod*
#     (see docs/RUNBOOK.md for how to get this there in the first place).
#   WIKITEXT_PATH  -- path to wikitext-2-raw/wiki.train.raw *on the pod*.
set -euo pipefail

POD_HOST="${POD_HOST:-1.2.3.4}"
POD_PORT="${POD_PORT:-43615}"
POD_SSH_KEY="${POD_SSH_KEY:-$HOME/.runpod/ssh/runpodctl-ssh-key}"
HF_USER="${HF_USER:-your-hf-username}"
MODEL_SRC_DIR="${MODEL_SRC_DIR:-/root/nemotron30b-bf16-src}"
WIKITEXT_PATH="${WIKITEXT_PATH:-/root/llama.cpp/wikitext-2-raw/wiki.train.raw}"
POD_POC_DIR="${POD_POC_DIR:-/root/poc}"
LOCAL_MODELS_DIR="${LOCAL_MODELS_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../models" && pwd)}"

# --- defaults for run-specific parameters ---
BITS=3
QUANT_RECIPE=""
QUANT_RECIPE_MODE="positional"
GROUP_SIZE=64
CALIB_CHUNKS=24
CALIB_CHUNK_TOKENS=512
MOE_SUBBATCH=12
SEQUENTIAL=""
RUN_NAME=""
DO_UPLOAD=1
DO_DOWNLOAD=1
CHECKPOINT_EVERY=0
CHECKPOINT_HF_REPO=""
RESUME_FROM_CHECKPOINT=""
CPU_THREADS=32
SKIP_SANITY_CHECK=""
NO_DASHBOARD=""
DASHBOARD_PORT="${DASHBOARD_PORT:-8420}"
LORA_ADAPTER=""
MLX_LM_GIT=""
COMPONENT_RECIPE=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --bits) BITS="$2"; shift 2 ;;
    --quant-recipe) QUANT_RECIPE="$2"; shift 2 ;;
    --quant-recipe-mode) QUANT_RECIPE_MODE="$2"; shift 2 ;;
    --group-size) GROUP_SIZE="$2"; shift 2 ;;
    --calib-chunks) CALIB_CHUNKS="$2"; shift 2 ;;
    --calib-chunk-tokens) CALIB_CHUNK_TOKENS="$2"; shift 2 ;;
    --moe-subbatch) MOE_SUBBATCH="$2"; shift 2 ;;
    --sequential) SEQUENTIAL="--sequential"; shift 1 ;;
    --run-name) RUN_NAME="$2"; shift 2 ;;
    --no-upload) DO_UPLOAD=0; shift 1 ;;
    --no-download) DO_DOWNLOAD=0; shift 1 ;;
    --checkpoint-every) CHECKPOINT_EVERY="$2"; shift 2 ;;
    --checkpoint-hf-repo) CHECKPOINT_HF_REPO="$2"; shift 2 ;;
    --resume-from-checkpoint) RESUME_FROM_CHECKPOINT="$2"; shift 2 ;;
    --cpu-threads) CPU_THREADS="$2"; shift 2 ;;
    --skip-sanity-check) SKIP_SANITY_CHECK=1; shift 1 ;;
    --no-dashboard) NO_DASHBOARD=1; shift 1 ;;
    --lora-adapter) LORA_ADAPTER="$2"; shift 2 ;;
    --mlx-lm-git) MLX_LM_GIT="$2"; shift 2 ;;
    --component-recipe) COMPONENT_RECIPE="$2"; shift 2 ;;
    *) echo "unknown flag: $1" >&2; exit 1 ;;
  esac
done

if [[ -z "$RUN_NAME" ]]; then
  SEQ_SUFFIX=""
  if [[ -n "$SEQUENTIAL" ]]; then
    SEQ_SUFFIX="-seq"
  fi
  LORA_SUFFIX=""
  [[ -n "$LORA_ADAPTER" ]] && LORA_SUFFIX="-lora"
  MTP_SUFFIX=""
  [[ -n "$MLX_LM_GIT" ]] && MTP_SUFFIX="-mtp"
  if [[ "$QUANT_RECIPE_MODE" == "component" ]]; then
    RUN_NAME="gptq-component-${COMPONENT_RECIPE}-g${GROUP_SIZE}${SEQ_SUFFIX}${LORA_SUFFIX}${MTP_SUFFIX}"
  elif [[ -n "$QUANT_RECIPE" ]]; then
    MODE_TAG=""
    [[ "$QUANT_RECIPE_MODE" == "sensitivity" ]] && MODE_TAG="-smart"
    RUN_NAME="gptq-${QUANT_RECIPE}${MODE_TAG}-g${GROUP_SIZE}${SEQ_SUFFIX}${LORA_SUFFIX}${MTP_SUFFIX}"
  else
    RUN_NAME="gptq${BITS}bit-g${GROUP_SIZE}${SEQ_SUFFIX}${LORA_SUFFIX}${MTP_SUFFIX}"
  fi
fi

if [[ "$CHECKPOINT_EVERY" -gt 0 && -z "$CHECKPOINT_HF_REPO" ]]; then
  CHECKPOINT_HF_REPO="${HF_USER}/nemotron-30b-a3b-${RUN_NAME}-checkpoint"
fi

SSH="ssh -i $POD_SSH_KEY -p $POD_PORT root@$POD_HOST"
SCP="scp -i $POD_SSH_KEY -P $POD_PORT"
HF_STAGE_DIR="/root/lightning30b-${RUN_NAME}-src"
HF_MLX_DIR="/root/lightning30b-${RUN_NAME}-mlx"
HF_REPO="${HF_USER}/nemotron-30b-a3b-${RUN_NAME}"
LOG_FILE="/root/pipeline-${RUN_NAME}.log"

echo "=== run: ${RUN_NAME} (bits=${BITS} group_size=${GROUP_SIZE} calib=${CALIB_CHUNKS}x${CALIB_CHUNK_TOKENS} moe_subbatch=${MOE_SUBBATCH} sequential=${SEQUENTIAL:-no}) ==="
echo "=== pod: ${POD_HOST}:${POD_PORT} ==="

echo "--- syncing poc/ to pod ---"
$SCP "$(dirname "${BASH_SOURCE[0]}")"/*.py "root@${POD_HOST}:${POD_POC_DIR}/"

if [[ -z "$NO_DASHBOARD" ]]; then
  # Dashboard runs ON the pod itself (--local mode: reads log files
  # directly, no SSH round-trip per poll) and is reached through a local
  # SSH -L tunnel -- both need a fresh (re)start every time this points at
  # a (possibly new) pod, since RunPod reassigns host/port on every pod
  # restart and a stale tunnel just silently connects to nothing.
  echo "--- (re)starting dashboard on pod + local SSH tunnel on :${DASHBOARD_PORT} ---"
  # 'nohup cmd & disown' inside a single non-interactive SSH command was
  # observed to NOT reliably survive the SSH session closing (the process
  # would vanish minutes later with no crash in its own log -- nohup only
  # blocks SIGHUP, it doesn't detach the process from the session's
  # process group). 'setsid' makes the process its own session leader
  # (confirmed via `ps -o pid,sid,pgid` all matching), which is immune to
  # ANY signal the closing SSH session might propagate. Two separate SSH
  # calls (kill, then launch) instead of one chained command, since that
  # combined form was also observed to intermittently fail outright.
  $SSH "pkill -f 'pipeline_dashboard.py --local' 2>/dev/null || true" || true
  sleep 1
  $SSH "setsid nohup python3 ${POD_POC_DIR}/pipeline_dashboard.py --local --port ${DASHBOARD_PORT} \
    > /root/pipeline_dashboard.log 2>&1 < /dev/null &"
  pkill -f "ssh.*-L ${DASHBOARD_PORT}:localhost:${DASHBOARD_PORT}.*${POD_HOST}" 2>/dev/null || true
  sleep 1
  nohup ssh -i "$POD_SSH_KEY" -p "$POD_PORT" -o StrictHostKeyChecking=no -N \
    -L "${DASHBOARD_PORT}:localhost:${DASHBOARD_PORT}" "root@${POD_HOST}" \
    > /tmp/pipeline_dashboard_tunnel.log 2>&1 &
  disown
  echo "--- dashboard: http://localhost:${DASHBOARD_PORT} ---"
fi

CHECKPOINT_ARGS=""
if [[ "$CHECKPOINT_EVERY" -gt 0 ]]; then
  CHECKPOINT_ARGS="--checkpoint-every ${CHECKPOINT_EVERY} --checkpoint-hf-repo ${CHECKPOINT_HF_REPO}"
fi
RESUME_ARGS=""
if [[ -n "$RESUME_FROM_CHECKPOINT" ]]; then
  RESUME_ARGS="--resume-from-checkpoint ${RESUME_FROM_CHECKPOINT}"
fi
SANITY_CHECK_STEP=""
if [[ -z "$SKIP_SANITY_CHECK" ]]; then
  SANITY_CHECK_STEP="&& python3 -u sanity_check_hf.py --model ${HF_STAGE_DIR}"
fi
GPTQ_MODEL_DIR="${MODEL_SRC_DIR}"
MERGE_STEP=""
if [[ -n "$LORA_ADAPTER" ]]; then
  GPTQ_MODEL_DIR="/root/nemotron30b-bf16-${RUN_NAME}-merged"
  # Idempotent: skip the merge (a full bf16 forward-load + save, the same
  # cost class as one GPTQ calibration pass) if a prior run already left a
  # complete merged checkpoint here.
  # Double quotes only below -- this whole block is later spliced into an
  # outer bash -c '"'"'...'"'"' single-quoted string; an unescaped literal
  # single quote here would terminate that string early and corrupt
  # everything after it.
  MERGE_STEP="if [ ! -f \"${GPTQ_MODEL_DIR}/config.json\" ]; then \
    pip install --quiet --break-system-packages peft && \
    python3 -u merge_lora.py --base ${MODEL_SRC_DIR} --adapter ${LORA_ADAPTER} --output ${GPTQ_MODEL_DIR}; \
  else echo \"merged model already present, skipping merge\"; fi \
  && echo LORA_MERGE_DONE \
  &&"
fi
MLX_LM_INSTALL_STEP=""
if [[ -n "$MLX_LM_GIT" ]]; then
  # Swaps in a patched mlx-lm fork/branch (e.g. one that preserves mtp.*
  # weights through sanitize()/convert instead of stock mlx-lm's silent
  # strip) right before the conversion step that actually needs it --
  # --force-reinstall because pip won't otherwise treat a git URL as newer
  # than an already-satisfied "mlx-lm" from PyPI.
  # Double quotes (see MERGE_STEP's comment above) -- single quotes here
  # would terminate the outer bash -c '...' string early on the pod.
  MLX_LM_INSTALL_STEP="pip install --quiet --break-system-packages --force-reinstall \"${MLX_LM_GIT}\" && echo MLX_LM_SWAP_DONE &&"
fi
GPTQ_BITS_ARGS="--bits ${BITS}"
MLX_CONVERT_CMD="mlx_lm.convert --hf-path ${HF_STAGE_DIR} --mlx-path ${HF_MLX_DIR} -q --q-bits ${BITS} --q-group-size ${GROUP_SIZE}"
if [[ "$QUANT_RECIPE_MODE" == "component" ]]; then
  if [[ -z "$COMPONENT_RECIPE" ]]; then
    echo "--quant-recipe-mode component requires --component-recipe (e.g. jang, jang-dense)" >&2
    exit 1
  fi
  # component mode (e.g. "jang", reverse-engineered from JANG_2L-CRACK's
  # published bit allocation -- see gptq_stock_convert.py's docstring)
  # assigns bits purely by component type (attention/mamba/moe_shared/
  # moe_routed_up/moe_routed_down/lm_head/embeddings), ignoring --bits/
  # --quant-recipe/layer position entirely. This is the recipe the current
  # production ...-JANG-GPTQ-... models actually use.
  GPTQ_BITS_ARGS="--quant-recipe-mode component --component-recipe ${COMPONENT_RECIPE}"
  MLX_CONVERT_CMD="python3 -u mlx_convert_recipe.py --hf-path ${HF_STAGE_DIR} --mlx-path ${HF_MLX_DIR} --group-size ${GROUP_SIZE} --mode component --component-recipe ${COMPONENT_RECIPE}"
elif [[ -n "$QUANT_RECIPE" ]]; then
  GPTQ_BITS_ARGS="--quant-recipe ${QUANT_RECIPE} --quant-recipe-mode ${QUANT_RECIPE_MODE}"
  # Neither stock mlx_lm.convert --quant-predicate NOR this project's own
  # custom converter can be trusted here without the switch_mlp naming fix
  # (see poc/mlx_convert_recipe.py's docstring and docs/session_findings_
  # 2026-09-11.md section 7q) -- routed MoE experts are named switch_mlp.fc1/
  # fc2 in this architecture's MLX port, not up_proj/down_proj, so any
  # predicate matching only the literal substring "down_proj" silently never
  # upgrades ~99% of a MoE block's parameters regardless of recipe. Always
  # use the fixed custom converter for both recipe modes.
  NUM_LAYERS_EXPR="\$(python3 -c \"import json; print(len(json.load(open('${HF_STAGE_DIR}/config.json'))['layers_block_type']))\")"
  if [[ "$QUANT_RECIPE_MODE" == "sensitivity" ]]; then
    MLX_CONVERT_CMD="python3 -u mlx_convert_recipe.py --hf-path ${HF_STAGE_DIR} --mlx-path ${HF_MLX_DIR} --group-size ${GROUP_SIZE} --mode sensitivity"
  else
    MLX_CONVERT_CMD="python3 -u mlx_convert_recipe.py --hf-path ${HF_STAGE_DIR} --mlx-path ${HF_MLX_DIR} --group-size ${GROUP_SIZE} --mode positional --recipe ${QUANT_RECIPE} --num-layers ${NUM_LAYERS_EXPR}"
  fi
fi

echo "--- launching GPTQ calibration (nohup, log: ${LOG_FILE}) ---"
[[ -n "$CHECKPOINT_HF_REPO" ]] && echo "--- checkpointing every ${CHECKPOINT_EVERY} blocks to https://huggingface.co/${CHECKPOINT_HF_REPO} ---"
# nohup+disown, not tmux: tmux sessions on this pod image were observed to
# vanish silently (no error, empty log) moments after launch for unclear
# reasons -- nohup survives the launching ssh connection closing just as
# well and has been reliable in practice.
CLEAN_STAGE_DIR="rm -rf ${HF_STAGE_DIR};"
if [[ "$RESUME_FROM_CHECKPOINT" == "$HF_STAGE_DIR" ]]; then
  CLEAN_STAGE_DIR=""
  echo "--- resuming in place from ${HF_STAGE_DIR}, not wiping it ---"
fi
$SSH "${CLEAN_STAGE_DIR} cd ${POD_POC_DIR} && nohup bash -c '
  ${MERGE_STEP}
  python3 -u gptq_stock_convert.py \
    --model ${GPTQ_MODEL_DIR} --output ${HF_STAGE_DIR} \
    --wikitext ${WIKITEXT_PATH} \
    ${GPTQ_BITS_ARGS} --group-size ${GROUP_SIZE} \
    --calib-chunks ${CALIB_CHUNKS} --calib-chunk-tokens ${CALIB_CHUNK_TOKENS} \
    --moe-subbatch ${MOE_SUBBATCH} --cpu-threads ${CPU_THREADS} ${SEQUENTIAL} ${CHECKPOINT_ARGS} ${RESUME_ARGS} \
  && echo GPTQ_STAGE_DONE \
  ${SANITY_CHECK_STEP} \
  && ${MLX_LM_INSTALL_STEP} \
  ${MLX_CONVERT_CMD} \
  && echo MLX_CONVERT_DONE \
' > ${LOG_FILE} 2>&1 < /dev/null & disown"

echo "--- job launched; tail with: ${SSH} 'tail -f ${LOG_FILE}' ---"
echo "--- waiting for MLX_CONVERT_DONE (this can take a while; Ctrl-C is safe, the pod job keeps running) ---"

$SSH "tail -f -n +1 ${LOG_FILE}" | grep -m1 -E "MLX_CONVERT_DONE|SANITY_CHECK_FAILED|Traceback|Error"

if $SSH "tail -50 ${LOG_FILE} | grep -q MLX_CONVERT_DONE"; then
  echo "=== conversion succeeded: ${HF_MLX_DIR} ==="
else
  echo "=== conversion FAILED -- check ${LOG_FILE} on the pod ==="
  exit 1
fi

if [[ "$DO_UPLOAD" == "1" ]]; then
  echo "--- uploading to HF: ${HF_REPO} ---"
  $SSH "cd /root && hf upload ${HF_REPO} ${HF_MLX_DIR} . --repo-type model"
fi

if [[ "$DO_DOWNLOAD" == "1" ]]; then
  echo "--- downloading to ${LOCAL_MODELS_DIR}/${RUN_NAME} ---"
  mkdir -p "${LOCAL_MODELS_DIR}/${RUN_NAME}"
  rsync -avz --partial --progress -e "ssh -i ${POD_SSH_KEY} -p ${POD_PORT}" \
    "root@${POD_HOST}:${HF_MLX_DIR}/" "${LOCAL_MODELS_DIR}/${RUN_NAME}/"
fi

echo "=== done: ${RUN_NAME} ==="
echo "Local model: ${LOCAL_MODELS_DIR}/${RUN_NAME}"
[[ "$DO_UPLOAD" == "1" ]] && echo "HF: https://huggingface.co/${HF_REPO}"
