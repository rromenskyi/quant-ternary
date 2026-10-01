#!/usr/bin/env bash
# Mac-side driver for all heavy compute on a RunPod GPU pod:
#   create pod -> setup (mlx[cuda] + mlx-audio fork) -> download bf16 model ->
#   capture calibration -> GPTQ/RTN LLM variants -> pull results -> DELETE pod.
#
#   scripts/pod_pipeline.sh --variant 'gptq3:--bits 3' --variant 'rtn3:--bits 3 --rtn'
#
# Every variant is "<name>:<gptq_llm.py flags>"; results land in $RESULTS/<name>/
# (llm.safetensors + quant.json + gptq_log.json), ready for splice_llm.py.
# The pod is deleted on exit (success, failure or ctrl-c) unless KEEP_POD=1;
# POD_ID=<id> reuses an existing pod instead of creating one.
set -euo pipefail
HERE="$(cd "$(dirname "$0")/.." && pwd)"
RPC="${RPC:-$HOME/.local/bin/runpodctl}"
GPU="${GPU:-NVIDIA A40}"
IMAGE="${IMAGE:-runpod/pytorch:1.0.3-cu1281-torch291-ubuntu2404}"
DISK_GB="${DISK_GB:-120}"
KEY="${POD_SSH_KEY:-$HOME/.runpod/ssh/runpodctl-ssh-key}"
RESULTS="${RESULTS:-$HERE/pod_results}"
BF16_REPO="${BF16_REPO:-mlx-community/NemotronLabs-VoiceChat-11B-bf16}"
FORK="${FORK:-https://github.com/ipsupport-llc/mlx-audio.git}"
FORK_REF="${FORK_REF:-b99f797}"
CALIB_CLIPS="${CALIB_CLIPS:-256}"
CAPTURE_PROCS="${CAPTURE_PROCS:-3}"
VOLUME_GB="${VOLUME_GB:-80}"        # /workspace on a pod volume, so `pod stop` keeps it
POD_END="${POD_END:-delete}"        # delete | stop | keep
# The captured calibration set is on the private research repo: pull it
# (CALIB_SOURCE=hf) instead of capturing again (capture). Results go up to
# the same repo (RESULTS_TO=hf) and come down from there to the Mac, or are
# rsynced (rsync). The HF token (HF_TOKEN_FILE) goes over ssh stdin into the
# one remote process's environment: never onto the pod's disk or into a log.
HF_REPO="${HF_REPO:-roman220220/NemotronLabs-VoiceChat-11B-gptq-research}"
HF_TOKEN_FILE="${HF_TOKEN_FILE:-$HOME/.cache/huggingface/token}"
CALIB_SOURCE="${CALIB_SOURCE:-hf}"  # hf | capture
RESULTS_TO="${RESULTS_TO:-hf}"      # hf | rsync
VARIANTS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --variant) VARIANTS+=("$2"); shift 2 ;;
    *) echo "unknown arg $1"; exit 2 ;;
  esac
done
[[ ${#VARIANTS[@]} -gt 0 ]] || { echo "need at least one --variant"; exit 2; }
LOG="$RESULTS/pipeline.log"
mkdir -p "$RESULTS"
log() { echo "$(date '+%F %T') $*" | tee -a "$LOG"; }

cleanup() {
  [[ -n "${POD_ID:-}" ]] || return 0
  case "$POD_END" in
    delete)
      log "deleting pod $POD_ID"
      "$RPC" pod delete "$POD_ID" >/dev/null 2>&1 || log "WARNING: pod delete failed -- delete $POD_ID by hand"
      "$RPC" pod list 2>/dev/null | grep -q "$POD_ID" && log "WARNING: $POD_ID still listed" || log "pod $POD_ID gone" ;;
    stop)
      log "stopping pod $POD_ID (volume kept; storage billing only)"
      "$RPC" pod stop "$POD_ID" >/dev/null 2>&1 || log "WARNING: pod stop failed -- stop $POD_ID by hand"
      "$RPC" pod get "$POD_ID" 2>/dev/null | grep -o '"desiredStatus": *"[A-Z]*"' | tee -a "$LOG" ;;
    *) log "leaving pod $POD_ID running (POD_END=$POD_END) -- it keeps billing" ;;
  esac
}
trap cleanup EXIT

if [[ -z "${POD_ID:-}" ]]; then
  log "creating pod: $GPU, $IMAGE, disk ${DISK_GB}GB"
  POD_JSON="$("$RPC" pod create --name voicechat-quant --gpu-id "$GPU" --image "$IMAGE" \
    --container-disk-in-gb "$DISK_GB" --volume-in-gb "$VOLUME_GB" --volume-mount-path /workspace \
    --ports '22/tcp' --wait --wait-timeout 15m)"
  POD_ID="$(python3 -c 'import json,sys; print(json.loads(sys.stdin.read())["id"])' <<<"$POD_JSON")"
  CREATED=1
  log "pod $POD_ID up"
elif ! "$RPC" pod get "$POD_ID" 2>/dev/null | python3 -c 'import json,sys; s=json.load(sys.stdin).get("ssh") or {}; sys.exit(0 if s.get("ip") and s.get("port") else 1)'; then
  # A stopped pod (POD_END=stop) comes back with its volume; ssh needs it running.
  log "starting pod $POD_ID"
  "$RPC" pod start "$POD_ID" >/dev/null
  for _ in $(seq 90); do
    "$RPC" pod get "$POD_ID" 2>/dev/null | python3 -c 'import json,sys; s=json.load(sys.stdin).get("ssh") or {}; sys.exit(0 if s.get("ip") and s.get("port") else 1)' && break
    sleep 10
  done
fi

read -r HOST PORT < <("$RPC" pod get "$POD_ID" | python3 -c '
import json, sys
ssh = json.load(sys.stdin)["ssh"]
print(ssh["ip"], ssh["port"])')
SSH=(ssh -i "$KEY" -p "$PORT" -o StrictHostKeyChecking=accept-new -o ServerAliveInterval=30 "root@$HOST")
RSYNC_E="ssh -i $KEY -p $PORT -o StrictHostKeyChecking=accept-new"
log "ssh root@$HOST -p $PORT"

step() {  # step <name> <remote command>; skipped when /workspace/.done_<name> exists
  local name="$1"; shift
  if "${SSH[@]}" "test -f /workspace/.done_$name"; then log "skip $name"; return; fi
  log "step $name"
  "${SSH[@]}" "set -euo pipefail; cd /workspace; $*; touch /workspace/.done_$name" 2>&1 | tee -a "$RESULTS/$name.log"
}

with_token() {  # with_token <name> <remote command>: like step, HF_TOKEN from stdin
  local name="$1"; shift
  if "${SSH[@]}" "test -f /workspace/.done_$name"; then log "skip $name"; return; fi
  log "step $name"
  "${SSH[@]}" "read -r HF_TOKEN; export HF_TOKEN; set -euo pipefail; cd /workspace; $*; touch /workspace/.done_$name" \
    < "$HF_TOKEN_FILE" 2>&1 | tee -a "$RESULTS/$name.log"
}

if [[ "$CALIB_SOURCE" == capture ]]; then
  rsync -rltz -e "$RSYNC_E" "$HERE/scripts" "$HERE/calib" root@"$HOST":/workspace/vcq/
else
  rsync -rltz -e "$RSYNC_E" "$HERE/scripts" root@"$HOST":/workspace/vcq/
fi

step setup "
  apt-get -qq update >/dev/null && apt-get -qq install -y git rsync >/dev/null
  python3 -m venv /workspace/venv && . /workspace/venv/bin/activate
  pip -q install -U pip && pip -q install 'mlx[cuda]' huggingface_hub hf_transfer numpy
  git clone -q $FORK /workspace/mlx-audio && git -C /workspace/mlx-audio checkout -q $FORK_REF
  pip -q install -e /workspace/mlx-audio
  python -c 'import mlx.core as mx; print(mx.default_device(), mx.__version__)'"

step deps "
  . /workspace/venv/bin/activate
  pip -q install -e '/workspace/mlx-audio[stt]' sentencepiece
  # torch only for the GPTQ Hessian inverse (float64 Cholesky on CUDA; MLX linalg is CPU-only
  # and takes minutes per 4480^2 matrix). Its cu128 wheels downgrade two libs mlx[cuda] pins,
  # so put those back afterwards (both frameworks then work; checked on MLX 0.32.2 / torch 2.11).
  pip -q install torch --index-url https://download.pytorch.org/whl/cu128
  pip -q install 'nvidia-cuda-nvrtc-cu12==12.9.*' 'nvidia-cufft-cu12==11.4.*'
  python -c 'import torch, mlx.core as mx; assert torch.cuda.is_available(); print(mx.compile(lambda a: a * 2)(mx.ones(2)))' 
  python -c 'import mlx_audio.sts, sentencepiece; print(\"imports ok\")'"

step download "
  . /workspace/venv/bin/activate
  HF_HUB_ENABLE_HF_TRANSFER=1 hf download $BF16_REPO --local-dir /workspace/vc-bf16 >/dev/null
  du -sh /workspace/vc-bf16"

if [[ "$CALIB_SOURCE" == hf ]]; then
with_token calib "
  . /workspace/venv/bin/activate
  hf download $HF_REPO --include 'calib/*' --local-dir /workspace/hfcalib >/dev/null
  rm -rf /workspace/calib && mv /workspace/hfcalib/calib /workspace/calib
  test \$(ls /workspace/calib/c*.npy | wc -l) -eq $CALIB_CLIPS"
else
# Capture is sequential per clip (batch 1, one 80 ms frame at a time), so one
# process leaves a big GPU mostly idle: run CAPTURE_PROCS shards side by side.
step capture "
  . /workspace/venv/bin/activate
  for i in \$(seq 0 $((CAPTURE_PROCS - 1))); do
    python /workspace/vcq/scripts/calib_capture.py --model /workspace/vc-bf16 \
      --audio /workspace/vcq/calib/audio --out /workspace/calib --limit $CALIB_CLIPS \
      --shard \$i/$CAPTURE_PROCS > /workspace/capture_\$i.log 2>&1 &
  done
  wait
  tail -n 1 /workspace/capture_*.log
  test \$(ls /workspace/calib/c*.npy | wc -l) -eq $CALIB_CLIPS"
fi

for v in "${VARIANTS[@]}"; do
  name="${v%%:*}"; flags="${v#*:}"
  step "gptq_$name" "
    . /workspace/venv/bin/activate
    python /workspace/vcq/scripts/gptq_llm.py --model /workspace/vc-bf16 --calib /workspace/calib \
      --out /workspace/out/$name $flags"
  mkdir -p "$RESULTS/$name"
  if [[ "$RESULTS_TO" == hf ]]; then
    with_token "upload_$name" "
      . /workspace/venv/bin/activate
      python /workspace/vcq/scripts/hf_upload.py --repo $HF_REPO --folder /workspace/out/$name --path-in-repo $name"
    log "pulling $name from HF"
    "${MAC_PY:-$HOME/gh/mlx-audio-fork/.venv/bin/python3}" -c "import sys; from huggingface_hub import snapshot_download as d; d(sys.argv[1], allow_patterns=[sys.argv[2] + '/*'], local_dir=sys.argv[3])" \
      "$HF_REPO" "$name" "$RESULTS/hf" && log "$name in $RESULTS/hf/$name"
  else
    log "pulling $name"
    rsync -a --partial -e "$RSYNC_E" root@"$HOST":/workspace/out/"$name"/ "$RESULTS/$name/"
  fi
done
rsync -a -e "$RSYNC_E" root@"$HOST":"/workspace/calib/manifest*.json" "$RESULTS/" || true
log "done"
