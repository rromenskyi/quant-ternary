#!/usr/bin/env bash
# Bootstrap a fresh RunPod pod for this project's GPTQ pipeline -- turns
# docs/RUNBOOK.md's manual copy-paste steps into a single, idempotent,
# reusable script. Needed the first time a pod is set up, and again any
# time a pod's container disk gets wiped (e.g. a RunPod disk-size change,
# which recreates the container even without a persistent volume attached).
#
# Every step checks whether its result already exists and skips it if so,
# so re-running this after a partial failure (or just to double-check
# everything is in place) is safe and cheap.
#
# Usage:
#   POD_HOST=1.2.3.4 POD_PORT=12345 ./setup_pod.sh
#   POD_HOST=1.2.3.4 POD_PORT=12345 ./setup_pod.sh --with-axolotl
set -euo pipefail

WITH_AXOLOTL=""
for arg in "$@"; do
  case "$arg" in
    --with-axolotl) WITH_AXOLOTL=1 ;;
    *) echo "unknown flag: $arg" >&2; exit 1 ;;
  esac
done

POD_HOST="${POD_HOST:?set POD_HOST}"
POD_PORT="${POD_PORT:?set POD_PORT}"
POD_SSH_KEY="${POD_SSH_KEY:-$HOME/.runpod/ssh/runpodctl-ssh-key}"
MODEL_SRC_DIR="${MODEL_SRC_DIR:-/root/nemotron30b-bf16-src}"
WIKITEXT_DIR="${WIKITEXT_DIR:-/root/llama.cpp/wikitext-2-raw}"
POD_POC_DIR="${POD_POC_DIR:-/root/poc}"
AXOLOTL_VENV="${AXOLOTL_VENV:-/root/axolotl_venv}"

SSH="ssh -i $POD_SSH_KEY -p $POD_PORT -o StrictHostKeyChecking=no root@$POD_HOST"

echo "--- syncing poc/*.py to ${POD_HOST}:${POD_POC_DIR} ---"
$SSH "mkdir -p ${POD_POC_DIR}"
scp -i "$POD_SSH_KEY" -P "$POD_PORT" -o StrictHostKeyChecking=no \
    "$(dirname "${BASH_SOURCE[0]}")"/*.py "root@${POD_HOST}:${POD_POC_DIR}/"

echo "--- installing base Python deps (skips already-satisfied ones) ---"
$SSH "pip install --quiet --break-system-packages \
    torch transformers accelerate huggingface_hub[hf_xet] \
    'mlx[cuda12]' mlx-lm pandas"

echo "--- source model (~62GB, skipped if already present) ---"
$SSH "
if [ -f '${MODEL_SRC_DIR}/config.json' ]; then
  echo 'source model already present, skipping download'
else
  hf download nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16 --local-dir '${MODEL_SRC_DIR}'
fi
"

echo "--- calibration corpus: wikitext-2-raw (skipped if already present) ---"
# The RUNBOOK's S3 URL for this is broken (redirect/cert issues observed
# this session) -- pull via the HF datasets mirror and convert from
# parquet instead, the approach that actually worked.
$SSH "
if [ -f '${WIKITEXT_DIR}/wiki.train.raw' ] && [ -f '${WIKITEXT_DIR}/wiki.test.raw' ]; then
  echo 'wikitext already present, skipping'
else
  mkdir -p '${WIKITEXT_DIR}'
  hf download Salesforce/wikitext --repo-type dataset --include 'wikitext-2-raw-v1/*' --local-dir /root/wikitext-hf
  python3 -c \"
import pandas as pd
for split, name in [('train', 'wiki.train.raw'), ('test', 'wiki.test.raw'), ('validation', 'wiki.valid.raw')]:
    df = pd.read_parquet(f'/root/wikitext-hf/wikitext-2-raw-v1/{split}-00000-of-00001.parquet')
    with open(f'${WIKITEXT_DIR}/{name}', 'w', encoding='utf-8') as f:
        f.write(''.join(df['text'].tolist()))
    print(f'wrote ${WIKITEXT_DIR}/{name}')
\"
fi
"

echo "--- LICENSE (copied from the source model, needed for HF model-card uploads) ---"
$SSH "cp -n '${MODEL_SRC_DIR}/LICENSE' /root/LICENSE 2>/dev/null || true"

if [[ -n "$WITH_AXOLOTL" ]]; then
  echo "--- Axolotl venv for ipsupport-code LoRA fine-tuning (isolated from the base env" \
       "above so a broken/conflicting dependency there can't take down the GPTQ pipeline) ---"
  $SSH "
if [ -x '${AXOLOTL_VENV}/bin/python3' ] && '${AXOLOTL_VENV}/bin/python3' -c 'import axolotl' 2>/dev/null; then
  echo 'axolotl venv already set up, skipping'
else
  python3 -m venv '${AXOLOTL_VENV}'
  '${AXOLOTL_VENV}/bin/pip' install --quiet --upgrade pip
  # Staged, not one shot: flash-attn's setup.py imports torch at build
  # time, so torch must already be installed before flash-attn is even
  # attempted (a single combined 'pip install axolotl[flash-attn]' fails
  # with 'ModuleNotFoundError: No module named torch' during the build).
  '${AXOLOTL_VENV}/bin/pip' install torch --index-url https://download.pytorch.org/whl/cu121
  '${AXOLOTL_VENV}/bin/pip' install packaging ninja wheel setuptools
  '${AXOLOTL_VENV}/bin/pip' install axolotl
  export PATH=/usr/local/cuda-12.8/bin:\$PATH
  export LD_LIBRARY_PATH=/usr/local/cuda-12.8/lib64:\${LD_LIBRARY_PATH:-}
  '${AXOLOTL_VENV}/bin/pip' install flash-attn --no-build-isolation
fi
"
fi

echo "=== pod ready: model at ${MODEL_SRC_DIR}, wikitext at ${WIKITEXT_DIR}, poc/ synced ==="
[[ -n "$WITH_AXOLOTL" ]] && echo "=== axolotl venv ready at ${AXOLOTL_VENV} ==="
