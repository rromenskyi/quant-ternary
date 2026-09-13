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
MODEL_REPO_ID="${MODEL_REPO_ID:-nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16}"
MODEL_SRC_DIR="${MODEL_SRC_DIR:-/root/nemotron30b-bf16-src}"
WIKITEXT_DIR="${WIKITEXT_DIR:-/root/llama.cpp/wikitext-2-raw}"
POD_POC_DIR="${POD_POC_DIR:-/root/poc}"
AXOLOTL_VENV="${AXOLOTL_VENV:-/root/axolotl_venv}"

SSH="ssh -i $POD_SSH_KEY -p $POD_PORT -o StrictHostKeyChecking=no root@$POD_HOST"

echo "--- checking HF auth ---"
# Check this FIRST, before any other step -- a wiped/missing token doesn't
# break read-only steps (public model download, wikitext download), so it
# stays invisible until the very first upload deep into the pipeline (this
# bit us once: a RunPod disk-resize wiped the token along with everything
# else, and it wasn't discovered until an hf upload failed after a long run).
HF_TOKEN_LOCAL_FILE="${HF_TOKEN_LOCAL_FILE:-$HOME/.runpod/hf_token}"
if $SSH "test -s ~/.cache/huggingface/token"; then
  echo "HF token file present"
elif [[ -s "$HF_TOKEN_LOCAL_FILE" ]]; then
  # Push straight over stdin -- never as a CLI arg, never echoed to this
  # script's own output, so it can't end up in shell history or a log.
  echo "no token on pod, pushing from local $HF_TOKEN_LOCAL_FILE"
  $SSH "mkdir -p ~/.cache/huggingface && cat > ~/.cache/huggingface/token" < "$HF_TOKEN_LOCAL_FILE"
  echo "HF token pushed to pod"
else
  echo "ERROR: no HF token found at ~/.cache/huggingface/token on the pod," >&2
  echo "and no local fallback at $HF_TOKEN_LOCAL_FILE either." >&2
  echo "Either save one locally (chmod 600 recommended):" >&2
  echo "  cat > $HF_TOKEN_LOCAL_FILE <<< 'hf_your_token_here' && chmod 600 $HF_TOKEN_LOCAL_FILE" >&2
  echo "or log in on the pod directly (paste the token via stdin, never as a CLI arg):" >&2
  echo "  ssh -i $POD_SSH_KEY -p $POD_PORT root@$POD_HOST \"mkdir -p ~/.cache/huggingface && cat > ~/.cache/huggingface/token\" <<< 'hf_your_token_here'" >&2
  exit 1
fi

echo "--- syncing poc/*.py to ${POD_HOST}:${POD_POC_DIR} ---"
$SSH "mkdir -p ${POD_POC_DIR}"
scp -i "$POD_SSH_KEY" -P "$POD_PORT" -o StrictHostKeyChecking=no \
    "$(dirname "${BASH_SOURCE[0]}")"/*.py "root@${POD_HOST}:${POD_POC_DIR}/"

echo "--- installing base Python deps (skips already-satisfied ones) ---"
# 'mlx[cuda12]' is NOT a real extra -- current mlx releases only expose
# 'mlx[cuda]' (pulls in the separate mlx-cuda-12 package with the actual
# libmlx.so runtime). The base 'mlx' wheel alone has no CUDA runtime on
# Linux at all, so getting this wrong doesn't just warn, it silently
# leaves MLX non-functional until someone notices at conversion time.
$SSH "pip install --quiet --break-system-packages \
    torch transformers accelerate huggingface_hub[hf_xet] \
    'mlx[cuda]' mlx-lm pandas pyarrow"

# The base install above can still land on a CPU-only mlx: this image's
# preinstalled torch pins nvidia-cublas-cu12/nvidia-cuda-nvrtc-cu12/
# nvidia-cufft-cu12 to exact 12.8.x builds, while mlx-cuda-12 wants exact
# 12.9.x -- pip's resolver then silently downgrades mlx itself to an old
# version that doesn't pull mlx-cuda-12 at all, rather than erroring
# (confirmed live: import succeeds, but mx.default_device() reports
# Device(cpu, 0), not gpu). Force mlx[cuda] to the newest release (which
# upgrades those nvidia-cu12 packages past torch's pin) -- CUDA 12.x
# libraries are ABI-compatible across minor versions in practice, and
# this was verified not to break torch's own .cuda() ops on the same pod
# earlier this session.
$SSH "pip install --quiet --break-system-packages --upgrade --force-reinstall 'mlx[cuda]'"
if ! $SSH "python3 -c \"import mlx.core as mx; import sys; sys.exit(0 if mx.default_device().type == mx.DeviceType.gpu else 1)\""; then
  echo "ERROR: mlx did not get a GPU device after the upgrade -- inspect manually:" >&2
  echo "  ssh -i $POD_SSH_KEY -p $POD_PORT root@$POD_HOST \"python3 -c 'import mlx.core as mx; print(mx.default_device())'\"" >&2
  exit 1
fi
echo "mlx confirmed on GPU"

echo "--- source model ${MODEL_REPO_ID} (skipped if already present) ---"
$SSH "
if [ -f '${MODEL_SRC_DIR}/config.json' ]; then
  echo 'source model already present, skipping download'
else
  hf download '${MODEL_REPO_ID}' --local-dir '${MODEL_SRC_DIR}'
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
if [ -x '${AXOLOTL_VENV}/bin/python3' ] && '${AXOLOTL_VENV}/bin/python3' -c 'import axolotl, flash_attn, cut_cross_entropy' 2>/dev/null; then
  echo 'axolotl venv already set up, skipping'
else
  python3 -m venv '${AXOLOTL_VENV}'
  # Activate (not just call binaries by full path) -- this is the ACTUAL
  # fix for flash-attn's build parallelism, more important than the
  # MAX_JOBS/TORCH_CUDA_ARCH_LIST tuning below: pip installs the 'ninja'
  # PACKAGE regardless, but torch's BuildExtension looks for the 'ninja'
  # EXECUTABLE via PATH lookup -- calling pip by full path without
  # activating the venv left venv/bin off PATH, so ninja's binary was
  # never found and the build silently fell back to a slow, effectively
  # single-file-at-a-time compilation (only nvcc's own --threads 4
  # per-file arch parallelism, no cross-file parallelism at all).
  source '${AXOLOTL_VENV}/bin/activate'
  '${AXOLOTL_VENV}/bin/pip' install --quiet --upgrade pip
  # Install axolotl FIRST and let its own dependency resolution pick
  # whatever torch it wants -- pre-pinning a specific torch build here is
  # pointless, since axolotl silently upgrades/replaces it anyway (observed:
  # a pre-installed torch+cu121 got replaced by torch 2.13.0+cu130, which
  # then mismatched this system's actual CUDA 12.8 toolkit and broke the
  # flash-attn build). Fix torch to match the REAL system CUDA toolkit only
  # AFTER axolotl has had its say.
  '${AXOLOTL_VENV}/bin/pip' install packaging ninja wheel setuptools
  '${AXOLOTL_VENV}/bin/pip' install axolotl

  CUDA_BIN=\$(dirname \$(readlink -f /usr/local/cuda/bin/nvcc 2>/dev/null || echo /usr/local/cuda-12.8/bin/nvcc))
  CUDA_HOME=\$(dirname \"\$CUDA_BIN\")
  CUDA_VER=\$(\"\$CUDA_BIN/nvcc\" --version | grep -oP 'release \K[0-9]+\.[0-9]+')
  TORCH_CU_TAG=\"cu\$(echo \$CUDA_VER | tr -d '.')\"
  echo \"--- system CUDA toolkit: \$CUDA_VER (\$TORCH_CU_TAG) at \$CUDA_HOME ---\"
  '${AXOLOTL_VENV}/bin/pip' install --force-reinstall \"torch\" --index-url \"https://download.pytorch.org/whl/\$TORCH_CU_TAG\"

  # TORCH_CUDA_ARCH_LIST is set (the standard torch-extension convention)
  # but this flash-attn release (2.8.3.post1) was observed to ignore it --
  # its setup.py has its OWN override instead: FLASH_ATTN_CUDA_ARCHS,
  # defaulting to \"80;90;100;120\" (all four archs, each folded into a
  # single 'nvcc -gencode ... -gencode ... -gencode ... -gencode ...'
  # invocation PER SOURCE FILE, confirmed by reading the cached sdist's
  # setup.py and a live nvcc command line on the pod). Building 4 archs
  # nobody asked for isn't just wasted time -- it multiplies the memory
  # footprint of every single compile job (the actual OOM root cause: even
  # MAX_JOBS=6 still OOM-killed a 4-arch build). This pod has exactly ONE
  # GPU architecture that matters -- restrict FLASH_ATTN_CUDA_ARCHS to it.
  ARCH=\$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1)
  export TORCH_CUDA_ARCH_LIST=\"\$ARCH\"
  export FLASH_ATTN_CUDA_ARCHS=\"\$(echo \"\$ARCH\" | tr -d '.')\"
  PY_TAG=\"cp\$(python3 -c 'import sys; print(f\"{sys.version_info[0]}{sys.version_info[1]}\")')\"

  # A from-source flash-attn build is ~20 minutes even after the arch fix
  # below -- a prebuilt wheel is only valid for the EXACT stack it was
  # built against (GPU arch, Python tag, CUDA/torch version), so gate this
  # on matching FLASH_ATTN_CUDA_ARCHS/Python tag rather than blindly trying
  # it (a wrong-arch wheel would still 'pip install' successfully but
  # produce broken/crashing kernels at runtime instead of failing loudly).
  FLASH_ATTN_WHEEL_REPO=\"\${FLASH_ATTN_WHEEL_REPO:-roman220220/flash-attn-wheel-cache}\"
  FLASH_ATTN_WHEEL_FILE=\"flash_attn-2.8.3.post1-\${PY_TAG}-\${PY_TAG}-linux_x86_64.whl\"
  CACHED_WHEEL_OK=\"\"
  if [[ \"\$FLASH_ATTN_CUDA_ARCHS\" == \"80\" && \"\$PY_TAG\" == \"cp312\" ]]; then
    echo \"--- arch/python match sm_80/cp312, trying cached wheel from \$FLASH_ATTN_WHEEL_REPO first ---\"
    if hf download \"\$FLASH_ATTN_WHEEL_REPO\" \"\$FLASH_ATTN_WHEEL_FILE\" --repo-type dataset --local-dir /root/flash_attn_wheel_cache 2>/dev/null \\
       && '${AXOLOTL_VENV}/bin/pip' install \"/root/flash_attn_wheel_cache/\$FLASH_ATTN_WHEEL_FILE\"; then
      CACHED_WHEEL_OK=1
      echo \"--- installed flash-attn from cached wheel, skipping source build ---\"
    else
      echo \"--- cached wheel unavailable, falling back to source build ---\"
    fi
  else
    echo \"--- GPU arch \$ARCH / python \$PY_TAG has no matching cached wheel, building from source ---\"
  fi

  if [[ -z \"\$CACHED_WHEEL_OK\" ]]; then
  # MAX_JOBS is a secondary safety net now that per-job memory is ~4x
  # lower (single-arch, not four) -- still derive it from THIS container's
  # actual cgroup memory limit rather than a fixed guess, since a RunPod
  # container's cgroup limit can be far below the host's reported total
  # ('free -h' showed ~2TB host RAM but the container's own cgroup capped
  # it at ~232GB).
  MEM_LIMIT_BYTES=\$(cat /sys/fs/cgroup/memory.max 2>/dev/null)
  if [[ -z \"\$MEM_LIMIT_BYTES\" || \"\$MEM_LIMIT_BYTES\" == \"max\" ]]; then
    MEM_LIMIT_BYTES=\$(cat /sys/fs/cgroup/memory/memory.limit_in_bytes 2>/dev/null || free -b | awk '/^Mem:/{print \$2}')
  fi
  MEM_LIMIT_GB=\$(( MEM_LIMIT_BYTES / 1024 / 1024 / 1024 ))
  CPU_COUNT=\$(nproc)
  PER_JOB_GB=10
  MAX_JOBS_BY_MEM=\$(( MEM_LIMIT_GB / PER_JOB_GB ))
  export MAX_JOBS=\$(( CPU_COUNT < MAX_JOBS_BY_MEM ? CPU_COUNT : MAX_JOBS_BY_MEM ))
  [[ \$MAX_JOBS -lt 1 ]] && export MAX_JOBS=1
  echo \"--- building flash-attn (FLASH_ATTN_CUDA_ARCHS=\$FLASH_ATTN_CUDA_ARCHS, cgroup mem limit=\${MEM_LIMIT_GB}GB, cpu=\${CPU_COUNT}, MAX_JOBS=\$MAX_JOBS) ---\"
  export PATH=\"\$CUDA_HOME:\$PATH\"
  export LD_LIBRARY_PATH=\"\$CUDA_HOME/../lib64:\${LD_LIBRARY_PATH:-}\"
  '${AXOLOTL_VENV}/bin/pip' install flash-attn --no-build-isolation
  fi

  # Axolotl's cut_cross_entropy plugin requires its own fork with
  # transformers support -- the stock PyPI cut-cross-entropy package lacks
  # it, and training fails immediately at model-load with an ImportError
  # naming this exact install command.
  '${AXOLOTL_VENV}/bin/pip' uninstall -y cut-cross-entropy 2>/dev/null || true
  '${AXOLOTL_VENV}/bin/pip' install 'cut-cross-entropy[transformers] @ git+https://github.com/axolotl-ai-cloud/ml-cross-entropy.git@4dfa522'
fi
"
fi

echo "=== pod ready: model at ${MODEL_SRC_DIR}, wikitext at ${WIKITEXT_DIR}, poc/ synced ==="
[[ -n "$WITH_AXOLOTL" ]] && echo "=== axolotl venv ready at ${AXOLOTL_VENV} ==="
