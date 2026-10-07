# Runbook: Qwen3.5 / Qwen3.5-MoE -> MLX (GPTQ JANG, vision kept)

One pipeline, `poc/qwen35_mlx_pipeline.sh`, resumable (re-run after any
interruption: finished steps are skipped, calibration resumes at the next
layer). Status lines go to `$WORK/logs/pipeline.log` (shown by
`gemma4-quant/poc/pipeline_dashboard.py`), step output to
`$WORK/logs/qwen35_<step>.log`. Nothing is uploaded unless `PUBLISH=1`. Why
each step is the way it is: [FINDINGS.md](FINDINGS.md).

## 1. Pod

What the 35B MoE (72 GB bf16) needed, 2026-10-06:

- **One A100 80 GB, secure cloud** (community pods vary in RAM). The model
  stays in CPU memory during calibration and the CPU conversion: take a pod
  with **>= 150 GB RAM** (ours had 250 GB). A 4B model runs on anything.
- **Volume >= 350 GB** for one MoE build at a time: bf16 snapshot 72 GB,
  per-layer GPTQ output ~67 GB, the assembled checkpoint ~67 GB, the MLX
  model 14-17 GB, venv 4 GB. Delete `gptq-<variant>` and `ongrid-<variant>`
  after a build is published; the pipeline can rebuild them.
- Image `runpod/pytorch:1.0.3-cu1281-torch291-ubuntu2404`, port 22/tcp.

```bash
runpodctl pod create --name <name> --gpu-id "NVIDIA A100-SXM4-80GB" --cloud-type SECURE \
  --image runpod/pytorch:1.0.3-cu1281-torch291-ubuntu2404 \
  --container-disk-in-gb 60 --volume-in-gb 350 --volume-mount-path /workspace --ports "22/tcp" --wait
```

The SSH host and port change on every start: read them from
`runpodctl pod get <id>`.

## 2. Code onto the pod

The pipeline needs `qwen35-quant/`, `cards_assets/`,
`gemma4-quant/poc/pipeline_lib.sh` and `nemotron-extreme-quant/poc/{gptq,quantize}.py`
(relative paths kept):

```bash
COPYFILE_DISABLE=1 tar czf - --exclude __pycache__ --exclude '._*' qwen35-quant cards_assets \
  gemma4-quant/poc/pipeline_lib.sh nemotron-extreme-quant/poc/gptq.py nemotron-extreme-quant/poc/quantize.py \
  | ssh <pod> 'mkdir -p /workspace/qt && tar xzf - --no-same-owner -C /workspace/qt'
```

## 3. Environment

Ubuntu 24.04 refuses pip into the system Python (PEP 668): a venv on the
volume that sees the image's torch, so it survives a pod stop.

```bash
python3 -m venv --system-site-packages /workspace/venv
```

Every run puts it first on `PATH`. `SETUP=1` installs transformers >= 5.8,
mlx[cuda], the ipsupport-llc/mlx-lm fork (`MLX_LM_REF`, default main) and a
torchvision that matches the image's torch (0.24 for torch 2.9, from the
cu128 index; the latest one fails with `operator torchvision::nms does not
exist`). The pipeline itself preloads the NCCL next to torch for the MLX
steps (mlx[cuda]'s NCCL breaks torch: `undefined symbol:
ncclCommWindowRegister`).

The volume is a network file system: pip and Python imports take minutes
there; large files read fast.

## 4. Run

Start it detached (an SSH drop must not kill it):

```bash
cd /workspace/qt/qwen35-quant/poc
nohup setsid bash -c "PATH=/workspace/venv/bin:\$PATH VARIANT=ornith-35b SETUP=1 WORK=/workspace \
  ./qwen35_mlx_pipeline.sh > /workspace/logs/run.log 2>&1" > /dev/null 2>&1 < /dev/null &
```

Presets: `ornith-35b` (8/6/6/3), `ornith-35b-small` (experts' gate / up 2,
down 3; `MAX_PPL_RATIO=1.5`), `frognano-4b` (8/6/4). Anything a preset sets
can be overridden: `MODEL_ID`, `RECIPE` (keys `attn`, `linear`, `mlp`,
`shared`, `experts`, or `experts_gate_up` / `experts_down`), `HF_REPO`,
`GROUP_SIZE`, `EMBED_BITS`, `HEAD_BITS`, `VISION_BITS`, `CALIB_CHUNKS`.

The MTP head: `MTP` (default `mtp=4`; per component `mtp_fc`, `mtp_attn`,
`mtp_mlp` / `mtp_shared`, `mtp_experts_gate_up`, `mtp_experts_down`; empty:
no head). It is calibrated after the layers and goes to
`model-mtp.safetensors`; another `MTP` on a finished run recalibrates only
the head (FINDINGS §4a). The check fails below `--min-mtp-accept` (0.5).

Two runs at once (another variant, or a publish of a finished one) need a
launcher of their own: `pkill -f qwen35_mlx_pipeline` over SSH also kills
the SSH command that names it.

Expected times on the A100 for the 35B MoE: download ~5 min, HF reference
~6 min, calibration ~25 min (~30 s a layer: if a layer takes minutes, the
calibration is CPU-bound, see FINDINGS), assembly ~5 min, conversion with
exact codes ~20 min, checks ~10 min.

A calibration resumes only with the recipe, group size and calibration
data its finished layers were made with (`progress.json`'s `made_with`);
with anything else it stops and says so: use another `WORK` or delete
`gptq-<variant>`. The check step is skipped on a re-run only if it passed.

Watch: `tail /workspace/logs/pipeline.log`. The step logs are appended across
runs: read them from the latest start, not the whole file.

## 5. Check before publishing

`$WORK/check-<variant>.json` has the vision cosine against HF, the image
answer and both perplexities against bf16 (`$WORK/ref-<model>/ref.json`).
The check step fails above `MAX_PPL_RATIO` (default 1.10). Write the
numbers into `cards/<repo>.md` (template rules: `HF_CARD_TEMPLATE.md`,
including the Disclaimer block) and, if the base repo has no LICENSE file,
`cards/licenses/<model>-LICENSE`.

## 6. Publish

The HF token goes to the pod on stdin only, never into a file or a command
line:

```bash
printf '%s' "$TOKEN" | ssh <pod> 'read -r T; export HF_TOKEN="$T"; export PATH=/workspace/venv/bin:$PATH;
  cd /workspace/qt/qwen35-quant/poc && VARIANT=ornith-35b WORK=/workspace PUBLISH=1 ./qwen35_mlx_pipeline.sh'
```

The card step copies the card, its banner and the license; the publish step
uploads the model directory. A card-only update: `hf upload <repo>
README.md README.md`. Some files only: `PUBLISH_FILES="model-mtp.safetensors
model.safetensors.index.json README.md"` (e.g. the MTP head for a model
already published; if its backbone is an older build, publish the whole new
build: the head is calibrated on this one).

Re-uploading a weight file keeps the old one in the repo's history, and
HF's `usedStorage` (what LLMTray showed as the size until #255) counts both:
squash it (`HfApi().super_squash_history(repo_id)`; the files stay, the old
commits go; ask first).

## 7. Afterwards

Measure speed and memory on a Mac from the published repo and add them to
the card. Stop the pod (or delete it when the work is finished).
