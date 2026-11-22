# Runbook: Inkling 2 Mini -> MLX (GPTQ JANG, vision kept) and GGUF

Two pipelines, both resumable, status lines in `$WORK/logs/pipeline.log`:
`poc/inkling2_mlx_pipeline.sh` (presets `mini`, `mini-small`) and
`poc/inkling2_gguf_pipeline.sh`. Why each step is the way it is:
[FINDINGS.md](FINDINGS.md).

## 0. Dry run first (CPU, seconds)

```bash
LIKE=<dir with Inkling-2-Mini's config.json + tokenizer files> PY=<python> ./poc/smoke_test.sh
```

Needs torch, torchvision, transformers >= 5.24, mlx and the fork
(`pip install -e ~/gh/mlx-lm` on the `inkling2` branch).

## 1. Pod

As qwen35-quant/docs/RUNBOOK.md, smaller: one A100 80 GB, secure cloud,
>= 125 GB RAM (the bf16 model, 53 GB, stays in CPU memory during the
calibration and the CPU conversion), volume >= 250 GB (snapshot 20 GB,
GPTQ layers ~55 GB, assembled bf16 ~53 GB, MLX 11-13 GB, F16 GGUF ~53 GB,
quantized GGUF ~16 GB).

## 2. Code onto the pod

```bash
COPYFILE_DISABLE=1 tar czf - --exclude __pycache__ --exclude '._*' inkling2-quant cards_assets \
  qwen35-quant/poc/gptq_qwen35.py qwen35-quant/poc/convert_mlx.py \
  gemma4-quant/poc/pipeline_lib.sh nemotron-extreme-quant/poc/gptq.py nemotron-extreme-quant/poc/quantize.py \
  | ssh <pod> 'mkdir -p /workspace/qt && tar xzf - --no-same-owner -C /workspace/qt'
```

## 3. Environment

`python3 -m venv --system-site-packages /workspace/venv`, then `SETUP=1`
(transformers >= 5.24, mlx[cuda], the fork at `MLX_LM_REF`, default
`main`: set `MLX_LM_REF=inkling2` until the branch is merged).

## 4. Run

```bash
cd /workspace/qt/inkling2-quant/poc
nohup setsid bash -c "PATH=/workspace/venv/bin:\$PATH VARIANT=mini SETUP=1 MLX_LM_REF=inkling2 WORK=/workspace \
  ./inkling2_mlx_pipeline.sh > /workspace/logs/run-mini.log 2>&1" > /dev/null 2>&1 < /dev/null &
```

Then `VARIANT=mini-small` (shares the download, data and reference), then
the GGUF pipeline (uses the MLX pipeline's `data/` for the perplexities):

```bash
nohup setsid bash -c "PATH=/workspace/venv/bin:\$PATH WORK=/workspace \
  ./inkling2_gguf_pipeline.sh > /workspace/logs/run-gguf.log 2>&1" > /dev/null 2>&1 < /dev/null &
```

Overrides: `MODEL_ID`, `RECIPE` (keys `attn`, `experts` or
`experts_gate_up` / `experts_down`), `HF_REPO`, `GROUP_SIZE`, `EMBED_BITS`,
`HEAD_BITS`, `VISION_BITS`, `CALIB_CHUNKS`; GGUF: `BASE_QUANT`,
`ATTN_TYPE`, `LLAMA_REF`.

## 5. After

Publish (`PUBLISH=1`, HF token at `$HF_HOME/token`, stdin only), copy the
check JSONs and logs into `docs/`, terminate the pod.
