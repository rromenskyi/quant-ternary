# Runbook: Gemma 4 → MLX (GPTQ JANG), GGUF (imatrix JANG), MTP drafter

Three pipelines, all in `poc/`, all resumable (re-run after any interruption
and finished steps are skipped), all writing to one status log that
`poc/pipeline_dashboard.py` shows live:

| Pipeline | Script | Runs on | Output |
|---|---|---|---|
| MLX GPTQ JANG (E4B or 26B) | `gemma4_mlx_pipeline.sh` | CUDA pod (calibrate with torch, splice with `mlx[cuda]`) | `roman220220/gemma-4-{E4B,26B-A4B}-it-gptq-mlx-jang` |
| GGUF imatrix JANG (26B) | `gemma4_26b_gguf_pipeline.sh` | CUDA pod (llama.cpp built there) | `roman220220/gemma-4-26B-A4B-it-GGUF-jang-imatrix` |
| MTP drafter, MLX 8-bit (26B) | `gemma4_mtp_drafter_pipeline.sh` | Mac (or pod with `mlx[cuda]`) | `roman220220/gemma-4-26B-A4B-it-assistant-mlx-8bit` |

Nothing is uploaded unless `PUBLISH=1`. The recipe and every gotcha these
scripts encode are explained in [FINDINGS.md](FINDINGS.md).

## 1. Pod

One A100 80GB is enough for everything (the 26B bf16 checkpoint is 51.6GB
and calibration keeps it resident).

- **Size the container disk at creation: ≥ 250GB** for the 26B MLX + GGUF
  runs together (bf16 checkpoint 52GB, GPTQ-corrected weights ~47GB, MLX
  output 15GB, f16 GGUF 52GB, GGUF outputs ~14GB). **Never** resize later
  with `runpodctl pod update --container-disk-in-gb`: that recreates the
  disk and wipes `/workspace` (lost a full calibration run to this).
- RunPod assigns a new SSH host/port on every start: re-read them
  (`runpodctl pod list`) after any restart.

```bash
export POD_HOST=1.2.3.4 POD_PORT=12345 POD_SSH_KEY=~/.runpod/ssh/runpodctl-ssh-key
POD="ssh -i $POD_SSH_KEY -p $POD_PORT root@$POD_HOST"

# Code to the pod (scripts + model cards; the cards are copied in last by the pipelines).
rsync -az -e "ssh -i $POD_SSH_KEY -p $POD_PORT" --exclude .venv --exclude __pycache__ \
  gemma4-quant/ root@$POD_HOST:/workspace/gemma4-quant/
$POD 'hf auth login --token $HF_TOKEN'   # only needed for PUBLISH=1
```

## 2. Dashboard (optional, recommended)

On the Mac, while anything below runs:

```bash
python3 gemma4-quant/poc/pipeline_dashboard.py --pod-host $POD_HOST --pod-port $POD_PORT --pod-ssh-key $POD_SSH_KEY
open http://localhost:8421
```

It shows each pipeline's steps with durations, per-layer calibration grids
(from `gptq_progress_*.json`), progress inside long steps (calibration
batch/example, splice shard, imatrix chunk, llama-quantize tensor), the
running step's log tail, GPU and disk. `--local --work /workspace` runs it
on the pod itself; forward the port with `ssh -L 8421:localhost:8421`.

## 3. MLX GPTQ JANG

```bash
$POD 'cd /workspace/gemma4-quant/poc && SETUP=1 VARIANT=26b nohup ./gemma4_mlx_pipeline.sh > /workspace/mlx_pipeline.out 2>&1 &'
```

`VARIANT=e4b` for the E4B, which adds the audio component. Steps:

1. `setup` (`SETUP=1` only): torch/transformers, `mlx[cuda]`, and
   ipsupport-llc/mlx-lm (`MLX_LM_REF`, default `main`). The fork is needed
   for Gemma 4 multimodal loading and the vision smoke test.
2. `download`: HF snapshot of `google/gemma-4-{26B-A4B,E4B}-it`.
3. `calibrate_<component>`: GPTQ, attention 8-bit / FFN + experts 4-bit,
   group 64 (`ATTN_BITS`, `FFN_BITS`, `GROUP_SIZE`). Resumes per layer. The
   step is skipped when the progress files already list every layer, which
   avoids loading 50GB just to find that out.
4. `splice`: `mx.quantize` + re-shard + `config.json` quantization dict.
   Also embeddings 8-bit RTN, uncalibrated leftover Linears 8-bit RTN, and
   dead KV-shared weights dropped.
   `--keep-float 'patch_embedder\.input_proj'` (+ `'router\.proj'` on 26B)
   keeps what must stay float.
5. `smoke`: `gemma4_smoke_test.py`. Checks a text reply with no leaked
   control tokens, vision on the COCO cats photo, audio transcription on
   E4B, and that no large float weight is left.
   - The audio check needs `$WORK/smoke-assets/speech.wav` (16kHz mono
     speech). The script makes it with `say` on a Mac. On a pod, copy one
     over or the audio check is skipped.
6. `card`: `cards/<repo>.md` → `README.md`, **last**, because a splice
   re-run overwrites it with Google's card.
7. `publish` (`PUBLISH=1` only): `hf upload`.

Re-run with `PUBLISH=1` once the smoke test has passed; everything else is
skipped.

## 4. GGUF (26B)

```bash
$POD 'cd /workspace/gemma4-quant/poc && nohup ./gemma4_26b_gguf_pipeline.sh > /workspace/gguf_pipeline.out 2>&1 &'
```

Steps:
1. `llama_build`: clone + CUDA build of current llama.cpp. Prebuilt
   binaries lack Gemma 4 MoE + vision.
2. `download`.
3. `fix_tokenizer`: the dict-form `extra_special_tokens` fix.
4. `calib_data`: bartowski's `calibration_datav3.txt`.
5. `convert_f16`.
6. `imatrix`: 200 chunks.
7. `quantize`: JANG. Base Q3_K_M, attention Q5_K, experts IQ3_S, then the
   in-place `token_type` CONTROL fix.
8. `mmproj_convert`, `mmproj_quantize`: vision tower to mmproj, Q8_0.
9. `drafter`: ollama's `gemma4:26b` draft layer, sha256-verified.
10. `ollama_files`: `Modelfile` with `FROM hf.co/…` + `DRAFT` +
    `RENDERER`/`PARSER gemma4`, plus `params`.
11. `smoke`: `llama-server --jinja --mmproj`. Checks a text reply with no
    control-token leak, and vision on the cats photo.
12. `card`.
13. `publish` (`PUBLISH=1` only): skips the f16 intermediates.

User-side ollama install is in the model card: download `Modelfile` + the
drafter GGUF, then `ollama create`.

## 5. MTP drafter (MLX 8-bit)

On the Mac:

```bash
cd gemma4-quant/poc
WORK=~/gemma4-work \
PY_TORCH=../.venv/bin/python \
PY_MLX="$HOME/Library/Application Support/LLMTray/mlx_server_venv/bin/python3" \
MAIN_MODEL=roman220220/gemma-4-26B-A4B-it-gptq-mlx-jang \
./gemma4_mtp_drafter_pipeline.sh
```

- `PY_TORCH`: a Python with torch + transformers ≥ 5 (for
  `Gemma4AssistantForCausalLM`). It's only used for the `parity` step,
  which is skipped without it.
- `PY_MLX`: needs ipsupport-llc/mlx-lm ≥ `d7b4f8c`.
- `MAIN_MODEL`: turns on `bench` (plain vs MTP, fresh + cached-prefix). It
  needs RAM for the 26B, so stop LLMTray's server first.

Steps:
1. `download`.
2. `parity`: fp32 vs transformers on the real weights.
3. `convert`: 8-bit, group 64.
4. `bench`: fails below `MIN_SPEEDUP`, default 1.15×.
5. `card`.
6. `publish`.

A run reproduces the published weights byte for byte.

## 6. Tests

```bash
python -m pytest gemma4-quant/poc/test_pipeline_tools.py -q     # plumbing: log/progress parsing, splice key rules
gemma4-quant/.venv/bin/python gemma4-quant/poc/test_26b_moe_calibrate.py   # MoE-expert GPTQ path, tiny real modules
```

The MTP decoding itself is tested in the mlx-lm fork
(`tests/test_gemma4_mtp.py`).

## Troubleshooting

| Symptom | Cause / fix |
|---|---|
| Dashboard says "no activity" but GPU busy | A step that logs rarely (imatrix warm-up, big shard save). The step log tail shows the truth. |
| `calibrate_*` restarts from scratch | Wrong `CORRECTED` dir. Progress lives in `$CORRECTED/gptq_progress_*.json`. |
| Smoke: vision says "no animals" | Something quantized `patch_embedder.input_proj` (see FINDINGS). |
| Smoke: `large unquantized weights` | A new uncalibrated Linear: RTN it (default) or, if it must stay float, add a `--keep-float` regex **with the reason** in the pipeline. |
| GGUF chat shows `<|channel>thought` | `token_type` fix skipped, or the client isn't using `--jinja` / the Modelfile. |
| MTP: no speed-up in LLMTray | Old runtime (needs mlx-lm ≥ `d7b4f8c`, #5). Run `bench`: the cached-prefix row must show a speed-up too. |
