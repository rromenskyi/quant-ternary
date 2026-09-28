# Runbook: NemotronLabs VoiceChat 11B, faster per frame on a base M5

Everything here is scripted in `voicechat-quant/scripts/`. The Mac only
**measures** things. Every calibration or GPTQ step runs on a RunPod GPU pod
(`pod_pipeline.sh`), and the pod is deleted when the script exits.
What the numbers mean is in [FINDINGS.md](FINDINGS.md).

## 0. Prerequisites

- Mac: the mlx-audio fork `ipsupport-llc/mlx-audio@llmtray` (`b99f797`) in
  `/Users/roman220/gh/mlx-audio-fork/.venv`, and
  `mlx-community/NemotronLabs-VoiceChat-11B-4bit` in the HF cache.
- Independent ASR for scoring: `mlx-community/whisper-large-v3-turbo`
  (1.6 GB). Its repo has no tokenizer, so `score.py` takes the processor
  files (~5 MB) from `openai/whisper-large-v3-turbo`:
  `python -c "from huggingface_hub import snapshot_download as s; s('openai/whisper-large-v3-turbo', allow_patterns=['*.json','*.txt'])"`.
- RunPod CLI `~/.local/bin/runpodctl` (≥ 2.14), authenticated. The SSH key is at
  `~/.runpod/ssh/runpodctl-ssh-key`.
- Before any Mac run: `pgrep -f mlx_lm.server; pgrep -f llmtray_music_runner`
  must print nothing. `run_matrix.sh` checks this itself and stops. Also
  look at what else is using the GPU (`ioreg -r -d 1 -c IOAccelerator | grep
  "Device Utilization"`). `vc_eval.py` records it in `run.json`.

```bash
cd ~/gh/quant-ternary/voicechat-quant
PY=/Users/roman220/gh/mlx-audio-fork/.venv/bin/python
export HF_HUB_OFFLINE=1
```

## 1. Eval and calibration audio (Mac, cheap)

```bash
python3 scripts/make_eval_audio.py            # eval/audio/q01..q20.wav (committed)
python3 scripts/make_calib_audio.py --n 512   # calib/audio/c000..c511.wav (not committed; deterministic)
```

`eval/questions.json` holds 20 short factual questions (Samantha/Daniel) and
the expected keywords for each. The calibration prompts are built from
templates plus 30 free-form turns (8 voices, 4 speaking rates). None of them
overlaps the eval set.

## 2. Speed + quality on the Mac

A full run of one variant: 20 questions, speech then silence, stopping when
the reply ends. It records per-component ms/frame and writes the reply WAVs:

```bash
$PY scripts/vc_eval.py --out results/x --act-dtype bf16 --scales-dtype bf16
```

Many variants at once, each in a fresh process, then scored:

```bash
scripts/run_matrix.sh eval/matrix_full.txt results/mac_full
# -> results/mac_full/table.md (+ per-variant run.json / score.json)
```

Scoring alone (Whisper transcripts are cached per dir):

```bash
$PY scripts/score.py results/mac_full/*/ --table results/mac_full/table.md
```

Speed numbers from separate processes are only comparable when the Mac is
idle. For small effects, use the interleaved in-process A/B bench. It loads
the model once and alternates the variants over several rounds:

```bash
$PY scripts/bench_ab.py --out results/ab.json --rounds 3 --ids q01,q04 \
  --variant 'base=' \
  --variant 'bf16=--act-dtype bf16 --scales-dtype bf16' \
  --variant 'best=--act-dtype bf16 --scales-dtype bf16 --tts-mog-gather --compile-tts-codes --rtn stt_model.perception:4:64 --rtn tts_model.tts_model.backbone:4:64'
```

Variant flags (`scripts/vc_variants.py`, shared by every script):

| flag | effect |
|---|---|
| `--act-dtype bf16` | perception (mel → conformer) and LLM inputs run in bf16 instead of float32 |
| `--scales-dtype bf16` | casts the float32 quantization scales/biases of the stock 4-bit checkpoint to bf16 |
| `--rtn PREFIX:BITS:GROUP` | in-memory RTN of every float Linear/Embedding (and 1x1 Conv1d, rewritten as Linear) under PREFIX; can be repeated |
| `--rtn-skip REGEX` | leave matching modules in float |
| `--tts-mog-gather` | exact rewrite: MoG head computes only the sampled mixture's `proj_mus` rows |
| `--compile-tts-codes` | `mx.compile` of the per-frame TTS code generator (random state threaded through) |
| `--model DIR` | any VoiceChat MLX dir, e.g. a spliced GPTQ model (§4) |

## 3. Pod: calibration capture + GPTQ of the LLM

```bash
scripts/pod_pipeline.sh \
  --variant 'rtn4:--bits 4 --rtn' \
  --variant 'gptq4:--bits 4' \
  --variant 'gptq3:--bits 3'
```

Env knobs: `GPU` (default `NVIDIA A40`, we used `NVIDIA A100-SXM4-80GB`
because no A40 was in stock), `DISK_GB=120`, `CALIB_CLIPS=256`,
`CAPTURE_PROCS=3`, `POD_ID=<id>` to reuse a running pod (it is then kept,
unless `DELETE_POD=1`), `KEEP_POD=1` to keep a pod the script created.
The pipeline's steps are resumable. Each step leaves `/workspace/.done_<step>`
on the pod:

1. `setup`: venv, `mlx[cuda]` (the same MLX 0.32.2 as the Mac), the fork at `b99f797`.
2. `deps`: fork extras (`[stt]`, sentencepiece).
3. `download`: `mlx-community/NemotronLabs-VoiceChat-11B-bf16` (22 GB).
   This is the fork's own unquantized conversion of
   `nvidia/NVIDIA-NemotronLabs-VoiceChat-11B` at the pinned source revision.
4. `capture`: `calib_capture.py`. It streams each calibration clip through
   the real bf16 duplex session (system-prompt prefill → speech → silence
   until the reply ends) and saves the LLM's fused per-frame input. The
   TTS/codec are stubbed because they never feed the LLM. The step is
   sequential and runs at batch 1, so it barely loads a big GPU;
   `CAPTURE_PROCS` runs shards in parallel on one GPU.
5. `gptq_<name>`: `gptq_llm.py <flags>`. Layer-by-layer sequential GPTQ in
   MLX-CUDA onto MLX's own affine grid. Output: `llm.safetensors`,
   `quant.json`, `gptq_log.json` (per-module relative output error).
   Useful flags: `--bits`, `--group-size`,
   `--override 'REGEX=BITS[:GROUP]'`, `--head-bits`, `--embed-bits`,
   `--rtn`, `--damp`.
6. The results are rsynced to `pod_results/<name>/`. The pod is deleted on
   exit (trap), and the script checks `runpodctl pod list` afterwards.

Always check by hand afterwards: `runpodctl pod list` must print `[]`.

## 4. Splice and measure a pod result on the Mac

```bash
$PY scripts/splice_llm.py --base '~/.cache/huggingface/hub/models--mlx-community--NemotronLabs-VoiceChat-11B-4bit/snapshots/*' \
   --llm pod_results/gptq3 --out models/vc-gptq3
$PY scripts/vc_eval.py --model models/vc-gptq3 --out results/mac_full/gptq3_exact \
   --act-dtype bf16 --tts-mog-gather --compile-tts-codes
```

The splice keeps every non-LLM tensor of the base (perception/TTS/codec in
bf16) and swaps in the new LLM, heads, embeddings and per-module
quantization config. It does no compute.

## 5. Cost

| item | rate | time | cost |
|---|---|---|---|
| see FINDINGS.md §Cost for the actual run | | | |

Estimate for a rerun on an A100-SXM4-80GB ($1.59/h secure): setup+download
~5 min, capture of 256 clips ~16 min (or ~6 min with `CAPTURE_PROCS=3`),
~5 min per RTN variant, ~10–15 min per GPTQ variant. About 1 h in total,
≈ $1.6. An A40 ($0.49/h) or RTX 6000 Ada ($0.84/h) with 48 GB is enough
and costs less, if one is in stock.
