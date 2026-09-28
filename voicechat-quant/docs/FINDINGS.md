# Findings: NemotronLabs VoiceChat 11B on a base M5

Goal: one 80 ms duplex frame in ≤ 80 ms on a base M5 (26 GB, Metal limit
~19 GB) without losing quality, for LLMTray's Voice Lab. How to reproduce
everything: [RUNBOOK.md](RUNBOOK.md).

Model: `nvidia/NVIDIA-NemotronLabs-VoiceChat-11B` (OpenMDW 1.1) = a streaming
FastConformer encoder (perception) + **Nemotron-Nano-9B-v2** (hybrid
Mamba-2/attention, 56 layers, hidden 4480) + a T5Gemma-based TTS + a NeMo
22.05 kHz codec. Baseline checkpoint: `mlx-community/NemotronLabs-VoiceChat-11B-4bit`
(LLM, embeddings, heads 4-bit g64; perception and TTS bf16), run through our
mlx-audio fork (`ipsupport-llc/mlx-audio@llmtray`).

## Results (20 spoken questions, 2026-09-27)

ms per 80 ms frame, mean over all frames of all questions; quality from
`score.py` (independent Whisper for the reply audio).

| variant | perception | LLM | TTS | codec | **total** | RTF | params GB | user WER | keyword acc | reply WER |
|---|---|---|---|---|---|---|---|---|---|---|
| base (mlx-community 4-bit) | 62.7 | 62.3 | 47.8 | 9.5 | **185.2** | 2.32 | 9.16 | 0.020 | 0.95 | 0.053 |
| exact: bf16 act/scales + TTS MoG gather + compiled codes | 30.4 | 61.4 | 42.7 | 11.6 | **150.3** | 1.88 | 8.57 | 0.020 | 1.00 | 0.058 |
| + perception/TTS RTN 8-bit (p8t8) | 25.5 | 61.8 | 36.3 | 12.2 | 139.1 | 1.74 | 7.44 | 0.020 | 0.95 | 0.052 |
| + perception/TTS RTN 4-bit (p4t4) | 20.3 | 59.4 | 30.0 | 11.1 | 123.5 | 1.54 | 6.84 | 0.013 | 0.95 | 0.064 |
| pod rtn4 LLM, exact | 17.5 | 49.0 | 26.5 | 6.7 | 101.2 | 1.27 | 8.57 | 0.020 | 1.00 | 0.053 |
| pod rtn4 LLM + p4t4 | 17.9 | 56.9 | 25.5 | 10.1 | 112.4 | 1.41 | 6.84 | 0.013 | 0.95 | 0.062 |
| **pod GPTQ-3 LLM, exact** | 18.0 | **42.5** | 25.6 | 7.5 | **95.0** | 1.19 | 7.61 | 0.020 | **1.00** | **0.043** |

- **GPTQ-3 is the best so far**: every answer right (20/20), the lowest reply
  WER, 95 ms/frame (−49% from the baseline), peak footprint 10.4 GB.
- **The LLM's share falls with bits** as expected for a bandwidth-bound step:
  62 → 49 (our 4-bit) → 42.5 ms (3-bit).
- **Noise between runs is large.** Perception runs the same bf16 weights in
  `exact` and `rtn4_exact` yet reads 30.4 vs 17.5 ms: `gpu_busy_before` in
  `run.json` shows other GPU load (browser, the pod's monitor) during some
  runs. Compare variants only from back-to-back runs on an idle machine;
  re-run the headline rows before publishing numbers.
- **JANG on the dense 9B is not a speed recipe**: the component recipe
  (attention 8, Mamba in/out 6, MLP up 4 / down 3, embeddings/heads 6)
  averages **5.62 bits/weight** here (6.3 GB) — Mamba projections are a large
  share of this dense model, unlike the 30B MoE where routed experts dominate.
  Kept on HF for a quality comparison; not measured yet.
- Perception already streams (cached FastConformer, one encoder frame per
  step, `use_perception_cache=True`); its cost is the per-step overhead
  (bf16 weights read + many small kernels at batch 1), the next target.

## Pod notes

- One A100 SXM 80 GB, ~1 h 35 min, ≈ $2.5: calibration capture (256 real
  duplex conversations, 16 min), then rtn4 (5 min), gptq3, gptq4, rtn3, jang.
- The GPTQ script ran on ~2 of 128 host cores (GPU at ~28%): the host-side
  Hessian solve and code packing weren't given threads. `gptq_llm.py
  --cpu-threads` (default: every core) now sets BLAS/OpenMP/torch threads.
- The pod had **no volume**: `/workspace` lived on the container disk, which
  a stop wipes. Everything worth keeping went to the private HF repo
  `roman220220/NemotronLabs-VoiceChat-11B-gptq-research` first
  (rtn4, gptq3, gptq4, rtn3, jang, `calib/` = the captured calibration set,
  `pod_scripts/`); the pod is stopped, not deleted. Next time: a network
  volume, and skip the capture by pulling `calib/` from HF.
- Uploads from the pod: the HF token went over ssh stdin into the upload
  process's environment only — never onto the pod's disk or into logs.

## Night 2026-09-27/28: fork-side speedups (ipsupport-llc/mlx-audio PR #3)

Measured in isolation on the base M5 (`scripts/bench_perception.py`,
`bench_llm_step.py`), then in the full session (`vc_eval.py`).

| change | alone | notes |
|---|---|---|
| perception in bf16 (mel frames arrived float32, promoting every matmul) | 51.0 → 17.3 ms | cosine 0.99985 vs the old path |
| + `linear_pos(pos_emb)` cached per window (was per layer per frame) | (in the above) | |
| + compiled steady-state conformer stack (`mx.compile`) | 17.3 → 13.6 ms | cosine 0.99984 |
| perception weights 8-bit / 4-bit RTN | 12.1 / 10.2 ms | cosine 0.9995 / 0.974; 4-bit keeps user WER 0.013, 20/20 answers |
| TTS: MoG head computes only the sampled mixture; compiled code generation | TTS 42.7 → ~26 ms in session | same sampled codes |
| LLM input cast to the weight dtype | — | no float32 LLM step/caches |
| compiled LLM decode (Mamba + MLP runs between attention layers) | 40.2 → 40.0 ms | **reverted**: the step is memory-bound |
| compiled codec `decode_step` | 5.9 → 5.3 ms | exact |

Full session, GPTQ-3 LLM + fork fixes: **~95–99 ms per 80 ms frame**
(185 upstream), 20/20 answers, reply WER 0.043. Where a frame goes now:
LLM 43–46, TTS 27–29 (Gemma3 backbone 12 at batch 2 for CFG, codes 8.5,
fusion 2.2), perception 14–19, codec 7.5.

- **The LLM is at the bandwidth floor.** 3-bit ≈ 4.4 GB read per frame in
  ~42 ms ≈ 105 GB/s effective on a base M5; compiling the step changed
  nothing. Faster only with fewer bits (the MLP runs are ~45% of it) or a
  smaller LLM.
- ~~The TTS backbone is launch-bound~~ — wrong, it's bandwidth-bound; see
  "Plan item 4" below.
- TTS without CFG (`--no-tts-guidance`, backbone batch 1): 106 ms vs 99 in
  the paired run, reply WER unchanged 0.044 — inconclusive: background GPU
  load was 25–27% during it vs 13–17%. Re-measure on an idle Mac.
- **The Mac isn't idle while measuring** (`gpu_busy_before` 13–17%: the UI,
  a task monitor): ±5 ms between runs. Headline numbers need an idle machine.

## Published (2026-09-28)

[roman220220/NemotronLabs-VoiceChat-11B-gptq-mlx-3bit](https://huggingface.co/roman220220/NemotronLabs-VoiceChat-11B-gptq-mlx-3bit)
(public, OpenMDW 1.1; card `cards/NemotronLabs-VoiceChat-11B-gptq-mlx-3bit.md`,
staged and uploaded by `scripts/hf_publish.py`: private upload, every file's
size checked, then public). It's the `models/vc-gptq3` splice, unchanged,
with NVIDIA's LICENSE and notice files. LLMTray Voice Lab's default from
v0.8.3-beta.2.

Final run on an idle Mac (`gpu_busy_before` 7–9%), fork `2dce524` (PR #3
merged), `results/mac_full/table_pr3final.md`:

| variant | perception | LLM | TTS | codec | **total** | p95 | RTF | peak GB | user WER | keyword acc | reply WER |
|---|---|---|---|---|---|---|---|---|---|---|---|
| gptq3 + fork 2dce524 | 16.4 | 39.5 | 24.4 | 5.9 | **87.7** | 98.7 | 1.10 | 9.12 | 0.020 | 1.00 | 0.043 |

Voice Lab's runner (`llmtray_voice_runner.py`, walkie-talkie over its frame
protocol) on the published files: loads in 4.3 s, warm-up 83 ms/frame
(rtf 1.03), right answer.

Parity of the compiled paths is tested on the **CPU**: Metal's fp32 matmul
rounds like tf32 (selected-means vs full projection: 1e-3 on the GPU,
2e-7 on the CPU), which would hide an indexing mistake in the tolerance.

## Plan item 3: mixed 2/3-bit LLM — closed, quality loss (2026-09-28)

GPTQ on the pod: a new A100 SXM pod, `uaaawubmr2qf66`, with an 80 GB
volume, stopped afterwards and not deleted. It ran about 48 min for
≈ $1.3. The calibration set came from HF, with no second capture. With
`--cpu-threads` it took about 6 s per layer and about 8 min per variant.
Each result went up to the private repo and came down to the Mac from
there. Base recipe: 3-bit g64, as in gptq3. The overrides:

| variant | override | LLM GB | perception | LLM | TTS | codec | **total** | keyword acc | misses | reply WER |
|---|---|---|---|---|---|---|---|---|---|---|
| gptq3 (reference, `table_pr3final`) | — | 4.37 | 16.4 | 39.5 | 24.4 | 5.9 | 87.7 | **1.00** | — | 0.043 |
| mlp2 | MLP up+down 2-bit g64 | 3.93 | 16.2 | **35.6** | 23.6 | 5.7 | **82.6** | 0.90 | blue+yellow = "gray"; Mona Lisa by "Francisco … Esquiza" | 0.011 |
| mlp2 + backbone 8-bit | + `--rtn tts…backbone:8:64` | 3.93 | 18.2 | 36.9 | 21.8 | 6.7 | 85.2 | 0.90 | same two | 0.017 |
| mlp2g32 | MLP up+down 2-bit g32 | 4.15 | 16.8 | 38.0 | 24.6 | 5.9 | 86.7 | 0.95 | elephant is "king of the jungle" | 0.086 |
| down2 | MLP down 2-bit g64 | 4.15 | 15.9 | 37.5 | 23.5 | 5.7 | 84.0 | 0.90 | jaguar = king of the jungle; water freezes at "0 °F" | 0.045 |

- **2-bit MLP costs facts.** Every 2-bit variant gets answers wrong that
  the 3-bit model gets right. The MLPs hold the model's knowledge. The
  speed gain is at most 4 ms (39.5 → 35.6); g32 or only `down_proj` keep
  less than half of it and still miss answers.
- **GPTQ-3 stays the published model.** Below 80 ms needs something else:
  - a smaller or distilled LLM;
  - skipping perception while the model speaks (walkie-talkie);
  - a GPTQ-calibrated 4-bit TTS backbone (RTN 8-bit is −4 ms and safe);
  - an M5 Pro or Max, with more bandwidth.
- The LLM parts are in `pod_results/mix/hf/<variant>` and on the private
  repo (`mlp2`, `mlp2g32`, `down2`). The spliced models were deleted; they
  can be rebuilt with `splice_llm.py` from these files.

## Plan item 4: compiled TTS backbone step — closed, no gain (2026-09-28)

Built it: static K/V buffers (grown in 512-frame chunks), a boolean mask
for the unfilled tail and for what the sliding layers' RotatingKVCache
would drop, one `mx.compile` graph per capacity. Exact: a CPU parity test
past the sliding window and across buffer growth, CFG on and off; bitwise
the same in bf16 on the real model. Branch `tts-backbone-compile` on
ipsupport-llc/mlx-audio, **not merged**.

It isn't faster, because the step isn't launch-bound. It reads 1.19 GB of
bf16 weights (the other TTS parts: MoG head 0.32 GB, subword encoder 0.05 GB)
in about 11 ms, which is ~108 GB/s, the same bandwidth floor as the LLM.
The earlier note that "4-bit weights didn't speed it up" was wrong: it
came from a noisy run. `scripts/bench_tts_step.py`, 200 frames at
batch 2 (CFG):

| backbone | eager ms | static-compiled ms |
|---|---|---|
| bf16 | 11.2–11.8 | 12.0 |
| RTN 8-bit g64 | 7.4–7.8 | 8.3 |
| RTN 4-bit g64 | 5.3–5.4 | 5.5 |

In the full session (`--rtn tts_model.tts_model.backbone:B:64`):

| variant | perception | LLM | TTS | codec | **total** | RTF | keyword acc | reply WER |
|---|---|---|---|---|---|---|---|---|
| gptq3, bf16 backbone (static compile) | 15.9 | 39.8 | 23.3 | 5.7 | 86.1 | 1.08 | 1.00 | 0.043 |
| gptq3 + backbone 8-bit | 16.7 | 39.7 | **20.4** | 5.9 | **84.2** | 1.05 | 1.00 | 0.049 |
| gptq3 + backbone 4-bit | 20.7 | 41.8 | 22.3 | 7.9 | 94.4 | 1.18 | 1.00 | 0.054 |

The 4-bit row ran under outside GPU load: every part slowed, including
perception and codec, which it doesn't touch. Its TTS column isn't
comparable. **Next for the TTS: an 8-bit backbone** (−4 ms, quality holds)
in the next checkpoint. For 4-bit, calibrate (GPTQ on captured TTS inputs)
rather than RTN.

## Plan (agreed with the user, 2026-09-28)

1. LLMTray v0.8.3-beta.2 with the fork speedups (Voice Lab).
2. Public HF release of the GPTQ-3 model with a template card (OpenMDW 1.1).
3. Then: a mixed 2/3-bit LLM (MLP in 2-bit, GPTQ on the pod).
4. Then: a compiled TTS backbone step (fixed-size KV buffer).

## Next

1. Below 80 ms needs ~20 ms more: a 2-bit/3-bit mixed LLM (MLP down 2-bit,
   GPTQ on the pod, calibration set on HF), and/or a compiled TTS backbone
   with a fixed KV buffer; TTS guidance off (batch 1) is worth a quality check.
2. Walkie-talkie mode only: skip perception while the model speaks (silent
   mic) once the encoder output has converged on silence.
4. ~~Package and publish the best variant~~ — done, see Published.
