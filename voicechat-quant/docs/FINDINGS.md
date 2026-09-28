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

## Next

1. Perception step: `mx.compile` of the one-frame step with fixed-size ring
   caches, precomputed positional terms, 4-bit weights (RTN holds quality:
   user WER 0.013). Target ~10 ms.
2. TTS step: the same (25.6 ms now).
3. Walkie-talkie mode only: skip perception while the model speaks (silent
   mic) once the encoder output has converged on silence.
4. Package the best variant as a full MLX checkpoint with a card from
   `HF_CARD_TEMPLATE.md` and publish (OpenMDW 1.1 notices, NVIDIA attribution).
