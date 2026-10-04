# gemma4-ternary findings

## Setup (2026-10-03)

- Host: DGX Spark (GB10, 121 GB unified memory, CUDA 13), torch 2.14.1+cu130,
  transformers 5.17.0. The venv uses a uv-managed Python 3.12: torch 2.14
  routes some ops (the RoPE `bmm` outer product) through Triton, which
  compiles its launcher against `Python.h`, absent from the system Python.
- bf16 matmul: 99 TFLOPS (8192³).

## Cost of ternary training on the Spark

Measured on the real model (Gemma 4 12B, 328 text Linears, 10.90B ternary
params), batch 2 × 2048, gradient checkpointing, SRAdamW (bf16 weights and
moments, stochastic rounding):

- Memory: ≈110 GB in use. fp32 master weights would not fit, hence the
  stochastically rounded bf16 update (verified unbiased: lr 1e-5 moves
  |w| by 1.00e-5 on average while only 25% of weights change a bf16 step).
- Speed: 180 tok/s with the eager quantizer, ≈320 tok/s after fusing it
  (`torch.compile`: 14.3 → 1.0 ms per 59M-weight matrix) and the optimizer
  update. A synthetic model of the same shape reached 420 tok/s.
- Teacher (bf16 forward, top-32): 1500 tok/s.
- So 50M tokens ≈ 9 h teacher + 43 h training on the Spark; 1B tokens would
  take about a month there, ≈1 day on one B200.
- Checkpoint: ≈65 GB (weights + two moments), written in ≈70 s.

## Measuring: chat, not raw web text

On 8 packed rows of raw web text + chat, the master scores PPL 52, and every
quantization looks bad: KL to the master 0.045 at 8-bit RTN, 0.27 on the
QAT grid itself (q4_0), 0.73 at MLX 4-bit RTN. Gemma 4 -it is very unsure
on raw text (see gemma4-quant's perplexity trap), and KL there overstates
the damage. The eval set is therefore chat from smoltalk's held-out test
split; training keeps 30% web text for breadth.

The eval is exact: the unmodified master against its own stored top-k gives
KL 1.3e-7. Top-1 agreement tops out near 98%, not 100%: bf16 logits tie.

## Ternary start

Rounding the master straight to ternary (group 64, on the mixed test rows) breaks it: KL 13.8, top-1 0.2%.
Six optimizer steps (49k tokens) of distillation took KL to 6.1.

## Pilot

50M tokens, group 128, top-32 KL distillation. Eval: 64 chat windows of
2048 tokens from smoltalk's test split, every token scored (user turns
too, so even the 4-bit references score high). References on the same
windows: q4_0 (the QAT grid) KL 0.358 / top-1 83.7%; MLX 4-bit RTN
(g64) 0.662 / 75.9%; 2-bit RTN at the ternary size (g128) 18.7 / 0.25%.
Ternary start (g128): KL 14.13, top-1 1.5%.

### Run 1: lr 3e-5, ternary from step 0 -- plateau

| step (tokens) | eval KL | top-1 |
|---|---|---|
| 150 (2.5M) | 5.96 | 4.6% |
| 300 (4.9M) | 5.89 | 5.1% |

Training loss fell 14.3 -> 5.8 in 20 steps, then sat at 5.6-5.7 for 380
steps. Not frozen weights: at step 399, 1.15% of ternary codes had flipped
(0.55-2.4% per matrix, sampled layers 0/12/24/47) and latent weights had
moved 1.4-4.6% of their magnitude. They moved without improving the model:
gradients through a network that starts at KL 14 carry little signal.
Stopped at step 399 (checkpoint kept in `run/`).

### Run 2: lr 1e-4, ternary projection ramped in over 150 steps

`w + lam * (ternary(w) - w)`, lam 0 -> 1 linearly over the first 150
optimizer steps (`--quant-warmup 150`); the gradient stays the identity.
Eval always scores the fully ternary model.

| step (tokens) | lam | eval KL | top-1 |
|---|---|---|---|
| 50 (0.8M) | 0.33 | 9.24 | 2.4% |
| 100 (1.6M) | 0.67 | 8.65 | 0.7% |
| 150 (2.5M) | 1.0 | 2.30 | 43.2% |
| 200 (3.3M) | 1.0 | 1.77 | 50.5% |
| 250 (4.1M) | 1.0 | 1.70 | 51.1% |
| 300 (4.9M) | 1.0 | 1.63 | 52.2% |

At equal tokens (4.9M): KL 1.63 vs 5.89, top-1 52% vs 5%. Both changes
went in together (time-boxed pilot), so this run does not say how much
each contributed.

Perplexity on the eval windows (20.2 at step 300) is below q4_0's (26.1):
the student is trained on smoltalk and fits its user turns better than
the master does. It is not a quality comparison; KL is.

Running on to 50M tokens. Later points: KL 1.30 / 57.3% at 10.6M tokens,
1.23 / 58.4% at 13.1M, 1.25 / 58.2% at 13.9M.

### First MLX export (step 871, 14.3M tokens)

`export_mlx.py` + `splice_mlx.py`: the 328 ternary Linears as stock MLX
2-bit (g128, scale = s, bias = -s), everything else from our 12B QAT MLX
build (6-bit embeddings). 3.95 GB. The packing was checked against
`mx.dequantize` and `mx.quantized_matmul`; MLX scores the same as training
on 4 eval rows (KL 1.18, top-1 61.6%; the 12B 4-bit build: 0.27 / 84.7%),
so the export is exact.

It does not generate: greedy and sampled replies are repetition and
fragments ("_capital_capital...", "<|channel>" loops) on every prompt. KL
1.2 nats/token teacher-forced is far from usable; the 4-bit builds sit at
0.27-0.36. A second gap: the chat training rows are rendered without the
`<|channel>thought\n<channel|>` prefix the generation prompt carries, so
the student never trained on the exact inference context; dropping the
prefix at inference didn't rescue it.

Decode on the base M5 (200 tokens, greedy, mlx-lm fork):

| model | size | tok/s | peak GB | answer |
|---|---|---|---|---|
| E2B phone MLX | 3.10 GB | 71.9 | 3.27 | correct |
| 12B ternary (step 871) | 3.95 GB | 27.0 | 4.09 | garbage |
| E4B QAT MLX | 5.95 GB | 37.0 | 6.04 | correct |
| 12B QAT MLX (4-bit) | 7.89 GB | 14.9 | 8.02 | correct |

The ternary 12B decodes 1.8x faster than its 4-bit build in half the
memory, but at this point E2B beats it on every axis. It earns its place
only if its quality passes E4B's at 2/3 of E4B's size.

A rough power-law fit of the curve (KL 1.59 at 5.7M -> 1.23 at 13.1M,
exponent ~0.31) puts MLX-4-bit-level KL (~0.6) near 0.13B tokens and
q4_0-level (~0.3) near 1.2B tokens. Three points, noisy; an indication,
not a forecast.
