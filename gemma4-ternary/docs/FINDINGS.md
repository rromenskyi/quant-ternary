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

Running on to 50M tokens.
