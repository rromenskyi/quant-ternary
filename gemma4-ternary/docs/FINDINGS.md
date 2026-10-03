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

Running: 50M tokens, group 128, lr 3e-5. Results to follow.
