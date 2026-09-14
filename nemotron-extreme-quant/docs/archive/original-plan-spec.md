# Specification

## Goal

Build a reproducible, backend-neutral post-training quantization pipeline
that pushes NVIDIA Nemotron hybrid models to **1.1–2.0 effective bits per
weight**, while preserving reasoning, coding, agentic behavior, instruction
following, and factuality.

| | bpw |
|---|---|
| Stretch target | ~1.125 (binary + group scales) |
| Primary target | 1.5–1.8 |
| Hard ceiling | 2.0 |

The current development target is
[`nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16`](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16).
The eventual target this pipeline needs to scale to is the much larger
`nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16`. The PoC (see
[`stage-a-b-poc-validation.md`](stage-a-b-poc-validation.md)) runs against a small dense Nemotron checkpoint instead,
specifically to validate the core quantization math before spending compute
on either MoE model.

## Research philosophy

This is not a Q4/Q8 conversion project. The techniques under study:

- Bonsai-style binary weight compression
- GPTQ / OBQ (second-order weight correction)
- QuIP-style incoherence processing (random/Hadamard rotation)
- Ternary quantization
- Activation-aware scaling
- Salient-weight / outlier preservation
- Sparse residual correction

**Constraint:** only publicly documented techniques and public reference
implementations may be used or adapted — see
[`bonsai-1bit-repro`](https://github.com/ThakiCloud/bonsai-1bit-repro) as
prior art for binary packing, group-wise quantization, and evaluation
methodology. No proprietary or confidential quantization IP goes into this
repository, since the end goal is a public release.

## Target architecture

Nemotron's `nemotron_h` family is a hybrid architecture, discovered from each
checkpoint's config rather than hardcoded:

- Mamba-2 state-space blocks
- Transformer attention blocks
- Mixture-of-Experts (MoE) feed-forward blocks, some with a shared expert
- Multi-Token Prediction (MTP) head
- Very long context (up to 256K)

The quantization policy system must treat each of these tensor classes
differently — see [`roadmap.md`](original-plan-roadmap.md) Phase 14/15 for the
Mamba- and MoE-specific studies this implies.

## Hardware & backend requirements

- Must run **without CUDA**. CPU is the required baseline backend.
- Intel XPU is a preferred backend where available.
- CUDA is optional, used mainly for the eventual cloud run.
- Every compute-heavy command takes an explicit `--device cpu|xpu|cuda`.
- Backend abstraction lives in `nemotron_quant/backends/`; algorithms
  themselves stay backend-neutral.
- **No silent fallback.** If an operation moves from the requested backend to
  another one, that must be logged explicitly.

## Resource constraints

Full BF16 weights are not assumed to fit in RAM or VRAM. Full-model
quantization streams: disk → CPU staging → compute → quantized output → disk
→ free original. Disk space is estimated and checked *before* starting a
full-model operation, failing early if insufficient.

## Deployment artifacts (hard requirement)

A quantized checkpoint that only exists in this project's internal storage
format is not a deliverable. The pipeline must also produce artifacts that
run in standard, unmodified inference engines.

### GGUF / llama.cpp

Must produce a GGUF file that loads and runs in **stock, unpatched
llama.cpp**.

Targets, in priority order:
1. **Q1_0** — llama.cpp's native 1-bit binary quantization (sign + block scale)
2. **TQ1_0** — native 1-bit ternary quantization, if available upstream
3. A custom GGUF type — only if it exactly matches the GGUF spec and a
   minimal llama.cpp PR would be acceptable

Forbidden: misrepresenting tensor layout to llama.cpp, mislabeling a
quantization type, or shipping a GGUF file that crashes or produces garbage.

Acceptance: `llama-cli -m out.gguf -p "def fib(n):" -n 50` runs cleanly, no
NaN, no crash.

### MLX / Apple Silicon

Must produce an artifact (safetensors + config) that runs via `mlx-lm` or
native MLX on Apple Silicon.

Targets:
1. Native 2-bit via `mlx.nn.quantize` — if the quality/size tradeoff is
   acceptable
2. A custom 1-bit backend — only if 1-bit *materially* outperforms 2-bit on
   coding/agent benchmarks at a similar effective bpw

Acceptance: `python -m mlx_lm.generate --model ./mlx_model --prompt "def fib(n):" --max-tokens 10`
runs cleanly.

### Constraints specific to GGUF/MLX export

- GGUF Q1_0 block size is fixed at 32 weights per FP16 scale, so our group
  size must be 32 or a multiple of it to map directly.
- MLX has no native 1-bit kernel; a 1-bit path needs a custom
  `QuantizedLinear` and kernel.
- llama.cpp and MLX do not natively support Mamba-2 blocks or MTP heads for
  this architecture yet. Strategy: export the attention+MoE backbone first,
  keep MTP as separate (unused-by-default) tensors, and document the
  limitation rather than silently dropping it.
- The same quantization recipe (policy, group sizes, residual %,
  calibration, rotation, GPTQ params) should drive both export targets. If
  GGUF and MLX end up needing different recipes to hit acceptable quality,
  that divergence must be documented and justified, not left implicit.

Export validation is a quality gate: if GGUF or MLX export fails, the
quantization recipe is not considered complete, regardless of how good its
offline reconstruction metrics look.

## Guiding principle

> Measure first. Compress second. Optimize third.
