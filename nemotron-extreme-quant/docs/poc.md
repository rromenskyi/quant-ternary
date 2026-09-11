# PoC Validation

## Purpose

Before investing weeks in the full pipeline (`roadmap.md`), prove that the
core methodology — binary/ternary quantization plus an optional sparse
residual — can hit **<2.0 effective bpw with acceptable quality loss** on
real Nemotron weights.

This is a **go/no-go gate**, run in two stages so the first, cheapest stage
can fail fast without needing a large download.

## Stage A — core methodology, dense model, fully local

Validates the quantization math itself (binary/ternary/residual, packing,
metrics) against real weights and real calibration activations, without any
MoE routing complexity. Small enough to run entirely on a laptop.

| Component | Scope |
|---|---|
| Model | [`nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16`](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16) — dense `nemotron_h` (Mamba + Attention), ~8 GB BF16, fits comfortably in 24 GB RAM |
| Data | A small local calibration set (coding + reasoning prompts), run through a real forward pass with hooks capturing real input activations |
| Layers | A handful of representative linear layers: an attention projection, a Mamba in/out projection, and an MLP gate/up/down |
| Methods | Naive binary · activation-aware binary · binary + magnitude residual · binary + activation-weighted residual · ternary |
| Hardware | CPU (MPS available but not used yet — no MPS backend implemented) |

## Stage B — MoE expert, on the real target

Once Stage A is green, validate the MoE-specific path against the actual
development target,
[`nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16`](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16).

| Component | Scope |
|---|---|
| Model | A single MoE expert (gate/up/down) from one layer, fetched via safetensors byte-range requests — no full 30B download |
| Data | Stage A's calibration set as synthetic input, since capturing *real* activations at an internal layer needs most of the model loaded (~60 GB BF16, doesn't fit locally) — real calibration for this stage happens on a cloud machine later |
| Methods | Same set as Stage A |

If Stage B's synthetic-activation numbers look promising, real-activation
validation on cloud hardware is what actually unblocks Phase 6 of the
roadmap.

## Success criteria (go/no-go)

| Metric | Threshold | Rationale |
|---|---|---|
| Activation output cosine (real calibration input) | ≥ 0.95 | Below this, error compounds catastrophically across layers |
| Weight cosine | ≥ 0.98 | Sanity check — bad weight reconstruction makes activation reconstruction meaningless |
| Effective bpw | ≤ 1.8 | Must clear the 2.0 target with residual overhead included |
| NaN / Inf | Zero | Numerical stability gate |

**All four must pass**, on at least 3 of the 5 sampled layers, for a **Go**.

## Deliverables

1. `poc/quantize.py` — standalone quantization methods (no framework
   dependency beyond torch/numpy)
2. `poc/evaluate.py` — weight/activation MSE, cosine, effective-bpw metrics
3. `poc/collect_acts.py` — forward-hook activation capture on the dense
   model (Stage A) / synthetic activation generation (Stage B)
4. `poc/extract_expert.py` — safetensors byte-range expert extraction
   (Stage B only)
5. `poc/results.md` — the results table below, filled in, plus a decision

### Results table template

| Method | Layer | Weight cos | Act cos | Eff. bpw | Verdict |
|---|---|---|---|---|---|
| Naive binary g128 | | | | | |
| Act-aware binary g128 | | | | | |
| Binary + 1% residual (magnitude) | | | | | |
| Binary + 3% residual (activation-weighted) | | | | | |
| Ternary g128 | | | | | |

## What this PoC does *not* need

- Full model inspection / tensor inventory (Phase 1)
- The policy/config system (Phase 2)
- Multi-block, multi-expert, or full-model runs
- The storage format / manifest / resumable quantization (Phase 18)
- Optimized kernels or inference (Phase 21)
- The evaluation suite — perplexity, coding, agent (Phase 22)
- Cloud packaging or GGUF/MLX export

## If it fails

| Failure mode | Pivot |
|---|---|
| All methods score act-cosine < 0.90 | Nemotron's architecture may be fundamentally incompatible with extreme quantization at this size — try a higher bpw target (2.5–3.0) or a different model |
| Only GPTQ-style correction works, and it's too slow/unstable | Invest in GPTQ optimization, or treat 4-bit as the practical floor |
| Residual needed > 5% to hit 0.95 cosine | Effective bpw exceeds 2.0 — report a negative result rather than force it |
| Stage A passes but Stage B's synthetic-activation numbers don't transfer | Real calibration data matters more than expected — prioritize a cloud run with real MoE activations before anything else |

## Outcome → next action

| Outcome | Action |
|---|---|
| **Go** | Proceed with the full Phase 0–6 implementation in `roadmap.md` |
| **Partial** (some methods work) | Implement only the working methods for Phases 3–6, defer the rest |
| **No-go** | Write up the negative result, archive the repo or pivot the approach |

Decision authority on go/no-go rests with whoever is funding the next 2–4
weeks of compute — this document supplies the numbers, not the decision.
