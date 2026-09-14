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

## Stage A results (2026-09-11)

The methodology in `docs/spec.md` (GPTQ + Hadamard rotation) reconstructs a
single layer's output almost exactly in isolation — 0.98-0.997 activation
cosine at 1.125 bpw across Mamba/Attention/MLP layers (`poc/results.md`).
Applying that in a one-shot pass across all 92 target layers of the full
model does not transfer: it collapses the model (perplexity 601.6x versus
FP16, generations degenerate into repeating a single token). Root cause,
confirmed by isolating it: naive one-shot calibration captures every layer's
activations against the *original, unquantized* model, then quantizes all
layers at once — so every layer downstream of the first is calibrated
against activation statistics that no longer hold once upstream layers are
actually quantized, and the error compounds across a recurrent (Mamba)
architecture.

The fix is standard GPTQ practice, just not what round 1 did: quantize block
0, re-derive block 1's calibration by running the calibration set through
the model with block 0 *already quantized*, and so on — implemented in
`poc/quantize_sequential.py`. Combined with salient-weight pinning (top ~3%
of weights by activation-weighted magnitude, selected in the *rotated*
domain and excluded from GPTQ's column loop entirely rather than patched in
afterward — `poc/methods.py`), this is a dramatically different picture:

| Blocks quantized | % of model | Perplexity ratio vs FP16 |
|---|---|---|
| 5 | 12% | 1.77x |
| 10 | 24% | 5.08x |
| 20 | 48% | 11.9x |
| 42 (full model) | 100% | **46.5x** |

The per-block growth rate decelerates as more blocks are added (roughly
1.24x/block over the first 5, 1.09x/block by block 20, 1.07x/block by block
42) — error is *not* compounding exponentially, which is what would make
this approach fundamentally unviable. But 46.5x is not remotely
"near-lossless": full-model generations at this setting are visibly
degraded (malformed recursion, repeated-digit loops on some prompts), a
different failure mode from round 1's total collapse but still not a
result you'd ship. Full data and generation samples: `poc/experiments_log.csv`
and the per-run `poc/quality_report_*.md` files.

An earlier attempt at *post-hoc* residual correction (patch GPTQ's largest
errors after quantization finishes, rather than pinning them before) made
activation cosine measurably *worse*, not better — GPTQ had already spent
its error-compensation budget assuming those positions would stay wrong in
a specific way, and overwriting them broke that balance. Pinning the same
salient fraction *before* GPTQ runs (so those positions are excluded from
column-loop error propagation instead of contributing zero error into it)
improved both weight and activation cosine monotonically instead. This is
implemented, not just theorized — see `gptq.py`'s `salient_mask` parameter.

**Revised verdict: PARTIAL.** The core methodology is validated far beyond
round 1's naive result (dramatically so — order-of-magnitude), and the
mechanism (sequential calibration + salient pinning) is sound and
demonstrably correct. But 3% salient at pure 1.125 bpw binary is not
sufficient for a full dense model to be near-lossless; unexplored knobs
before calling this done: a higher salient fraction (5-10%), ternary
instead of pure binary on the layers that turn out most sensitive, and a
proper sensitivity scan (roadmap.md Phase 13) to find which specific layers
need it rather than a uniform 3% everywhere. None of this blocks Stage B
prep, which proceeded in parallel (see below) — the finding that matters
for scale is that the *method* works and *how much* correction it needs
scales with how much of the model gets quantized, not that it's a dead end.

## Stage B — MoE expert, on the real target

Once Stage A is green, validate the MoE-specific path against the actual
development target,
[`nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16`](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16).

| Component | Scope |
|---|---|
| Model | A single MoE expert (gate/up/down) from one layer, fetched via safetensors byte-range requests — no full 30B download |
| Data | Stage A's calibration set as synthetic input, since capturing *real* activations at an internal layer needs most of the model loaded (~60 GB BF16, doesn't fit locally) — real calibration for this stage happens on a cloud machine later |
| Methods | Same set as Stage A |

**Prep done 2026-09-11** (see `docs/archive/stage-b-prep-notes.md` for full detail): rented
a RunPod RTX A6000 pod to validate the cloud setup before committing to the
real 30B run. `mamba-ssm`/`causal-conv1d` installed and ran real CUDA
kernels — Nano-4B generation went from ~0.4 tok/s on the local CPU
reference path to **25.2 tok/s once warm**, a ~60x speedup, which reframes
Stage B's expected wall-clock entirely. One negative finding worth carrying
forward: running the GPTQ column-loop itself on the GPU (`--gptq-device
cuda`) is *slower* than CPU, not faster — many small sequential ops pay CUDA
kernel-launch overhead with nothing to batch. The right split, now
implemented in `quantize_sequential.py` as independent `--device` /
`--gptq-device` flags, is model forward pass on GPU, GPTQ math on CPU.
Still unsolved and unblocked by any of the above: the routed-expert
calibration-data-sparsity problem (`docs/archive/stage-b-prep-notes.md`'s main section) —
Stage A's 279-token calibration set works for a dense model but would give
most of 128 routed experts single digits of samples each, nowhere near
enough for a GPTQ Hessian.

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
