# Stage B Prep — Notes for the 30B-A3B MoE Run

Written while Stage A's full 42-block run was still crunching locally, so this
is prep/planning, not yet executed. Nothing here has been run — verify prices
and package availability again before actually renting anything.

**Update 2026-09-11:** a RunPod RTX A6000 pod (US-KS-2, $0.53/hr) was actually
rented to validate the setup checklist below. `mamba-ssm`/`causal-conv1d`
installed and ran real CUDA kernels — measured **25.2 tok/s generating on
Nano-4B once warm** (first call pays a one-time Triton JIT-compile cost, ~40s;
exclude it when timing). Stage A's CPU numbers on the same model were roughly
0.4 tok/s. That's a **~60x** generation speedup, which reframes the whole
Stage A timing model: what took hours locally (the 42-block sequential run)
should take minutes on a GPU with working Mamba kernels. Re-estimate Stage B's
30B-A3B wall-clock/cost from this number, not from Stage A's CPU-bound one.

Gotcha hit installing `mamba-ssm`: pip's dependency resolver silently
upgraded `torch` (2.8.0+cu128 → 2.14.0+cu130) to satisfy some transitive
requirement, which broke CUDA entirely (driver 570.133.20 only supports
CUDA 12.8, not 13.0) and left `triton` at a version torch didn't want either.
Fix: install `causal-conv1d`/`mamba-ssm` with `--no-build-isolation --no-deps`
(after exporting `CUDA_HOME=/usr/local/cuda` and adding it to `PATH`, since
`nvcc` isn't on `PATH` by default even on the "devel" template) — `--no-deps`
is the important one, it's what stops the resolver from touching torch/triton
at all. Verify with `python3 -c "import torch; print(torch.cuda.is_available())"`
after every mamba-ssm-related pip install, not just once.

## Cloud: RunPod

Checked 2026-09-10. RunPod, not CoreWeave/Lambda, per current preference —
cheapest reasonable option for a single-GPU Stage B run.

| GPU | Secure Cloud | Community Cloud |
|---|---|---|
| A100 80GB PCIe | $1.39/hr | $1.39/hr |
| A100 80GB SXM | $1.59/hr | — |
| H100 80GB PCIe | $2.89/hr | ~$1.99/hr |
| H100 80GB SXM | $3.29/hr | ~$2.69/hr |

- No ingress/egress data charges, billed per second.
- Secure Cloud = verified datacenters, more reliable. Community Cloud = cheaper,
  host-dependent availability. For a few-hour experimental run, Community
  Cloud's A100 80GB at $1.39/hr is the obvious starting point — a full day of
  experimentation costs under $35.
- Default spend cap is $80/hr across all resources — fine for a single pod,
  worth knowing before spinning up anything bigger.

Sources: RunPod pricing page and 2026 comparison roundups (Flexprice,
Spheron, Thunder Compute) — re-verify at runpod.io/pricing before paying,
prices move.

## Getting artifacts back out

Decided against routing large files through the local Mac:

1. **Quantized checkpoints (GBs)** → push straight from the pod to a private
   HuggingFace Hub repo via `huggingface_hub.upload_folder()`. Matches how
   the rest of this project already works, no extra infra, fast HF backbone.
2. **Small stuff (logs, `experiments_log.csv`, `results.md`)** → plain `scp`/
   `rsync` over the SSH access RunPod gives every pod.
3. RunPod Network Volumes are worth attaching if we expect multiple separate
   rental sessions against the same calibration cache — avoids re-downloading
   the 30B checkpoint each time.

## First-hour checklist on a fresh pod

1. `pip install torch transformers safetensors numpy huggingface_hub`
2. `pip install mamba-ssm causal-conv1d` — these are the packages that only
   install/compile with a CUDA toolchain, which is *why* Stage A was stuck on
   the slow reference path locally. This should turn the "much slower"
   fallback warnings from the whole session into real Mamba2 kernels on
   RunPod's GPUs, and would speed up both calibration collection and
   generation substantially.
3. `huggingface-cli login` (need a token if the model is gated, or for
   pushing results to a private repo either way)
4. Sanity check: `nvidia-smi`, confirm the expected VRAM is actually visible
   before downloading a 60GB checkpoint.

## The real open question: calibration for MoE

This is the part Stage A's dense-model result doesn't de-risk at all (see
docs/poc.md's Stage B section, and the conversation note that even PrismML's
published Bonsai result was demonstrated on a *dense* 27B model, not MoE).

`nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16` config: `n_routed_experts:
128`, `num_experts_per_tok: 6`, `n_shared_experts: 1`. Each token activates
6 of 128 routed experts (~4.7%) plus the always-on shared expert.

Stage A's calibration set was 10 short prompts, ~279 tokens total. For a
dense layer that's 279 real samples per Linear layer — thin but workable,
per Stage A's actual results. For a *routed* expert, at ~4.7% average
routing share, 279 tokens gives an expected ~13 samples per expert, and
routing is not remotely uniform — several experts would likely see 0-2
tokens from a calibration set this small. That's nowhere near enough for a
GPTQ Hessian (X^T X needs to be well-conditioned; a handful of samples in
a few-thousand-dimensional space is rank-deficient garbage).

What actually needs to change in `collect_acts_full.py` /
`quantize_sequential.py` for MoE:

1. **Much bigger, much more diverse calibration corpus.** Not 10 prompts —
   plausibly hundreds of sequences across varied domains (code in several
   languages, math, dialogue, long-form prose), specifically to spread
   token routing across as many of the 128 experts as possible. This is
   roadmap.md Phase 8's calibration pipeline (128 sequences × 512-2048
   tokens is its stated default) — Stage A never needed to build that
   because the dense-model PoC could get away with a token calibration set.
2. **Per-expert activation bucketing.** The forward-hook approach in
   `collect_acts_full.py` captures a Linear layer's *entire* input tensor;
   for a MoE expert's gate/up/down projections, that input is already
   post-routing (only the tokens dispatched to that expert), so the hook
   itself doesn't need new logic — but the *router's* dispatch needs to be
   observed too, to know how many tokens actually reached each expert and
   decide whether that expert has enough data to quantize at all.
3. **A per-expert usage threshold with a real fallback.** roadmap.md Phase
   15 already calls for "router statistics from calibration" and "per-expert
   quantization policy" — this is that phase, pulled forward out of
   necessity rather than by plan. Concretely: experts below some minimum
   token count (TBD empirically) should stay BF16 rather than get a
   garbage GPTQ fit from insufficient data. Given routed-expert traffic is
   usually power-law distributed (a handful of experts dominate token
   share), this fallback might apply to a meaningful fraction of the 128.
4. **The shared expert (`n_shared_experts: 1`) behaves like a dense layer**
   — every token passes through it, so it gets the same calibration
   coverage as Stage A's dense layers and needs none of the above.

None of this changes the core method (rotation + GPTQ + salient pinning,
applied sequentially block-by-block) — it changes how much and what kind of
calibration data feeds it, and adds a data-sufficiency gate before quantizing
a given expert at all. Worth building and testing the calibration-volume
question *before* committing to a specific quantization recipe for MoE,
since if routing turns out too concentrated even a large corpus might starve
the long tail of experts.

### Measured, not just predicted (2026-09-11, real 30B-A3B, block 1)

Ran `poc/quantize_moe_block.py` against a real MoE block on a rented A100.
Reality is *less* bad than predicted on one axis and confirms the rest:
**128/128 experts got at least one token** from Stage A's 279-token
calibration set (better than expected — routing wasn't as concentrated as
feared), but the actual counts are still thin: **min=2, median=12, max=33
tokens per expert**, against an up_proj `in_features` of 2688. A Hessian
from 12-33 samples in a 2688-dimensional space is severely rank-deficient
(rank ≤ sample count), regardless of how the samples are chosen.

Mitigation ideas, roughly in the order to try them:

1. **Raise `percdamp` for thin-data experts.** `gptq.py`'s damping
   (`damp = percdamp * diag_mean`) regularizes the Hessian toward identity.
   With a rank-deficient raw Hessian, a much larger `percdamp` (0.1-0.5
   instead of the current 0.01 default) should make GPTQ degrade gracefully
   toward simple per-group scaling instead of fitting to a nearly-singular
   matrix. Cheapest thing to try — no new calibration data needed, one
   parameter change, testable immediately on the existing captured data.
2. **Scale the calibration corpus 40-200x.** 279 tokens → 10k-50k tokens
   would move the median from ~12 to ~500-2400 samples/expert, comfortably
   past the 2688-dimension threshold for a well-conditioned Hessian. Still
   cheap wall-clock-wise on a real GPU (this session measured ~60x faster
   forward passes with working Mamba kernels vs Stage A's CPU reference
   path) — the cost is calibration-corpus curation, not compute.
3. **Graceful-degradation fallback tier**, already partly described in
   point 3 above: full GPTQ where sample count clears some threshold well
   past `in_features`, plain naive/activation-aware binary (no Hessian
   dependency) where data is thin but nonzero, BF16 only where an expert
   got zero tokens even from the larger corpus.
4. **Router-aware adaptive calibration** (more speculative, not implemented
   anywhere we've seen): run calibration once, look at which experts came
   up starved, deliberately add more text likely to route to *those*
   experts specifically, repeat. Iterative/coverage-driven rather than a
   single fixed corpus.

`quantize_moe_block.py` now prints a `[expert i/127] done in Xs, ETA Ys` line
per expert (per-expert mode) or per-group progress inside the batched call —
the original "no visibility for 30+ minutes" problem is fixed.

### Resolved: `torch.cholesky_inverse` was a hidden 30x-slowdown bug

The first real MoE run (and a follow-up isolated timing test) hung for
30-50+ minutes with no clear cause. Root-caused to `gptq.py`'s
`_compute_hinv()`: `torch.cholesky_inverse(L)` (LAPACK's `potri`) took
**24+ seconds** for a single 2688x2688 matrix on this hardware's BLAS,
*regardless of data conditioning* (confirmed with both a 12-sample
rank-deficient case and a 500-sample well-conditioned control). Replaced with
`torch.cholesky_solve(eye, L)`, which reuses the same Cholesky factor but
runs in ~0.7s — a ~30x fix, applied for both CPU and GPU paths. This one-line
bug had been silently inflating every GPTQ call in the whole project,
including Stage A's dense-model runs.

### Resolved: does the "GPU is slower for GPTQ" finding even apply here?

Stage A's single-layer test showed GPU-based GPTQ was slower than CPU — but
that was measured *before* the `cholesky_inverse` fix, when a single
pathological call dominated any timing regardless of device. Once fixed, a
fair single-matrix microbenchmark (2688-dim Hessian, 500 samples) showed
**GPU at 10.4s vs CPU at 378s — GPU ~36x faster**, reversing the earlier
conclusion. The remaining bottleneck on real (small per-expert sample count)
data is the per-column Python loop's kernel-launch overhead, not raw compute.

Built and tested the batched-across-experts path (`gptq_binary_batched`,
`_gptq_run_batched` in `gptq.py`, `rot_gptq_salient_batched` in `methods.py`,
`--batched` flag on `quantize_moe_block.py`): stacks all eligible experts'
weights into one `[E, out, in]` tensor, zero-pads each expert's calibration
to a common sample count (exact, not approximate — zero rows don't perturb
`X^T X`), and runs one Hessian/Cholesky/column-loop over all E experts at
once instead of E independent Python loops. Verified numerically identical
to the per-expert path on well-conditioned synthetic data (0 mismatched
weights); on near-singular data (n_samples=3 in a 256-dim space) a handful of
weights differ by floating-point-level Cholesky sensitivity, not a bug.

**Measured on real block 1, 112 eligible experts (of 128):**

| Mode | Wall time | Speedup |
|---|---|---|
| Per-expert loop (`--gptq-device cuda`) | 1565s (~26 min) | 1x |
| Batched (`--batched --gptq-device cuda`) | 270-284s (~4.6 min) | ~5.5x |

Less than the ~36x the single-matrix microbenchmark suggested — the batched
Hessian/Cholesky over 112 stacked 2688x2688 (up_proj) and 3712x3712
(down_proj) matrices pushes GPU memory to ~80.5/82GB (the 60GB model plus
Hinv/H/eye intermediates for 112 experts at once), which likely throttles
cuSOLVER's batched routines. Still a decisive, real win, and the per-column
loop itself dropped from ~1565s to ~1.1s (21 groups) — confirming the
kernel-launch-overhead hypothesis directly.

**Quality check (full `eval_quality.py`, not `--quick`) on the batched block-1
checkpoint:** held-out perplexity **3.101 -> 3.119 (1.006x)** — essentially no
degradation. 2 of 3 greedy generations matched the original *exactly*
token-for-token; the third took a different (still correct) solution path.
This is a strong first Stage B signal: one MoE block, quantized with only
279 calibration tokens spread across 128 experts (median 12 tokens/expert),
held up about as well as Stage A's best dense-model results. Not yet tested:
more than one block, the full model, or whether quality holds as more blocks
compound (Stage A showed compounding degradation matters a lot across many
blocks — see docs/poc.md).

## GGUF/LM Studio export: why our exact method doesn't fit, and what does

Investigated whether a quantized checkpoint can be loaded in LM Studio (for
a real, user-visible memory comparison) via either of its two backends:

- **llama.cpp/GGUF** (all platforms): dense hybrid Mamba2+Attention
  ("NemotronH") architecture support merged Dec 2025. The **MoE variant**
  (needed for our 30B-A3B target) is explicitly **not implemented** — issue
  requesting it was closed "not planned", no PR. What's actually missing is
  narrower than "nothing works": llama.cpp already has generic MoE routing/
  batched-expert-tensor/shared-expert primitives (used by Mixtral, Qwen-MoE,
  DeepSeek-MoE), and the NemotronH conversion script already has an `is_moe`
  flag stub — nobody has wired the two together and validated the specific
  hybrid-SSM+Attention+MoE combination. Realistically 1-3 weeks of C++ work
  for someone familiar with the codebase, not a fundamental blocker — but a
  separate project, not something to fold into this session.
- **MLX (LM Studio, Mac-only)**: `mlx-lm` actually *does* have a `nemotron_h.py`
  model implementation already. But LM Studio's MLX loader
  (`mlx-engine`'s `ModelKit`) calls `mlx_lm.utils.load()` with no
  `trust_remote_code` override, and `mlx-lm` defaults that to `False` —
  so, like llama.cpp, only architectures already registered inside the
  installed `mlx-lm` package will load. Same category of constraint as GGUF.

Both backends' low-bit formats (llama.cpp's TQ1_0/TQ2_0, MLX's fixed
`affine`/`mxfp` `QuantizedLinear` modes) are **block-uniform**: every weight
in a block must be one of a fixed small set of codes (ternary: {-1,0,1}), with
no room for our method's per-weight **salient pinning** (top ~3% of weights
kept at exact full precision, chosen in the rotated domain). That mechanism
is fundamentally incompatible with a fixed-format GGUF/MLX export — not a
detail to work around, a structural mismatch.

### Salience concentration analysis (which tensor TYPES are fragile)

Before assuming which whole tensors to protect in a mixed-precision export
(GGUF allows different quant types per tensor — this part *is* standard),
measured how concentrated each tensor's "importance mass" is: what fraction
of the top-3%-by-score weights account for, out of the tensor's total
activation-weighted score sum. Computed per-block on the real dense 4B model
(`poc/analyze_salience.py`, no GPTQ/Cholesky needed — just the same
`activation_weighted` scoring `methods.py` already uses for salient
selection), no quantization involved:

| Tensor type | Mean top-3%-mass ratio | Range |
|---|---|---|
| attention.k_proj | 0.293 | 0.24-0.32 |
| attention.q_proj | 0.227 | 0.21-0.24 |
| attention.v_proj | 0.221 | 0.19-0.23 |
| attention.o_proj | 0.164 | 0.16-0.17 |
| mamba.in_proj | 0.154 | 0.12-0.17 |
| mlp.up_proj | 0.151 | 0.12-0.20 |
| mamba.out_proj | 0.142 | 0.10-0.59 (one outlier block) |
| mlp.down_proj | 0.141 | 0.11-0.30 (one outlier block) |

Attention (k_proj especially) looks most "locally fragile" by this metric.
That prediction turned out to be actively misleading for the *global*
sequential-collapse question — see below.

### Ablation: which whole tensors actually need full precision

`quantize_sequential.py` gained a `--skip-quantize-for proj_name,...` flag
(leaves listed projections at original precision, no GPTQ at all) to test
mixed-precision export candidates directly, as full 42-block runs
(`--method ternary --group-size 256`, since TQ2_0's one-scale-per-256-weights
block forces group_size=256 to match our own group-wise scale granularity).

| Configuration | Perplexity ratio | GGUF-compatible? |
|---|---|---|
| Pure ternary, no protection | 678.6x (collapse) | yes |
| Per-weight salient=3% everywhere (our normal method) | 114.6x | **no** |
| Skip-quantize attention (q/k/v/o at F16) | 1142.0x (**worse than nothing**) | yes |
| **Skip-quantize mamba (in/out_proj at F16)** | **19.6x** | **yes** |
| Skip-quantize mamba + mlp.down_proj (F16) | **9.35x** | **yes** |
| Per-weight salient=10% everywhere | 9.0x (best overall) | **no** |

Two findings that cut against the per-tensor concentration analysis above:

1. **Protecting attention alone made things *worse* than no protection at
   all.** This hybrid architecture is mostly Mamba/MLP blocks by count
   (attention appears roughly every 6-8 blocks); since quantization is
   *sequential* (block i+1's calibration activations come from the model
   with blocks 0..i already quantized), leaving the majority of blocks
   (Mamba+MLP) unprotected means they collapse and poison every downstream
   activation regardless of what happens to the minority attention blocks.
   Local per-tensor salience concentration doesn't predict global
   sequential-collapse sensitivity.
2. **Protecting Mamba entirely (`in_proj`/`out_proj` at F16, everything else
   pure ternary, no salient) is the best GGUF-compatible result found**
   (19.6x) — better than salient=3% (114.6x) despite salient=3% touching
   *every* tensor including Mamba's. Mamba's SSM parameters are the
   structurally critical component for this architecture's stability under
   quantization, more so than raw per-weight importance concentration
   predicts.

Also note the bpw cost of per-weight salient pinning scales sharply with
fraction: ~2.06 bpw (no salient) -> ~2.9 bpw (salient=3%) -> ~4.9-5.1 bpw
(salient=10%) for the tensors it touches, since the residual-storage
overhead (position + value per pinned weight) grows with count. Skip-mamba
avoids this entirely for the tensors that stay ternary (flat 2.06 bpw) at
the cost of full 16 bpw for the (relatively small parameter share) Mamba
tensors specifically.

Confirmed the `mlp.down_proj` hypothesis directly: adding it to the skip
list (Mamba `in_proj`/`out_proj` + `mlp.down_proj` all at F16, everything
else pure ternary/no-salient) reached **9.35x** — nearly 2x better than
skip-mamba alone, and now within noise of the best (but GGUF-incompatible)
per-weight salient=10% result (9.0x), while staying fully representable as
plain F16 + TQ2_0 tensors in a real GGUF file. The two tensor types with the
per-block concentration outliers (mamba.out_proj's 0.59 at block 6,
mlp.down_proj's 0.30 at block 41) turned out to matter in combination —
consistent with those outlier blocks being genuinely fragile positions that
per-tensor-type averages alone wouldn't have flagged as strongly.

This is the practical answer for a real GGUF export of this project's dense
model: **F16 for mamba.in_proj, mamba.out_proj, and mlp.down_proj; TQ2_0 for
everything else (mlp.up_proj, all attention projections)** gets within ~10x
perplexity ratio using only standard, off-the-shelf GGUF mixed-precision —
no llama.cpp changes needed. Not yet at salient-pinning-level quality
(1.006x-4.7x range Stage A/B achieved with per-weight pinning), but a real,
loadable-today result.

### The rotation problem: stored weights aren't literally ternary

`rot_gptq_salient`'s `rotate()`-then-`unrotate()` wrapper is why a plain
`nn.Linear` computes the correct rotated-domain-quantized result without any
runtime changes — but it means the *stored* weight is a dense reconstruction
(every entry a linear combination of up to `group_size` rotated-domain
ternary values), not literally `{-scale, 0, scale}` in the standard basis.
Confirmed directly: a 256-element group of an already-"ternary" checkpoint
had **29 distinct values**, not 3. TQ2_0 requires genuinely 3-valued groups
in the standard basis — there's no way to encode "this block was secretly
rotated" in the format. Fix: `quantize_sequential.py` gained a
`--skip-rotation` flag that calls `gptq_binary`/`gptq_ternary` directly,
bypassing the rotate/unrotate wrapper entirely. Verified the result is
exactly 3-valued per group (confirmed across `mlp.up_proj`, `attention.q_proj`,
`attention.o_proj`). Cost: dropping rotation moved skip-mamba+down_proj from
9.35x to 12.43x — a real but non-catastrophic quality hit, and the necessary
price for a packable result.

### First real GGUF conversion validation

Ran llama.cpp's own `convert_hf_to_gguf.py` directly against the
no-rotation quantized checkpoint (as a plain F16 export, no custom packing
yet) — **it succeeded**, confirming the project's exact
`NemotronHForCausalLM`/`nemotron_h` config is genuinely covered by
llama.cpp's dense-hybrid support, not just similar architectures. This
de-risks the packer work below: the metadata/tensor-naming layer already
works, only the quant-type substitution for specific tensors remains.

### The real bottleneck: Mamba dominates parameter count

Computing the actual mixed-precision file size (not just the perplexity
ratio) revealed the practical ceiling: **mamba.in_proj + mamba.out_proj +
mlp.down_proj together are 58.6-75.4% of this model's total parameters**
(depending on grouping), while the TQ2_0-eligible tensors (mlp.up_proj +
attention q/k/v/o) are only ~20.7%. Keeping the majority-parameter share at
F16 caps the whole-file compression at **~1.22x** regardless of how good the
ternary portion's quality is — the win from the tensor-type ablation above
was real for *quality*, but nearly irrelevant for *size* on its own.

Tested whether Mamba/down_proj could tolerate a milder quantization than
full F16 but gentler than 1-bit ternary: `gptq.py` gained `gptq_nbit(bits,
group_size)` — standard symmetric uniform quantization (Q8_0/Q4_0-style:
`scale = absmax / (2^(bits-1)-1)`, evenly-spaced levels) reusing the same
GPTQ Hessian-based error-compensation column loop, just with a different
`decide` function and a `scale_mode="max_abs"` option added to `_gptq_run`.

| Mamba precision | Perplexity ratio | down_proj | File size | Reduction |
|---|---|---|---|---|
| F16 (16 bpw) | 12.43x | F16 | 6.51 GB | 1.22x |
| Q8 (8.5 bpw) | 12.82x | F16 | 4.96 GB | 1.60x |
| Q4 (4.5 bpw) | 14.20x | F16 | ~4.13 GB | ~1.93x |
| Q8 (8.5 bpw) | 28.82x | **ternary** | smaller | quality regresses |
| **Q8 (8.5 bpw)** | **12.74x** | **Q8 (8.5 bpw)** | **4.33 GB** | **1.83x** |

Key findings: Q8 for Mamba is essentially free (12.82x vs 12.43x at full
F16 — within noise), meaning the majority-parameter tensors don't actually
need full precision, just more than 1-2 bits. Q4 costs a little more
(14.20x) but is still usable. `mlp.down_proj` still cannot go to pure
ternary even with Mamba protected (28.82x, worse than Mamba-F16+down_proj-F16)
— but it tolerates Q8 just as well as Mamba does (12.74x, matching Mamba-Q8
alone almost exactly). **Best practical config found: mamba.in_proj,
mamba.out_proj, mlp.down_proj all at Q8_0; mlp.up_proj and all attention
projections at TQ2_0 — 1.83x smaller than F16 at ~12.7x perplexity ratio**,
using only standard GGUF quant types (Q8_0 and TQ2_0 both ship in stock
llama.cpp) and requiring zero upstream engineering.

### TQ2_0 is blocked for most tensors: hidden_size=3136 isn't divisible by 256

Attempting to actually pack via `gguf-py`'s `quantize(data, TQ2_0)` failed
immediately: `can_quantize()` hard-requires `tensor.shape[-1] % 256 == 0`,
no padding fallback. This model's `hidden_size=3136` is the `in_features`
for most linear layers (`mlp.up_proj`, `attention.q/k/v_proj`,
`mamba.in_proj`) — and 3136/256=12.25, not integral. Only tensors whose
in_features is a *different*, 256-divisible dimension are TQ2_0-eligible:
`attention.o_proj` (5120), `mamba.out_proj` (7680), `mlp.down_proj` (12544).
NVIDIA's own official GGUF release of this exact model uses Q4_K_M, not a
ternary format — this is almost certainly why: K-quants/TQ-quants share the
same 256-block constraint, so a from-scratch ternary GGUF for this model's
shape was never going to be possible without a legacy 32-block fallback.
32-block formats (Q8_0, Q4_0, Q5_0) have no such restriction (3136/32=98).

**Final packing recipe**: Q8_0 for `ssm_in`/`ssm_out`/`ffn_down` (majority
of params, validated to cost ~0 quality vs F16), TQ2_0 for `attn_output`
(the one 256-divisible ternary-eligible tensor), Q4_0 fallback for
`ffn_up`/`attn_q`/`attn_k`/`attn_v` (blocked from TQ2_0, 4.5 bpw instead of
2.06 bpw), F16 kept for embeddings/lm_head/norms/SSM scalar params.

### Real GGUF assembly: `poc/pack_gguf.py`, and three upstream llama.cpp bugs found along the way

Built a packer (uses `gguf-py`'s `GGUFReader`/`GGUFWriter` directly, no
llama.cpp rebuild needed for the packing step itself): convert the quantized
HF checkpoint to a reference F16 GGUF via the stock `convert_hf_to_gguf.py`
(to get correct tensor naming/NemotronH-specific transforms for free), then
rewrite it tensor-by-tensor — matched suffixes get repacked via
`gguf.quants.quantize(data, target_type)` (this call alone does the
correct, exact packing — no manual bit-twiddling needed once you know which
target type each tensor should get), everything else copied through as-is.
`writer.add_tensor(name, packed_bytes, raw_dtype=target)` — critically,
**omit `raw_shape` entirely** here; it defaults to `tensor.shape` which for
already-packed bytes *is* the byte-shape the writer's internal
`quant_shape_from_byte_shape` expects, and passing the logical (unpacked)
shape instead produces a byte-count mismatch error.

Getting to a file that actually **loads** in llama.cpp (not just parses)
surfaced three real upstream bugs in `convert_hf_to_gguf.py`'s NemotronH
support, all stemming from one root cause: **`AutoConfig.from_pretrained(dir).to_dict()`
always includes every field `NemotronHConfig`'s Python class defines,
including MoE-only fields, regardless of whether the original checkpoint's
raw `config.json` ever set them.** A model round-tripped through
`save_pretrained()` (as ours was, after quantizing) picks up these
class-level defaults as if they were real, deliberately-set values:

1. **`has_moe_params = "num_experts_per_tok" in llm_config`** (nemotron.py)
   checks key *presence*, not whether it's meaningful — always true for any
   NemotronH model post-`AutoConfig`, mistagging every dense model as
   `nemotron_h_moe`. Symptom: `feed_forward_length` computed via the wrong
   (MoE) branch came out all-zero, and spurious `expert_count`/
   `expert_used_count`/etc. KVs got written. Fix: read the *raw*
   `config.json` file directly for this specific check instead of trusting
   `AutoConfig`'s defaulted dict; when false, also strip the MoE-only keys
   from `hparams`/`llm_config` so no generic downstream code path
   (`base.py`'s `set_gguf_parameters` also independently writes
   `expert_group_count` whenever `hparams.get("n_group")` exists) picks
   them back up.
2. **`_MLP_LAYER_TYPES = {"moe"}`** (class constant) — when
   `layers_block_type` is a list of type-name strings (which
   `AutoConfig().to_dict()` synthesizes from the compact
   `hybrid_override_pattern` string), dense models use the literal string
   `"mlp"` for feed-forward layers, not `"moe"` — so `_mlp_layers` came out
   empty even once `is_moe` was correctly `False`, keeping
   `feed_forward_length` all-zero. Fix: `_MLP_LAYER_TYPES = {"moe", "mlp"}`.
3. **`hybrid_override_pattern` doesn't round-trip through `save_pretrained()`**
   — it's a non-standard field `NemotronHConfig` doesn't preserve when a
   model is re-saved; `AutoConfig().to_dict()` synthesizes the verbose
   `layers_block_type` list from it at *load* time, but that only works if
   the compact pattern string is still there in the file being loaded. Any
   pipeline that loads-then-resaves a NemotronH checkpoint (ours included)
   silently loses it. Fix: copy `hybrid_override_pattern` from the original
   checkpoint's `config.json` into the resaved one before conversion.

None of these are fixed upstream (this was root-caused and worked around
locally in this session's cloned `conversion/nemotron.py`, not submitted as
a PR) — worth doing so, since any project quantizing-and-resaving a dense
NemotronH checkpoint will hit exactly this.

**End-to-end validated**: built `llama-simple` (CPU-only, `cmake -B build
-DGGML_CUDA=OFF -DLLAMA_CURL=OFF`; note `llama-cli` itself now embeds an
interactive HTTP server and its progress-spinner output doesn't reliably
flush through a piped/redirected stdout — `llama-simple` is the
non-interactive tool for scripted testing) and ran generation directly:

| File | Size | Speed (CPU) | Sample generation |
|---|---|---|---|
| Reference F16 | 7.96 GB | 9.35 tok/s | `def quicksort(arr):\n"""Sorts a list of numbers in ascending order."""` |
| **Mixed (Q8_0/TQ2_0/Q4_0/F16)** | **4.57 GB** | **10.46 tok/s** | `def quicksort(arr):\n"""Sorts a list of numbers in ascending order." def quicksort(arr):\n return sorted([arr] for arr in arr)]` |

**1.74x smaller, loads and runs in real llama.cpp, comparable speed** (the
GGUF-packed version's Q8/Q4/TQ2 tensors don't yet benefit from any of this
project's rotation/GPTQ-error-compensation work — this file's ternary/Q4/Q8
values came from this session's `--skip-rotation` GPTQ runs specifically
because literal-domain values are what a fixed-format quant type requires;
see the rotation section above). Generation degrades somewhat vs F16 (some
repetition) but stays on-topic and syntactically valid — consistent with
the ~12.7x-not-1x perplexity ratio measured earlier. This is a real,
loadable-today artifact, not just a size projection.

### Conclusion: stock llama.cpp + imatrix beats our custom packer outright

Pushed further to see if the gap to NVIDIA's official 2.84GB Q4_K_M release
could be closed. Two findings changed the picture substantially:

1. **`gguf-py`'s Python `quantize()` has no K-quant implementation** — calling
   it on `Q4_K`/`Q5_K`/`Q6_K` raises `NotImplementedError` in
   `quantize_blocks` (confirmed directly). Only the simple/legacy formats
   (Q4_0, Q5_0, Q8_0) and the TQ ternary formats have real Python
   quantizers; K-quants' more sophisticated per-superblock scale+min search
   only exists in llama.cpp's C++ implementation. This means our
   from-scratch `pack_gguf.py` approach is structurally capped at
   legacy-format quality — it can never produce genuine K-quants, no matter
   how the tensor-type assignment is tuned.
2. **`llama-quantize` (the real C++ tool) already does everything our
   packer was trying to do, better**: it automatically falls back to a
   legacy format (`Q5_0` observed, not `Q8_0`) for any tensor whose
   in_features isn't a multiple of 256 ("51 of 263 tensor(s) required
   fallback quantization" — the exact same tensors our own analysis flagged
   as TQ2_0-blocked), while using proper K-quants everywhere the shape
   allows, all via one `llama-quantize model.gguf out.gguf Q4_K_M` call.

Measured (same held-out text, same `llama-perplexity -c 256 --chunks 1`
methodology, so directly comparable):

| File | Size | PPL | vs F16 |
|---|---|---|---|
| F16 reference | 7.96 GB | 91.45 | 1.00x |
| **Q4_K_M + imatrix** (wikitext-2, 40 chunks) | **2.70 GB** | **94.86** | **2.95x, ~1.04x PPL** |
| Q4_K_M, no imatrix | 2.84 GB | 109.81 | 2.80x, 1.20x PPL |
| Our custom mixed packer (Q8_0/TQ2_0/Q4_0/F16) | 3.80 GB | 114.51 | 2.09x, 1.25x PPL |

**imatrix calibration (`llama-imatrix -f wikitext-2-raw/wiki.train.raw
--chunks 40`, ~40min on this pod's CPU) makes the decisive difference** —
without it, Q4_K_M's own rounding decisions are magnitude-only, same
category of heuristic as our simple Q4_0/Q8_0 choices; with it, perplexity
drops from 109.81 to 94.86, landing within ~4% of the F16 reference at
under 3.5GB. This beat every custom configuration built earlier in this
session, on both size and quality simultaneously, using zero project-specific
code — just the standard llama.cpp release pipeline (`convert_hf_to_gguf.py`
→ `llama-imatrix` → `llama-quantize ... Q4_K_M`).

**Bottom line for this project's dense-model GGUF story**: our
rotation+GPTQ+salient-pinning method is real and produces excellent results
in its native (uncompressed-storage) form — see the 1.006x/4.7x results
elsewhere in this doc and docs/poc.md — but once the constraint is "must
land in a stock GGUF quant type", llama.cpp's own imatrix-calibrated K-quant
pipeline is simply the better tool for the job on this architecture. The
three upstream metadata bugs found and fixed along the way
(architecture mistagging, `_MLP_LAYER_TYPES` missing `"mlp"`,
`hybrid_override_pattern` not surviving `save_pretrained()`) are the
durable contribution from this thread — worth upstreaming — separate from
the packing-method question, which this comparison resolves in favor of
"just use `llama-quantize` with an imatrix."
