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
