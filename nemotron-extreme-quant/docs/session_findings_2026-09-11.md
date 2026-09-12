# Session Findings — 2026-09-11 (Stage B + GGUF export)

Consolidated summary of a very long session. Full detail lives in
`docs/stage_b_prep.md` (updated throughout); this file is the fast-scan
index of what happened and why it matters, written mid-session as
insurance against context loss.

## 1. Core bug fix: `torch.cholesky_inverse` was silently 30x slow

`poc/gptq.py`'s `_compute_hinv()` used `torch.cholesky_inverse(L)` (LAPACK
`potri`), which took **24+ seconds** for a 2688-dim Hessian on this
hardware's BLAS, *regardless of data conditioning*. Replaced with
`torch.cholesky_solve(eye, L)` — same math, ~30x faster (0.7s). This had
been inflating every GPTQ call in the whole project, not just Stage B.
**This fix invalidated the earlier "GPU is slower for GPTQ" conclusion** —
once fixed, GPU beat CPU 36x on a single-matrix microbenchmark (10.4s vs
378s).

## 2. Batched-across-experts GPTQ: ~5.5x speedup for MoE

Built `gptq_binary_batched`/`_gptq_run_batched` (gptq.py) and
`rot_gptq_salient_batched` (methods.py): stacks all same-shaped routed
experts into one `[E, out, in]` tensor, zero-pads calibration to a common
sample count (exact, not approximate), and does one Hessian/Cholesky/
column-loop over all E experts instead of E independent Python loops.
Verified numerically identical to per-expert path on well-conditioned data.

**Measured on real block 1 of the 30B-A3B MoE model, 112/128 eligible
experts**: per-expert loop 1565s vs batched 270-284s — collapse from
~128x-more Python-loop-launch-overhead was real but not the full 36x the
microbenchmark suggested (batched Hessian pushes GPU memory to ~80.5/82GB,
likely throttling cuSOLVER).

## 3. Stage B MoE quality: validated, strong result

Quantized real MoE block 1 (rotation+GPTQ+ternary→binary, salient=3%,
group_size=128, batched) and measured: **held-out perplexity 3.101 →
3.119 (1.006x)** — essentially no degradation, with 279 calibration tokens
spread across 128 experts (median 12 tokens/expert). This is the project's
best-method result and is NOT affected by anything below — it used a
completely different recipe (binary, group_size=128, no `--skip-rotation`)
than the GGUF-export experiments.

**Open question the user asked to verify next**: does this hold up under
an independent measurement (llama.cpp's own perplexity tool on a real GGUF
conversion of the 30B-A3B model), or could 1.006x be a measurement
artifact of the PyTorch eval harness? Not yet resolved — see "Next" below.

## 4. GGUF export investigation (dense 4B model)

Full attempt to package this project's method into a real, loadable GGUF
file. Three sequential surprises, each initially assumed to be the last
blocker:

### 4a. Rotation breaks the "literally ternary" requirement

`rot_gptq_salient`'s rotate-then-unrotate wrapper produces a *dense*
reconstruction in the stored weight (every entry a linear combination of
rotated-domain ternary values) — verified directly: a "ternary" tensor had
29 distinct values per 256-group, not 3. Fixed with a new
`--skip-rotation` flag on `quantize_sequential.py` (calls `gptq_binary`/
`gptq_ternary` directly). Verified exactly 3-valued after the fix. Cost:
skip-mamba+down_proj config moved from 9.35x to 12.43x perplexity ratio —
real but survivable.

### 4b. `hidden_size=3136` blocks TQ2_0/K-quants for most tensors

TQ2_0, TQ1_0, and all K-quants (Q4_K/Q5_K/Q6_K) require the block axis to
be divisible by 256 — hard-coded, no padding fallback
(`gguf-py`'s `can_quantize()`: `shape[-1] % block_size == 0`). This
model's hidden_size (3136 = 64×49) is **not** divisible by 256 — it's
divisible by 32 (legacy Q4_0/Q5_0/Q8_0 formats only). Only tensors whose
in_features is a *different* dimension (attention.o_proj=5120,
mamba.out_proj=7680, mlp.down_proj=12544 — all divisible by 256) are
K-quant/TQ-eligible. This is almost certainly why NVIDIA's own official
GGUF release uses Q4_K_M (mixed K-quant + legacy fallback) rather than a
pure ternary format — **the model's hidden_size was never chosen with GGUF
block-size alignment in mind** (that constraint postdates most model
architectures by design; it's a llama.cpp implementation detail, not a
property the model was designed against). Padding hidden_size to 3328
would unlock it but breaks RMSNorm statistics (its denominator would
include the padding elements) without also patching the norm kernel —
assessed as not worth the engineering risk for a ~0.23GB gain.

### 4c. Three real upstream llama.cpp bugs (found, patched locally)

All from one root cause: `AutoConfig.from_pretrained(dir).to_dict()`
always includes every field `NemotronHConfig`'s Python class defines
(including MoE-only fields), regardless of whether the checkpoint's raw
`config.json` set them. A checkpoint round-tripped through
`save_pretrained()` (as ours was, post-quantization) picks these up as if
deliberately set:

1. `has_moe_params = "num_experts_per_tok" in llm_config` (nemotron.py) —
   checks key *presence* via `AutoConfig`, not meaningfulness → always
   true → mistags every dense NemotronH model as `nemotron_h_moe`. Fixed:
   read the *raw* `config.json` for this specific check; strip MoE-only
   keys from `hparams` when false (a *second* leak existed: `base.py`'s
   generic `set_gguf_parameters` independently writes
   `expert_group_count` whenever `hparams.get("n_group")` exists).
2. `_MLP_LAYER_TYPES = {"moe"}` (class constant) — dense models' verbose
   `layers_block_type` list (which `AutoConfig` synthesizes from the
   compact `hybrid_override_pattern` string) uses the literal string
   `"mlp"`, not `"moe"`, for feed-forward layers → `_mlp_layers` came out
   empty even after fixing #1 → `feed_forward_length` computed as
   all-zero. Fixed: `_MLP_LAYER_TYPES = {"moe", "mlp"}`.
3. `hybrid_override_pattern` doesn't survive `save_pretrained()` —
   `AutoConfig().to_dict()` only synthesizes the per-layer type list from
   it at *load* time; a model resaved without the original file present
   loses it silently. Fixed: copy `hybrid_override_pattern` from the
   original checkpoint's `config.json` into the resaved one before
   conversion.

Patch saved at `docs/patches/llamacpp_nemotron_dense_moe_mistag_fix.diff`
— worth upstreaming, affects anyone quantizing-and-resaving a dense
NemotronH checkpoint. Symptom without the fix: `llama-cli`/`llama-simple`
fails to load with `check_tensor_dims: tensor 'blk.1.ssm_in.weight' not
found` (or similar) even though `convert_hf_to_gguf.py` reports success —
**conversion succeeding is not proof the file loads**.

### 4d. `gguf-py`'s Python `quantize()` has no K-quant implementation

Confirmed directly: calling `quantize(data, Q4_K)` raises
`NotImplementedError` in `quantize_blocks`. Only TQ formats and legacy
formats (Q4_0/Q5_0/Q8_0) have real Python quantizers in `gguf-py` — K-quant
math (per-superblock scale+min search) only exists in llama.cpp's C++.
**This caps any from-scratch Python packer (`poc/pack_gguf.py`) at
legacy-format quality — it cannot produce genuine K-quants no matter how
tensor-type assignment is tuned.** Use `llama-quantize` (the real binary)
for K-quants; it has a `--tensor-type name=type` flag for per-tensor
overrides if a custom mix is still wanted on top of its own recipe.

### 4e. Final verdict: stock `llama-quantize` + imatrix wins outright

Built `poc/pack_gguf.py` (uses `gguf.GGUFReader`/`GGUFWriter`,
`gguf.quants.quantize()` — no llama.cpp rebuild needed for packing itself)
and iterated through several hand-picked tensor-type mixes. Then compared
against the stock tool. Same held-out text, same `llama-perplexity -c 256
--chunks 1` methodology throughout:

| File | Size | PPL | vs F16 |
|---|---|---|---|
| F16 reference | 7.96 GB | 91.45 | 1.00x |
| **`llama-quantize ... Q4_K_M` + imatrix** (wikitext-2, 40 chunks, ~40min CPU) | **2.70 GB** | **94.86** | **2.95x, ~1.04x PPL** |
| `llama-quantize ... Q4_K_M`, no imatrix | 2.84 GB | 109.81 | 2.80x |
| Our best custom `pack_gguf.py` mix (Q8_0 majority + TQ2_0 + Q4_0 fallback) | 3.80 GB | 114.51 | 2.09x |

**imatrix calibration is what actually matters** — it's the difference
between 109.81 and 94.86 (landing within ~4% of F16), and it applies
uniformly via one flag on the standard tool. Every hand-tuned custom
config built this session (mamba-Q8 ablations, Q4_K-for-256-divisible
retries, etc.) lost to "just run the standard pipeline with an imatrix."

**Practical conclusion for this project's dense-model GGUF story**: the
rotation+GPTQ+salient-pinning method is real and excellent in its native
(uncompressed-storage, PyTorch-side) form — see §3 above and
`docs/poc.md` — but once the constraint is "must be a stock GGUF quant
type," llama.cpp's own imatrix-calibrated K-quant pipeline is simply
better-engineered for this than a from-scratch packer can be (no K-quant
math in Python, legacy-fallback logic already handles the 3136-not-256
constraint, imatrix calibration already solved). The bug fixes (§4c) are
the durable, reusable contribution from this thread.

## 4f. Separate bug found: `llama-cli` interactive/chat mode hangs for this architecture

Not a quantization issue — a distinct upstream bug. `llama-simple` (raw
completion, no chat template) works fine and was used for all the
size/quality comparisons above. But `llama-cli`'s interactive/chat mode
(what LM Studio's chat UI drives under the hood) **hangs indefinitely**
when given any input (tested with `hi`) on this NemotronH hybrid
Mamba+Attention model — confirmed by launching it both piped
(`echo hi | llama-cli ...`) and interactively, in both cases with zero
token-generation progress after 120+ seconds, and critically: **the
process ignored `timeout 120` entirely** (CPU time stayed flat, i.e.
genuinely stuck, not just slow) and had to be `kill -9`'d manually.

This is the direct explanation for the user's real-world symptom: LM
Studio's chat window either produced garbled/irrelevant text (consistent
with LM Studio falling back to raw-completion-style behavior when its own
chat-template handling doesn't engage properly) or froze. **The GGUF file
itself is fine** (llama-simple output was fine, and the model loaded with
correct metadata after the §4c fixes) — this is specifically about
llama.cpp's conversation/chat-loop code path interacting badly with this
architecture's Mamba recurrent state handling, most likely something in
how the hybrid-memory (attention KV-cache + Mamba recurrent-state) module
manages state across the chat loop's multi-call structure, as opposed to
`llama-simple`'s single straight-through generation call. Not
investigated further this session (out of scope, pod time/cost focused on
the size/quality question) — worth a follow-up if this project moves
toward recommending llama.cpp/LM Studio as a serving path for NemotronH
users generally, since it would block ordinary chat use even with a
perfectly good GGUF.

## 5. Where things are, mid-verification

- Final dense-model artifact (`nano4b-q4km-imatrix.gguf`, 2.7GB, PPL 94.86)
  uploading to a private HF repo (`roman220220/nemotron-nano-4b-q4km-imatrix-gguf`)
  for local download and real LM Studio testing.
- **In progress**: reproduce the same pipeline (convert → imatrix →
  Q4_K_M → llama-perplexity) against the *real* 30B-A3B MoE model
  (`/root/lightning30b` on the pod), now that this session's llama.cpp
  clone is recent enough to plausibly include the MoE support bartowski's
  own 30B-A3B GGUF release depends on (their README: "made with llama.cpp
  release b10362 — if this model's architecture is newly supported, you'll
  need that release or newer"). Goal: get an independent, llama.cpp-side
  perplexity number for the MoE case to cross-check the PyTorch-measured
  1.006x from §3 — not because there's a specific reason to doubt it, but
  because it's the project's best result and deserves a second, differently
  -implemented measurement before being trusted as the headline number.
- Pod (`nemotron-stageb-moe`, RunPod A100 PCIe, $1.59/hr) has been running
  ~6h50min (~$10.75) as of this writing. User has explicitly authorized
  continuing to spend to get this verification done properly.
- Disk cleaned twice this session (ablation checkpoints, then dense-GGUF
  intermediates) to make room for the 30B conversion, which needs ~60GB
  for an F16 intermediate alone.

## 7. Full-model 30B-A3B GPTQ quantization + MLX exploration (second half of session)

Picking up after §5: the llama.cpp-side verification against the real 30B
MoE model succeeded (F16 vs Q4_K_M+imatrix on wikitext-2, 20 chunks, both
statistically indistinguishable — ratio 0.994, well within combined error
bars). This confirmed the architecture tolerates *stock* quantization well.
The user then asked for a much smaller (~14GB) build for a 24GB-RAM MacBook,
which led to the following, in order:

### 7a. GGUF has a hard floor around ~18.8GB for this architecture

Checked bartowski's own `NVIDIA-Nemotron-3.5-Lightning-30B-A3B-GGUF` — even
his most aggressive quant (`IQ2_XXS`, ~2 bits) is **18.84GB**, not
meaningfully smaller than `IQ4_XS` (18.92GB). Confirms §4's finding
generalizes: `hidden_size=2688` not divisible by 256 forces most tensors
into legacy-format fallbacks regardless of the requested K-quant/I-quant
level, creating a floor GGUF can't quantize below. To get materially
smaller, needed a format without GGUF's 256-alignment requirement — MLX's
affine quantization has no such constraint.

### 7b. MLX supports this architecture natively; runs on Linux+CUDA too

`mlx-lm` has a real (non-stub) `nemotron_h.py` implementation including MoE
and Mamba2. Less obviously: **MLX now has a real CUDA backend** (`pip
install "mlx[cuda12]"`), not just Metal — confirmed working on the A100 pod
(`mx.default_device()` → `Device(gpu, 0)`), including quantized matmul
support (added in mlx 0.31.0+). This meant the whole MLX pipeline — convert,
quantize, and eventually any custom inference code — could run on the
rental GPU instead of requiring the user's own Mac, sidestepping their
24GB-RAM constraint for the *conversion* step (only the final artifact needs
to fit on their machine).

Stock `mlx_lm.convert -q --q-bits 3` on the original BF16 checkpoint: 3.504
bits/weight, **13.8GB**, ~4 min on the A100. Sanity-check generation was
coherent ("capital of France" continued correctly) but a later C++ example
in LM Studio showed clear structural breakdown (mismatched class/variable
names) — expected, since this is plain RTN (no calibration, no GPTQ)
applied uniformly including to the routed experts, which are ~98% of this
model's parameters.

**Important deployment constraint discovered**: LM Studio's MLX engine
(`mlx-engine`) vendors its own isolated `mlx-lm` via `venvstacks` and never
passes `trust_remote_code=True` to `load_model()` — so it **cannot load a
custom/non-standard MLX architecture**, confirmed by multiple open
lmstudio-ai/mlx-engine issues (#29, #172, #607). A custom quantized-linear
layer (needed for rotation+salient, see below) would only be usable via
`mlx_lm.server --trust-remote-code` from the terminal, not the LM Studio
app directly. This capped the ambition for a "drop-in LM Studio" artifact
to whatever fits mlx-lm's *stock* model classes.

### 7c. Honest full-model custom GPTQ quantization: three real bugs found and fixed

The user explicitly wanted this project's own research (GPTQ + ternary +
salient), not just a stock quantizer, applied to the *entire* 30B model —
previously this had only been done for a single MoE block (§3's 1.006x
result). Building `poc/quantize_full_moe_model.py` (new) to do this
end-to-end surfaced three real infrastructure bugs, each initially
mistaken for "this is just going to be slow":

1. **Mamba's `causal_conv1d` kernel has no CPU fallback in this
   environment.** `transformers`' `use_kernel_func_from_hub_with_fallback`
   decorator always prefers an *installed* package over the pure-PyTorch
   reference implementation, with **no runtime device check** — so loading
   the model with `device_map="cpu"` doesn't make it CPU-only, it makes it
   crash (`RuntimeError: Expected x.is_cuda() to be true`) the moment a
   Mamba block's forward pass runs. The model **must** stay resident on
   GPU for every forward pass; there is no way around this short of
   uninstalling `causal_conv1d` process-wide (which would also kill the
   fast path for legitimate GPU runs).

2. **Full model (65GB bf16) + batched MoE Hessian on the same GPU OOMs.**
   With the model fully resident on an 80GB A100 (~77GB used), only ~2-3GB
   was free — not enough for a batched-Hessian call across 100+ experts at
   once (`Tried to allocate 3.45 GiB... 2.30 GiB is free`). Fixed by (a)
   sub-batching experts (`--moe-subbatch`, default 12) instead of batching
   all valid experts in one call, and (b) switching calibration capture
   from **sequential** (one truncated forward pass per block, à la
   `quantize_sequential.py`) to **one-shot** (single forward pass per
   calibration chunk, all target modules hooked simultaneously against the
   pristine model) — sequential's repeated forward passes don't fit this
   model's memory profile as cheaply.

3. **The real showstopper: PyTorch's default CPU thread pool (252 threads)
   makes the non-batched dense-quantize path catastrophically slow.** Mamba/
   attention/shared-expert use a single (non-batched) `gptq_nbit` call with
   a ~2700-iteration Python column loop. On CPU with the default 252-thread
   pool, each tiny per-column op pays full thread-pool sync overhead —
   measured **7+ CPU-minutes and still running** for one mamba block that
   should take ~20s. Fixed with one line, `torch.set_num_threads(16)`,
   called once globally — after which the same block took **4.2s**. This
   is the single highest-leverage fix of this half of the session; without
   it, the "~2 hour" budget the user approved would have been wildly
   insufficient (extrapolated: hours, not minutes).

After all three fixes: full 52-block one-shot quantization (23 moe + 23
mamba + 6 attention) completed in **610s (~10 min)**, with **128/128
experts hit on every single MoE block** (vs the original single-block
test's 112/128 with the tiny 10-prompt calibration set — this run used 24
wikitext-2 chunks x 512 tokens = 12,288 total tokens instead).

### 7d. Plain ternary (no rotation, no salient) on all 23 MoE blocks: complete failure

First full-model attempt used `gptq_ternary_batched` directly (no
protection) for routed experts, matching what seemed like the pragmatic
"minimal functionality" choice given time pressure. Generation was
**garbage** — not merely degraded, actually broken: `"The capital of
France is 100,000,000,000,000,000,000,000,000,000"`. This is a much more
severe failure than any single-block ablation predicted, and initially
looked like it might be an implementation bug. The user's instinct from
early in the session ("без салиента — жопа по качеству") was right: rather
than debug further, went straight to adding rotation + salient protection
back in — this is the empirical confirmation that salient pinning is not
optional polish, it's load-bearing, and this generalizes from the single
MoE block tested in §3 to the *whole* model.

### 7e. Rotation + salient (no unrotate) added; fixes coherence, but doesn't fully close the gap

Added `rot_salient_ternary_batched_packable()` (`poc/methods.py`, new) —
same rotate → GPTQ-ternary → salient-pin recipe as the existing
`rot_gptq_salient_batched`, but returns the **rotated-domain** result and
the salient mask explicitly, instead of calling `unrotate()` internally.
This is necessary groundwork for eventually packing into a real low-bit
MLX format (a real packer needs literal {-scale, 0, +scale} values in a
known basis, not `unrotate()`'s dense linear-combination reconstruction —
see `poc/rotation.py`'s docstring). For this round, `unrotate()` is still
called on the *caller* side so the resulting checkpoint is immediately
testable via a normal HF forward pass.

Ported the Hadamard rotation to MLX (`poc/rotation_mlx.py` +
`poc/test_rotation_mlx.py`, new) and validated it matches the PyTorch
version bit-for-bit (max diff ~4.8e-7, floating-point noise only) — this
will be needed for the eventual custom MLX inference layer regardless of
which packing approach wins.

Result with rotation+salient (`salient_fraction=0.03`, same as §3),
applied to the full 52-block one-shot run:
- **Small-scale sanity check (blocks 0-3 only)**: generation fully
  coherent ("The capital of France is Paris. / The capital of Germany is
  Berlin. / ..."). Confirms rotation+salient genuinely fixes the complete
  breakdown from §7d, not just improves it slightly.
- **Full 52-block run**: generation is a mix — simple factual recall
  survives, but structured/code prompts still degrade (`def fibonacci(n):`
  continued with `1000000000000...`) and even the good example shows
  excessive greedy-decoding repetition loops.
- **Honest perplexity (wikitext-2 test split, 20 chunks x 256 tokens, HF
  pipeline, both models loaded via the same script for a fair comparison)**:
  F16 = **12.62**, quantized = **39.97** → **ratio 3.17x**. Real,
  measurable degradation — not catastrophic, but far short of §3's 1.006x
  single-block result.

**Hypothesis tested and REJECTED**: suspected one-shot calibration
(§7c.2's fix) — which captures every block's activations against the
*pristine* model, so no block's GPTQ ever accounts for upstream blocks'
quantization error — explained the gap from §3's 1.006x single-block
result. Built `--sequential` mode (`capture_single_block_activations`,
early-exit-per-block, block i's activations captured with blocks 0..i-1
already quantized in place, exactly matching `quantize_sequential.py`'s
approach) and re-ran the full 52-block quantization this way. Took 2094s
(~35 min, roughly 1.5x one-shot's 610s — matched the pre-run estimate of
"roughly double, not hours").

**Result: sequential PPL = 39.29 vs one-shot's 39.97** (F16 = 12.62) —
ratio 3.11x vs 3.17x. **A ~1.7% difference, essentially noise.** Sequential
calibration does NOT meaningfully close the gap to §3's 1.006x. The
one-shot-vs-sequential hypothesis is rejected; whatever separates the
full-52-block result from the single-block result is not primarily a
calibration-ordering effect. Plausible remaining explanations, none yet
tested: (a) cumulative effect of 23 independently-lossy ternary MoE blocks
compounds regardless of calibration order — i.e. this is a "many lossy
layers stacked" ceiling, not a calibration artifact; (b) `group_size=64`/
`salient_fraction=0.03` (chosen for MLX-packing convenience and matching
§3's recipe respectively) aren't the right hyperparameters at full-model
scale; (c) §3's single-block 1.006x was itself a favorable/unrepresentative
sample (one specific block, not necessarily typical of all 23). Reported
this directly rather than the originally-hoped-for "sequential fixes it"
result — worth flagging since it changes the shape of what's left to try
(hyperparameter sweep or accepting ~3x as this method's real ceiling on
this architecture, rather than "just fix the calibration order").

### 7f. Practical notes for reproducing any of the above

- **`torch.set_num_threads(16)`** (or similar) is now load-bearing for any
  script combining GPU batched work with CPU-based non-batched
  `gptq_nbit`/`gptq_binary`/`gptq_ternary` calls on a many-core host — this
  wasn't needed on smaller/earlier hardware in this project and is easy to
  miss (looks like "the GPU path is slow", not "the CPU path is
  catastrophically slow").
- **Disk discipline stayed tight all session**: repeatedly hit ~11GB free
  (94% used) with a 62GB source + 59GB output + assorted intermediates on a
  150GB volume; `df -h` occasionally read stale by 50-70GB for a few
  seconds after a large `rm -rf` (overlay filesystem lag) — always `sync`
  and re-check before trusting a low-free-space reading enough to act on
  it.
- **`pkill -f <pattern>` self-matches** when the pattern text is also
  present in the invoking shell command itself (e.g. `ssh host "pkill -f
  quantize_full_moe_model; echo done"` kills the SSH session's own shell,
  since that string is literally present in its argv) — this killed
  multiple debugging sessions this half of the session. Use `ps aux | grep
  <pattern> | grep -v grep` to find the real PID and `kill <pid>` directly,
  or the classic `pkill -f '[q]uantize_full_moe_model'` bracket trick.
- **B2 (Backblaze) credentials** exist for this project at
  `~/.config/b2/nemotron-quant.env` per earlier setup — not yet wired up to
  actually move an artifact there this session.

### 7g. `salient_fraction` is the dominant lever, not calibration order

Directly tested the user's own hypothesis after 7e's sequential-calibration
rejection: raise `salient_fraction` from 0.03 to 0.10 (one-shot calibration,
otherwise identical setup). Result: **PPL = 19.54, ratio 1.55x** (vs F16
12.62) — down from 3.17x at 0.03. This is the single biggest quality lever
found this session, confirming (b) from 7e's list of remaining explanations
over (a)/(c): the recipe's quality ceiling is much more sensitive to how
many weights get full-precision protection than to calibration ordering.

### 7h. Direct-to-MLX packing: naive salient storage would have been bigger than GGUF

With one-shot vs sequential settled (7e/7g) and `salient_fraction` identified
as the key knob, moved to actually packing the ternary+rotation+salient
result into MLX's on-disk format (`poc/pack_mlx.py`, new) instead of a dense
HF checkpoint — the whole point being a real low-bit artifact, not just a
perplexity number. Building this surfaced one more size-critical finding
before any full run: storing salient positions as explicit `(row, col,
value)` tuples — the obvious approach, and what `poc/rotated_switch_linear.py`
(new: a `SwitchLinear`-compatible module using `mx.gather_qmm` directly on
hand-packed ternary codes, since our {-scale,0,+scale} 3-level convention
maps exactly onto MLX's 2-bit affine layout with one unused code) originally
assumed — costs `int32 + int32 + float16 = 10 bytes/entry`. At
`salient_fraction=0.10` and this model's shapes (~5M weights/expert × 128
experts × 23 MoE blocks × 2 projections), that alone projected to **~37GB**,
bigger than plain GGUF. Caught via arithmetic before running the packer,
not after. Redesigned to a **packed bitmap** (1 bit/weight, fixed cost
regardless of fraction) + a values-only array (fraction × 2 bytes/weight) —
`poc/pack_mlx.py`'s `extract_salient()` — cutting the projected total to
~19-21GB. The bitmap is unpacked into explicit `(row, col)` arrays once, at
model *load* time (`poc/sparse_salient_mlx.py`'s `decode_salient_bitmap`),
not per forward pass — cheap on disk, but see 7k for why this reappeared as
a *runtime memory* problem instead.

Also built, alongside the packer: a genuinely novel **Metal scatter-add
kernel** (`poc/sparse_salient_mlx.py`'s `salient_correction`, using
`mx.fast.metal_kernel(..., atomic_outputs=True)`) for the sparse salient
correction — MLX has no sparse tensor type and no scatter-add in its
standard ops, so this is the same problem SpQR (Dettmers et al.) solves with
hand-written CUDA; this is (as far as could be determined) the first MLX
port of that idea. Validated against a slow numpy reference including a
deliberate hash-collision case (forcing duplicate output indices) to confirm
atomic accumulation actually happens, not silent overwrite
(`poc/test_sparse_salient_mlx.py`).

Two more real bugs found and fixed while building the packer itself, both
before any full run (caught by a 4-block sanity test):
1. **k-count off-by-one**: `pack_mlx.py`'s own salient-count formula used
   `round()`, `poc/methods.py`'s `_select_salient_mask` used `int()`
   truncation — disagreed by 1 for some shapes, breaking the packer's
   exact-count assumption. Also found `_select_salient_mask` itself could
   over-select on ties (threshold-based mask, not top-k-indices-based) —
   fixed at the source to always return exactly `k` via `torch.topk(...)
   .indices` directly, which every caller in the project depends on.
2. **Catch-all copy-through re-inflated the packed model 5x**: the "copy
   everything else in this block, unquantized" loop matched MoE's raw HF
   parameter names (`experts.up_proj`/`down_proj`) as "not yet written"
   (since the packed keys are named `switch_mlp.fc1/fc2`, a different
   scheme) and copied the full unquantized `[128, out, in]` expert tensors
   on top of the already-packed ones. A 4-block test came out **7.8GB**
   instead of the expected ~1.5GB; fixed with an explicit skip-list for
   already-handled raw prefixes, verified 7.8GB → 3.0GB on the same test.

### 7i. Full pipeline built: real 8-bit packing for mamba/attention/shared-expert, custom MLX model file

Closed a "we did the compute but didn't bank it" gap: mamba/attention/
shared-expert projections were being GPTQ-quantized to 8 bits for the
*compute* (Hessian error compensation) but then stored back as plain fp16 —
no actual size benefit from that work. Fixed by exposing `gptq_nbit`'s
internal per-group scale (`return_scale=True` in `poc/gptq.py`) so
`pack_mlx.py`'s `pack_nbit_codes()` packs the *exact* scale GPTQ used
(recomputing it from the already-quantized output would silently under-scale
any group where no weight happened to hit the extremal code) into MLX's
real affine N-bit layout — confirmed correct only for `bits ∈ {2,4,8,16}`;
5/3/6-bit would need cross-word bit-packing this simple packer doesn't do
(a real, acknowledged limitation, not yet needed since 8-bit is safe).

To actually *load* the packed model, wrote `poc/mlx_model_ternary.py` (new):
a custom `mlx_lm` architecture file (wired up via `config.json`'s
`model_file` key + `trust_remote_code=True`, plus a `quantization` key so
`mlx_lm`'s generic `nn.quantize()` pass auto-converts every plain `nn.Linear`
with a matching `.scales` key — mamba/attention/shared-expert — to 8-bit,
while leaving the custom MoE class and the unquantized embeddings/lm_head
alone). Reuses `mlx_lm.models.nemotron_h`'s own
`NemotronHMamba2Mixer`/`NemotronHAttention`/`NemotronHMLP`/`MoEGate`
unchanged; only the routed-expert MoE class (`NemotronHMoETernary` +
`TernarySwitchMLP`, wrapping two `RotatedTernarySwitchLinear`s) is custom.
`pack_mlx.py` now also copies this file plus `rotated_switch_linear.py`/
`rotation_mlx.py`/`sparse_salient_mlx.py` into the output directory and
patches `config.json` accordingly, so the packed directory is
self-contained and loadable via `mlx_lm.load(path, trust_remote_code=True)`.

A full 52-block pack at `salient_fraction=0.10` completed in **2107s** (~35
min) and produced a **21GB** `model.safetensors` — bigger than the 17GB
stock `mlx-community` 4-bit MLX conversion of the same model, because the
salient overlay's own overhead (bitmap + values, ~2.6 bits/weight at 0.10)
pushes routed-expert weights to **~4.87 bits/weight effective**, worse than
plain 4-bit RTN on pure size (the bet is that GPTQ error-compensation +
salient protection buys back more quality than that costs — not yet
conclusively shown to be worth it, see 7l).

### 7j. Three real bugs found loading the packed model in `mlx_lm` (none in the quantization math — all in the loader/model-file glue)

First `mlx_lm.load(..., trust_remote_code=True)` attempt failed 3 times in a
row, each a genuine bug in the new glue code, not the underlying
quantization:

1. **Missing `gate.e_score_correction_bias`**: the source HF PyTorch model
   has no such parameter (confirmed via a direct `named_parameters()`
   dump), but `mlx_lm`'s stock `MoEGate` always allocates one and
   `load_weights(strict=True)` refuses to load without it. Fixed in
   `mlx_model_ternary.py`'s `sanitize()` by injecting `mx.zeros(...)` for
   any missing bias — a no-op addition to routing scores, so this doesn't
   change behavior, just satisfies the stock class's shape contract.
2. **`conv1d.weight` layout mismatch**: PyTorch's `Conv1d` weight is
   `[out_channels, 1, kernel_size]`; MLX's `nn.Conv1d` expects
   `[out_channels, kernel_size, 1]`. Stock `mlx_lm.models.nemotron_h`'s own
   `Model.sanitize()` handles this (`v.moveaxis(2, 1)`) but our custom
   `sanitize()` didn't replicate it, since it doesn't inherit from the stock
   `Model` class. Fixed by copying the same transform in.
3. **Salient-correction reshape broke on real multi-token/multi-expert
   shapes** (only found once actually running `generate`, not caught by the
   earlier synthetic unit test): `RotatedTernarySwitchLinear`'s salient path
   assumed `indices.shape[0]` was the token count, true only for the *first*
   projection (`fc1`/up-proj, where `x` really is one row per token and the
   kernel internally loops over all `top_k` experts per row). `fc2`/down-proj
   receives `x` that *already* has a distinct row per `(token, expert-slot)`
   pair — output of `fc1` for each of the `top_k` chosen experts
   individually — so each row there maps to exactly one expert, not
   `top_k` of them. Fixed by deriving row count from `x_rot`'s own size
   (`x_rot.size // padded_in`) rather than assuming a relationship to
   `indices`'s shape, and inferring whether `indices` supplies one entry per
   row (fc2-like) or its own trailing `top_k` axis on top of per-token rows
   (fc1-like) from size alone — since `n_rows * top_k_eff * output_dims`
   always equals the target output's size either way, reshaping the flat
   kernel output directly into that target shape is correct without
   tracking which case applies.

After all three fixes, the model loads and runs a full forward pass across
every layer without further shape errors — the quantization/packing math
itself was correct; every bug found was in adapting stock `mlx_lm`
conventions to the custom loader.

### 7k. The bitmap redesign (7h) solved on-disk size but created a *runtime memory* problem instead

Even after loading correctly, generation failed: `RuntimeError: [METAL]
Command buffer execution failed: Insufficient Memory`, with `mlx_lm`
reporting the model **requires 39.7GB**, against the user's 24GB Mac's
~18.2GB MLX-recommended ceiling (roughly 76% of physical RAM). Root cause:
`decode_salient_bitmap` (7h) is cheap *on disk* but decodes into **explicit,
permanently-resident** `(row, col, value)` arrays at load time — the
`int32 row + int32 col + float16 value` layout costs 10 bytes/entry, and at
`salient_fraction=0.10` this project's shapes have **~2.94 billion** total
salient entries across all 23 MoE blocks × 128 experts × 2 projections —
**~27.6GB** just for decoded salient indices/values, on top of the ~11.9GB
"everything else" (ternary base + 8-bit dense + embeddings). The compact
on-disk representation and the decoded in-memory representation have very
different size profiles, and only the on-disk one had been checked before
this point.

Quick partial fix: switched `row`/`col` from int32 to int16 (every value is
≤2688, comfortably within int16's range) — `sparse_salient_mlx.py`'s
`decode_salient_bitmap` and `rotated_switch_linear.py`'s placeholder buffers
both updated, `salient_correction()`'s forced `.astype(mx.int32)` upcast
removed (Metal's implicit short→int widening handles the kernel-local read
fine). Verified correctness against the existing kernel unit test (still
passes, sub-µs diffs) before re-testing. Result: **39.7GB → 28.5GB**
required — an 11.2GB reduction, matching the predicted savings from halving
just the row/col portion — but still **~10.3GB over** the ~18.2GB ceiling.

Back-of-envelope for what fits: base (non-salient) resident memory ≈
11.9GB, leaving only ~6.3GB of budget for decoded salient data under the
18.2GB ceiling — equivalent to `salient_fraction ≈ 0.036`, barely below the
0.03 setting that gave the much worse 3.17x PPL ratio in 7e. Scaling
`salient_fraction` linearly, even the 0.06 variant computed in 7l (whose
decoded cost would be ~0.6× the 0.10 figure, ≈10.6GB) would still land
around 22.5GB total — **still over budget**. This is an open problem, not
yet solved: either (a) a real on-the-fly bitmap-reading kernel (no
persistent decoded index arrays at all — requires a rank/select structure
over the bitmap, meaningfully more kernel complexity, not yet started), (b)
also shrinking the salient *values* array (e.g. int8 instead of float16,
~2.9GB further savings, not sufficient alone), or (c) accepting a
`salient_fraction` low enough to fit (~0.03-0.04) and its correspondingly
worse quality. Not yet resolved at time of writing.

### 7l. `salient_fraction=0.06` + sequential calibration: PPL result

Ran `quantize_full_moe_model.py --salient-fraction 0.06 --sequential` (dense
HF checkpoint, for perplexity testing — not yet packed to MLX) to get a
data point between 7e's 0.03 (3.17x) and 7g's 0.10 (1.55x), and to check
whether sequential calibration's near-null effect (7e) still holds at a
different `salient_fraction`. Quantization took 1936s (~32 min, consistent
with sequential's ~1.5x one-shot-time overhead from 7e). Perplexity (F16 vs
quantized, same wikitext-2-test methodology, 40 chunks × 512 tokens this
time for tighter error bars):
- F16 baseline: **PPL = 8.115** (mean nll 2.094 ± 0.057)
- salient=0.06, sequential: **PPL = 14.936** (mean nll 2.704 ± 0.077)
- **Ratio: 1.84x**

Fits the monotonic curve cleanly: 0.03→3.17x, 0.06→1.84x, 0.10→1.55x.
Confirms (again) that `salient_fraction` is doing essentially all the work;
sequential calibration's effect remains small at this operating point too
(consistent with 7e's rejection of the calibration-order hypothesis). Not
yet packed to MLX/tested for actual memory fit (see 7k — the arithmetic
suggests it likely still doesn't fit under the 18.2GB ceiling even at 0.06).

### 7m. The "real" fix from 7k: rank/select kernel reads the bitmap directly, no decode at all

Built the option (a) sketched in 7k: instead of decoding `salient_bitmap`
into explicit `(row, col)` arrays at load time, keep the bitmap resident
as-is and resolve "flat position of the j-th set bit" *inside the Metal
kernel* using a small rank/select index — a standard succinct-bitvector
technique (Jacobson's rank/select), not a novel structure; what's specific
here is applying it so the decoded-index memory blowup from 7k never
happens at all.

`poc/sparse_salient_mlx.py`'s new `build_rank_index()` (numpy, load-time)
computes a per-expert checkpoint array: `checkpoint[e, c]` = cumulative
popcount of `bitmap[e]` before chunk `c` (chunks of `chunk_words` 32-bit
words, default 32 words = 1024 bits). At query time (inside the new
`salient_correction_bitmap` kernel, `_kernel_bitmap`): binary-search the
checkpoint for the containing chunk (`O(log num_chunks)`), linearly scan at
most `chunk_words` words using Metal's builtin `popcount` to find the exact
word, then scan at most 32 bits within that word — all per-thread, no
shared state. `_popcount32` (numpy SWAR bit-twiddling) builds the index
without a Python loop.

**Sizing win**: the checkpoint index costs `numel/(chunk_words*32)` int32
entries — at `chunk_words=32` that's `numel/1024` words × 4 bytes =
`numel/256` bytes, roughly 1/8th the bitmap's own `numel/8` bytes, and the
bitmap itself (fixed cost, independent of `salient_fraction`, unlike the
old decoded row/col arrays) replaces ~12GB of decoded int16 row/col data at
`salient_fraction=0.10`. Measured end-to-end on the real packed model:
required memory for generation dropped **28.5GB → 20.9GB** — matching the
predicted ~21GB back-of-envelope from 7k almost exactly. Still ~2.7GB over
the Mac's ~18.2GB ceiling at 0.10, but the 0.06 pack (smaller `values`
array, same fixed bitmap+index cost) is expected to land close to fitting
and is being tested next.

**Three real bugs found and fixed while building this** (same "correct math,
wrong plumbing" pattern as 7j):
1. **Explicit `device`-address-space pointer redeclaration is a hard Metal
   compile error** when the input's actual auto-generated buffer address
   space isn't `device` (can be `constant` for some inputs, not controlled
   by the caller) — `device const int* cp = checkpoint + offset;` failed to
   compile with `cannot initialize a variable of type 'const device int
   *thread' with an lvalue of type 'const constant int32_t *thread'`.
   Fixed by never redeclaring a typed pointer at all — index the named
   kernel input directly (`checkpoint[cp_base + mid]`), exactly like the
   working non-bitmap kernel already did; this sidesteps the address-space
   question entirely since indexing doesn't require re-typing the pointer.
2. **`mx.fast.metal_kernel` reuses a stale compiled pipeline when the same
   kernel object is called twice with different template values in one
   process** — calling one `mx.fast.metal_kernel(name="X", ...)` object
   first with `chunk_words=32` then again with `chunk_words=7` produced
   silently wrong results (exactly 2x off in the worst case) on the *second*
   call, despite passing correct template arguments each time; either value
   used *alone* in a fresh process was exact to float32 noise. Root cause
   not fully diagnosed (name-based compiled-kernel caching in MLX/Metal is
   suspected, not confirmed) — worked around by giving the kernel a name
   that includes `chunk_words` and caching one kernel object per distinct
   value actually used (`_kernel_bitmap_cache`). Doesn't affect real usage
   (this project always uses one fixed `chunk_words` throughout a whole
   model), but is a real footgun worth remembering if `mx.fast.metal_kernel`
   is ever reused with varying template values elsewhere.
3. Neither of the above showed up in the *original* single-expert diagnostic
   kernel used to isolate them (which has no multi-expert offset arithmetic
   and no repeated differently-templated calls) — a reminder that a
   minimal repro needs to match the *specific* conditions of the failure
   (multi-expert indexing, repeated calls with varying template args), not
   just "the same kernel logic," or it won't reproduce the bug at all.

Wired the new path into `poc/rotated_switch_linear.py`'s
`RotatedTernarySwitchLinear` (stores `salient_bitmap`/`salient_checkpoint`/
`salient_val` instead of `salient_row`/`salient_col`/`salient_val`, calls
`salient_correction_bitmap`) and `poc/mlx_model_ternary.py`'s `sanitize()`
(builds `salient_checkpoint` via `build_rank_index` instead of decoding to
row/col, keeps the bitmap itself as a normal loaded weight rather than
popping it). Validated against the same numpy reference used for the
original row/col kernel, plus a dedicated bitmap-specific test in
`poc/test_sparse_salient_mlx.py` (`salient_correction_bitmap` vs
`reference()`, across `chunk_words` values including one that doesn't
evenly divide the bitmap).

**Measured end-to-end result** (real packed model, `salient_fraction=0.10`,
loaded via `mlx_lm.generate(..., trust_remote_code=True)` on the user's
Mac): required memory **28.5GB → 20.9GB**. Still ~2.7GB over the ~18.2GB
ceiling at 0.10; the `salient_fraction=0.06` pack (same fixed bitmap+index
cost, smaller `values` array) is expected to land close to fitting and was
being downloaded to test at the time of writing.

**Not yet done, a further lever if 0.10 still doesn't fit**: `salient_val`
is still stored as float16 (2 bytes/entry) — quantizing it to int8 with a
per-expert scale (negligible overhead, ~128 floats total) would roughly
halve that array's cost. At `salient_fraction=0.10` this is ~2.9GB of
further savings (the values array is ~5.87GB at that fraction, per 7k's
sizing), which back-of-envelope would bring the 0.10 pack to roughly
**~18GB** — right at the edge of the ceiling instead of 2.7GB over it.
Unlike the bitmap/rank-index change, this needs re-running the packer (a
~35 min pod job) plus a small kernel change (dequantize `int8 * scale`
on read instead of reading `T` directly), so it wasn't done speculatively —
worth doing only if the 0.06 pack turns out to need it too, or if 0.10's
better quality (1.55x vs 0.06's 1.84x ratio) is worth the extra engineering.

**Update**: the int8-values change (above) was in fact implemented and
shipped this session — see §7n — once it became clear the memory picture
wasn't done improving yet even after the bitmap/rank-index fix.

### 7n. int8 salient values, and a second, much more serious bug: the atomic kernel's own correctness

Implemented the int8 quantization sketched above (`poc/pack_mlx.py`'s
`extract_salient` now returns `(bitmap, value_int8, value_scale)` instead
of `(bitmap, value_fp16)`, one scale per expert) directly in the packer
this time (the user: "а еще лучше сразу в скрипт сборки" — "even better,
straight into the build script"), plus a **version-tolerant loader**:
`mlx_model_ternary.py`'s `sanitize()` now quantizes on the fly for any
older pack that has `salient_bitmap` but no `salient_val_scale` key, so one
loader handles both on-disk formats without needing to know which
`pack_mlx.py` version produced a given model directory (the user's
suggestion: "нельзя конвертер написать на загрузке?" — "can't we write a
converter at load time?").

While testing this on the actual downloaded `salient_fraction=0.06` pack
(not synthetic data), generation ran successfully end-to-end for the first
time — and then broke on the *second* forward pass: `"Hello"` followed by
a string of `<unk>` tokens. Traced with a manual layer-by-layer forward
pass (checking `mx.isnan` after every block) down to `NaN` first appearing
inside `MoEGate`'s routing scores, on **layer 3's** gate specifically
(layer 1's identical code path was clean) — ruling out anything in the
custom ternary/salient code, since `MoEGate` is stock, unmodified `mlx_lm`
code. `mx.disable_compile()` did not fix it, ruling out an `@mx.compile`
cache issue despite the symptom's surface resemblance to one.

The actual root cause, found via a synthetic probe kernel (a trivial
`atomic_fetch_add` into a single fixed index of a 100,000-element output,
called repeatedly): **`mx.fast.metal_kernel`'s `atomic_outputs=True` output
buffer is not guaranteed zero on a fresh call.** The probe's one written
cell kept *incrementing* across separate calls (1.0, 2.0, 3.0, ...) instead
of resetting — MLX evidently reuses the same physical buffer across
separate invocations of the same compiled kernel without re-zeroing it. Our
`salient_correction_bitmap` kernel (the one built in §7m) only ever
atomic-adds into the small fraction of an expert's output rows that are
actually salient, so on every call after the first, *every* row's value was
`this call's correct contribution + all previous calls' contributions`,
compounding without bound across a model's ~50+ forward passes during
generation until it produced `NaN`. This also retroactively explains an
earlier-logged, only-partially-understood finding (§7m item 2, "kernel
reuses a stale compiled pipeline when called with different template
values") — it was never specifically about `chunk_words` changing between
calls; it was this same buffer-reuse issue, and any two calls to the same
kernel object (same or different template) exhibited it.

First attempted a masking fix (multiply the kernel's raw output by a
precomputed "does this expert/row have >=1 salient bit" mask, zeroing rows
the kernel legitimately never touched) — this was insufficient and still
failed a repeated-identical-call regression test, because rows the kernel
*did* legitimately write also carry forward the previous call's value
(accumulation, not just leftover garbage in untouched cells).

**Real fix**: found `mx.array`'s official `.at[idx].add(values)` scatter-
add primitive (analogous to JAX's `.at[]`), which starts from a genuinely
fresh `mx.zeros(...)` every call and is confirmed correct including with
duplicate indices. Split the custom kernel into two pieces: (a) a
*non-atomic* Metal kernel that only resolves bitmap positions into
`(row, col)` arrays (`_make_kernel_resolve`) — safe because every thread
writes to a unique output cell, so there is no accumulation and therefore
no dependence on the buffer's prior contents — and (b) a plain MLX-level
gather (`x_rot[token_idx, col]`) + dequantize + `out.at[token_idx, slot_idx,
row].add(contribution)` for the actual scatter-add, starting from a fresh
zero array every call. `row`/`col` outputs use int16 (halves this
per-forward-pass *transient* cost; unlike the old fully-decoded arrays,
these are never persisted, only computed and freed within one
`salient_correction_bitmap` call). Added a repeated-call regression test to
`poc/test_sparse_salient_mlx.py` specifically to catch this class of bug in
the future (a single-call test cannot: the original buggy kernel passed a
single isolated call every time).

**End-to-end validated on the real 30B model** (`salient_fraction=0.06`):
generation runs stably for many steps with no NaN, using `mlx_lm.generate`
directly (not just a synthetic test). Peak memory came in at ~18.4-19.8GB
depending on exact settings — right at the edge of the Mac's default
~18.2GB MLX-recommended ceiling, occasionally triggering `[METAL] Command
buffer execution failed: Insufficient Memory` in `mlx_lm`'s CLI specifically
(slightly more overhead than a bare script) despite ~19GB of the machine's
24GB being free at the OS level. Root cause: macOS's own
`iogpu.wired_limit_mb` sysctl (a hard GPU-wired-memory cap, separate from
MLX's own soft `mx.set_memory_limit()`, which already defaults to a much
larger 1.5x headroom) was set below what the model now needs. Raised via
`sudo sysctl iogpu.wired_limit_mb=22000` (user-authorized, documented in
MLX's own `set_wired_limit` docstring as the correct lever) — after which
CLI generation via chat template completed without error, ~19.4-19.8GB
peak. Quality at `salient_fraction=0.06` on an instruction-style prompt was
rough (degenerate output on a chat-formatted prompt) — roughly consistent
with the already-measured 1.84x perplexity ratio at this operating point
(see §7l), a real quality cost, not a remaining correctness bug.

### 7o. The custom kernel's real throughput ceiling, and a pivot to a simpler, faster, likely-better-value recipe

With correctness fixed (§7n), tested the custom ternary+salient model in a
realistic setting: a coding-agent-style request through `mlx_lm.server`.
Two new problems surfaced immediately, both eventually traced to genuine
findings rather than one-off flakiness:

1. **A 601-token prompt spiked peak memory to ~36GB** even though the
   single-token case was fine — `salient_correction_bitmap`'s resolve/
   gather/scatter arrays are shaped `[n_tokens, top_k, k]`, and `k` (this
   project's per-expert salient count) is hundreds of thousands, so this
   scales directly with prompt length during prefill. Fixed by chunking
   the token dimension internally (`_TOKEN_CHUNK`, forcing `mx.eval()`
   per chunk to free transients before the next) — brought the same
   601-token prompt back to ~19.5GB.
2. **Generation throughput measured at ~7 tokens/sec**, independent of
   `_TOKEN_CHUNK` size (tested 8 and 32 — nearly identical tok/s, ruling
   out "too many small Python-loop iterations" as the bottleneck; the
   compute itself is the ceiling, not dispatch overhead). For comparison,
   measured on the *same hardware*, same architecture, different
   quantization:
   - Stock 3-bit MLX conversion (plain RTN, `mlx_lm.convert -q --q-bits
     3`, no custom kernel): **67 tok/s** via `mlx_lm generate`, 13GB on
     disk.
   - Same stock 3-bit model via LM Studio: **26.8 tok/s** (screenshot-
     confirmed by the user) — slower than raw `mlx_lm generate` but still
     ~4x this project's custom kernel.
   - This project's custom ternary+salient model (`salient_fraction=0.06`):
     **~7 tok/s**, confirmed independently via both `mlx_lm.server` and
     direct `mlx_lm generate` (ruling out the server as the cause for
     *this* model specifically — see point 3 below for a *separate*,
     real server-only bug found on the stock model).
   
   This is a genuine, current limitation of the resolve-then-gather-then-
   scatter design (§7n) — not a quick parameter fix. it hasn't been
   root-caused further (candidates: the `.at[].add()` scatter-add itself,
   the fancy-indexing gathers, or fundamental overhead of a per-token
   custom kernel dispatch at this k-scale) as of this writing.

3. **Separately, `mlx_lm.server` itself was unreliable for the *stock* 3-bit
   model too** — a request that completed instantly via `mlx_lm generate`
   hung/OOM'd via `mlx_lm.server` in this project's testing. Root cause not
   isolated (didn't block on it once the decision was made to prefer
   `mlx_lm.generate`/a stock model for the immediate goal) — worth
   revisiting if `mlx_lm.server` specifically is needed later (e.g. for an
   OpenAI-compatible endpoint a coding agent can point at).

**The honest size/quality math that triggered the pivot** (user's own
question: "3-бит модель размером 13 гиг, что мы делаем не так?"):
- Stock 3-bit RTN: 3 bits + per-group scale/bias overhead
  (`2×16/group_size=64` ≈ 0.5 bit) ≈ **3.5 bits/weight** →
  30B × 3.5/8 ≈ **13.1GB** (matches the observed 13GB exactly).
- This project's ternary+salient recipe at `salient_fraction=0.06`: 2-bit
  ternary base + ~0.5 bit group overhead + a **mandatory 1 bit/weight
  bitmap** (fixed cost regardless of `salient_fraction` — see §7h) + the
  salient values themselves (0.06 × 8-bit int8 ≈ 0.48 bit) ≈ **~4 bits/
  weight** — i.e. this recipe's own bitmap tax alone puts the *floor*
  above stock 3-bit's total cost, before counting the actual salient
  values. Measured accordingly bigger (17-20GB vs 13GB) *and* ~10x slower,
  with quality never directly compared against this specific baseline
  (only against F16, not against stock 3-bit RTN) until this session.

Given 3+ bits doesn't need incoherence processing to avoid catastrophic
breakdown the way 1-2 bit does (§7d's finding was specific to very low
bit-widths), the user proposed a pivot: **apply this project's real
research contribution — GPTQ Hessian-based error compensation — at a
plain N-bit width with NO rotation and NO salient overlay**, then pack
with *stock* `mlx_lm.convert`. This should give the exact same size and
speed as naive RTN (same MLX-native quantized-linear path, `mx.gather_qmm`,
no custom kernel or loader at all) while being calibration-aware instead
of blind-round-to-nearest — a strictly-more-honest quality/speed/size
trade-off than the ternary+salient path at 3+ bits, if it works as
expected. Not yet measured (pod was mid-restart, GPU unavailable, at time
of writing) — see §7p.

Built for this: `gptq_nbit_batched()` (`poc/gptq.py`, new) — the
batched-across-experts analogue of the already-existing single-expert
`gptq_nbit()`, needed since routed MoE experts are ~all of this model's
parameters and the single-expert path's Python column loop would launch
per-column kernels 128x (once per expert) instead of once (batched across
all 128 simultaneously) — same reasoning as the existing `gptq_ternary_
batched`/`gptq_binary_batched`. Also added `scale_mode`/`return_scale`
to the shared `_gptq_run_batched` helper (mirroring the single-expert
`_gptq_run`'s existing support for these), needed so the batched n-bit
path can report the *exact* per-group scale it used (required for a
downstream packer/converter to reproduce the same codes, not a
re-derived approximation — see §7i for why this specific correctness
detail mattered before).

`poc/gptq_stock_convert.py` (new): applies `gptq_nbit`/`gptq_nbit_batched`
uniformly to every linear layer (mamba in/out proj, attention q/k/v/o,
MoE routed experts, shared-expert) at a single CLI-specified `--bits`/
`--group-size` — no rotation, no salient mask, no hardcoded bit-width
(the user explicitly asked for this to be parametric, not another
hardcoded-constant script like earlier ones this session: "ты бы не
хардкодил эти параметры а делал универсально через параметры запуска").
Saves a standard HF checkpoint (weights already sitting exactly on the
target affine quantization grid) for a subsequent *stock*
`mlx_lm.convert -q --q-bits N --q-group-size N` to pack — deliberately
does not implement its own N-bit packing (avoids this session's earlier
found limitation that a naive packer only handles bits∈{2,4,8,16}
correctly, not 3/5/6 — see §7i's "5-bit doesn't pack cleanly" finding;
letting stock `mlx_lm.convert` do the packing sidesteps needing to solve
that at all).

`poc/run_pipeline.sh` (new) + `docs/RUNBOOK.md` (new): after a full night
of ad-hoc SSH one-liners, the user asked for real orchestration ("самое
время сделать нормальную оркестрацию... что бы эксперименты были
воспроизводимы") — a single parametrized script (bits/group-size/
calibration settings as flags, not constants; run names and output paths
derived from parameters so different configs never collide) that syncs
code to the pod, runs the GPTQ stage, runs stock `mlx_lm.convert`,
uploads to HF, and rsyncs the result to this Mac, plus a from-scratch
runbook covering both this new path and the existing custom-kernel path
(§7h-7n) side by side. The custom ternary+rotation+salient work is
explicitly NOT abandoned (user: "не теряем наши наработки по ядру!!!" /
"мы еще не сдались окончательно" — keep the kernel work, this is an
additional/comparison path, not a replacement) — `poc/pack_mlx.py` and
the custom Metal kernel remain in the repo and documented in the runbook
as "Path B".

## 6. Files touched this session (for reference)

- `poc/gptq.py` — cholesky fix, batched-across-experts functions, `gptq_nbit`
- `poc/methods.py` — `rot_gptq_salient_batched`
- `poc/quantize_sequential.py` — `--skip-quantize-for`, `--skip-rotation`,
  `--nbit-quantize-for`/`--nbit-bits`/`--nbit-group-size`, `--group-size`
- `poc/quantize_moe_block.py` — `--batched`, per-expert progress logging
- `poc/eval_quality.py` — `device_map` load fix (unrelated pre-existing bug)
- `poc/pack_gguf.py` — new: custom GGUF mixed-precision packer (superseded
  by the imatrix+Q4_K_M conclusion, kept as reference/fallback)
- `poc/analyze_salience.py` — new: per-tensor-type salience concentration
  analysis (used during the ablation phase, not referenced in final verdict)
- `docs/patches/llamacpp_nemotron_dense_moe_mistag_fix.diff` — new: the
  three-bug fix to llama.cpp's `conversion/nemotron.py`, worth upstreaming
- `docs/stage_b_prep.md` — full narrative, updated throughout this session
- `poc/experiments_log.csv` — every perplexity-ratio measurement from this
  session appended (dense-model ablations: nosalient/salient3/skipattn/
  skipmamba/skipmambadown/norot-final/mamba-q8/mamba-q4/q8-downternary/
  q8-all; MoE: lightning30b-moe-block1-batched-poc)
- `poc/quantize_full_moe_model.py` — new (§7c-7e): full-model GPTQ
  quantization driven by `layers_block_type`, one-shot and `--sequential`
  calibration modes, `--moe-subbatch` for OOM-safe batched expert GPTQ
- `poc/methods.py` — added `rot_salient_ternary_batched_packable` (§7e):
  rotate+GPTQ-ternary+salient without the final `unrotate()`, for future
  MLX packing
- `poc/rotation_mlx.py`, `poc/test_rotation_mlx.py` — new (§7e): MLX port
  of the Hadamard rotation, validated bit-identical to the PyTorch version
- `poc/ppl_wikitext.py` — new (§7e): direct HF/PyTorch perplexity on a
  wikitext-2 split, for comparing checkpoints without a GGUF round-trip
  (useful when disk is too tight for an extra ~60GB F16 GGUF conversion)
