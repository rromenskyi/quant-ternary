# Findings — Quantization and Fine-Tuning of NemotronH Models

This document distills the technical findings from this project's research into
low-bit quantization and LoRA fine-tuning of NVIDIA's NemotronH model family
(the hybrid Mamba2+Attention+MoE "Nemotron-3/3.5" line, both the dense 4B
variant and the 30B-A3B mixture-of-experts variant), organized by topic rather
than by when things happened. It exists as a reference for anyone extending or
reusing this codebase: what was measured, what broke, and what the fix or
conclusion was.

A full chronological research log — exact commands, dates, pod costs, and
session-by-session narrative — is preserved at
[`docs/archive/session-log.md`](archive/session-log.md) for anyone who wants
raw provenance for any of the findings below.

`docs/archive/` also holds this project's **original, superseded plan**: a
pure binary/ternary quantization approach, documented in
`archive/original-plan-spec.md` and `archive/original-plan-roadmap.md`, with
its Stage A/B proof-of-concept validation results in
`archive/stage-a-b-poc-validation.md` and `archive/stage-b-prep-notes.md`.
That earlier track hit a real ceiling: sequential GPTQ calibration plus
salient-weight pinning got binary (1.125 bits/weight) quantization of the full
dense 4B model only to a **"partial" verdict** — 46.5x perplexity degradation
ratio at the full-model scale (`archive/stage-a-b-poc-validation.md`), good
enough to validate the core methodology but not good enough to ship. The
project then pivoted to the **GPTQ-affine mixed-precision approach** (3–8
bits/weight, component-type bit allocation) that this document covers, which
produced the models that actually shipped.

---

## 1. Quantization methodology

### 1.1 GPTQ vs. naive round-to-nearest (RTN): when calibration actually helps

The project's central quantization primitive is GPTQ (Hessian-based
sequential error compensation, implemented in `poc/gptq.py`): after quantizing
each column of a weight matrix, the resulting error is redistributed onto the
not-yet-quantized columns using calibration-activation second-moment
statistics, rather than rounding every column independently as naive RTN does.

Whether this actually beats RTN turned out to be **bit-width- and
domain-dependent**, not a universal win:

- **At 3-bit, uniform, whole-30B-A3B-model, wikitext-2 calibration/eval**:
  GPTQ (PPL 6.24) measurably beat naive RTN (PPL 6.54) at the same bit width
  and group size (`poc/gptq_stock_convert.py`, §1.4 below). This is the
  project's first *bit-exact-verified* confirmation that Hessian correction
  helps at this operating point.
- **At 8-bit, on a dense 4B model, for code-generation capability
  specifically**: naive RTN *matched or slightly beat* GPTQ, both by
  wikitext perplexity (11.3868 vs. 11.3914 — noise-level) and, more
  importantly, by an actual compile-and-run C++ test (RTN 4/9 passes vs.
  GPTQ's 2/9, unquantized baseline 6/9). See §2.3 for the full ablation and
  the working theory (GPTQ's calibration corpus contained no C++, so its
  error compensation had no signal to preserve C++-specific weight
  directions, and may have traded that capability away for extra fidelity on
  calibration-represented patterns).

**Takeaway**: GPTQ's advantage over RTN is real but is a property of "does
this bit-width/data regime give the Hessian correction something useful to
work with," not an unconditional law. Perplexity computed only against
calibration-similar text will not reveal a capability regression in a
divergent domain — see §2.3 and the general PPL-vs-capability caveat below.

### 1.2 The affine-quantization grid-matching bug (GPTQ output must land on MLX's actual grid)

The pipeline for the "plain N-bit, no rotation, no salient overlay" GPTQ path
is two independent quantization passes: `poc/gptq_stock_convert.py` runs GPTQ
and writes Hessian-corrected weights into a plain bf16 HF checkpoint, then
stock `mlx_lm.convert -q` re-derives its *own* per-group scale/bias from those
bf16 values and re-quantizes from scratch. If GPTQ's internal grid doesn't
match MLX's actual affine-quantization grid bit-for-bit, this second pass
silently discards all of GPTQ's error compensation and picks different codes
than the ones actually calibrated against.

Three wrong formulas were tried and measured, in order, before the real one
was found:

1. A symmetric-style formula (levels from `2^bits - 1`, no independent
   min/max) → PPL 27.68 (worse than naive RTN's 6.54).
2. Independent min/max, but with a bug in the step calculation (forgot to
   subtract the bias term) — caught by a synthetic unit test before spending
   a pod run on it.
3. Independent min/max with the step bug fixed — the `unique_token_ratio`
   sanity heuristic gave a **false-positive PASS** (gibberish text has high
   lexical diversity too — a real, unfixed limitation of that heuristic).
   Real PPL: 6,074,737. A further-refined version of this formula gave PPL
   17.82 (coherent, but still worse than RTN's 6.54).

None of these matched MLX's real behavior, which is neither simple
symmetric-absmax nor simple independent-min/max but a specific hybrid — found
by reading `mlx/backend/metal/kernels/quantized.h`'s `affine_quantize`
function directly:

```
n_bins = 2^bits - 1
scale = max((w_max - w_min) / n_bins, eps=1e-7)
side = |w_min| > |w_max|            # whichever extreme has larger magnitude
scale = side ? scale : -scale        # sign follows the dominant extreme
edge = side ? w_min : w_max
q0 = round(edge / scale)
if q0 != 0: scale = edge / q0         # rescale so `edge` lands exactly on an integer code
bias = (q0 == 0) ? 0 : edge
code = clamp(round((w - bias) / scale), 0, n_bins)
dequant = code * scale + bias
```

Implemented as `_affine_scale_bias()` in `poc/gptq.py`, and verified **100%
bit-exact** against real `mx.quantize(mode="affine")` across 5 random test
cases (varied shapes/seeds/scales; max diff ~1e-6, float32 noise floor) —
verified correct *before* committing to another full pod run, not after.

With this fix, uniform 3-bit GPTQ (group size 64, one-shot calibration, full
52-block 30B-A3B model) measured PPL 6.24 against bf16's 5.11 reference and
naive RTN's 6.54 — see the table in §1.4.

### 1.3 Component-type bit allocation (the "JANG" recipe) beats layer-position heuristics

A third-party mixed-precision release (`JANG_2L-CRACK`, MLX format) achieved
PPL 5.43 at a claimed ~3.73 bits/weight — better than either this project's
uniform-3-bit GPTQ (6.24) or its first mixed-precision attempts. Its
`config.json` embeds a fully-inspectable `quantization` dict (readable via a
single-file `hf_hub_download`, no full download needed), which revealed its
strategy is **not** layer-position-based (unlike llama.cpp's Q4_K_M-style
heuristics) — it assigns bits purely by **component type**, uniformly across
every layer:

| component | bits | rough share of total params |
|---|---|---|
| attention q/k/v/o_proj | 8 | small (6/52 layers) |
| mamba in_proj/out_proj | 6 | moderate (23/52 layers) |
| MoE shared_experts (up+down) | 8 | small (1 expert/block, not 128) |
| MoE routed `switch_mlp.fc1` (up-equivalent) | 4 | huge (128 experts × 23 blocks) |
| MoE routed `switch_mlp.fc2` (down-equivalent) | 3 | huge (128 experts × 23 blocks) |
| embeddings | 6 | — |
| lm_head | 8 | — |
| `mtp.layers.0.eh_proj` (unused draft head) | 2 | negligible |

The strategy: give generous precision (8-bit) to whatever is *cheap in total
parameter count* (attention, shared experts), moderate precision (6-bit) to a
moderate-sized component (mamba), and reserve the lowest precision (3–4 bit)
for the single largest pool of parameters (routed experts, ~93% of the
model) — with an asymmetric up(4)/down(3) split within routed experts that
this project's own recipes hadn't considered. JANG's README describes it only
as "3.73-bit affine (MLX)" with no mention of GPTQ/Hessian/calibration —
strong circumstantial evidence it is naive RTN with a hand-tuned
bit-allocation map, not activation-aware quantization.

This project's own layer-position-based ("positional") and Hessian-saliency-
based ("sensitivity") per-layer recipes were both outperformed once the
JANG component-type map was reverse-engineered and applied through this
project's own GPTQ pipeline (see §1.4's table) — a well-tuned positional or
component heuristic proved hard to beat with a simple per-layer local-saliency
metric.

Implemented as the `jang` component recipe (MoE models) and `jang-dense`
(dense models, no MoE speed/size tension since nothing is sparsely activated)
in `poc/gptq_stock_convert.py` and `poc/mlx_convert_recipe.py`
(`COMPONENT_BIT_RECIPES`).

### 1.4 The `switch_mlp.fc1`/`fc2` naming bug: two mixed-precision releases silently never upgraded the routed experts

After running both the positional (`mixed_3_6`) and sensitivity-based
mixed-precision recipes, inspecting the *actual packed tensor shapes* on
HuggingFace (via an HTTP range request against the safetensors header — no
full download needed) revealed that **neither release's "6-bit upgrade" had
ever touched the routed MoE experts**. This architecture's MLX port names
routed-expert weights `switch_mlp.fc1`/`switch_mlp.fc2`, not `up_proj`/
`down_proj` — and both stock `mlx_lm.convert --quant-predicate` and this
project's own layer-selection script matched paths against the literal
substring `"down_proj"`, which `switch_mlp.fc2` never contains. Since routed
experts are ~99% of a MoE block's parameters, the "6-bit" recipes were only
ever upgrading `v_proj`/`shared_experts.down_proj`/`lm_head` — silently
discarding the recipe's actual intent on the parameters that matter most.

Confirmed directly: layer 3's `switch_mlp.fc2.weight` packed shape was
`[128, 2688, 174]` in the buggy positional release even though GPTQ had
calibrated it at `down_bits=6` — `174 = 1856*3/32` (3-bit packing), not
`348 = 1856*6/32` (6-bit). GPTQ's Hessian correction had computed the right
6-bit-calibrated values; the MLX repacking step silently re-quantized them at
3-bit anyway.

Fixed with a unified converter, `poc/mlx_convert_recipe.py`, matching
`down_proj` OR `switch_mlp.fc2` (and supporting both `--mode positional` and
`--mode sensitivity`). The MoE router (`gate`, a raw `mx.array` weight, not an
`nn.Linear`-like module) needed no equivalent fix — `mlx_lm`'s own
`quantize_model` already excludes it via `hasattr(module, "to_quantized")`.

Re-running both releases with the fix, full 52-block 30B-A3B model,
wikitext-2 (20×512-token chunks):

| model | PPL | size | bits/weight |
|---|---|---|---|
| bf16 (reference) | 5.11 | — | 16 |
| mixed_3_6, positional (fixed) | 5.81 | 16GB | 4.215 |
| mixed_3_6, sensitivity (fixed) | 5.90 | 16GB | 4.338 |
| ~~mixed_3_6, positional (pre-fix)~~ | ~~5.92~~ | ~~14GB~~ | ~~3.548~~ |
| ~~mixed_3_6, sensitivity (pre-fix)~~ | ~~5.95~~ | ~~14GB~~ | ~~3.549~~ |
| uniform 3-bit GPTQ | 6.24 | 13.8GB | 3.5 |
| naive RTN, uniform 3-bit | 6.54 | 13GB | 3.5 |
| JANG_2L-CRACK (3rd party) | 5.43 | 16GB | ~3.73 (their own label) |
| JANG component recipe, this project's GPTQ | **5.24** | — | 4.237 |

Applying JANG's reverse-engineered component-type allocation *through this
project's calibrated GPTQ pipeline* (rather than JANG's presumed naive RTN)
beat JANG's own PPL outright (5.24 vs. 5.43) — confirming that bit-allocation
strategy and calibration method are independent, additive levers.

### 1.5 The decode-throughput-vs-size trade-off of component-type bit allocation

The JANG-style component recipe wins on PPL but costs real decode throughput
on-device, and the mechanism is now fully understood. A controlled test
(`mlx_lm.generate`, no server, no KV-cache quantization, same prompt,
`--max-tokens 300 --temp 0`, 2 runs each) on the same Mac hardware:

| model | avg bpw | gen tok/s (2 runs) | peak mem |
|---|---|---|---|
| uniform 3-bit (this project) | 3.0 | 65.4 / 68.6 | 13.98GB |
| uniform 4-bit (mlx-community) | 4.0 | 53.8 / 53.5 | 17.93GB |
| jang-component (this project, GPTQ) | 4.237 | 42.1 / 42.8 | 16.88GB |
| JANG_2L-CRACK (3rd party, presumed RTN) | ~3.73 (claim) | 42.7 / 43.1 | 16.87GB |

**The real driver is component-type bit-allocation *structure*, not average
bits/weight.** This project's jang-component release and the third-party
JANG_2L-CRACK model run at essentially identical speed (42–43 tok/s) despite
different calibration methods (GPTQ vs. presumed RTN) *and* different claimed
average bpw (4.237 vs. 3.73) — because both share the exact same
component→bits map. Calibration method only chooses the quantized values
within a bit-width slot; it has zero effect on which Metal kernels get
dispatched, so runtime speed is identical regardless of which method produced
better PPL.

Uniform-bit-width scaling is also not quite linear with bytes: 4-bit (53.7
tok/s avg) is *faster* than pure linear byte-scaling from 3-bit (65 ×
3/4 ≈ 48.75 predicted) would suggest — plausibly because 3-bit packing
doesn't align to byte boundaries (8 weights per 3 bytes, needing cross-byte
bit-shifting to unpack) while 4-bit packs exactly 2 values/byte (trivial
mask+shift); the byte savings still win out overall, just by less than naive
linear scaling predicts.

**Why the trade-off is structural, not incidental**: JANG's recipe bumps
attention/mamba/shared-experts to high bits specifically because they are
*cheap in total parameter count* — but those same three components are the
ones active on *every* token (dense), while the huge-but-low-bit routed
experts are only ever partially active per token (sparse, top-k of 128). So
"cheap on disk" and "cheap per decoded token" are opposite properties for this
architecture: optimizing bit-allocation for size directly pessimizes decode
speed, with no way to lower the dense components' bits back down without
reverting most of the PPL gain they bought. This is a genuine Pareto frontier,
not a bug — pick uniform 3-bit for interactive/latency-sensitive local use,
the component recipe for quality-sensitive/offline use. The only unexplored
lever is intermediate points on the frontier (e.g. bump only attention, leave
mamba/shared-experts low).

### 1.6 GPTQ-Hessian calibration domain vs. capability: the 4B model's C++ regression

See §2.3 for the full ablation, but the quantization-methodology-relevant
conclusion is: **GPTQ's Hessian correction is calibration-domain-specific.**
It optimizes weight-reconstruction fidelity for the calibration corpus's
activation distribution. Neither wikitext-2 nor this project's tool-calling
SFT data contains any C++, so GPTQ had no signal to preserve C++-specific
weight directions — and its calibrated correction may have actively traded
C++ fidelity for extra precision on calibration-represented patterns, a trade
naive RTN doesn't make in either direction. At 8-bit, on the same
attention-only-LoRA 4B base model:

| method (same 8.503 bpw, same base) | C++ compile-pass (of 9) | wikitext PPL |
|---|---|---|
| this project's GPTQ (Hessian-calibrated) | 2/9 | 11.3914 |
| stock `mlx_lm.convert` (naive RTN, no calibration) | **4/9** | 11.3868 |
| unquantized bf16 (reference) | 6/9 | 11.4260 |

All three land within measurement noise of each other on PPL, while the
compile-and-run test cleanly separates them — see the general "PPL is not a
sufficient capability check" caveat in §2.3. This predicts GPTQ's usual PPL
win over RTN (§1.1, §1.4) still holds for text resembling the calibration
corpus, just not for capabilities entirely absent from it — worth
remembering for any small model expected to retain broad,
calibration-underrepresented capabilities (e.g. additional programming
languages).

### 1.7 Rotation is necessary for genuinely-ternary storage, but is expensive, and dense-model bit budgets don't need it

Early in the project, an incoherent-processing (Hadamard rotation) + GPTQ +
salient-weight-pinning recipe (`poc/rotation.py`, `poc/methods.py`'s
`rot_gptq_salient*` family) was the primary approach for very low bit-widths
(binary/ternary, ~1–2 bits/weight). Two findings from this track matter for
anyone reusing it:

- **Rotation breaks a literal "3-valued ternary" storage requirement.**
  `rot_gptq_salient`'s rotate-then-unrotate wrapper produces a *dense*
  reconstruction in the stored weight (every entry becomes a linear
  combination of rotated-domain ternary values) — verified directly: a
  "ternary" tensor had 29 distinct values per 256-element group, not 3.
  Fixed with a `--skip-rotation` flag on `poc/quantize_sequential.py` (calls
  `gptq_binary`/`gptq_ternary` directly, no rotation), verified exactly
  3-valued after the fix. Cost: on a skip-mamba+down_proj config, this moved
  perplexity ratio from 9.35x to 12.43x — real, but survivable at that
  operating point.
- **Plain ternary with no protection is a complete failure at full-model
  scale, not just "degraded."** Applying `gptq_ternary_batched` directly
  (no rotation, no salient pinning) to all 23 MoE blocks of the 30B-A3B model
  produced outright broken generation (`"The capital of France is
  100,000,000,000,000,000,000,000,000,000"`), a far more severe failure than
  any single-block ablation predicted. Adding rotation + salient protection
  back in fixed coherence — empirical confirmation that salient pinning is
  load-bearing at this bit-width, not optional polish, and that this
  generalizes from a single MoE block to the whole model.
- **At 3+ bits/weight, none of this incoherence-processing machinery is
  needed.** GPTQ affine quantization at moderate bit-widths (3/6/8, §1.1–1.4)
  does not exhibit the catastrophic breakdown seen at 1–2 bit, so the
  simpler plain-N-bit-GPTQ-then-stock-`mlx_lm.convert` pipeline (no rotation,
  no salient overlay, no custom MLX kernel or loader) is both simpler and
  produces smaller/faster/better-understood artifacts for 3-bit-and-up use
  cases. The rotation+salient custom-kernel path (`poc/pack_mlx.py`,
  `poc/rotated_switch_linear.py`, `poc/sparse_salient_mlx.py`) remains in the
  repo as a validated, working "Path B" for anyone who specifically needs
  sub-3-bit storage, but is not the recommended default.

### 1.8 `salient_fraction` is the dominant quality lever at low bit-widths, not calibration order

For the rotation+ternary+salient recipe, `salient_fraction` (the proportion
of weights per group stored at full precision rather than ternary) dominates
quality far more than calibration ordering. Measured on the full 52-block
30B-A3B model, one-shot calibration, wikitext-2:

| `salient_fraction` | PPL ratio vs. bf16 |
|---|---|
| 0.03 | 3.17x |
| 0.06 | 1.84x |
| 0.10 | 1.55x |

A competing hypothesis — that **one-shot calibration** (capturing every
block's activations against the pristine, unquantized model, so no block's
GPTQ accounts for upstream blocks' already-introduced error) explained most
of the gap between this full-model result and an earlier single-MoE-block
result (1.006x ratio, §1.9) — was tested and **rejected**. A `--sequential`
calibration mode (block `i`'s activations captured with blocks `0..i-1`
already quantized in place, matching `poc/quantize_sequential.py`'s approach)
was built and re-run on the full model: sequential PPL 39.29 vs. one-shot's
39.97 (both at `salient_fraction=0.03`) — a ~1.7% difference, essentially
noise. Sequential calibration does *not* meaningfully close the gap;
`salient_fraction` is the lever that matters. (Sequential calibration costs
roughly 1.5–2x one-shot's runtime — 2094s vs. 610s for the full 52-block
model — for that near-null quality benefit at this bit-width; see §1.10 for
where sequential *does* pay off.)

Implemented in `poc/quantize_full_moe_model.py` (`--salient-fraction`,
`--sequential`).

### 1.9 GPTQ batching across MoE experts: real speedup, with a caveat about GPU memory pressure

Because routed MoE experts share shape, `poc/gptq.py`'s
`gptq_binary_batched`/`_gptq_run_batched` (and `poc/methods.py`'s
`rot_gptq_salient_batched`) stack all same-shaped experts into one `[E, out,
in]` tensor and run a single Hessian/Cholesky/column-loop across all `E`
experts instead of `E` independent Python loops. Verified numerically
identical to the per-expert path on well-conditioned data.

Measured on a real MoE block (112/128 eligible experts, 30B-A3B model): the
per-expert loop took 1565s vs. the batched version's 270–284s — a real ~5.5x
collapse from avoiding per-expert Python-loop-launch overhead, though not the
full ~36x a smaller microbenchmark suggested (batched Hessian computation
pushes GPU memory to ~80.5/82GB on an 80GB card, likely throttling cuSOLVER).

On the single MoE block this was first validated against (rotation+GPTQ+
ternary→binary, `salient_fraction=0.03`, `group_size=128`, batched, 279
calibration tokens spread across 128 experts): held-out perplexity moved
3.101 → 3.119, a 1.006x ratio — essentially no degradation. This remains the
project's best single-block result; it used a different recipe (binary,
group_size=128) than the full-model ternary experiments in §1.8, and §1.8's
gap-explanation experiments concluded the difference is not primarily a
calibration-ordering artifact — plausible remaining explanations (untested):
cumulative effect of 23 independently-lossy ternary blocks compounding
regardless of order, hyperparameter differences (`group_size`), or this
single block being a favorable/unrepresentative sample.

For very large models that must run fully GPU-resident, batching all
eligible experts in one call can itself OOM; `poc/quantize_full_moe_model.py`
exposes `--moe-subbatch` (default 12) to sub-batch experts instead of
batching all of them in a single Hessian call.

### 1.10 One-shot vs. sequential calibration: negligible difference at moderate bit-widths, real cost at extreme low bit-widths

On the dense 4B model, at the `jang-dense` component recipe (3/6/8-bit,
5.778 bits/weight average), one-shot and sequential calibration produced
statistically indistinguishable perplexity (10.2463 vs. 10.2432 on
wikitext-2, a 0.03% relative difference) while one-shot ran in 515s vs.
sequential's 979s — essentially free speedup with no quality cost at these
bit-widths. Sequential calibration's value proposition is compensating
already-quantized upstream blocks' error in downstream blocks' calibration;
at 3/6/8-bit, per-block error is small enough that there is little for it to
compensate. This is consistent with §1.8's near-null result for sequential
calibration at ternary/binary bit-widths on the MoE model too — sequential
calibration does not appear to be a strong lever at *any* bit-width tested in
this project, though it was expected a priori to matter most at extreme low
bits (1–2 bit), which is where the original binary/ternary Stage A track
(`docs/archive/stage-a-b-poc-validation.md`) showed the largest, most
cascading per-block error.

### 1.11 Dense-model universality gaps in a pipeline first built for MoE

Extending the full-model GPTQ+component-recipe pipeline
(`poc/quantize_full_moe_model.py`, `poc/gptq_stock_convert.py`) from the
30B-A3B MoE model to a dense NemotronH variant
(`nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16`; 42 layers: 21 mamba, 17 plain
"mlp", 4 attention, no MoE) surfaced real gaps the MoE-only testing had never
hit:

- Block-type dispatch was hardcoded to `kind in ("mamba", "attention")`
  throughout both scripts — this model's "mlp" blocks (40% of it) silently
  fell through an "unknown kind, skipping" fallback and were left
  unquantized. Fixed with a moe/non-moe split plus a generic
  `dense_projection_names()` helper: known kinds use their validated
  projection lists, anything else auto-discovers every `nn.Linear` on the
  block's mixer.
- This checkpoint's `config.json` has no `layers_block_type` key — only a
  compact `hybrid_override_pattern` string. `AutoConfig` derives the
  per-layer list from that pattern only as a live post-load attribute, never
  writing it back to the file. Fixed by reading
  `model.config.layers_block_type` (post-load) instead of parsing the raw
  `config.json`.
- Added a `jang-dense` component recipe (attention=8, mamba=6, mlp=3 — no
  MoE speed/size tension here since nothing is sparsely activated, so bits
  go purely by component size) plus the matching `mlx_convert_recipe.py`
  predicate branch. Also fixed a latent `KeyError`: the recipe's `low_bits`
  placeholder hardcoded the MoE-only key `cbits["moe_routed_down"]`, which
  doesn't exist for a dense-only recipe — replaced with `min(cbits.values())`.

Result (wikitext-2, 20×512 chunks):

| | size | PPL |
|---|---|---|
| bf16 (unquantized) | 7.78GB | 9.7990 |
| jang-dense, sequential | 2.7GB (5.778 bpw) | 10.2432 |
| jang-dense, one-shot | 2.7GB (5.778 bpw) | 10.2463 |

Only **+4.5% PPL for a 2.9x size reduction** — much better than the original
binary/ternary Stage A track's results on this same model (4.7x–46.5x PPL
degradation, `docs/archive/stage-a-b-poc-validation.md`), because this uses
proper GPTQ-affine calibration at moderate bit-widths (3/6/8) instead of
extreme 1–2 bit.

### 1.12 MoE-specific naming and shape gotchas (cross-reference)

Several bugs in this project were caused by MoE-specific naming/shape
conventions that don't show up on dense models. Consolidated list (details
in the sections cited):

- `switch_mlp.fc1`/`switch_mlp.fc2` (not `up_proj`/`down_proj`) for routed
  experts — silently excluded from bit-upgrade predicates matching on
  `"down_proj"` (§1.4).
- The MoE router (`gate`) is a raw `mx.array`, not an `nn.Linear`-like
  module — already correctly excluded from quantization by `mlx_lm`'s own
  `hasattr(module, "to_quantized")` check, no fix needed (§1.4).
- `gate.e_score_correction_bias`: the source HF PyTorch model has no such
  parameter, but `mlx_lm`'s stock `MoEGate` class always allocates one, and
  strict weight-loading refuses to load without it — fixed by injecting
  zeros for any missing bias at load time (a no-op addition to routing
  scores). See §3 for the loader-glue context this came from.
- A block's row-count semantics differ between the up-projection (`fc1`,
  one row per token) and down-projection (`fc2`, one row per
  `(token, expert-slot)` pair, since `fc1`'s output already has a distinct
  row per chosen expert) — a custom kernel that assumed the same row/index
  relationship for both broke only on real multi-token/multi-expert
  generation, not on a synthetic single-token unit test. Fixed by deriving
  row count from the data's own size rather than assuming a relationship to
  the routing-index tensor's shape.

---

## 2. LoRA fine-tuning findings

### 2.1 `reasoning_content` must be present and non-empty, or the model never learns to close `<think>`

A LoRA fine-tune of the dense 4B model (targeting attention+MLP together)
exhibited a serious functional bug in real usage: for any message not
obviously requiring a tool (e.g. "hi"), the client received an *empty* reply
— `content: ""`, no tool call, `finish_reason: stop`. Comparing raw
`mlx_lm.generate` output against a working 30B model on the identical prompt
showed the mechanism directly: given a prompt ending in the chat template's
forced generation-prefix `<|im_start|>assistant\n<think>\n`, the 30B model
writes reasoning, closes with `</think>`, then answers — the 4B model writes
its answer immediately and **never emits `</think>` at all**. The client's
OpenAI-compatible layer, finding no closing tag, classifies the entire output
as `reasoning_content` and leaves `content` empty. The model is not
malfunctioning at generation — the whole reply is trapped on the wrong side
of an unclosed tag.

**First hypothesis (only partially right): zero non-tool training
examples.** The original SFT set had 369 tool calls across 110 conversations
and zero conversations ending in a plain reply with no tool call. Adding 116
more (English balance/volume) and 18 synthetic tool-calling conversations
(235 total, including explicit "just reply" examples) did **not** fix the
bug — unchanged after retraining.

**Actual root cause**, found by reading axolotl's `ChatTemplateStrategy`
(`chat_template.py`) and the tokenizer's own jinja template directly, not by
further trial and error: the jinja template only renders a populated
`<think>...</think>` block when

```jinja
{%- if message.reasoning_content is defined and message.reasoning_content is string and message.reasoning_content | trim | length > 0 %}
    {%- set content = "<think>\n" ~ message.reasoning_content ~ "\n</think>\n" ~ (message.content | default('', true)) %}
```

— i.e. `reasoning_content` must be present and non-empty after trimming. None
of this project's SFT data (original or newly-added synthetic) ever set
`reasoning_content` on assistant turns; every message was authored as plain
`{"role": "assistant", "content": "..."}`. Without it, the template's
fallback path collapses the turn to `<think></think>` immediately followed by
`content` — no newline between the tags, no space before the content — a
completely different token sequence from what inference forces
(`<|im_start|>assistant\n<think>\n`, tag-then-newline, expecting *something*
before a closing tag). The model was trained only on the collapsed
empty-adjacent form and never on a genuinely opened-then-closed `<think>`
block, so at inference — dropped into a real opened `<think>\n` — it had no
learned pattern for "nothing to reason about, just close and answer," and
fell back to answering inline without ever closing the tag.

**Verified directly, no training run needed**, by importing axolotl's own
`ChatTemplateStrategy` and tokenizer against the exact dataset/config used
for training and calling `tokenize_prompt()` on individual records to inspect
the rendered text and label mask:

```python
tokenizer = load_tokenizer(cfg)  # cfg.tokenizer_config = cfg.base_model first
strategy = load_strategy(tokenizer, cfg, cfg.datasets[0])
tok = strategy.tokenize_prompt(json.loads(line))
tokenizer.decode(tok["input_ids"])                                  # rendered text
tokenizer.decode([t for t, l in zip(tok["input_ids"], tok["labels"]) if l != -100])  # trainable span only
```

Without `reasoning_content`: rendered as `<think></think>Hey! What are we
working on today?<|im_end|>`, trainable span 10 tokens. With
`reasoning_content: "No tool needed here."` added to the same message:
rendered as `<think>\nNo tool needed here.\n</think>\nHey!
...<|im_end|>`, trainable span now includes the `</think>` token and is 30
tokens long — exactly the pattern the model needs to see. (A separate axolotl
warning, "Last turn is not trainable... dataset design issue," looked
directly relevant during the original run but turned out to be about the
correctly-non-trainable tool-result turn at the end of tool-heavy
conversations — confirmed only by printing per-turn `should_train` values via
a temporary source patch, not by assuming the warning meant what it sounded
like.)

**Fix**: added a short, contextually-appropriate `reasoning_content` to every
synthetic assistant turn (e.g. "Casual greeting, no task given yet — just
reply and see what's needed" for greetings; "User wants: {task}. I'll use a
tool to do this directly rather than describing how" before tool calls),
re-verified via the same direct-tokenization check before spending another
full train+quantize+convert cycle.

**Open question, unresolved**: the 30B-A3B model, fine-tuned on the exact
same `reasoning_content`-free dataset, does *not* exhibit this bug — it
reliably closes `<think>` even for plain replies. Working theory (not
independently verified): the larger model's pretraining gave it a strong
enough "always reason first" prior that a small number of LoRA examples
showing the collapsed pattern weren't enough to override it, while the 4B
model's weaker prior was overridden by training data that structurally never
demonstrates a closed, non-empty think block. Would need a controlled
model-size ablation holding the dataset bug constant to confirm.

### 2.2 Joint attention+MLP LoRA training catastrophically breaks general code competency, independent of quantization

Real usage of the 4B LoRA+quantized model (after the §2.1 fix) surfaced
garbled, non-compiling C++ and degenerate repetition on a follow-up
translation request — much worse than the `<think>` bug, and invisible to
wikitext perplexity. This was root-caused with a proper eval instead of
eyeballing output: for each (temperature, prompt) pair, generate with N
different seeds, extract the code block, and **actually compile it**
(`g++ -std=c++20`) or **actually execute it** (Python), counting real
pass/fail — this catches syntax errors, missing includes, and logic bugs
that reading the text alone misses.

**Temperature was refuted as the cause.** A single anecdotal sample
suggested `temperature=1.0` (the deployment's configured value) was too
aggressive and `temperature=0.4` looked clean. A proper sweep (3 seeds × 3
temps × 2 prompts = 18 generations, actual compile/run) showed pass rate flat
across temperature (3/6, 4/6, 3/6) — the one clean low-temp sample had been
luck. (A separate methodology bug caught along the way: the eval harness's
first version used `max_tokens=600`, which truncated the longer C++ prompt
mid-function and miscounted the resulting syntax error as a real failure —
fixed by raising to 900 before drawing conclusions.)

**Cause #1: training both attention and MLP LoRA modules together breaks
code competency, independent of quantization.** Isolated by testing the
un-fine-tuned base model, the merged (bf16, unquantized) LoRA model, and the
quantized model separately (compile-and-run pass rate, 9 generations: 3
seeds × 3 temps, C++ sort task):

| variant | C++ compile-pass | overall (C++ + Python) |
|---|---|---|
| base bf16 (no LoRA) | 6/9 | 15/18 |
| full LoRA (attn+mlp), bf16 merged | **0/9** | 8/18 |
| full LoRA (attn+mlp) + jang-dense quantized | 3/9 | 10/18 |

The base model is fine at C++; the LoRA fine-tune broke it, and quantization
on top of the broken LoRA looks *slightly* better (3/9) than the unquantized
broken LoRA (0/9) — plausibly because quantization's rounding noise perturbs
the badly-overfit weights away from whatever specific failure mode the LoRA
converged to, not because quantization helps per se.

Ablated *which* targeted modules caused this by loading the already-trained
(r=32, 4 epochs) adapter and zeroing its `lora_B` matrices for either the MLP
(`up_proj`/`down_proj`) or the attention (`q/k/v/o_proj`) modules before
merging — testing each half of the jointly-trained adapter alone:

| variant | C++ compile-pass |
|---|---|
| attention-only half of the joint adapter (MLP zeroed) | 6/9 |
| MLP-only half of the joint adapter (attention zeroed) | 7/9 |
| both halves together (the actual trained adapter) | 0/9 |

Neither half alone reproduces the damage — both are indistinguishable from
base. This rules out "too much LoRA capacity/overfitting" as the mechanism:
retraining from scratch with rank and epochs both halved (r=16, 2 epochs
instead of r=32, 4 epochs), still targeting attention+MLP together, was
**still broken** (merged bf16: 1/9 C++, quantized: 3/9 — statistically
indistinguishable from the r=32/4-epoch run). The damage tracks *training
attention and MLP LoRA modules simultaneously*, not capacity or training
length — consistent with an interference effect between the two
simultaneously-adapted sub-circuits, not simple overfitting/catastrophic
forgetting.

**Fix**: retrain with `lora_target_modules` restricted to
`q_proj/k_proj/v_proj/o_proj` only (dropping `up_proj`/`down_proj`
entirely), same r=32/alpha=64/4-epochs otherwise. Merged bf16 result matches
base exactly (6/9, 15/18). `poc/run_4b_lora_retrain.sh` exposes
`LORA_TARGET_MODULES` as an explicit env var for this rather than a
hardcoded list.

### 2.3 Ablation methodology: PPL alone did not catch either the `<think>` bug or the code-competency regression

Both findings in §2.1 and §2.2 were invisible to wikitext-2 perplexity — a
metric used throughout this project's quantization work — and were only
caught by testing the actual deployed behavior (chat-formatted generation,
and compile-and-run code checks). The general methodology takeaway,
consolidated here since it recurs across §1.6/§1.1 and §2.1/§2.2:

- **Perplexity measures fit to a reference text distribution, not
  task/capability correctness.** A model can score better on the metric it
  was optimized for (e.g. a chat-template-formatted training objective) while
  scoring worse on a differently-formatted probe of superficially related
  text — a LoRA-tuned 4B model scored *worse* on raw-flattened-text
  perplexity (wikitext PPL 12.02 vs. the untuned quantized baseline's 10.24,
  and SFT-data-as-raw-text PPL 2.92 vs. 2.84) despite axolotl's own
  chat-template-formatted held-out eval showing clear improvement during
  training (eval_loss 0.49 → eval_ppl 1.57 over 83 steps). This is a
  measurement-methodology mismatch (structured chat-template tokens vs. raw
  flattened text), not a contradiction — and the functional check that
  actually matters (correct tool-calling behavior) had already been validated
  separately and was not undermined by the worse raw-text PPL number.
- **At high bit-widths (8-bit), PPL differences across quantization methods
  can vanish into noise while a task-specific behavioral test still cleanly
  separates them** — see §1.6's table (GPTQ 2/9, RTN 4/9, bf16 6/9 on
  compile-and-run, but all three indistinguishable on PPL).
- **A capability regression from LoRA fine-tuning specifically requires
  testing the actual capability**, not just held-out loss on the training
  distribution or a general-purpose perplexity probe — the joint
  attention+MLP interference in §2.2 was completely invisible until an
  actual compile-and-run test was built.

### 2.4 Practical LoRA config notes

- On the MoE (30B-A3B) model, LoRA training must target
  `lora_target_parameters` (not `lora_target_modules`) for the MLP, since
  routed experts are 3D parameter tensors (`[experts, out, in]`), not plain
  `nn.Linear` modules — this also forces `lora_dropout: 0` (the exotic
  parameter-wrapper path this requires doesn't support dropout).
- On the dense 4B model, MLP weights are plain `nn.Linear`, so
  `lora_target_modules` works directly with no such restriction, and
  `lora_dropout` is unconstrained.
- Merging a LoRA adapter into the dense model touches exactly the tensor
  count implied by the target-module list — e.g. 4 attention blocks × 4
  projections + 17 mlp blocks × 2 projections = 50/263 tensors touched,
  confirmed to match exactly, verifying the LoRA only affected what was
  configured. No expert-weight-drop warning appears for the dense model
  (unlike the MoE model's merge) since there are no MoE experts to drop.

---

## 3. Infrastructure and tooling

### 3.1 `torch.cholesky_inverse` is silently ~30x slower than the mathematically-equivalent alternative

`poc/gptq.py`'s Hessian-inverse computation used `torch.cholesky_inverse(L)`
(LAPACK `potri`), which took 24+ seconds for a 2688-dim Hessian on this
project's hardware, **regardless of data conditioning**. Replaced with
`torch.cholesky_solve(eye, L)` — identical math, ~30x faster (0.7s). This had
been inflating every GPTQ call across the whole project, not just one
experiment, and had previously produced a now-invalidated conclusion that
"GPU is slower than CPU for GPTQ" — once fixed, GPU beat CPU by 36x on a
single-matrix microbenchmark (10.4s vs. 378s).

### 3.2 `torch.set_num_threads(16)` is load-bearing for CPU-side GPTQ column loops on many-core hosts

The non-batched, single-matrix GPTQ path (`gptq_nbit`/`gptq_binary`/
`gptq_ternary`, used for mamba/attention/shared-expert projections that
aren't batchable across experts) runs a Python loop of ~2700 iterations, one
per column. With PyTorch's default CPU thread pool (252 threads on this
project's host), each tiny per-column op paid full thread-pool
synchronization overhead — measured at 7+ CPU-minutes and still running for
one mamba block that should take ~20s. Adding a single
`torch.set_num_threads(16)` call, once, globally, brought the same block down
to 4.2s. This is easy to miss because the symptom looks like "the GPU path is
slow," not "the CPU path is catastrophically slow" — it is load-bearing for
any script combining GPU batched work with CPU-based non-batched
`gptq_nbit`/`gptq_binary`/`gptq_ternary` calls on a many-core host.

### 3.3 Mamba's `causal_conv1d`/`mamba_ssm` kernels have no CPU fallback once installed

`transformers`' `use_kernel_func_from_hub_with_fallback` decorator always
prefers an *installed* `causal_conv1d` package over the pure-PyTorch
reference implementation, with no runtime device check — so loading a model
with `device_map="cpu"` does not make it CPU-only once `causal_conv1d` is
installed; it crashes (`RuntimeError: Expected x.is_cuda() to be true`) the
moment a Mamba block's forward pass runs. The model must stay fully
GPU-resident for every forward pass; there is no way around this short of
uninstalling `causal_conv1d` process-wide (which would also remove the fast
path for legitimate GPU runs).

Separately, `mamba_ssm`'s own pip dependency resolution can silently
force-upgrade `torch` to an incompatible CUDA-stack version (observed:
2.11.0+cu128 → 2.14.0) when installed without care, which would break an
already-built `flash-attn` and any pinned torch version elsewhere in the
stack. Fixed by installing with `--no-build-isolation --no-deps`
(`mamba_ssm`'s setup.py only needs torch *importable* at build time for
arch/version detection, not any specific version), plus a hard
version-unchanged assertion in `setup_pod.sh` so this cannot silently
regress again.

For **training** (not just inference/calibration), a full model resident on
GPU with a real backward pass through `mamba2_chunk_scan`'s reference
PyTorch fallback (used whenever `causal_conv1d`/`mamba_ssm` are not
installed) can OOM by trying to allocate a single 45GB tensor — calibration
scripts print this fallback as a warning every run but never hit the memory
wall since calibration is forward-only; training's backward pass does.
Installing both packages (with the `--no-build-isolation --no-deps` caveat
above) resolves it.

### 3.4 GGUF's 256-alignment requirement creates a hard size floor for this architecture

TQ2_0, TQ1_0, and every K-quant format (Q4_K/Q5_K/Q6_K) in `gguf-py`/
llama.cpp require the quantized axis to be divisible by 256
(`gguf-py`'s `can_quantize()`: `shape[-1] % block_size == 0`), with no
padding fallback. Both models in this project have a `hidden_size` not
divisible by 256 (dense 4B: 3136 = 64×49; 30B-A3B: 2688) — divisible only by
32 (legacy Q4_0/Q5_0/Q8_0 formats). Only tensors whose in_features is a
*different* dimension (e.g. `attention.o_proj`, `mamba.out_proj`,
`mlp.down_proj`) are K-quant/TQ-eligible; everything gated on `hidden_size`
falls back to a legacy format regardless of the requested quant level. This
is almost certainly why NVIDIA's own official GGUF releases use Q4_K_M
(mixed K-quant + legacy fallback) rather than a pure low-bit format — the
constraint postdates most model architectures and is a llama.cpp
implementation detail, not something these models were designed against.
Confirmed to generalize to the 30B-A3B model too: even a third party's most
aggressive quant (IQ2_XXS, ~2 bits) came out at 18.84GB, barely smaller than
their IQ4_XS (18.92GB) — the 256-alignment floor caps how small *any* GGUF
quant of this architecture can get, independent of the requested bit level.
(Padding `hidden_size` to the next multiple of 256 would unlock K-quants but
breaks RMSNorm statistics, whose denominator would then include the padding
elements, without also patching the norm kernel — assessed as not worth the
engineering risk for a small size gain.)

`gguf-py`'s pure-Python `quantize()` also has **no K-quant implementation at
all** — confirmed directly, `quantize(data, Q4_K)` raises `NotImplementedError`
in `quantize_blocks`; only TQ and legacy formats have real Python quantizers.
This caps any from-scratch Python packer (`poc/pack_gguf.py`) at
legacy-format quality; genuine K-quants require the real `llama-quantize`
binary (which supports a `--tensor-type name=type` flag for per-tensor
overrides on top of its own recipe, if a custom mix is still wanted).

**imatrix calibration, not custom tensor-type tuning, is what actually
matters for GGUF quality.** Same held-out text, same
`llama-perplexity -c 256 --chunks 1` methodology, dense 4B model:

| File | Size | PPL | vs. F16 |
|---|---|---|---|
| F16 reference | 7.96 GB | 91.45 | 1.00x |
| `llama-quantize ... Q4_K_M` + imatrix (wikitext-2, 40 chunks) | 2.70 GB | 94.86 | 2.95x, ~1.04x PPL |
| `llama-quantize ... Q4_K_M`, no imatrix | 2.84 GB | 109.81 | 2.80x |
| Custom `pack_gguf.py` mix (Q8_0 majority + TQ2_0 + Q4_0 fallback) | 3.80 GB | 114.51 | 2.09x |

Every hand-tuned custom tensor-type config tried lost to "just run the
standard `llama-quantize` pipeline with an imatrix" — the imatrix flag alone
is the difference between 109.81 and 94.86 PPL (landing within ~4% of F16).

### 3.5 Three real upstream llama.cpp bugs mistagging dense NemotronH models as MoE

All three share one root cause: `AutoConfig.from_pretrained(dir).to_dict()`
always includes every field `NemotronHConfig`'s Python class defines
(including MoE-only fields), regardless of whether the checkpoint's raw
`config.json` actually set them. A checkpoint round-tripped through
`save_pretrained()` (as happens after quantization) picks these up as if
deliberately set:

1. `has_moe_params = "num_experts_per_tok" in llm_config`
   (`convert_hf_to_gguf.py`'s `nemotron.py`) checks key *presence* via
   `AutoConfig`, not meaningfulness — always true, mistagging every dense
   NemotronH model as `nemotron_h_moe`. Fixed by reading the *raw*
   `config.json` for this specific check and stripping MoE-only keys from
   `hparams` when false (a second leak existed: `base.py`'s generic
   `set_gguf_parameters` independently writes `expert_group_count` whenever
   `hparams.get("n_group")` exists).
2. `_MLP_LAYER_TYPES = {"moe"}` (class constant): dense models' verbose
   `layers_block_type` list (synthesized by `AutoConfig` from the compact
   `hybrid_override_pattern` string) uses the literal string `"mlp"`, not
   `"moe"`, for feed-forward layers — so `_mlp_layers` came out empty even
   after fixing #1, and `feed_forward_length` computed as all-zero. Fixed:
   `_MLP_LAYER_TYPES = {"moe", "mlp"}`.
3. `hybrid_override_pattern` does not survive `save_pretrained()` —
   `AutoConfig().to_dict()` only synthesizes the per-layer type list from it
   at *load* time, so a model resaved without the original file present
   loses it silently. Fixed by copying `hybrid_override_pattern` from the
   original checkpoint's `config.json` into the resaved one before
   conversion.

Patch saved at
`docs/patches/llamacpp_nemotron_dense_moe_mistag_fix.diff` — worth
upstreaming, since it affects anyone quantizing-and-resaving a dense
NemotronH checkpoint. Symptom without the fix: `llama-cli`/`llama-simple`
fails to load (`check_tensor_dims: tensor 'blk.1.ssm_in.weight' not found` or
similar) even though `convert_hf_to_gguf.py` itself reports success —
**conversion succeeding is not proof the resulting file loads.**

### 3.6 `llama-cli`'s interactive/chat mode hangs indefinitely on this architecture

Distinct from the quantization/conversion bugs above: `llama-cli`'s
interactive/chat mode (what LM Studio's chat UI drives) hangs indefinitely
given any input on this NemotronH hybrid Mamba+Attention architecture —
confirmed both piped (`echo hi | llama-cli ...`) and interactive, in both
cases zero token-generation progress after 120+ seconds, and critically the
process ignored `timeout 120` entirely (CPU time stayed flat — genuinely
stuck, not slow) and required `kill -9`. `llama-simple` (raw completion, no
chat template) works fine and was used for all conversion/quality
comparisons in this project. This directly explains a real-world symptom
where LM Studio's chat window either produced garbled/irrelevant text or
froze entirely — the GGUF file itself was fine (correct metadata, correct
`llama-simple` output); the bug is specifically in llama.cpp's
conversation/chat-loop code path interacting badly with this architecture's
hybrid KV-cache + Mamba recurrent-state handling across the chat loop's
multi-call structure, as opposed to `llama-simple`'s single straight-through
generation call. Not root-caused further in this project — worth a
follow-up before recommending llama.cpp/LM Studio generally as a serving
path for NemotronH models, since it would block ordinary chat use even with
a perfectly good GGUF.

### 3.7 MLX on Linux+CUDA: a real backend, with real gaps

`mlx-lm` has a genuine (non-stub) NemotronH implementation, including MoE
and Mamba2 support. Less obviously, **MLX now has a real CUDA backend**
(`pip install "mlx[cuda]"` — **not** `mlx[cuda12]`, which is not a real
extra at all; the base `mlx` wheel alone has no CUDA runtime on Linux),
confirmed working on an A100 pod (`mx.default_device()` → `Device(gpu,
0)`), including quantized matmul support (added in MLX 0.31.0+). A
base install can still silently land on a CPU-only `mlx`: this project's
pod image's preinstalled torch pins `nvidia-cublas-cu12`/
`nvidia-cuda-nvrtc-cu12`/`nvidia-cufft-cu12` to exact 12.8.x builds, while
`mlx-cuda-12` wants exact 12.9.x — pip's resolver then downgrades `mlx`
itself to an old version that doesn't pull in `mlx-cuda-12` at all,
*without erroring* (import succeeds, but `mx.default_device()` reports
`Device(cpu, 0)`). The fix baked into `poc/setup_pod.sh`: force
`--upgrade --force-reinstall 'mlx[cuda]'` after the base install, then
hard-assert `mx.default_device().type == gpu` before proceeding — CUDA
12.x libraries are ABI-compatible across minor versions in practice, so
upgrading the `nvidia-cu12` packages past torch's pin doesn't break
torch's own `.cuda()` ops. This meant the entire MLX pipeline (convert,
quantize, and any custom inference code) could run on a rental GPU rather
than requiring a Mac, sidestepping a target machine's RAM constraint for
the *conversion* step — only the final artifact needs to fit on the
target machine.

Deployment gap: **LM Studio's MLX engine (`mlx-engine`) cannot load a
custom/non-standard MLX architecture.** It vendors its own isolated
`mlx-lm` via `venvstacks` and never passes `trust_remote_code=True` to
`load_model()` (confirmed via multiple open lmstudio-ai/mlx-engine
issues). A custom quantized-linear layer (as built for the
rotation+salient path, §1.7) is only usable via
`mlx_lm.server --trust-remote-code` from the terminal, not the LM Studio
app directly — capping any "drop-in LM Studio artifact" ambition to
whatever fits MLX-LM's *stock* model classes.

`mlx_lm.server` was also separately found to be unreliable in this
project's testing on a *stock* (non-custom) model — a request that
completed instantly via `mlx_lm.generate` would hang or OOM via
`mlx_lm.server`. Root cause not isolated; worth revisiting if the server's
OpenAI-compatible endpoint is specifically needed later.

### 3.8 Custom Metal-kernel pitfalls (relevant to `poc/sparse_salient_mlx.py`)

Building a hand-written Metal scatter-add kernel (for the salient-weight
correction pass in the rotation+ternary+salient path, `poc/pack_mlx.py` /
`poc/rotated_switch_linear.py` / `poc/sparse_salient_mlx.py`) surfaced three
reusable pitfalls with `mx.fast.metal_kernel`:

1. **`atomic_outputs=True` output buffers are not guaranteed zero on a fresh
   call.** A trivial probe kernel that atomic-adds into one fixed index of an
   output buffer, called repeatedly, kept *incrementing* across separate
   calls (1.0, 2.0, 3.0, ...) instead of resetting — MLX evidently reuses the
   same physical buffer across separate invocations of the same compiled
   kernel without re-zeroing it. In this project's actual kernel, this meant
   every call after the first accumulated all previous calls' contributions
   on top of the current call's, compounding without bound across a
   generation's ~50+ forward passes until it produced `NaN`. The real fix
   was **not** to mask untouched output rows (insufficient — legitimately
   written rows also carried forward stale accumulation) but to use `mx.array`'s
   `.at[idx].add(values)` scatter-add primitive, which starts from a
   genuinely fresh `mx.zeros(...)` every call and was confirmed correct
   including with duplicate indices. The custom kernel was split into a
   non-atomic resolve step (bitmap position → `(row, col)`, safe because every
   thread writes a unique cell) plus a plain MLX-level gather + dequantize +
   `.at[].add()` scatter — add a repeated-call regression test for this class
   of bug specifically, since a single-call test cannot catch it.
2. **Explicit `device`-address-space pointer redeclaration is a hard Metal
   compile error** when an input's actual auto-generated buffer address
   space isn't `device` (it can be `constant` for some inputs, not
   controlled by the caller) — e.g. `device const int* cp = checkpoint +
   offset;` fails to compile if `checkpoint` was auto-typed `constant`.
   Avoid by never redeclaring a typed pointer; index the named kernel input
   directly instead.
3. **A kernel object reused across calls with different template values can
   silently return wrong results on the second call** (observed: values
   exactly 2x off) — not confirmed as MLX-name-based pipeline caching, but
   worked around by giving the kernel a name that includes the template value
   and caching one kernel object per distinct value actually used. This
   symptom retroactively turned out to be the *same* underlying
   buffer-reuse issue as #1 above, not something specific to template-value
   changes — any two calls to the same kernel object exhibit it, regardless
   of whether the template argument changed between calls.

A minimal repro needs to match the *specific* conditions of a failure
(multi-expert indexing, repeated calls) — neither of the above showed up in
the original single-expert/single-call diagnostic kernel used to first
isolate the problem.

### 3.9 Sizing pitfall: on-disk compactness and in-memory resident size are different budgets

A bitmap-based salient-weight encoding (1 bit/weight, replacing an earlier
per-entry `(row, col, value)` scheme that would have cost 10 bytes/entry and
projected to ~37GB — bigger than plain GGUF) solved the *on-disk* size
problem, but `decode_salient_bitmap`'s original design decoded the bitmap
into explicit, permanently-resident `(row, col, value)` arrays at model
*load* time — reintroducing the same per-entry cost, just at runtime instead
of on disk. At `salient_fraction=0.10` on the 30B-A3B model (~2.94 billion
total salient entries across all MoE blocks/experts/projections), this cost
~27.6GB just for decoded salient indices/values, on top of ~11.9GB for
everything else — a model reported as needing 39.7GB against an
~18.2GB MLX-recommended ceiling on a 24GB Mac.

The fix that actually closed this (rather than incremental byte-shaving)
was to **never materialize the decoded arrays at all**: a Jacobson-style
rank/select index over the bitmap (a per-chunk cumulative-popcount
checkpoint array, computed once at load time in plain numpy) lets a Metal
kernel resolve "flat position of the j-th set bit" via binary search + local
`popcount`, entirely inside the kernel, at query time — no persistent
decoded index arrays, ever. Measured end-to-end: 28.5GB → 20.9GB required
memory, matching the back-of-envelope prediction. The general lesson: check
the *resident* memory profile of a compact on-disk format separately from
its on-disk size — they are genuinely different budgets, and only the
on-disk one is obvious from the file listing.

Related, smaller levers in the same investigation: switching `row`/`col`
indices from int32 to int16 (every value fits, ≤2688) cut ~11GB;
quantizing salient values from float16 to int8 with a per-expert scale
saves roughly another 2.9GB at `salient_fraction=0.10`. macOS's own
`iogpu.wired_limit_mb` sysctl (a hard GPU-wired-memory cap, separate from
MLX's own `mx.set_memory_limit()`) can also independently block a model that
otherwise fits in free RAM — raise via
`sudo sysctl iogpu.wired_limit_mb=<value>` (documented in MLX's own
`set_wired_limit` docstring as the correct lever).

### 3.10 A custom per-token salient-correction kernel does not scale to long prompts without chunking, and has a real throughput ceiling

Two further problems surfaced only under realistic (not single-token) usage
of the rotation+ternary+salient model via `mlx_lm.server`:

- A 601-token prompt spiked peak memory to ~36GB even though the
  single-token case was fine, because the resolve/gather/scatter arrays in
  the salient-correction kernel are shaped `[n_tokens, top_k, k]` (`k` being
  the large per-expert salient count), scaling directly with prompt length
  during prefill. Fixed by chunking the token dimension internally, forcing
  `mx.eval()` per chunk to free transients before the next — brought the
  same prompt back to ~19.5GB.
- Generation throughput measured at ~7 tok/s, independent of chunk size
  (ruling out per-chunk Python-loop overhead as the bottleneck — the compute
  itself is the ceiling). For comparison, on the same hardware/architecture:
  stock 3-bit MLX RTN conversion measured 67 tok/s via `mlx_lm generate`
  (13GB on disk) and 26.8 tok/s via LM Studio; this project's custom
  ternary+salient kernel (`salient_fraction=0.06`) measured ~7 tok/s via both
  `mlx_lm.server` and direct `mlx_lm generate` — a genuine, unresolved
  throughput limitation of the resolve-then-gather-then-scatter kernel
  design, not a quick parameter fix.

**The size/quality/speed math that motivated moving away from this path for
3-bit-and-up use cases**: stock 3-bit RTN costs ≈3 bits + per-group
scale/bias overhead (`2×16/group_size=64` ≈ 0.5 bit) ≈ 3.5 bits/weight → a
30B model lands at ≈13.1GB (matches the observed 13GB exactly). This
project's ternary+salient recipe at `salient_fraction=0.06` costs 2-bit
ternary base + ~0.5-bit group overhead + a **mandatory 1 bit/weight bitmap**
(fixed cost regardless of `salient_fraction`, §1.7/§3.9) + the salient values
themselves (0.06 × 8-bit int8 ≈ 0.48 bit) ≈ ~4 bits/weight — i.e. the
bitmap's fixed tax alone puts this recipe's size floor above stock 3-bit
RTN's *total* cost, before even counting the salient values. This is why
§1.7 concludes the rotation+salient machinery is best reserved for
sub-3-bit targets, where naive RTN's own quality collapses and there is no
competing cheap alternative.

### 3.11 Miscellaneous operational notes

- `pkill -f <pattern>` self-matches when the pattern text is also present in
  the invoking shell command's own argv (e.g. running
  `ssh host "pkill -f quantize_full_moe_model; echo done"` kills that SSH
  session's own shell, since the pattern string appears literally in its
  command line). Use `ps aux | grep <pattern> | grep -v grep` to find the
  real PID and `kill <pid>` directly, or the bracket trick
  `pkill -f '[q]uantize_full_moe_model'`.
- A visible `df -h` free-space reading can lag actual state by tens of GB
  for a few seconds after a large `rm -rf` on an overlay filesystem — `sync`
  and re-check before trusting a low-free-space reading enough to act on it.
- A packer/converter's "copy everything else in this block, unquantized"
  catch-all loop is a real footgun for MoE checkpoints specifically: it can
  match a MoE model's raw HF parameter names (e.g. `experts.up_proj`) as
  "not yet written" when the packed output uses different key names (e.g.
  `switch_mlp.fc1`), and copy the full unquantized `[experts, out, in]`
  tensor on top of the already-packed one — inflating output size by ~5x in
  one observed case (7.8GB vs. an expected ~1.5GB on a 4-block test). Fix
  with an explicit skip-list for already-handled raw prefixes, not a
  name-based "already written" heuristic.
- A packer's own salient-count formula and the mask-selection function it
  depends on can disagree by exactly one entry if one uses `round()` and the
  other uses `int()` truncation for the same fractional count — breaking any
  downstream assumption of an exact count. A mask built from a fixed
  threshold can also over-select on ties. Fix at the source: always select
  exactly `k` elements via `torch.topk(...).indices`, not a
  threshold-derived mask.
- Any from-scratch N-bit weight packer that packs multiple sub-byte codes
  into words only handles bit-widths that divide evenly into a byte/word
  boundary without cross-word bit-shifting (`bits ∈ {2, 4, 8, 16}` for this
  project's packer) — 3/5/6-bit packing needs real cross-word bit-packing
  that a simple packer does not implement. Where this matters, prefer
  letting a mature stock tool (e.g. `mlx_lm.convert`) do the final packing
  step instead of re-implementing bit-packing.

## 4. MTP (Multi-Token Prediction) preservation

### 4.1 The current production model has zero `mtp.*` weights, and neither GPTQ nor `mlx_lm.convert` is where they got lost

`nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16` ships a DeepSeek-style
MTP head (config: `num_nextn_predict_layers`, `mtp_layers_block_type`) as a
separate safetensors shard — 270 `mtp.*` tensors, ~1.34B params (bf16),
~4.06% of the model's total 65.8GB. This project's own
`...-JANG-GPTQ-ipsupport-code-lora` has 0 of 729 keys starting with
`mtp.` (confirmed directly against the published `model.safetensors.
index.json`). The actual cause: HF transformers' `NemotronHForCausalLM`
has `_keys_to_ignore_on_load_unexpected = [r"mtp.*"]`, so **any**
`transformers.AutoModelForCausalLM.from_pretrained()` call — the LoRA
merge step and `gptq_stock_convert.py`'s own loading, both PyTorch-based —
silently drops these weights on load, before GPTQ or `mlx_lm.convert` (or
which `mlx-lm` version is installed) ever get a chance to see them. Stock
`mlx_lm`'s own `sanitize()` also strips `mtp.*` unconditionally, but by
the time a checkpoint reaches that step in this project's pipeline, the
weights are already gone — patching only that layer (as
`ipsupport-llc/mlx-lm@nemotron-h-mtp` initially did, mirroring
`AirRunner/mlx-lm`'s Qwen3.5 work and pierre427's abandoned
`nemotron-h-mtp-head` PR upstream) is necessary but not sufficient for a
pipeline that routes through PyTorch at all.

Confirmed by direct reproduction: round-tripping a real `mtp.*`-bearing
checkpoint through plain `AutoModelForCausalLM.from_pretrained(dtype=
torch.bfloat16).save_pretrained(...)` — no GPTQ, no LoRA, nothing else —
already drops all `mtp.*` keys (824 keys, 0 `mtp.*`, vs. the source's 272).

### 4.2 Fix: extract before, inject after — never let `mtp.*` touch a PyTorch step

`poc/extract_mtp_weights.py` pulls `mtp.*` straight out of the untouched
bf16 source via `mlx.core.load` (reading only the shard(s) that actually
contain them, identified from the index — no need to touch the other
~62GB), before the LoRA merge or GPTQ ever run. `poc/inject_mtp_weights.py`
splices them back into the already-`mlx_lm.convert`-ed model afterward,
using `mlx_lm`'s own model classes directly (`ipsupport-llc/mlx-lm@
nemotron-h-mtp`'s `nemotron_h.Model`/`sanitize()` — no transformers
involved at this point at all). `run_pipeline.sh`'s `--mlx-lm-git` flag
now triggers both steps around the existing merge/GPTQ/convert chain,
rather than (as first assumed) just needing to swap the `mlx_lm.convert`
step's package. Verified end-to-end against a real checkpoint (not just
synthetic weights): extract → simulated transformers strip → `mlx_lm.
convert -q` → inject → self-speculative decoding output is bit-exact
against plain greedy decoding, both at a flat bit-width and via the
`--component-recipe jang` path.

### 4.3 A real upstream `mlx_lm` bug: the fused single-step Mamba/SSM kernel is wrong when replaying a captured (not live) state

Independent of this project's pipeline: the reference MTP self-speculative
decoding driver (`ipsupport-llc/mlx-lm@nemotron-h-mtp`'s
`nemotron_h_mtp_generate_step`, modeled on `AirRunner/mlx-lm`'s Qwen3.5
implementation) rejects a bad MTP draft by rolling the model's Mamba/KV
caches back from a 2-token speculative block to 1 kept token, via
`Model.rollback_speculative_cache` — for the Mamba side, by replaying the
SSM update from a *captured* pre-update state (an `ssm_sink` snapshot),
not letting the cache's own live state carry forward. `mlx_lm.models.ssm.
ssm_update` auto-dispatches any `seq_len == 1` call on GPU to
`ssm_update_kernel`, a fused Metal kernel that assumes `state` is *its own
immediately-prior output* — precisely violated by a `keep == 1` rollback
replay, which is a `seq_len == 1` call fed a state captured earlier in the
same forward pass, not produced by that kernel a moment before. Confirmed
by forcing `mx.set_default_device(mx.cpu)` (which always takes the safe
`ssm_attn` scan path regardless of `seq_len`): the SAME rollback call that
diverges by up to 1.73 (raw SSM state, arbitrary units) on GPU matches a
fresh forward to ~1e-9 on CPU. Existing coverage never caught this because
the one pre-existing rollback unit test only exercised `keep=2, block_size
=4` (`seq_len=2`), which never reaches the single-step kernel at all.
Fixed by calling `ssm_attn` directly in the replay path instead of going
through `ssm_update`'s auto-dispatcher. `keep==1` is exactly the reject
case for k=1 MTP speculation — the common case — so this would have
silently corrupted the Mamba state on every rejected draft on real Apple
Silicon hardware.

### 4.4 `transformers`' `{"__float__": "Infinity"}` config encoding needs decoding in `mlx_lm`, not patching around

Any config that has round-tripped through `save_pretrained()` (LoRA merge,
GPTQ) re-serializes `NemotronHConfig`'s full resolved field set, including
computed defaults not present in the original checkpoint's config.json.
`time_step_limit`'s default upper bound is `float("inf")`, and plain JSON
has no literal for that — `transformers` encodes it as
`{"__float__": "Infinity"}` instead of a bare number. `mlx_lm`'s
`ModelArgs.__post_init__` didn't know this convention and passed the raw
dict straight into `mx.clip()`, crashing on the model's very first forward
pass. Fixed at the source (a small `_decode_hf_float()` helper in
`nemotron_h.py`'s `ModelArgs.__post_init__`) rather than a one-off
config.json patch-up step in this project's own pipeline scripts, since
every checkpoint that goes through a PyTorch-based step in this pipeline
will hit this, not just the MTP path specifically.

### 4.5 The MTP head gets its own bit tier, not the backbone's most aggressive one

The injected head isn't GPTQ-calibrated (plain round-to-nearest via
`mlx_lm.utils.quantize_model`, reusing its divisibility-skip safety rather
than a bare `nn.quantize` call, which crashes instead of skipping a
weight whose last dim isn't divisible by `group_size`). Correctness is
unaffected by the head's precision either way — a degraded draft is just
*rejected more often* by the verify pass, never wrong (see 4.2's bit-exact
result) — but a badly-quantized head defeats the entire point of doing
this (a low accept rate means little to no speedup). `COMPONENT_BIT_
RECIPES["jang"]` therefore carries dedicated `mtp_attention`/`mtp_moe_
shared`/`mtp_moe_routed_up`/`mtp_moe_routed_down`/`mtp_fusion` keys (8/8/
6/6/8 bit) instead of reusing `moe_routed_up`/`moe_routed_down`'s 4/3-bit
tier — ~4% of total size buys meaningfully better accept rate.
`inject_mtp_weights.py` reuses `mlx_convert_recipe.py`'s own component-mode
predicate builder (extracted to a module-level `make_component_quant_
predicate()` so both scripts share it) — every `mtp.*` path already
matches one of that predicate's existing checks (same submodule names as
the backbone) except `eh_proj` (the MTP-only embed/hidden fusion
projection), which the predicate now also handles.
