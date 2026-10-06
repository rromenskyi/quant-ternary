# Gemma 4 quantization — findings

> **Note 2026-10-06:** the MLX GPTQ builds here were converted by MLX
> re-quantization, which changes some GPTQ codes (group extremes unused after
> error feedback); see [`docs/GPTQ_EXACT_CODES.md`](../../docs/GPTQ_EXACT_CODES.md).

Real, load-bearing lessons from GPTQ-quantizing both Gemma 4 variants
(E4B multimodal; 26B-A4B MoE) to MLX, and building a GGUF from the 26B.
Everything here was hit in practice, not theorized.

## Models

| Variant | Repo | Modalities | Notes |
|---|---|---|---|
| E4B | `google/gemma-4-E4B-it` | text + vision + **audio** | Per-layer embeddings (`embed_tokens_per_layer`, 5.6GB bf16); KV-shared layers (`num_kv_shared_layers=18`) |
| 26B-A4B | `google/gemma-4-26B-A4B-it` | text + vision (**no audio**, `audio_config: null`) | 128-expert / top-8 MoE, hybrid dense+MoE every layer; `num_kv_shared_layers=0`; `attention_k_eq_v=true` |

Published (MLX, JANG-mixed attn 8-bit / ffn 4-bit, everything ≤8-bit, no bf16 left):
- `roman220220/gemma-4-E4B-it-gptq-mlx-jang` (~6.3GB)
- `roman220220/gemma-4-26B-A4B-it-gptq-mlx-jang` (~15GB, from 51.6GB bf16)

## JANG recipe (both models)

Mixed precision *by role*, not uniform bit width:
- **Attention** (`q/k/v/o_proj`) → 8-bit GPTQ. Small, disproportionately error-sensitive.
- **Feed-forward** (dense `gate/up/down_proj`, and 26B's 128 routed experts) → 4-bit GPTQ. The overwhelming majority of parameters.
- **Router** (26B MoE) → left untouched. Routing precision is disproportionately sensitive; quantizing it wrecks expert selection.
- **Embeddings** → 8-bit RTN (round-to-nearest). GPTQ's Hessian correction needs a shared-input matmul; a pure lookup table has none, so RTN, honestly labeled.

Calibration on REAL data: diverse text prompts, real COCO photos (vision),
real LibriSpeech clips (E4B audio). Not synthetic noise.

## Bug: vision tower silently NOT quantized (first E4B "8-bit" release)

The splice script's vision key prefix was `model.vision_tower.` but the real
checkpoint nests vision layers under `model.vision_tower.encoder.layers.N`.
All 112 vision corrections silently failed to match any real key → the
entire vision tower shipped as plain bf16 despite the "8-bit" name, and
~7.2GB of embeddings (dominated by `embed_tokens_per_layer`, 5.6GB) were
never in the calibration target list at all. **Always verify corrected keys
actually match real checkpoint keys before shipping** — a
matched/unmatched/sample-unmatched count would have caught this instantly.
Fixed prefixes + added embedding RTN; the old buggy repo was deleted.

## Bug: KV-shared layers crash with quantized KV cache (real production crash)

Gemma 4's KV-shared layers reuse an earlier layer's `(keys, values)`
directly, with no cache object of their own (`cache=None`). But
`scaled_dot_product_attention`'s quantized-attention dispatch keys off
`hasattr(cache, "bits")` — so once the source layer's cache crossed
`quantized_kv_start` and became a `QuantizedKVCache`, the shared layer got
a quantized `(packed, scales, biases)` tuple but was routed to the
unquantized SDPA path → `TypeError: ... Invoked with types: array, list,
list`. Fix (ipsupport-llc/mlx-lm#1, merged): thread the source layer's
actual cache object through to the shared layer so dispatch sees the real
cache. LLMTray also now forces `kv-bits=0` for any model with
`num_kv_shared_layers>0` as a belt-and-suspenders guard
(`ModelDiscovery.disallowsQuantizedKV`).

## 26B MoE architecture (verified directly against real checkpoint + transformers source)

- **Hybrid every layer**: each of the 30 decoder layers computes BOTH a
  dense MLP path and a routed-experts path, combined additively
  (`hidden_states_1 + hidden_states_2`). The router runs on the
  *pre-dense-MLP residual*. My earlier "layer 29 has no dense MLP"
  observation was a false positive from reading only one of the two
  safetensors shards — all layers have both.
- **Experts are raw stacked `nn.Parameter`**, not `nn.Linear` submodules:
  `experts.gate_up_proj [128, 1408, 2816]`, `experts.down_proj
  [128, 2816, 704]`, consumed inside a Python loop. So per-expert
  calibration hooks the whole `Gemma4TextExperts` module, replicates its
  real `expert_mask`/`token_idx` gathering to bucket rows per expert, then
  runs batched GPTQ across all 128 at once. `down_proj`'s calibration input
  is recomputed as `act_fn(gate)*up` from the real (uncorrected)
  `gate_up_proj`, exactly as the real forward feeds it.
- **`attention_k_eq_v=true`**: some attention layers genuinely have NO
  `v_proj` (v reuses k). Detect via the resolved submodule being absent,
  not a fixed layer pattern.
- **mlx-lm's `gemma4_text.py` already had full MoE support** (SwitchGLU
  Router/Experts, `sanitize()` splitting stacked `gate_up_proj` into
  `switch_glu.gate_proj`/`up_proj`) — E4B just never exercised it
  (`enable_moe_block=false` there). No new model code needed for 26B.

## Pod / infrastructure lessons (paid for in wall-clock)

- **`runpodctl pod update --container-disk-in-gb` WIPES `/workspace`.**
  Resizing the container disk recreates it from the image — the 26B
  checkpoint, all in-progress calibration batches, and every installed pip
  package were gone after the resize+restart. If you need more disk mid-run:
  transfer everything off first, or provision the bigger disk at pod
  creation. (Also: the SSH port changes after any pod restart — re-read
  `runpodctl pod get`.)
- **MLX runs on CUDA pods** (`pip install "mlx[cuda]" mlx-lm`), so the
  entire GPTQ *splice* (mx.quantize + re-shard) runs ON THE POD — no need
  to scp 47GB of corrected weights to a Mac. This was the single biggest
  time saver once realized; the earlier E4B run needlessly round-tripped
  through a Mac.
- **MLX streams are thread-local.** An `mx.array` built on a
  `ThreadPoolExecutor` worker can't be `mx.eval()`'d from the main thread
  ("no Stream(gpu/cpu, N) in current thread"), and it stays bound even
  after a worker-local eval. Solution: parallelize ONLY the pure
  torch→numpy prefetch on workers; do every MLX op (array/quantize/eval)
  on the main thread.
- **`mx.quantize` on CPU is ~100× slower than GPU** for the MoE splice
  (217s vs 2.2s per ~4GB shard). The op is elementwise (round+pack), so
  GPU utilization *looks* low (0%), but it's still far faster than CPU —
  keep `mx.set_default_device(mx.gpu)`.
- **`mx.quantize` only supports group_size ∈ {32, 64, 128}**, and it must
  evenly divide the input dim. Vision's `intermediate_size=4304` (=2⁴·269)
  divides by none of them. Rather than leave those tensors bf16, zero-pad
  `gate/up/down_proj` to 4352 (next multiple of 64) — mathematically exact
  (GELU(0)·0=0, padded activations hit zero-weighted `down_proj` columns),
  declared in `config.json`'s `vision_config.intermediate_size` so the
  model is built at the padded width with no inference changes.

## Bug: final splice overwrites the model card

The splice copies all non-weight files from the original checkpoint
verbatim (config, tokenizer, ...) — including the original `README.md`. If
you write your own card into the output dir BEFORE a splice re-run, the
re-run clobbers it with Google's original, which carries
`library_name: transformers` and no `mlx` tag. LLMTray's HF browser filters
by `filter=mlx`, so the model became invisible there. Fix: write the card
LAST (or via a separate `upload_file` after `upload_folder`), and never
between splice runs.

## GGUF path (shipped: roman220220/gemma-4-26B-A4B-it-GGUF-jang-imatrix)

Result: text `gemma4-26b-a4b-jang-iq3s.gguf` (~12.5GB, attention Q5_K /
experts IQ3_S / imatrix) + `mmproj-gemma4-26b-q8.gguf` (~0.8GB, vision).
Both verified by generation (PPL is unusable here — see below).

### GGUF gotchas hit in practice (all now handled by the pipeline)

- **Tokenizer conversion crash → silently broken chat.** transformers 5.17
  crashes converting this tokenizer: `extra_special_tokens` is a LIST
  (`["<|video|>"]`) but `_set_model_specific_special_tokens` calls `.keys()`
  on it. DON'T fix by deleting the field — with AutoTokenizer failing the
  converter mis-registers control tokens and chat/thinking breaks (raw
  completion still works, so it's easy to miss). Fix = convert to the DICT
  form (`{"video_token": "<|video|>"}`). See `gemma4_26b_fix_tokenizer.py`.
- **Control tokens tagged NORMAL → leak into chat as literal text.** Even
  with a good tokenizer, `convert_hf_to_gguf.py` tags Gemma 4's control
  tokens (`<|turn>`, `<turn|>`, `<|channel>`, `<channel|>`, `<|think|>`,
  `<|tool*>`, image/audio/video markers) as NORMAL/USER_DEFINED — it reads
  the type from the base vocab and ignores `special: true` on the
  added_tokens. They then decode to literal text and leak into output
  ("<|channel>thought..."); a correctly-tagged GGUF (stock q3km) keeps them
  hidden. `llama-cli` masked this partially; `ollama` showed it fully. Fix =
  surgical in-place byte patch of the `token_type` array (flip those ids to
  CONTROL=3) — no rewrite, no requant, instant. See
  `gemma4_gguf_fix_token_types.py`, wired into the pipeline after every
  quantize (`--from-tokenizer` derives the id list; nothing hardcoded).
- **Always launch GGUF with `--jinja`** or chat/thinking breaks the same way.
- **Ollama needs its native RENDERER/PARSER — a HF repo can't provide them.**
  Even with correct token_type, `ollama run hf.co/<repo>` still leaked
  `<|channel>thought … <channel|>` into chat. The official ollama `gemma4`
  manifest doesn't use a real Go template at all (`{{ .Prompt }}`); its config
  sets `"renderer": "gemma4", "parser": "gemma4"`, and the Go parser
  (ollama `model/parsers/gemma4.go`) is what strips thinking AND parses Gemma
  4's native tool-call syntax (`<|tool_call>call:name{…}<tool_call|>` with
  `<|"|>` string delimiters) — no Go template can replicate that. HF's ollama
  integration only reads `template` / `params` / `system` files. Fix: ship a
  `Modelfile` with `FROM hf.co/<repo>` + `RENDERER gemma4` + `PARSER gemma4`
  (both accepted by ollama's Modelfile parser, not yet in its docs);
  `gemma4_26b_gguf_pipeline.sh` step `ollama_files` generates it. The
  official gemma4's separate `gemma4-assistant` DRAFT model is shipped too —
  see "MTP drafter" below.
- **Diagnosing without the full file:** the `token_type` array is in the
  GGUF metadata at the front, so HTTP-range the first ~60MB and read it with
  `gguf.GGUFReader` (zero out `tensor_count` in the header copy first, else
  the reader chokes building truncated tensors).

## GGUF method notes

GGUF (llama.cpp) can't consume our GPTQ-corrected weights directly — its
k-quant super-block grid doesn't match `mx.quantize`'s affine grid, so the
Hessian correction is re-quantized away. The GGUF-world equivalent of our
calibration is **imatrix** (importance matrix) computed on the SAME real
corpus, plus **JANG expressed via `llama-quantize` per-tensor overrides**
(attention kept high, experts pushed lower). llama.cpp gained native
Gemma 4 (incl. MoE + vision mmproj) support only recently, in the
`conversion/gemma.py` module refactor — a prebuilt binary won't have it, so
llama.cpp must be built from current source (CUDA build, ~5-10 min on the
pod; the pipeline's `llama_build` step does it). See RUNBOOK.

## MTP drafter (speculative decoding) — Ollama + MLX

- **Ollama:** the official `gemma4:26b` draft layer is Google's
  `gemma-4-26B-A4B-it-assistant` as GGUF `Q8_0` (arch `gemma4-assistant`,
  462MB, sha256 `6326fb9f…`). Republished byte-identical in the GGUF repo;
  the Modelfile adds `DRAFT ./gemma4-26b-assistant-q8_0.gguf` +
  `PARAMETER draft_num_predict 3`. `DRAFT` only accepts a local file (not
  `hf.co/…`), hence the separate download. The GGUF pipeline's `drafter`
  step fetches it from the ollama registry by digest, with a checksum check.
- **MLX:** stock mlx-lm has no support. Added in ipsupport-llc/mlx-lm#3
  (`models/gemma4_assistant.py` + `generate.gemma4_mtp_generate_step`,
  used via `--draft-model`). How the drafter works: it has no KV cache of its
  own. All 4 layers attend to the main model's last full-attention and last
  sliding-attention KV. Input is `concat(main embed(tok)*sqrt(2816),
  main final normed hidden)`, the query sits at the fixed position of `tok`
  for every draft step, no mask. Output is tied-embedding logits plus
  `post_projection(h)`, which is fed back as the next step's hidden.
  Published 8-bit: `roman220220/gemma-4-26B-A4B-it-assistant-mlx-8bit`
  (446MB). Acceptance is the same as bf16.
- **Gotchas found:**
  - mlx-lm's generic speculative decoding can't run Gemma 4 past 1024
    tokens: sliding layers use `RotatingKVCache`, which is not trimmable once
    it wraps. The verify forward is always multi-token, and that leaves the
    rotating buffer in temporal order, so slicing its tail off is an exact
    rollback.
  - The generator must leave the cache at exactly prompt + yielded tokens
    when the consumer stops (at max_tokens or EOS, mid-drafts or at the
    bonus token). The server stores the cache under the emitted tokens and
    trims/extends it for the next request. Getting this wrong made every
    later request that reused the cache produce garbage. mlx's own
    `speculative_generate_step` does the same in a `finally`. **The
    Nemotron-H MTP step had the same hole:** it left the cache one token
    short (the bonus/correction token) on exit, so the server dropped the
    last token of every assistant reply from the multi-turn context.
    Confirmed by test and on the real 30B checkpoint, fixed in
    ipsupport-llc/mlx-lm#4.
  - With `kv_bits`, `maybe_quantize_kv_cache` swaps entries in the list
    it's given, so the step must use the caller's list, not a slice.
  - **Offline benchmarks with an empty cache can't show the server's case.**
    `mlx_lm.server` passes only the uncached tail of the prompt, with a
    cache that already holds the prefix (at least the chat template's
    opening tokens). The drafter's RoPE position was computed from the tail
    length alone, so it was off by the cached length and the drafter
    guessed wrong almost every time. It was +48% offline but did nothing in
    LLMTray (33.1 vs 32.5 tok/s, k=1 best, larger k slower: the signature
    of near-zero acceptance). Fixed in ipsupport-llc/mlx-lm#5.
    `gemma4_mtp_bench.py` now measures a cached-prefix scenario too, and
    fails below `--min-speedup`.
- **Numbers** (26B JANG, 26GB Mac, greedy, k=3): 35.8 → 52.9 tok/s short
  prompt, 32.7 → 41.7 at 3.9k context, ~70% of tokens drafted. Greedy
  output diverges from plain decoding only at near-ties (batched-verify vs
  single-token kernels).

## Bug: uncalibrated Linears shipped in bf16 (found by the smoke test)

GPTQ calibration only corrects the Linears it hooks, and the splice only
RTN'd embeddings. Every other quantizable Linear was copied through as bf16,
which breaks the "nothing left in bf16" rule without anything noticing. On the
**published E4B JANG**: 14 audio-tower tensors (`relative_k_proj` ×12,
`output_proj`, `subsample_conv_projection.input_proj_linear`, ~30MB), plus
~110MB of dead bf16 `k_proj`/`v_proj`/`k_norm` for KV-shared layers 24–41,
which mlx-lm's sanitize() drops at load anyway. On the **published 26B**:
`embed_vision.embedding_projection` (3.2M elements). Routers are
bf16 on purpose.

Fix (`splice_common.py`, both splices): `--rtn-leftovers-bits 8` RTN's
every leftover whose module is quantizable **in mlx-lm's own module tree**
(built lazily from config.json). Shape alone isn't enough: a 2D raw
parameter that isn't a Linear breaks at load if its weight is packed.
`--drop-kv-shared-dead` omits the dead tensors. Validated on a copy of the
published E4B with 14 RTN'd and 54 dropped: text unchanged, vision "two
cats", audio transcription word-for-word identical, audio embeddings
cosine ≥ 0.9997 vs the bf16 original.

**Decision (2026-09-23): not republished, not worth a pod run on its own.**
What the published repos actually carry:

| Repo | Real leftover | Intentional float | Dead on disk |
|---|---|---|---|
| E4B JANG | 14 audio tensors, ~30MB (≈0.5% of 6.3GB) | `patch_embedder.input_proj` 0.6M | 54 KV-shared k/v/k_norm, ~110MB (never loaded) |
| 26B JANG | `embed_vision.embedding_projection` 6.5MB (~3MB saved at 8-bit, of 15GB) | routers 21.6MB, `patch_embedder.input_proj` 1.8MB | none |

No measurable speed/memory/quality gain; bf16 is if anything slightly more
precise. The pipeline now handles it automatically, so the next real
re-release (new recipe / calibration) ships clean. Until then the live HF
cards still say "Nothing is left in bf16"; the corrected wording is in
`cards/` and goes out with that re-release.

**Trap: `vision_tower.patch_embedder.input_proj` must stay float.**
mlx-lm's `gemma4_vision.PatchEmbedder` casts pixels to
`input_proj.weight.dtype`. Once quantized, that dtype is uint32, and the
model answered "There are no discernible animals" on the cats photo.
The pipelines pass `--keep-float 'patch_embedder\.input_proj'` (and
`'router\.proj'` on 26B) to both the splice and the smoke test; nothing is
hardcoded.

`gemma4_smoke_test.py` fails any release with a large (≥100k elements) float
weight not covered by `--keep-float`. That's how both findings surfaced.

## mx.quantize: valid parameters

Bits ∈ {2, 3, 4, 5, 6, 8} (1 and 7 raise), group_size ∈ {32, 64, 128}
(16/48/80/96/112 raise), and the group must divide the input dim. LLMTray's
KV-cache settings used to offer 0…8 bits and 16…128-step-16 groups; the
invalid ones crashed the server on the first quantized cache
(ipsupport-llc/llmtray#4, now off / 4 / 8 and 32 / 64 / 128).

## Pipelines + dashboard

Everything above is automated in three resumable pipelines. They share one
step log, and `pipeline_dashboard.py` shows them live. Commands are in
[RUNBOOK.md](RUNBOOK.md).

## Gemma 4 E2B: when NOT to quantize — the vendor already shipped QAT (2026-09-27)

We set out to quantize `google/gemma-4-E2B-it` to GGUF + MLX for "max quality
at min size". The useful result was a **negative** one: for this model our
quantization is redundant, because Google shipped a **quantization-aware-trained
(QAT)** release of the whole model and the community has already ported it to
every format we'd target. Worth writing down so the next popular small model
gets a 30-second check before a pod is rented.

### What exists already

- `google/gemma-4-E2B-it-qat-q4_0-gguf` — Google's QAT, 4-bit, whole model
  (~3.12 GB text). Also `...-qat-q4_0-unquantized` (the QAT weights dequantized
  back to bf16).
- `mlx-community/gemma-4-E2B-it-qat-{4,5,6,8}bit` (+ mxfp4/nvfp4/bf16, and
  `-assistant-*` for the MTP drafter) — the QAT checkpoint already in MLX, at
  every bit width. So the MLX niche (our LLMTray target) is covered too.

### Why our PTQ can't win here

- **QAT beats PTQ by construction, and only the vendor can do it.** QAT
  fine-tunes with fake-quant in the forward pass so the weights *learn* to be
  4-bit-robust; PTQ (GPTQ/imatrix — what our pipelines do) only picks rounding
  after the fact. No calibration recipe recovers what QAT bakes in.
- **JANG buys nothing on a dense model.** JANG's real lever is MoE: spend bits
  asymmetrically because ~93% of params are routed experts (26B / Nemotron —
  where JANG beat uniform and third-party releases). E2B is dense, so JANG
  degenerates to a mild "attn 8 / ffn 4" mix with no structural win.
- **QAT robustness is tied to its own grid.** Even quantizing the
  `qat-q4_0-unquantized` weights ourselves (GPTQ/JANG, an affine/mixed grid)
  wouldn't reliably carry the QAT benefit, which was trained against the q4_0
  grid specifically.

### The perplexity red herring (and how it was diagnosed)

Raw wikitext PPL on this model in llama.cpp comes out absurdly high, which
looked like a bug. It is not ours:

| model of `google/gemma-4-E2B-it` (same binary, wiki.test.raw, -c 512, 20 chunks) | PPL |
|---|---|
| ggml-org **official** Q8_0 (near-lossless reference) | 233 |
| our Q4_K_M + imatrix + JANG | 215 |
| our F16 | 154 |
| Google **QAT** q4_0 (different, QAT-trained checkpoint) | 69 |

Our build matches (slightly beats) the official ggml-org GGUF of the same base
→ **our quantization is correct.** The ~230 absolute is a property of the
metric on an instruct/multimodal model tuned for chat, not a defect (instruct
tuning inflates raw-text PPL). The QAT release scoring 69 is a *different
checkpoint*, not a like-for-like comparison — do not read it as "q4_0 format
beats Q4_K_M" (it doesn't; K-quants win at equal bits). Judge such models by
generation, not raw PPL.

### Process rule

Before quantizing a **popular** model, check for (a) a vendor QAT release and
(b) existing community MLX/GGUF builds. If both exist and cover your formats,
your effort is better spent on models **without** that coverage — our own
fine-tunes (ipsupport-code LoRA), MoE models where JANG actually wins
(Nemotron, 26B), or niche/larger models the community hasn't done well.

### What was kept (reusable, for non-QAT models)

- `gemma4_mlx_pipeline.sh VARIANT=e2b` (reuses the E4B multimodal path;
  MODEL_ID/HF_REPO env-overridable).
- `gemma4_e2b_gguf_pipeline.sh` — a dense (no-MoE, no-drafter) GGUF pipeline
  with an env-driven recipe and a wikitext-PPL step.
- SETUP now installs `datasets`/`soundfile` (the audio-calibration path needed
  them; only worked on E4B before because they were preinstalled).
- mmproj Q8_0 on E2B hits a llama.cpp `GGML_ASSERT` (audio conv1d tensor) →
  the pipeline falls back to an F16 mmproj.

Nothing was published for E2B.

## Gemma 4 QAT in MLX on the q4_0 grid (2026-09-30, in progress)

The 2026-09-27 conclusion ("the vendor's QAT is already in MLX, don't
quantize") was half right. The QAT is the thing to ship, but the MLX ports
of it are not on the grid it was trained for. LLMTray's 8 GB pick was
`mlx-community/gemma-4-e2b-it-4bit`, a plain RTN 4-bit with no QAT at all.

### What the QAT release is (checked)

- `google/gemma-4-*-it-qat-q4_0-gguf` is **llama.cpp's q4_0 of the
  `-qat-q4_0-unquantized` master weights, bit for bit**. Every block's fp16
  `d` and every code matched on E2B (`blk.0.attn_q`, `blk.0.ffn_down`,
  `blk.20.attn_output`; then all 275 Q4_0 tensors in the converter).
- The "unquantized" weights are therefore *not* on the grid themselves:
  only ~26% of blocks round-trip unchanged (`qat_grid_check.py`). They are
  the master weights QAT kept, and q4_0 of them is the model Google tuned.
- Q4_0 in the GGUF = exactly the text decoder's Linears: q/k/v/o, MLP
  gate/up/down, per-layer input gate and projection. 275 on E2B, i.e.
  35 layers × 7 + 15 KV layers × 2. The embeddings are Q6_K. The
  vision/audio towers are a separate F16 mmproj and were never QAT-trained
  (0% of their blocks on the grid).

### Why the community MLX ports miss

- `mlx-community/gemma-4-*-it-qat-4bit`: affine 4-bit, **group 64**, and on
  E2B **all 105 MLP Linears at 8-bit**. Group 64 can't hold two q4_0 blocks'
  scales, so even the 4-bit part is refitted per group, off the QAT grid.
  It is also not a 4-bit model: 4.33 GB (E2B), 28.8 GB (31B).
- A plain `mlx_lm.convert -q --q-group-size 32` has the right group but
  fits each group's min/max, not q4_0's `d = absmax_signed / -8`.

### The conversion (`poc/qat_aligned_convert.py`)

MLX affine 4-bit, group 32: `x = scale·q + bias`. With `scale = d`,
`bias = -8d`, and q4_0's codes packed into uint32 (8 per word, value j at
bits 4j), this is q4_0's grid exactly. The packing was checked with
`mx.dequantize` (diff 0.0).
- One deliberate loss: MLX keeps scales in the model dtype (bf16), so the
  fp16 `d` is rounded to bf16. That is under 1/64 of a step per weight.
  fp16 scales are accepted, but every quantized matmul then returns float32.
- `--gguf` compares every q4_0 tensor's `d` and codes with Google's GGUF and
  fails on any difference. E2B: 275/275 identical.
- Everything else follows Google's split:
  - embeddings: RTN 6-bit, group 64 (≈ Q6_K);
  - vision/audio towers: RTN 8-bit, which halves Google's F16 mmproj;
  - norms: copied.
- E2B: **3.94 GB**. Google's GGUF + mmproj is 4.34 GB; mlx-community
  qat-4bit is 4.33 GB.

### E2B quality (`poc/qat_eval.py`, wikitext-2 test, 64 × 512, BOS per window)

| model | size | PPL | KL to google bf16 | top-1 agree |
|---|---|---|---|---|
| google/gemma-4-E2B-it bf16 (not QAT) | 10 GB | 200.9 | — | — |
| QAT master weights, bf16 | 10 GB | 63.6 | 0.293 | 80.0% |
| **ours, q4_0 grid** | **3.94 GB** | 68.3 | **0.330** | **78.3%** |
| mlx-community qat-4bit (MLP at 8-bit) | 4.33 GB | 63.7 | 0.363 | 76.8% |
| mlx-community 4-bit RTN (LLMTray's 8 GB pick until now) | 3.6 GB | 231.6 | 0.646 | 69.1% |

- Ours at 68.3 sits where Google's q4_0 GGUF measured (69, the 09-27 llama.cpp
  run). It *is* that model.
- mlx-community's lower PPL buys MLP at 8-bit with 0.4 GB more, yet it
  agrees less with the original (KL, top-1).
- Plain RTN 4-bit is far behind every QAT variant.

A first run without BOS per window gave PPL in the thousands for every
model: Gemma needs BOS at the start of each window, as llama.cpp's
perplexity provides.

### All four models (2026-09-30)

| model | q4_0 tensors = GGUF | ours | mlx-community qat-4bit |
|---|---|---|---|
| E2B | 275/275 | 3.94 GB | 4.33 GB |
| E4B | 342/342 | 5.85 GB | 6.80 GB |
| 12B (`gemma4_unified`) | 328/328 | 7.74 GB | 10.99 GB |
| 31B | 410/410 | 20.21 GB | 28.8 GB |

The 31B vision MLP is 4304 wide, divisible by neither 64 nor 32. Such
tensors stay bf16: the converter now picks the widest group that divides
the row, or none. The 12B's vision/audio projections aren't quantizable
modules in mlx-lm, so they're copied (mlx-lm drops them at load).

### Against the QAT master weights (128 × 512, BOS per window)

KL and top-1 agreement are measured to the bf16 QAT master weights, the
model the quantization should reproduce. The non-QAT original is a
different checkpoint: even the master weights are 0.29 KL away from it
(E2B), so that reference mostly measured the QAT fine-tune itself.

| E2B | size | PPL | KL to QAT master | top-1 |
|---|---|---|---|---|
| QAT master bf16 | 10 GB | 66.1 | — | — |
| ours, q4_0 grid | 3.94 GB | 70.9 | **0.054** | **89.1%** |
| mlx-community qat-4bit (MLP 8-bit) | 4.33 GB | **66.2** | 0.067 | 87.6% |
| mlx-community 4-bit RTN | 3.6 GB | 239.7 | 0.853 | 66.3% |

| E4B | size | PPL | KL to QAT master | top-1 |
|---|---|---|---|---|
| QAT master bf16 | 15 GB | 42.4 | — | — |
| **ours, q4_0 grid** | **5.85 GB** | **44.3** | **0.041** | **90.5%** |
| mlx-community qat-4bit | 6.80 GB | 45.4 | 0.055 | 89.0% |
| ours, GPTQ JANG (from the non-QAT original; LLMTray's 16 GB pick) | 6.8 GB | 82.3 | 0.369 | 78.3% |

- E4B: the grid conversion beats mlx-community on every metric with 1 GB less.
- The old GPTQ's KL isn't comparable, since it was made from another
  checkpoint. Its PPL and agreement are still well behind.
- E2B: closer to the master (KL, top-1), but higher PPL. That gap is what
  the sensitivity scan below closes.
- 12B: suspect, not used. The master itself scores PPL 684, and KL is 0.55
  for both quantizations. The `gemma4_unified` text path in the pinned
  mlx-lm doesn't behave on raw text; investigate before trusting any 12B
  number.

### QAT + a few Linears at 8-bit (`poc/qat_sensitivity.py`, E2B)

Each of the 277 candidates was raised to 8-bit (group 64) on its own, from
the all-q4_0 model:
- 275 text Linears;
- the 2 embeddings, 6 → 8 bit.

Each one was scored by the KL it removes on 16 windows, per MB it adds,
then taken greedily under a size budget and measured on the full 128
windows. Built in memory from the master weights.

| budget | raised | text part | PPL | KL | top-1 |
|---|---|---|---|---|---|
| 0 (pure q4_0) | 0 | 3.40 GB | 70.9 | 0.054 | 89.2% |
| **+50 MB** | 99 | 3.45 GB | **64.1** | **0.034** | **91.3%** |
| +100 MB | 126 | 3.50 GB | 64.5 | 0.030 | 91.8% |
| +200 MB | 166 | 3.60 GB | 65.7 | 0.025 | 92.8% |
| +400 MB | 201 | 3.80 GB | 65.8 | 0.018 | 94.0% |
| +800 MB | 245 | 4.20 GB | 65.6 | 0.012 | 95.4% |

- With +50 MB the model beats mlx-community's qat-4bit on all three
  metrics: PPL 64.1 vs 66.2, KL half, top-1 91.3% vs 87.6%. It is still
  ~0.35 GB lighter.
- What earns the bits isn't the big MLPs, which mlx-community raised
  wholesale, but ~100 small sensitive Linears.
- PPL below the master's (64.1 < 66.1) is noise in raw-text PPL. KL and
  top-1 are the measures of fidelity.

Past +100 MB, KL and top-1 keep improving while PPL stays flat. +100…200 MB
is the knee.

### One pipeline for every size (`poc/qat_mlx_pipeline.sh`)

`MODEL=E2B|E4B|12B|31B [BUDGET_MB=100] [BASELINES=...] [PUBLISH=1 HF_REPO=...]`.
Steps:
1. download Google's master weights and GGUF;
2. scan;
3. convert (q4_0 checked against the GGUF, plus the budget point's Linears
   at 8-bit via `--raise-json/--raise-budget-mb`);
4. eval against the QAT master;
5. card, then publish.

Every step resumes. The scanner keeps the master weights in host memory
and builds one candidate's 8-bit version at a time, so a 31B fits an 80 GB
GPU.

### E2B from disk (`MODEL=E2B BUDGET_MB=100`)

| E2B | size | PPL | KL to QAT master | top-1 |
|---|---|---|---|---|
| QAT master bf16 | 10 GB | 66.1 | — | — |
| **ours, q4_0 grid + 126 Linears 8-bit** | **4.04 GB** | **64.5** | **0.030** | **91.8%** |
| mlx-community qat-4bit | 4.33 GB | 66.2 | 0.067 | 87.6% |

- The converter wrote what the scan measured in memory: 64.48 vs 64.47.
- The 149 Linears left at 4-bit still carry the GGUF's codes (scales rounded to bf16).

### E2B speed on the M5 (`poc/qat_speed.py`, mlx-lm `be4b6c7`, 2 rounds, order alternated, 15 s idle before each)

| E2B | decode | prefill | peak memory | PPL |
|---|---|---|---|---|
| **ours, q4_0 grid + 126 Linears 8-bit** | **67.5 tok/s** | 1224 tok/s | 4.13 GB | **64.5** |
| mlx-community qat-4bit | 54.7 tok/s | 1052 tok/s | 4.42 GB | 66.2 |
| mlx-community 4-bit RTN (LLMTray's 8 GB pick) | 83.7 tok/s | 1365 tok/s | 3.64 GB | 239.7 |

- Ours is 23% faster than mlx-community's qat build, which has every MLP at
  8-bit.
- Ours is 19% slower than plain RTN, the price of group 32 and the raised
  Linears, at a quarter of its PPL.

### Loading in LLMTray's runtime (mlx-lm fork `97b75f3`)

- mlx-community's `gemma-4-E2B-it-qat-4bit` **doesn't load**: `Expected
  shape (128, 3, 3, 1) but received (128, 3, 1, 3)` for
  `audio_tower.subsample_conv_projection.layer0.conv.weight`.
- Ours failed the same way at first: the converter tagged its files
  `{"format": "mlx"}`, so sanitize() took the raw (torch-layout) audio convs
  as already in MLX layout. The pod's fork (`be4b6c7`) loaded them anyway.
  Fixed: no format tag, the convention the GPTQ splice always followed.
  Now it loads and answers in LLMTray's runtime.

### 12B: a chat-only checkpoint, not an mlx-lm bug

The 12B QAT master scores raw text like noise: PPL 684. After "The capital
of France is" its top tokens are digits; after "... Paris. The capital of
Germany is" it says " Paris". **transformers gives the same tokens**, with
BOS, on the same weights, so this is the checkpoint, not mlx-lm's
`gemma4_unified`. With the chat template the first token is `<|channel>`,
as it should be.
- Raw-text PPL/KL are meaningless for it.
- `--chat` (eval and scan, `CHAT=1` in the pipeline) scores the text as the
  model's reply to "Continue this text.", over the text's own tokens only.

Next:
- the chosen recipe written by the converter and re-measured from disk;
- the same scan on E4B and 31B, where 31B also needs to fit a 32 GB Mac
  (~21 GB GPU limit);
- speed on the M5.

### 31B (2026-10-01, pod `uaaawubmr2qf66`, A100 80 GB)

`MODEL=31B BUDGET_MB=400`, raw-text scan (no `CHAT=1`), 128 × 512 windows.

| build | size | PPL raw | KL vs master | top-1 agree |
|---|---|---|---|---|
| master (bf16, `-qat-q4_0-unquantized`) | 62 GB | 1786 | -- | -- |
| q4_0 grid, nothing raised | 19.45 GB | 1734 | 0.1007 | 87.66% |
| +50 MB (7 Linears 8-bit) | | 1742 | 0.0935 | 88.39% |
| +100 MB (11) | | 1735 | 0.0932 | 88.45% |
| +200 MB (19) | | 1742 | 0.0925 | 88.53% |
| **+400 MB (36), from disk** | 19.85 GB | 1696 | **0.0905** | **88.62%** |
| +800 MB (62) | | 1646 | 0.0827 | 89.30% |
| +400 MB, `--chat` eval | | 1600 (master 1592) | 0.0789 | 89.72% |

- **31B is chat-only like the 12B**: the master itself scores PPL ~1600-1800
  on raw text, also with the `--chat` framing ("Continue this text.").
  Raw-text PPL says nothing here; KL and top-1 against the master do.
- **It answers sensibly** from the +400 MB build (mlx-lm on CUDA, chat
  template): "The capital of France is Paris." with its thought channel;
  a correct three-sentence Rayleigh explanation. Long-context output (a
  1500-token prompt) and per-position PPL are still to check -- the pod was
  stopped before that run (`/tmp/pos.py`, to redo).
- The 8-bit raising buys less than on E2B/E4B: KL 0.101 -> 0.091 for 400 MB
  (E4B: 0.041 -> 0.026 for 100 MB). The scan ran on raw text; for a
  chat-only checkpoint `CHAT=1` should order the Linears better -- redo the
  scan with it before publishing.
- 19.85 GB of weights: over a 32 GB Mac's ~21 GB GPU limit once the KV
  cache is added; it's a 48 GB+ model (or raise `iogpu.wired_limit_mb`).

Pipeline trouble on the way, for the runbook:
- **The volume quota (250 GB) ran out mid-convert** and the process died
  without a word in the pipeline log twice (`mx.save_safetensors`: "Unable
  to write ... bytes" only shows when run in the foreground). `df` shows the
  whole MooseFS cluster (118 T free), not the quota: check with
  `du -sh /workspace`. Freed: the old E4B GPTQ build, E4B r0, the E4B GGUF,
  mlx-community baselines (all superseded or re-downloadable).
- The cgroup sat at its 250 GB memory limit (page cache, `memory.events`
  max 5912) with no OOM kill: not the cause.

### Why the big Gemma 4 `-it` score raw text in the thousands (2026-10-01)

Three independent implementations, the same 31B wikitext windows (512
tokens, BOS):

| implementation | weights | PPL |
|---|---|---|
| llama.cpp `llama-perplexity` (built on the pod, CPU) | Google's own q4_0 GGUF | 2468 |
| HF transformers 5.17 | bf16 `-qat-q4_0-unquantized` master | 2640 |
| mlx-lm (our fork) | the same master / our r400 build | 1786 / 1696 |

- **Not our conversion, not mlx-lm**: Google's GGUF in llama.cpp and the
  master in transformers score the same.
- **The checkpoint**: it's every Gemma 4 with `attention_k_eq_v: True` (12B,
  26B, 31B); E2B and E4B (`False`) score 44-66. 26B-A4B-it -- not even QAT,
  our GPTQ build -- scores 73k on the same windows.
- The config and weights agree (no `v_proj` in the full-attention layers, no
  separate `lm_head`: tied, as configured).
- **What it does on raw text**: it falls into attractor tokens. After
  "Fraser Ayres , Sophie Stanton" it puts 0.58 on " same" and 0.31 on " own"
  (" and" is the text); " own" and `<|channel>` come up everywhere; 6-8% of
  the probability sits on control tokens. The start of a window is fine
  ("= Robert Boulter =" -> " Robert", 0.72); then it drifts.
- **In chat it's sound**: coherent answers, an accurate summary of a
  1500-token wikitext passage.
- So for these models a raw-text PPL is no quality measure, and the "chat"
  framing ("Continue this text.") is only a little better (1592). KL and
  top-1 to the master stay valid comparisons of a quantization; a PPL that
  means something needs chat-formatted text (real assistant replies scored).
- The chat-framed 31B scan was stopped at 100/411 (RunPod balance) and then
  cancelled for this check; the raw-scan r400 build stands
  (`31B-qat-mlx-r400-raw` on the pod volume).

### 31B, chat-framed scan with the thinking channel closed (2026-10-02)

The `--chat` framing left the model's turn open for its reasoning; with
`enable_thinking=False` (`<|channel>thought\n<channel|>`) it scores the text
as the reply (`qat_text.py`). Same wikitext windows, 26B: raw 67363, the old
`--chat` 82867, the fixed one 26.7; E4B: raw 46.5, fixed 22.6. Then the 31B
scan again, from scratch (`CHAT=1`, 128 eval windows, resumable now):

| build | PPL | KL to master | top-1 |
|---|---|---|---|
| master (bf16) | 26.94 | -- | -- |
| q4_0 grid, nothing raised | 27.69 | 0.0213 | 94.89% |
| +50 MB (6 Linears 8-bit) | 27.80 | 0.0211 | 94.87% |
| +100 MB (12) | 27.77 | 0.0209 | 94.87% |
| +200 MB (20) | 27.70 | 0.0205 | 94.89% |
| **+400 MB (35), from disk** | **27.82** | **0.0206** | **94.93%** |
| +800 MB (64) | 27.76 | 0.0194 | 95.10% |

- The real faithfulness of the build: KL 0.02, top-1 95% -- E4B's league,
  not the 0.09 / 88.6% the broken framing reported.
- **8-bit Linears buy little on the 31B**: the whole curve sits within KL
  0.019-0.021; Google's q4_0 grid is already close to the master. The
  raw-text scan's r400 measured KL 0.0222, top-1 94.87% (PPL 24.1 vs the
  master's 23.2, 32 windows): about the same.
- PPL across different models isn't comparable with this test: the
  instruct models are badly calibrated, the bigger more confident, so the
  mean is carried by confident misses (26B's median token PPL 3.16 vs E4B's
  5.84, while its mean PPL is higher). Against its own master it is a fair
  measure.
- Pod time: the scan ran 22:19-06:02 UTC on an A100 (~$12).

### E2B for phones (2026-10-02)

**Our recipe, step by step** (`mobile_sweep.sh`, 128 chat windows, master PPL 36.04):

| build | PPL | KL | top-1 |
|---|---|---|---|
| published r100 (4.04 GB) | 36.38 | 0.0227 | 92.73% |
| r0 (no 8-bit Linears) | 37.52 | 0.0382 | 90.40% |
| r0 + PLE 4-bit | 37.70 | 0.0397 | 90.31% |
| + token embeddings 4-bit | 37.85 | 0.0480 | 89.27% |
| + towers 4-bit | 37.89 | 0.0481 | 89.28% |
| + PLE 3-bit | 37.73 | 0.0586 | 88.29% |

- The +100 MB of 8-bit Linears halve E2B's KL: keep them.
- **PLE 6 -> 4 bit is almost free** (KL +0.0015, ~0.6 GB): half of E2B's
  size is its per-layer embeddings (1.91 GB at 6-bit).
- Token embeddings below 6-bit cost (they're tied to the output head);
  PLE 3-bit costs. The towers don't show in a text eval.

**Google's mobile QAT** (`google/gemma-4-E2B-it-qat-mobile-transformers`,
"wNa8o8"): per-row symmetric int2/int4/int8, MLP layers 15+ and the token
embeddings / an untied lm_head at 2-bit, attention 4-bit, PLE 4-bit, the
audio tower 2-bit; static int8 activation scales on every Linear and KV-cache
scales. `qat_mobile_convert.py` maps it onto MLX exactly (the packed bytes are
MLX's words; scales to bf16): 2.57 GB, 122.5 tok/s and 2.43 GB peak on the M5
(our q4_0 E2B: 70.3 tok/s, 3.80 GB).
- The conversion matches transformers: KL 0.0021, top-1 97.4% (bf16
  activations both), PPL 65.03 vs 65.06 (32 windows).
- **But it needs its int8 activations**: transformers with them 41.26, without
  65.06 (KL 0.86 between the two); without them the audio tower hears nothing
  ("Please provide the audio file..."); vision still works.
- So it's a scheme for a phone NPU runtime (Google ships it as LiteRT-LM).
  In MLX it would need activation fake-quant in the model code, in llama.cpp
  there's none -- and even with it, 41 vs our recipes' ~36.5. Not published;
  our own recipe (PLE 4-bit, 8-bit Linears kept, towers 4-bit) is the phone
  build, in MLX and GGUF.

**The phone build** (`e2b_phone.sh`: PLE 4-bit, token embeddings 6-bit, the
+100 MB 8-bit Linears, towers 4-bit):
- MLX: 3.10 GB (r100 4.04), PPL 36.44, KL 0.0242, top-1 92.48% (128 chat
  windows; r100 36.38 / 0.0226 / 92.73%). -0.94 GB for KL +0.0016.
- GGUF (llama.cpp, raw wikitext, 32 chunks of 512, against our bf16 GGUF of
  the master, PPL 42.84 in the KL run's scoring -- the ratios below are to
  it; plain `llama-perplexity` puts it at 43.63): text `Q4_0`, `per_layer_token_embd` `Q4_K`,
  `token_embd` `Q6_K`, the raised Linears `Q8_0` via `--tensor-type`
  (`gguf_types.py` maps the MLX scan's modules to GGUF names).
  | GGUF | size | PPL | to bf16 |
  |---|---|---|---|
  | Google's `gemma-4-E2B_q4_0-it.gguf` | 3.35 GB | 46.47 | x1.085 |
  | **ours** | **2.86 GB** | **42.24** | **x0.986** |
  - all 149 of our `Q4_0` tensors are byte-identical to Google's (the same
    QAT grid); the gain is the `Q8_0` Linears, PLE `Q4_K` costs ~nothing.
  - llama-perplexity's KLD lines didn't match the grep; only PPL recorded.

### Review of the pipeline (2026-10-02, Codex on PR #18)

- `qat_mlx_pipeline.sh` reused any `<MODEL>-sens.json`: a raw-text scan
  could pick a `CHAT=1` build's Linears. The scan JSON (and its `.partial`)
  now records its settings (master, text, ctx, windows, embed/raised bits,
  chat); the pipeline reuses it only if chat, the window counts, the master
  and the text match --
  an older scan without settings is refused (move it aside).
- The resume compared only the base KL, which a different `--high-bits`
  doesn't change: it now needs the same settings too.
- The cards said the 4-bit weights are "identical, bit for bit" to Google's
  GGUF; the codes are, the fp16 scales are rounded to bf16. Fixed in
  `qat_card.py`, the five MLX cards here and on the Hub.

### Published: 12B, text + vision + audio (2026-10-02)

- **[roman220220/gemma-4-12B-it-qat-mlx](https://huggingface.co/roman220220/gemma-4-12B-it-qat-mlx)**:
  7.89 GB, `MODEL=12B CHAT=1 BUDGET_MB=200` (44 Linears at 8-bit: 43
  attention -- 17 V, 15 K, 10 O, 1 Q -- and 1 MLP). 128 chat windows:
  | build | size | PPL | KL | top-1 |
  |---|---|---|---|---|
  | master | -- | 22.26 | -- | -- |
  | +0 MB | 7.63 GB | 22.92 | 0.0307 | 92.78% |
  | **+200 MB (from disk)** | **7.89 GB** | **22.77** | **0.0253** | **93.44%** |
  | +800 MB (scan) | | 22.56 | 0.0198 | 94.14% |
  | mlx-community qat-4bit | 10.99 GB | 22.82 | 0.0259 | 93.35% |
  | mlx-community 4bit (no QAT) | 6.74 GB | 26.70 | 0.340 | 78.78% |
  - Same faithfulness as mlx-community's QAT build at 3.1 GB less (theirs
    keeps all 144 MLP Linears at 8-bit), ~1.5x faster decode on the M5
    (14.8 vs 9.8 tok/s, 9.2 vs 6.4 in a throttled pair), peak 8.1 vs 11.2 GB.
- **Multimodal.** The 12B is `gemma4_unified`: no towers -- raw 48 px patches
  through a `vision_embedder` (LN -> Linear -> LN, + factorized 2D posemb, LN)
  and raw 640-sample audio frames, each via embed_vision / embed_audio. Our
  mlx-lm fork loaded it text-only and dropped those weights; ported in
  ipsupport-llc/mlx-lm#22 (`17ab9af`), LLMTray runtime pin bumped (#214).
  - Bug found on the way: the pixels were cast to `patch_dense.weight.dtype`,
    uint32 once quantized -- every pixel became 0 ("no animals in this
    picture"). Features now match mlx-vlm's on the same checkpoint (cos >=
    0.9998, bf16); preprocessing identical.
  - The 12B uses bidirectional attention within an image; without the image
    spans set (`set_vision_spans`, as the server does) an encoder-free model's
    patches never see each other. `gemma4_smoke_test.py` sets them now.
  - The converter quantizes the three projections at 8-bit (`other`) once the
    fork has them -- the pod's venv must carry the fork before `convert`.
- Smoke (M5): text; vision "There is a red fox in this picture."; audio: it
  recognizes the spoken pangram but discusses it instead of transcribing
  (the smoke check looks at the whole output); peak 8.4 GB.
- Pod volume: freed ~55 GB (31B build/GGUF, E2B bf16 GGUF/KLD, phone builds,
  old raw 12B build); the existing mlx-community 12B copy was renamed to the
  pipeline's baseline dir instead of downloaded again.

### Published: E2B phone (2026-10-02)

- **[roman220220/gemma-4-E2B-it-qat-phone-mlx](https://huggingface.co/roman220220/gemma-4-E2B-it-qat-phone-mlx)**:
  3.10 GB. `gemma4_smoke_test.py` on the M5 passed with the 4-bit towers:
  text; vision "A red fox is in this picture."; audio word for word; peak
  4.2 GB. Decode 70.2 tok/s, peak 3.26 GB (the q4_0-grid build: 70.3 tok/s,
  3.80 GB) -- the PLE bits save memory, not time.
- **[roman220220/gemma-4-E2B-it-qat-phone-GGUF](https://huggingface.co/roman220220/gemma-4-E2B-it-qat-phone-GGUF)**:
  2.86 GB text + Google's `gemma-4-E2B-it-mmproj.gguf` unchanged (0.99 GB).
  llama.cpp CPU build on the pod (`-DGGML_CUDA=ON` failed: no nvcc in PATH):
  text, vision (a fox as "a small mammal with reddish-orange fur") and audio
  (word for word) through `llama-mtmd-cli --jinja`.
- Cards: `cards/gemma-4-E2B-it-qat-phone-{mlx,GGUF}.md`, hand-written
  (`qat_card.py` is the q4_0-grid card).
- The pod's host had no free GPU for ~3 h after the stop (a 0-GPU resume is
  refused too); a retry loop restarted it.

### Published: 31B (2026-10-02)

- **[roman220220/gemma-4-31B-it-qat-mlx](https://huggingface.co/roman220220/gemma-4-31B-it-qat-mlx)**:
  20.65 GB, +400 MB recipe from the chat scan (35 Linears at 8-bit, almost
  all attention K/V); PPL 27.82 vs the master's 26.94, KL 0.0206, top-1
  94.93% (128 chat-framed windows).
- `gemma4_smoke_test.py` passed before upload: text; vision ("There is a red
  fox in this picture."); peak 23.0 GB on the A100. Its unquantized check
  needed `--keep-float` for the vision tower's 27 MLP down projections
  (width 4304: not a multiple of MLX's group sizes; ~270 MB bf16, said on
  the card).
- On the pod: Pillow was missing from the venv (transformers' Gemma 4 image
  processor needs it), and the HF token lives on the container disk, which a
  pod restart wipes -- copy it again over ssh stdin before any upload.
- `qat_card.py` grew `--no-audio`, `--chat-only`, `--memory-note`,
  `--float-note` for it.

### Published (2026-09-30)

- **[roman220220/gemma-4-E2B-it-qat-mlx](https://huggingface.co/roman220220/gemma-4-E2B-it-qat-mlx)**:
  new repo, 4.07 GB with its tokenizer (weights 4.04 GB), +100 MB recipe (126 Linears at 8-bit).
- **[roman220220/gemma-4-E4B-it-qat-mlx](https://huggingface.co/roman220220/gemma-4-E4B-it-qat-mlx)**:
  5.99 GB with its tokenizer (weights 5.95 GB), +100 MB recipe (114 Linears at 8-bit). It replaced the GPTQ JANG
  build **in its own repo**, which was renamed from
  `gemma-4-E4B-it-gptq-mlx-jang`:
  - every old file was deleted in the upload commit;
  - `super_squash_history` left one commit, so the old weights aren't in the
    history;
  - HF redirects the old name (checked: a file under the old name resolves
    to the new repo), so LLMTray 0.8.3, whose wizard names the old repo,
    downloads the new build.
- Both passed `gemma4_smoke_test.py` before upload:
  - text;
  - vision: "Fox" for the z-image fox;
  - audio: a `say` clip transcribed word for word;
  - no large unquantized weight.
- Cards: `cards/gemma-4-E{2,4}B-it-qat-mlx.md`, generated by `poc/qat_card.py`
  from the measured numbers. The "raised" note comes from the scan:
  - E2B: 63 per-layer gates/projections, 54 attention projections, 9 MLP.
  - E4B: 60 per-layer gates/projections, 53 attention projections, 1 MLP.
- E4B speed on the M5:
  - ours: 34.9 tok/s, 6.07 GB peak;
  - mlx-community qat-4bit: 26.0 tok/s, 6.91 GB;
  - old GPTQ: 33.6 tok/s, 6.77 GB.
