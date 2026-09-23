# Gemma 4 quantization — findings

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
cosine ≥ 0.9997 vs the bf16 original. **Not republished yet**: both HF
repos still have the leftovers until the pipeline is re-run with
`PUBLISH=1`.

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
