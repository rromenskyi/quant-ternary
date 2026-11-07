# Findings: NVIDIA Dory 1.0 4B -> MLX (GPTQ JANG)

## 1. Architecture (from `nvidia/NVIDIA-Nemotron-Labs-Dory1.0-4B-BF16/config.json`)

Looped / nGPT-style, dense, 4B, bf16 7.95 GB, single `model.safetensors`, no
`transformers` modeling file (only vLLM and llama.cpp have their own; we
already have a tested `mlx_lm/models/dory.py` in the fork, merged 2026-11-07).

- `hidden_size` 2560, `intermediate_size` 7168, `vocab_size` 131072
- 72 raw layers = 16 input + 40 recurrent + 16 output (the card's "8 input /
  20 recurrent / 8 output" counts attn+mlp pairs as one logical block; the
  raw HF layer count, which is what matters for calibration, is double that)
- Every raw layer is *either* attention *or* MLP, never both (`layer_types`
  alternates `{sliding_attention,full_attention}` / `mlp` 1:1) -- unlike a
  normal transformer block
- `num_attention_heads` 16, `num_key_value_heads` 4, `head_dim` 256,
  `sliding_window` 1024, full-attention every 8th slot
- `n_recurrent_loops` 2 (default at this checkpoint's release; llama.cpp /
  vLLM expose it as a runtime override, 1-4)
- `recurrent_kv_cache_mode` "per_loop"
- nGPT scale params, all trainable, all tiny (kept float, never quantized):
  per attention layer `sqk`, `s_gate` (hidden_size,); per MLP layer
  `suv_gate`, `suv_up` (intermediate_size,); per layer `attn_alpha` /
  `mlp_alpha` (hidden_size,); model-level `logit_scale` (vocab_size,)
- Checkpoint tensor names: `backbone.embeddings.weight`,
  `backbone.layers.{i}.mixer.{q,k,v,g}_proj.weight` (attention) or
  `backbone.layers.{i}.mixer.{gate,up,down}_proj.weight` (MLP),
  `backbone.layers.{i}.{attn,mlp}_alpha`, `backbone.layers.{i}.mixer.{sqk,
  s_gate,suv_gate,suv_up}`, `lm_head.weight`, `lm_head.logit_scale`
- License: OpenMDW-1.1 (commercial + non-commercial; `license: other` on HF,
  `license_name: openmdw-1.1`), not a standard OSI license -- card needs the
  license text under `cards/licenses/` same as other non-Apache bases.

## 2. Why the usual layer-by-layer GPTQ chain doesn't apply to the recurrent block

Every other model we've quantized (`gptq_qwen35.py`, `gptq.py`) is a strict
chain: layer *i* is calibrated on the (already-quantized) output of layer
*i-1*, one pass through the stack. Dory's 40 recurrent layers are the *same*
physical `nn.Module`s, invoked `n_recurrent_loops` (2) times per forward,
with `state = justnorm(h)` after each loop and
`h = loop_inject(state, input_h)` (re-mixing with the *original* input
layers' output) before the next. A naive "run each layer once, chain the
hidden state" driver would both (a) only exercise the recurrent block once
instead of twice, missing the `loop_inject`/`justnorm` wrapper entirely
(that control flow lives in `Model.__call__`, not inside a single layer),
and (b) have no way to "quantize layer i, then move on" when layer i is
called again two loops later with a *different* weight than at the time of
its first call -- same physical weight, so there is only ever one
quantization of it, not two.

**Design decision**: treat the whole recurrent block as one sequential GPTQ
pass (chain-quantized exactly like a normal dense stack, 40 layers in
order) driven by the *first* loop's real forward (with hooks on every
targeted Linear submodule -- the hook list-append pattern already in
`gptq_qwen35.calibrate_layer` doesn't care how many times a module fires,
so this reuses that code unchanged for the intra-loop chain). Then, with
the block now fully quantized, re-run loop 2 (and any further configured
loops) for real -- using the just-quantized weights, through the actual
`loop_inject`/`justnorm` orchestration -- to get the real hidden state the
output layers will see at inference, and chain-quantize the output layers
on that. This keeps the account's usual "calibrate on the actually-quantized
predecessor's output" accuracy property at the loop-boundary and
input/output-boundary granularity; it does not attempt intra-loop
sequential refinement across loop iterations, because there is nothing to
refine -- the second loop's weights are identical to the first's by
construction.

Calibrated at `n_recurrent_loops=2` (the checkpoint's shipped default). A
build for a different loop count would need its own calibration run; not
planned unless requested.

## 3. Status

Run on a rented A100 80GB (2026-11-07), recipe `attn=8,mlp=6`, group size
64, calibrated on wikitext-2 train + the installed transformers/torch
Python sources (same corpora and split as the other pipelines here):

| | bf16 | this build |
|---|---|---|
| wikitext-2 perplexity | 14.033 | 14.090 (+0.41 %) |
| held-out Python code perplexity | 3.739 | 3.748 (+0.24 %) |
| size | 7.95 GB | 3.5 GB |

Both well under the pipeline's 1.10x gate. The ratio staying this close to
1.0 after 2 recurrent loops is itself a check on §2's design: if
chain-quantizing the recurrent block off only the first loop's activations
were a bad approximation of what the block actually needs to represent
across both loops, it would show up here as a much larger perplexity hit,
not a 0.2-0.4 % one.

Two real bugs found and fixed on the first live run against the actual
checkpoint (neither showed up against the tiny synthetic checkpoint the
smoke tests use):
- `AutoTokenizer.from_pretrained` loads `AutoConfig` first, and
  transformers' generic `layer_types` field validator rejects this
  checkpoint's config.json (it uses `"mlp"` as a layer-type value, which
  isn't in that validator's fixed attention-pattern list) --
  `StrictDataclassClassValidationError` before the tokenizer itself ever
  loads. Fixed with the same stub-`PretrainedConfig` fallback
  `mlx_lm.tokenizer_utils.load()` already uses for this exact class of
  problem (`gptq_dory.load_tokenizer`).
- Forgot to copy `nemotron-extreme-quant/poc/{gptq,quantize}.py` and
  `qwen35-quant/poc/convert_mlx.py` to the pod (the same cross-project
  relative-path convention `qwen35_mlx_pipeline.sh`'s RUNBOOK documents) --
  `ModuleNotFoundError` at the `reference`/`convert` steps.
