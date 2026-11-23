# Findings: Inkling 2 Mini -> MLX (GPTQ JANG) and GGUF

## 1. The checkpoint

`thinkingmachines/Inkling-2-Mini` (Apache 2.0, released 2026-11-20),
`model_type: inkling2_loop`, one 19.5 GB shard plus `mtp.safetensors`:

- 24 decoder layers, hidden 2304, GQA 24 q / 8 kv heads of 128, sliding
  window 1024 in 20 layers (`local_layer_ids`), full attention in layers
  5, 11, 17, 23. No RoPE: relative position bias, log scaling and L2
  clipping, per-token attention sinks, a sigmoid output gate, short
  convolutions on K / V (dynamic in the full layers) and after attention /
  MLP, block attention residuals every 6 layers.
- MoE: 128 routed experts, top 8, sigmoid routing with a correction bias,
  softcapped SwiGLU (gate 4, up 25), no shared expert.
- The loop: layers 6-17 run 3 times with shared weights
  (`loop_config`: core 6-18, 3 passes, gate rank 64). Passes 2 and 3
  restart from the pre-loop states and mix in messages read from the
  previous pass's blocks (per-layer Q / K / V / MLP message gates).
- Experts in NVFP4 (modelopt: E2M1 codes, an e4m3 scale per 16 values,
  per-expert `amax / outer_scale_denominator`); everything else bf16.
  There is no bf16 release. 26.4 B parameters, 24.5 B of them experts.
- Vision: 23-layer ViT, hidden 1024, 40 x 40 pixel blocks per image token.
- A DSpark draft (`mtp.safetensors`, 0.64 GB bf16), see section 7.

## 2. The MLX port

`mlx_lm.models.inkling2` in the ipsupport-llc/mlx-lm fork is the upstream
draft PRs ml-explore/mlx-lm#2061 (Inkling 2) and #2086 (Mini's loop) by
jundot, with the parts of the new upstream layout it imports
(`mlx_lm/layers`, `mlx_lm/kernels/bitlinear.py`), not the whole layout
refactor (#2082). Fork commits on the `inkling2` branch.

Checked against transformers 5.24 (`modeling_inkling2_loop.py`, the
reference) on a tiny random checkpoint in the published layout
(`poc/make_tiny_checkpoint.py`): logits agree to 4e-6 in fp32, with the
loop, the message gates, the dynamic convolutions and the NVFP4 experts
(MLX's native nvfp4 with the checkpoint's global scale). Cached decoding
matches the full forward past the sliding window, with multi-token steps.

Three things found on the way:

- **TF32.** The port's own tests `test_cached_matches_forward`,
  `test_batch_matches_single` and `test_loop` failed with a 1e-4
  tolerance in fp32. MLX on M5 runs fp32 matmuls as TF32 by default; with
  `MLX_ENABLE_TF32=0` all eight pass. Not a cache bug.
- **Layer types.** transformers takes them from `layer_types` (default:
  every 6th layer full), mlx-lm and llama.cpp from `local_layer_ids`.
  They agree for the published config; the tiny checkpoint sets both.
- **Fork fixes** (commit 9451e77): bf16 experts load as they are (only
  uint8 tensors are packed NVFP4 / MXFP8); `save_config` keeps
  `vision_config` for inkling2 (the fork's allowlist dropped it, and the
  converted model lost its vision tower); `load_processor(config)` knows
  inkling2.

## 3. Calibration (`poc/gptq_inkling2.py`)

transformers loads the snapshot with the NVFP4 experts dequantized to bf16
(`Inkling2LoopDequantizeExperts`), so GPTQ starts from the exact released
weights. Same grid, Hessian math and output files as
`qwen35-quant/poc/gptq_qwen35.py`, whose expert calibration it imports
(one change there: an experts module with its own `_apply_gate` is asked
for the activation, Inkling's `act_fn(gate, up)`).

Each chunk carries `(hidden, block_states)`: the block attention residual
reads every earlier block, not only the last hidden state.

**The loop**, as Dory's (dory-quant/docs/FINDINGS.md section 2): one weight
per looped layer, so the looped layers are chain-quantized once, on the
first pass's inputs. Then passes 2 and 3 run for real on the quantized
layers, and layers 18-23 are calibrated on their output.
`poc/test_driver.py`: the layer-by-layer driver reproduces the
transformers forward exactly (max diff 0) with float weights.

**Resume bug, fixed before any real run.** A run cut off inside or after
the loop and resumed gave different post-loop layers: the resumed process
loads the float model again, and passes 2-3 then ran float looped layers.
The resume now loads every finished layer's quantized weights back.
Tested: cut before layers 1, 3 and 5, resumed, all six layer files
byte-identical to an uninterrupted run.

Kept float: attention sink projections, the dynamic convolutions' weight
projections, the message gates, the router (as mlx-lm's quant_predicate
and llama.cpp's converter).

## 4. Assemble and convert

- `assemble_checkpoint.py` writes the calibrated weights under the
  checkpoint's names (`poc/names.py`), the experts as bf16 `w13` / `w2`
  with gate / up rows interleaved again (transformers splits them, the
  checkpoint and mlx-lm interleave them; the permutation is exact for GPTQ,
  which works per row), drops the NVFP4 companions and the
  `quantization_config`. The result loads in transformers and mlx-lm with
  the same logits (3e-6, fp32).
- `convert_mlx.py` quantizes at the recipe's bits and writes GPTQ's own
  codes (`docs/GPTQ_EXACT_CODES.md`). Vision stays bf16 by default.
  Checked per tensor: every quantized decoder tensor dequantizes to the
  GPTQ grid within 0.62 % of the tensor's max (bf16 scales / biases).
- A tiny random model's logits are no test of a quantized model: the same
  weights with bf16 instead of fp32 activations already give cosine min
  0.34 (the routing flips). The smoke test checks weights there.

## 5. Sizes (planned presets)

| preset | recipe | experts | approx. size |
|---|---|---|---|
| source | NVFP4 experts, bf16 rest | 4.5 bit | 19.5 GB |
| `mini` | attn 8, experts 3 | 3.5 bit | ~13 GB |
| `mini-small` | attn 8, experts gate/up 2, down 3 | ~2.8 bit | ~11 GB |

Embeddings and head 8 bit, vision bf16. Re-quantizing the experts at 4
bits would not pay: the source is already 4.5 bits per expert weight.

## 6. GGUF

llama.cpp master has Inkling 2 (#32217, 2026-11-14) and the Mini loop
(#32694, 2026-11-20). Its converter dequantizes the NVFP4 experts itself
and writes the vision tower as an mmproj. Open there as of 2026-11-22:
perplexity NaN for inkling2_loop on CUDA (#32710), prompts over ~96
tokens decoding to '?' (#32778). The smoke test sends a longer prompt.

Result (2026-11-23, master `7778de0c3`, A100): `Q4_K_M` experts, `Q6_K`
attention / embeddings / head, imatrix 200 chunks: 16,922 MiB (5.45 bpw).
`llama-perplexity` 40 chunks, F16 -> quant: wikitext 6.768 -> 6.785
(+0.24 %), Python stdlib 2.061 -> 2.062 (+0.06 %). No NaN on the A100; the
smoke answers (a prompt over 96 tokens, a code question) and an image
through `llama-mtmd-cli` with the F16 mmproj (640 MB) were correct. GGUF's
perplexity setup differs from check_mlx.py's: not comparable with the MLX
numbers.

## 7. The DSpark draft (speculative decoding)

`mtp_config` calls it an MTP head; it is a block drafter. transformers has
no implementation; vLLM has the reference
(`vllm/models/inkling2/nvidia/dspark.py`, `speculator.py`), llama.cpp a
port in an open PR (#32218). What it does, read from vLLM:

- Target features: the hidden state at the input of layer visits 5, 23,
  41 (the loop's passes counted: 48 visits) and the final output (48).
- Context: for every committed position, per draft layer, a softmax-mixed
  (`fuse_weight`), RMS-normed (`dflare_norm`) sum of the four features,
  projected to keys (`k_proj_ctx`, per-head norm, RoPE) and values. Local
  draft layers keep the last 1024 positions.
- Block: [anchor = embed_norm(embed(last token)), 7 x `mask_embed`] at
  positions P .. P + 7, three dense layers (GQA 24/8, q/k norm, RoPE
  theta 1e6, SiLU MLP 9216), attending with no mask to the context and the
  whole block; mask slot i drafts the token at P + i.
- Markov head, slot by slot: a GRU-like step over [state, w1(previous
  token), slot hidden, f_rnn] (`rnn_proj`, 5120 -> 3 x 256; f_rnn = the
  last committed position's features, mixed with the last `fuse_weight`
  row), `tanh` latent -> `hidden_out` -> the LM head adds a bias to the
  slot's logits, softcap 30.

The mlx-lm port (fork commit a57464a): `inkling2/dspark.py`, features from
`TextModel(capture=...)`, `generate.dspark_generate_step` (stream_generate
picks it when the model has the draft). The rejected drafts roll back
exactly: key/value caches as for Gemma 4's drafter, the convolution
states from the windows the verify forward recorded. Tested with a random
model: greedy output equal to plain decoding in fp32, oracle drafts for
partial acceptance, the cache after an early stop. In bf16 a multi-token
verify rounds unlike single steps, so a near tie can flip (the tiny random
model diverged at token 6, fp32 identical): check_mlx.py records whether
the greedy answer is the same, and fails on the acceptance only (tokens
per verify step, all 7 drafts, below 1.5).

Shipped as `model-mtp.safetensors` (keys `dspark.*`), 8-bit RTN; the
recurrence (`rnn_proj`, `hidden_out`) stays bf16, as vLLM keeps it in
fp32. Not GPTQ: it only decides which tokens are tried, never the output.
Not in the GGUF: llama.cpp's DSpark support is still a PR.

## 8. Results: `mini` (attn 8, experts 3), 2026-11-23

One A100 80 GB (RunPod, secure cloud). Calibration 12 min (about 27 s a
layer; the looped layers' passes 2-3 a few seconds), assemble 2 min,
convert + exact codes 8 min, checks 15 min.

| | size | PPL text | PPL code |
|---|---|---|---|
| release, transformers (NVFP4 experts) | 19.5 GB | 8.985 | 2.466 |
| `mini`, MLX | 13.1 GB + 0.34 GB draft | 9.292 (+3.4 %) | 2.526 (+2.4 %) |

- Vision vs transformers: cosine mean 0.9988, min 0.9888; the test image
  described correctly.
- GPTQ's codes differ from MLX's own re-quantization in 18.8 % of the
  packed words (Ornith, 3-bit: 8.3 %).
- DSpark, greedy, all 7 drafts: 1.80 tokens per verify step on wikitext,
  3.63 on the Python stdlib. Speed-up on the A100 with mlx[cuda] 1.09x
  (not a Mac number). The greedy answer with DSpark differed from plain
  decoding at token 17: a near tie, " This" 29.296 against " Let" 29.248
  in one forward (bf16 steps near the softcap are 0.125), not a rollback
  error.

Two failures on the way, both fixed: `layer_kwargs` left `message_input`
on the CPU (the CPU dry run could not show it), and the fork's `setup.py`
listed its packages by hand, so a pip install from git had no
`mlx_lm.models.inkling2` (nor `mlx_lm.kernels` / `mlx_lm.processors`):
now `find_packages` (fork commit 2c487e1).

## 9. Status

- Published 2026-11-23: roman220220/Inkling-2-Mini-gptq-mlx-jang (13.1 GB),
  -small (11.1 GB), Inkling-2-Mini-GGUF-jang-imatrix (17.8 GB + mmproj).
  The fork's inkling2 + DSpark is in its main (ipsupport-llc/mlx-lm#65).
  Pod terminated (about 3.5 h of A100).
- Not measured: speed and memory on a Mac.
- Inkling 2 Small (594 B, `inkling2`): needs per-layer loading from the
  safetensors (1.2 TB in bf16 does not fit in a pod's RAM) and the
  MXFP8 experts dequantized; not started.
