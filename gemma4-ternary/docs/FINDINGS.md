# gemma4-ternary findings

## Setup (2026-10-03)

- Host: DGX Spark (GB10, 121 GB unified memory, CUDA 13), torch 2.14.1+cu130,
  transformers 5.17.0. The venv uses a uv-managed Python 3.12: torch 2.14
  routes some ops (the RoPE `bmm` outer product) through Triton, which
  compiles its launcher against `Python.h`, absent from the system Python.
- bf16 matmul: 99 TFLOPS (8192³).

## Cost of ternary training on the Spark

Measured on the real model (Gemma 4 12B, 328 text Linears, 10.90B ternary
params), batch 2 × 2048, gradient checkpointing, SRAdamW (bf16 weights and
moments, stochastic rounding):

- Memory: ≈110 GB in use. fp32 master weights would not fit, hence the
  stochastically rounded bf16 update (verified unbiased: lr 1e-5 moves
  |w| by 1.00e-5 on average while only 25% of weights change a bf16 step).
- Speed: 180 tok/s with the eager quantizer, ≈320 tok/s after fusing it
  (`torch.compile`: 14.3 → 1.0 ms per 59M-weight matrix) and the optimizer
  update. A synthetic model of the same shape reached 420 tok/s.
- Teacher (bf16 forward, top-32): 1500 tok/s.
- So 50M tokens ≈ 9 h teacher + 43 h training on the Spark; 1B tokens would
  take about a month there, ≈1 day on one B200.
- Checkpoint: ≈65 GB (weights + two moments), written in ≈70 s.

## Measuring: chat, not raw web text

On 8 packed rows of raw web text + chat, the master scores PPL 52, and every
quantization looks bad: KL to the master 0.045 at 8-bit RTN, 0.27 on the
QAT grid itself (q4_0), 0.73 at MLX 4-bit RTN. Gemma 4 -it is very unsure
on raw text (see gemma4-quant's perplexity trap), and KL there overstates
the damage. The eval set is therefore chat from smoltalk's held-out test
split; training keeps 30% web text for breadth.

The eval is exact: the unmodified master against its own stored top-k gives
KL 1.3e-7. Top-1 agreement tops out near 98%, not 100%: bf16 logits tie.

## Ternary start

Rounding the master straight to ternary (group 64, on the mixed test rows) breaks it: KL 13.8, top-1 0.2%.
Six optimizer steps (49k tokens) of distillation took KL to 6.1.

## Pilot

50M tokens, group 128, top-32 KL distillation. Eval: 64 chat windows of
2048 tokens from smoltalk's test split, every token scored (user turns
too, so even the 4-bit references score high). References on the same
windows: q4_0 (the QAT grid) KL 0.358 / top-1 83.7%; MLX 4-bit RTN
(g64) 0.662 / 75.9%; 2-bit RTN at the ternary size (g128) 18.7 / 0.25%.
Ternary start (g128): KL 14.13, top-1 1.5%.

### Run 1: lr 3e-5, ternary from step 0 -- plateau

| step (tokens) | eval KL | top-1 |
|---|---|---|
| 150 (2.5M) | 5.96 | 4.6% |
| 300 (4.9M) | 5.89 | 5.1% |

Training loss fell 14.3 -> 5.8 in 20 steps, then sat at 5.6-5.7 for 380
steps. Not frozen weights: at step 399, 1.15% of ternary codes had flipped
(0.55-2.4% per matrix, sampled layers 0/12/24/47) and latent weights had
moved 1.4-4.6% of their magnitude. They moved without improving the model:
gradients through a network that starts at KL 14 carry little signal.
Stopped at step 399 (checkpoint kept in `run/`).

### Run 2: lr 1e-4, ternary projection ramped in over 150 steps

`w + lam * (ternary(w) - w)`, lam 0 -> 1 linearly over the first 150
optimizer steps (`--quant-warmup 150`); the gradient stays the identity.
Eval always scores the fully ternary model.

| step (tokens) | lam | eval KL | top-1 |
|---|---|---|---|
| 50 (0.8M) | 0.33 | 9.24 | 2.4% |
| 100 (1.6M) | 0.67 | 8.65 | 0.7% |
| 150 (2.5M) | 1.0 | 2.30 | 43.2% |
| 200 (3.3M) | 1.0 | 1.77 | 50.5% |
| 250 (4.1M) | 1.0 | 1.70 | 51.1% |
| 300 (4.9M) | 1.0 | 1.63 | 52.2% |

At equal tokens (4.9M): KL 1.63 vs 5.89, top-1 52% vs 5%. Both changes
went in together (time-boxed pilot), so this run does not say how much
each contributed.

Perplexity on the eval windows (20.2 at step 300) is below q4_0's (26.1):
the student is trained on smoltalk and fits its user turns better than
the master does. It is not a quality comparison; KL is.

Running on to 50M tokens. Later points: KL 1.30 / 57.3% at 10.6M tokens,
1.23 / 58.4% at 13.1M, 1.25 / 58.2% at 13.9M.

### First MLX export (step 871, 14.3M tokens)

`export_mlx.py` + `splice_mlx.py`: the 328 ternary Linears as stock MLX
2-bit (g128, scale = s, bias = -s), everything else from our 12B QAT MLX
build (6-bit embeddings). 3.95 GB. The packing was checked against
`mx.dequantize` and `mx.quantized_matmul`; MLX scores the same as training
on 4 eval rows (KL 1.18, top-1 61.6%; the 12B 4-bit build: 0.27 / 84.7%),
so the export is exact.

It does not generate: greedy and sampled replies are repetition and
fragments ("_capital_capital...", "<|channel>" loops) on every prompt. KL
1.2 nats/token teacher-forced is far from usable; the 4-bit builds sit at
0.27-0.36. A second gap: the chat training rows are rendered without the
`<|channel>thought\n<channel|>` prefix the generation prompt carries, so
the student never trained on the exact inference context; dropping the
prefix at inference didn't rescue it.

Decode on the base M5 (200 tokens, greedy, mlx-lm fork):

| model | size | tok/s | peak GB | answer |
|---|---|---|---|---|
| E2B phone MLX | 3.10 GB | 71.9 | 3.27 | correct |
| 12B ternary (step 871) | 3.95 GB | 27.0 | 4.09 | garbage |
| E4B QAT MLX | 5.95 GB | 37.0 | 6.04 | correct |
| 12B QAT MLX (4-bit) | 7.89 GB | 14.9 | 8.02 | correct |

The ternary 12B decodes 1.8x faster than its 4-bit build in half the
memory, but at this point E2B beats it on every axis. It earns its place
only if its quality passes E4B's at 2/3 of E4B's size.

A rough power-law fit of the curve (KL 1.59 at 5.7M -> 1.23 at 13.1M,
exponent ~0.31) puts MLX-4-bit-level KL (~0.6) near 0.13B tokens and
q4_0-level (~0.3) near 1.2B tokens. Three points, noisy; an indication,
not a forecast.

### Run 2 stopped at step 901 (14.8M tokens); run 3 on fixed chat data

Continuing on data whose chat turns lack the inference prefix would only
train the wrong context further. `prepare_data.py` now renders every model
turn as the generation prompt does (`<|turn>model\n<|channel>thought\n<channel|>`,
checked: 7/7 model turns in a sample row). New data and teacher
(`data_v2`, `teacher_v2`, ≈9 h of teacher); run 3 (`run_v2_init901`)
starts from run 2's step-901 latent weights (`--init-ckpt`: weights only,
fresh optimizer and lr warmup, no quantization warmup since the weights
are already ternary-trained). The references are recomputed: the eval
rows changed.

## Run 3 data: the master writes the replies (2026-10-04)

The chat-prefix fix (above) still left two mismatches with inference: it
gave the prefix to every model turn, history included, while a real
prompt carries it only on the turn being generated; and nothing covered
the thinking mode. Following LLM-QAT (data generated by the model itself
keeps its output distribution better than its pre-training data), the
data is now written by the master:

- `gen_teacher_data.py`: one assistant turn of a smoltalk conversation at
  random is the target; the turns before it are rendered by the chat
  template as they are (history without the prefix), then
  `add_generation_prompt` with `enable_thinking` on for half of the
  prompts; the master samples the reply (T 1.0, top-p 0.95, top-k 64).
  Checked on samples: history turns plain, the generated turn opens with
  `<|channel>thought\n` and either `<channel|>` or a real reasoning trace;
  smoltalk's function-calling prompts produce `<tool_call>` replies.
- Train: 18,247 replies (9,106 thinking), 16.3M generated tokens, 22.6M
  with prompts; 80% of finished within the caps (768 tokens, 2,560 when
  thinking). Packed with 20% FineWeb-edu into 41 shards (21.5M tokens).
- Eval: 253 replies (111 thinking) to smoltalk's test split, the same way.

### vLLM on the Spark, what it took

- `transformers.generate` reached 47 tok/s (batch 16): a static batch
  waits for its longest reply (4 of 16 had finished). vLLM 0.30.0
  (continuous batching, paged KV cache) installs with pip into a user
  venv (`uv pip install vllm --torch-backend=cu130`), no Docker/sudo.
- The multimodal 12B fails in vLLM (`Gemma4UnifiedVisionConfig` has no
  `num_soft_tokens`). vLLM's text-only `Gemma4ForCausalLM` maps
  `model.language_model.*` weights itself, so `masters/gemma-4-12B-text`
  is the text config (`model_type: gemma4_text`) plus symlinks to the same
  safetensors: no copy.
- vLLM JIT-builds kernels and needs `ninja` on PATH (the venv's).
- Per-request `seed` in SamplingParams made generation several times
  slower (a 4,096-prompt shard was still unfinished after 4.5 h; the same
  work at the later rate takes ~2.7 h). One engine seed instead.
- With long thinking replies decode becomes KV-cache bound: 390 tok/s at
  the start of a shard fell to ~240 as replies grew (KV cache 88% full,
  256 running). `kv_cache_dtype="fp8"` doubled the early rate (780) and
  raised the shard average from 233-243 to 413-650 tok/s (2.7x on most
  shards), with the same finish rate. Only the sampled text depends on
  it: the training labels come from the bf16 teacher pass.
- `prompt throughput: 0` in vLLM's stats is the prefill rate; it is 0
  while every prompt of a shard is already admitted (`Waiting: 0`) and
  the last long replies finish. `generation throughput` is the output rate.
- Exit: the script ends with `os._exit` (streaming readers crash
  interpreter teardown), which skipped vLLM's shutdown. Its `EngineCore`
  child lived on with 109 GB and the step's stdout, so the pipeline's
  `tee` never saw EOF: the pipeline hung for 70 min with the GPU idle.
  The script now kills its multiprocessing children before `os._exit`.

## Run 3 so far (2026-10-05)

New eval (teacher-written replies to smoltalk's test prompts; every model
scores lower KL on the master's own text): q4_0 KL 0.105 / top-1 91.8% /
PPL 5.43; MLX 4-bit RTN 0.248 / 87.1% / 5.68; 2-bit RTN 20.1 / 0.2%.

| step (run-3 tokens) | eval KL | top-1 | PPL |
|---|---|---|---|
| 0 (run 2's step 901) | 1.667 | 55.8% | 15.6 |
| 50 (0.8M) | 1.150 | 64.1% | 8.80 |
| 150 (2.5M) | 1.229 | 62.8% | 9.32 |
| 250 (4.1M) | 1.143 | 64.8% | 8.61 |
| 350 (5.7M) | 1.091 | 65.5% | 8.02 |
| 450 (7.4M) | 1.020 | 66.8% | 7.66 |

Run 2's weights carried over: the step-901 model started at KL 1.67 on the
new data, not 14, and needed 50 steps to adapt to the format. Then
roughly flat while lr sat near 1e-4, falling again from step ~300 as the
cosine schedule lowers it.

### Chat at step 424 (KL ≈1.07), MLX 2-bit on the Mac

It talks now (run 2's step 871 produced only fragments and loops): Gemma's
reply style, markdown, a fenced code block, the thinking channel with
Gemma-like bullet reasoning. The content is wrong: "The capital of
Australia is Sydney."; the Fibonacci function is code-shaped nonsense; a
Russian question drifts off topic into a repetition loop; the thinking
answer subtracts (3 apples - 2 bags = 1) and loops. Russian is weakest:
the teacher-written data is nearly all English (smoltalk) — a full run
needs multilingual prompts.

Copying exports from the Spark ran at 0.6-0.7 MB/s this time (10 MB/s on
the 4th): ~75 min for 3 GB.

## Run 3 final (step 1281, 21.0M tokens; 35.8M since the ternary start)

| step | eval KL | top-1 | PPL |
|---|---|---|---|
| 750 (12.3M) | 0.904 | 69.0% | 6.92 |
| 1000 (16.4M) | 0.769 | 71.6% | 6.17 |
| 1150 (18.8M) | 0.728 | 72.5% | 6.07 |
| 1281 (21.0M) | **0.709** | **72.9%** | **5.87** |

References on the same eval: q4_0 0.105 / 91.8% / 5.43; MLX 4-bit RTN
0.248 / 87.1% / 5.68. PPL is within ~3% of MLX 4-bit at half its size;
KL is still ~3x its. The eval PPL wobbles ±1-5% between points while KL
falls monotonically (KL is the training target; PPL also scores the
human-written user turns and is sensitive to a few unlikely tokens).

Chat (same 4 prompts as step 424, greedy, MLX 2-bit on the Mac):
- Thinking mode now works end to end: "3 apples + 2 bags x 4" is planned
  in Gemma-style bullets, computed (2 x 4 = 8, 3 + 8 = 11), the thought
  channel closes and a clean numbered answer says 11. At step 424 it
  computed 3 - 2 = 1 and looped.
- Still wrong: "the capital of Australia is Sydney"; the iterative
  Fibonacci becomes a garbled Binet formula; a Russian question is read
  as a text to process ("please provide the context...") and answered in
  English — the data is ~99.8% English.

### Moving files off the Spark: HF, not scp

scp from the Spark ran at 0.6-0.7 MB/s; `hf upload` of the same 3 GB to a
private repo took 82 s (~37 MB/s) and `hf download` on the Mac 58 s. The
single ssh stream was the bottleneck, not the uplink.

### In LLMTray (step 1281, `~/.lmstudio/models/roman220220/gemma-4-12B-it-ternary-pilot`)

Loads as a plain folder (config + complete weight index), 23.8 tok/s with
thinking. In a multi-turn chat it lags one turn: "Where is Sydney?" got
the previous answer again ("The capital of Paris is Paris"), "What do you
know about London?" got "Sydney is a ... city in the UK" with London
landmarks (Tower of London, the Thames) mixed into invented ones. Two
suspects: the 8 global-attention layers (1 KV head, K = V) carry the
long-range lookup that tracks the latest turn, and ternary may hit them
hardest — the attn4 probe (4-bit attention) tests that; and multi-turn
is only 27% of the training replies.
