# Qwen3.5 family (qwen3_5, qwen3_5_moe): findings

One pipeline, `poc/qwen35_mlx_pipeline.sh` (README), for dense and MoE
checkpoints. Models: FrogNano-4B-2609 (dense, below), Ornith-1.5-35B-A3B
(MoE, at the end).

## 0. GPTQ codes vs `mlx_lm.convert` (2026-10-06)

The cross-project write-up: [`docs/GPTQ_EXACT_CODES.md`](../../docs/GPTQ_EXACT_CODES.md).

The calibrate-then-`mlx_lm.convert` flow (this project's and the earlier
ones: Nemotron, Gemma 4, FrogNano) assumed that MLX re-quantizing GPTQ's
on-grid weights re-derives the same codes. It doesn't, in general: MLX
recomputes each group's scale and bias from the group's min and max, and
GPTQ's error feedback (later columns of a group absorb earlier columns'
rounding error) often leaves the extreme codes 0 and 2^b - 1 unused, so the
re-derived grid is narrower and some codes move by one step. Measured on a
tiny random qwen3_5_moe in the dry run: a 3-bit expert tensor had 7.8 % of
its codes changed; 12.5 % of all packed words differed. At 8 and 6 bits the
differences were bf16 rounding only.

Fix: `gptq_qwen35.py` saves GPTQ's scale and bias per group, and
`convert_mlx.py --gptq-work` rewrites the decoder's quantized tensors after
`mlx_lm.convert` with GPTQ's exact codes (packed as MLX packs them: a
little-endian bit stream in uint32 words, verified bit for bit against
`mx.quantize` for 2, 3, 4, 5, 6 and 8 bits), scales and biases. After the
fix the MLX dequantized weights equal GPTQ's to bf16 rounding (2.4e-4 at a
3-bit step of 2e-2). The FrogNano release predates the fix.

# FrogNano-4B-2609

Base: [microsoft/FrogNano-4B-2609](https://huggingface.co/microsoft/FrogNano-4B-2609)
(Qwen3.5-4B architecture, `qwen3_5`, 4.66 B parameters: 32 decoder layers,
24 Gated DeltaNet linear-attention + 8 gated full-attention (3:1), dense MLP
2560 -> 9216, tied 248 320-token embeddings, a 333 M-parameter Qwen3-VL vision
tower, one MTP layer). Released: `roman220220/FrogNano-4B-2609-gptq-mlx-jang`.

## 1. Recipe search (2026-10-05)

GPTQ (`poc/gptq_qwen35.py`), sequential over the decoder, 64 x 512 tokens of
wikitext-2 train, group 64, MLX affine grid (`gptq_nbit(scheme="affine")`
from `nemotron-extreme-quant/poc/gptq.py`). Perplexity on wikitext-2 test,
40 x 512 tokens (`poc/ppl_hf.py`, decoder on-grid, embeddings bf16):

| Recipe (attn / linear attn / MLP) | PPL | vs bf16 |
|---|---|---|
| bf16 | 12.3656 | — |
| 8 / 6 / 3 | 13.3771 | +8.2 % |
| **8 / 6 / 4** | **12.5209** | **+1.25 %** |

Nemotron 3 Nano 4B's `jang-dense` (8 / 6 / 3) cost only +4.5 % there; here
the MLP is ~2.26 B of the 3.5 B decoder parameters, and 3 bits on it cost
8 %. 4 bits on the MLP is ~0.3 GB more for 7 points of perplexity: released.

Calibration time on one L40S: 285 s per run (PyTorch fallbacks for the gated
delta rule and causal conv1d -- flash-linear-attention wanted a newer Triton
than torch 2.6 ships, causal-conv1d's wheel didn't match; not needed at 4B).

## 2. MLX conversion

`poc/convert_mlx.py`: the decoder at the recipe's bits (the GPTQ codes
reproduce exactly), embeddings 8-bit RTN, `in_proj_a` / `in_proj_b` bf16,
vision tower 8-bit except its position embedding. 3.2 GB, 6.08 bits per weight
overall (3.5 GB with the tower in bf16). MLX perplexity: **12.5476 (+1.5 %)**;
the 0.03 over the HF number is the 8-bit embeddings.

The first releases didn't carry the MTP layer (mlx-lm's qwen3_5 dropped `mtp.*` then); since 2026-10-07 it is a pipeline step (§4a).

### Vision tower at 8 bits: the position embedding stays bf16

`--vision-bits 8` on the whole tower: features at cosine 0.76 mean (0.26 min)
against HF. Cause: `fast_pos_embed_interpolate` builds the interpolation
weights with `dtype=self.pos_embed.weight.dtype`, which for a quantized
embedding is uint32 -- the bilinear weights round to 0 / 1. With `pos_embed`
kept bf16 and every other tower layer at 8 bits: cosine 0.9909 mean (bf16
tower: 0.9938), same image answer, 0.3 GB less. Released that way.

## 3. Vision in MLX (mlx-lm fork, PR ipsupport-llc/mlx-lm#23)

Stock mlx-lm (and our fork's main) load qwen3_5 text-only: `sanitize` drops
`model.visual.*` and `save_config` drops `vision_config` (the same bug fixed
for Gemma 4 earlier). The fork branch `qwen3-5-vision` adds the Qwen3-VL
vision tower (ported from mlx-vlm 0.7.3, MIT), interleaved mRoPE in the
full-attention layers, Qwen2VL-style preprocessing without torch, and the
server path.

Checked against HF transformers 5.18 on one 640 x 480 image (`poc/check_mlx.py`
plus the pod script that saved `ref_vision_feats.npy`):

- pixel_values: max abs diff 3.7e-9; grid (1, 30, 40) identical;
- 3D positions and the rope delta (-280) identical to `get_rope_index`;
- vision features: cosine 0.9938 mean, 0.795 min. Not a porting error: MLX
  bf16 vs MLX fp32 gives the same spread (0.895 min) on low-norm tokens
  (uniform background patches), and fp32 vs HF is 0.9955 mean;
- the 8/6/4 model describes the image as HF bf16 does.

Images are capped at 1 MP (the checkpoint allows 16.7 MP, i.e. up to ~16k
tokens per image).

## 4. Speed and memory

MacBook Air M5, 26 GB, MLX 0.32.3: 38.1 tokens/s decoding, 3.92 GB peak
(short prompts; LLMTray had Gemma 4 E2B loaded at the same time, so this is
a lower bound).

## 4a. The MTP head (2026-10-07)

Qwen3.5 checkpoints carry a multi-token-prediction head (`mtp.*`: `fc` over
[normed embedding of token t+1, normed hidden state at t], one full-attention
decoder layer like the backbone's, a norm; the output head is shared). Our
first releases dropped it. The mlx-lm fork drafts with it since
ipsupport-llc/mlx-lm#27 (self-speculative decoding, drafts checked against the
backbone's own samples, so any temperature and the same output).

What the head needs, measured on FrogNano-4B (M5, greedy unless said):

| Question | Answer |
|---|---|
| Which hidden state goes in | the backbone's **final (normed)** one: first-draft acceptance 0.89 vs 0.83 before the norm |
| Chaining a second draft | on the head's own output after `mtp.norm` (2.11 vs 1.97 accepted per step at 3 drafts) |
| Bits | RTN 4-bit = RTN 8-bit (1.62 / 0.96 accepted per step, code / Russian); 68 MB |
| Draft length | fixed 2–3 drafts made sampled text **slower** (×0.65–0.95); the fork picks 0–3 per step from measured acceptance and step time (`DraftLength`): never below ×1.0 in these runs (the head's prefill pass is a small fixed cost) |
| A smaller draft vocabulary | dropped: the head's step is ~80 % output head (248K vocabulary), but Qwen's Russian tokens sit at ids > 150K, and a corpus-free cut (low ids, or ids seen in the chat) took Russian acceptance from 0.96 to 0.17–0.44 |
| Memory | +0.1 GB peak at a 16K-token prompt (the head's KV cache is quantized like the backbone's) |

Speed with the head (FrogNano, decode tok/s vs plain): code ×1.37–1.54
greedy, ×1.09–1.32 at temperature 1.0; English text ×1.21 / ×1.08; Russian
×1.17 / ×1.03.

**In the pipeline** the head is calibrated after the last decoder layer, on
what it reads at inference: the calibrated backbone's final hidden states
(`hidden.pt`) and the embeddings of the next tokens. GPTQ like any layer,
recipe key `mtp` (or `mtp_fc`, `mtp_attn`, `mtp_mlp` / `mtp_shared`,
`mtp_experts_gate_up`, `mtp_experts_down`; `bits_for` falls back to shorter
prefixes). The layers' resume key leaves the head's bits out, so another head
recipe recalibrates only the head. `convert_mlx.py` writes its exact codes and
moves it to `model-mtp.safetensors`: an installed model gets the head as one
file. `check_mlx.py` measures first-draft acceptance (`--min-mtp-accept`,
default 0.5).

**Raw head norms.** Qwen3.5's norms are zero-centered in the checkpoint
(MLX adds 1). #27 told a raw head from a converted one by the values ("near
0"); a trained head's aren't (FrogNano's raw `mtp.norm` averages 2.58), so the
first pipeline build loaded the head without its 1s and the check measured
acceptance **0.000**. Fixed in ipsupport-llc/mlx-lm#28: by the tensor names
(raw: `mtp.*`; converted: `language_model.mtp.*`).

**FrogNano results** (GPTQ 4-bit head): first-draft acceptance 0.885 on
wikitext-2 test, 0.913 on the Python stdlib, with the backbone of the same
build. The previously published backbone predated the exact-codes fix (§0);
on it the new head reached only 0.837 / 0.851, so the whole new build was
published (PPL 12.551, +1.5 %, as before) and the repo's history squashed
(HF's `usedStorage`, which LLMTray showed as the size, counted the replaced
3.4 GB file too).

**Ornith results.** The head (one MoE layer, 256 experts; 476 MB at 4 bits)
was calibrated on the 8/6/6/3 build. On that build's own backbone: 0.761 /
0.824; on the published 8/6/6/3 weights 0.808 / 0.811 and on the published
2/3-bit "small" weights 0.854 / 0.827 (one base model, so one head file
serves both repos; the published backbones were kept, only the head and the
index were uploaded). The new 8/6/6/3 build's perplexity: +5.6 % text,
+15.3 % code (published: +6.8 / +15.7). Speed on a Mac isn't measured: on a
26 GB Mac the 17 GB model fills the default GPU limit, and the head (0.48 GB)
leaves no room for the prompt cache there.

## 4b. An ipsupport-code LoRA for FrogNano (2026-10-07)

`lora/lora_pipeline.sh`: the agent's own conversations, LoRA, merge, the
MTP head retrained, the release pipeline, a behavior gate.

**Data** (`lora/build_dataset.py`, `lora/synth.py`; private HF dataset
`roman220220/ipsupport-code-sft`). The earlier Nemotron dataset was lost
with its pods: rebuilt from `~/.config/ipsupport-code/traces.jsonl` (863
goals, 5,433 tool calls). A captured real request gives the system prompt
and the 8 tool schemas (`file`, `run`, `git`, `web`, `help`, `calc`,
`done`, `agent`; every call `{"action", "params"}`).

- One goal = one conversation; 720 episodes -> 511 kept (60 duplicates).
- Malformed calls (unknown tool / action, missing params, params as a JSON
  string...) and their error observations are cut out: the corrected call
  that followed stays. Actions the schema doesn't list (parser leakage the
  agent repaired, e.g. `shell\n<parameter=command>...`) drop the episode;
  an empty action of a single-action tool becomes that action. The agent's
  stand-in final ("(done — finished without a written summary...)") drops it.
- 225 of them are plain replies **with the tools available** — the case
  the Nemotron LoRA never saw (§2.5 of nemotron-extreme-quant: it called
  tools on "привет").
- Real traces barely use git (≤ 3 calls per action), calc, help, done,
  agent, web.stackexchange: 243 synthetic conversations cover them (RU/EN,
  one language per conversation; a few scenarios have fewer distinct
  combinations than `--per-scenario`). The `help` lessons start from a
  deliberately wrong call; that turn is marked `"weight": 0` and kept out
  of the loss (added after this release: its adapter learned from it too).

**Training** (`lora/train_lora.py`): rendered with FrogNano's own template
(tool arguments as objects, as mlx_lm.server passes them); loss on the
assistant turns after their think block only (the data has no reasoning:
the template's empty `<think></think>` stays out of the loss). Attention
only (full-attention q/k/v/o, delta-rule in_proj_qkv / in_proj_z /
out_proj; MLP untouched: adapting both cost Nemotron-Nano-4B its coding),
r 16, alpha 32, lr 1e-4, 2 epochs, 729 conversations up to 16K tokens.
12.4 M trainable parameters; ~30 min on an A100.

| | held-out loss | bf16 PPL text | bf16 PPL code |
|---|---|---|---|
| FrogNano | 0.8045 | 12.369 | 3.166 |
| + LoRA | **0.5992** | **12.166** | 3.171 (+0.15 %) |

Coding perplexity held; text got better.

**The MTP head** sits on hidden states the LoRA moved: the base's head on
the merged model (requantized by the pipeline) drafts 0.865 / 0.894 first
tokens right (wikitext / stdlib) against 0.885 / 0.913 on the base.
`lora/train_mtp.py` retrains it on the merged backbone (frozen): the
agent's conversations plus wikitext / code chunks, the head's own task
(hidden at t + token t+1 -> token t+2), lr 5e-5, 2 epochs, ~35 min on an
A100. Teacher-forced top-1 on the held-out conversations went 0.556 ->
0.967, but that counts the system prompt and tool schemas, which the head
memorizes and which are never drafted. On plain text the requantized head
drafts **0.776 / 0.850** (wikitext / stdlib) against the old head's 0.865 /
0.894. Released with the retrained head (the user's call: the model serves
the agent); the honest measure would be acceptance on the agent's own
answers only, not done. A head retrain should be judged by acceptance, not
by top-1 over whole conversations.

**Release build** (the pipeline's `frognano-4b-ipsupport-code` preset, the
same 8/6/4 recipe): MLX PPL 12.308 text / 3.224 code (base build 12.551 /
3.229).

**Behaviour** (`lora/eval_lora.py --backend mlx`, both 8/6/4 builds, the
agent's system prompt and tools, temperature 1.0 / top-p 0.95):

| | valid first step (75) | same tool + action as the reference | greeting tool calls (80) | of them not `done` |
|---|---|---|---|---|
| FrogNano 8/6/4 | 64 | 26 | 4 | 1 |
| + LoRA | **74** | 26 | 7 | 5 |

Valid first moves went up; the greeting reflex got slightly worse (a bare
"ок, понял" / "как дела?" sometimes starts work: `file list`, `file read`,
`run shell`). With 24 samples the first reflex run showed 0 vs 1 — noise at
that size; `EVAL_REFLEX_SAMPLES` (10 per greeting) is now the default. 6 of
the LoRA's 7 calls came on Russian messages: in the real sessions a short
Russian "ок" / "понял" mostly comes in the middle of a task, where going on
working is right. The gate failed on it; `PUBLISH=1` would have stopped
here — published by hand on the owner's call (the model serves an agent
that sends real goals, not small talk), the weakness on the card. Next
time: plain replies to bare acknowledgements outside a task in the data,
the reflex split by language.

This release ran its steps one by one on the pod (the same scripts);
`lora_pipeline.sh` as a whole hasn't run end to end yet (a quoting bug in
its train / merge / mtp steps was found in review and fixed). The gate now
also compares the same-tool rate on goals whose reference calls a tool (a
model that never calls a tool passed the old checks).

Published: `roman220220/FrogNano-4B-2609-gptq-mlx-jang-ipsupport-code-lora`
(public), adapter `roman220220/FrogNano-4B-2609-ipsupport-code-lora` (private).

## 5. Base-model notes that shape the card

- A coding agent for Microsoft's Leaf harness, not a general assistant; its
  agent benchmarks (SWE-bench Verified 61.5 % Avg@3 etc.) are Microsoft's,
  bf16, not re-run on the quantization.
- Image/video are inherited from Qwen3.5-4B, not post-trained or evaluated,
  and unsupported by Microsoft. They work as Qwen3.5-4B's do.
- License: the repo metadata says MIT, the card says Apache 2.0 (derived from
  Qwen3.5-4B, Apache 2.0). Released under Apache 2.0, which satisfies both.
- Sampling: temperature 0.6, repetition penalty 1.0, 8 192 tokens per turn,
  ~131K context.

# Ornith-1.5-35B-A3B (2026-10-06)

Base: [ornith-ai/Ornith-1.5-35B-A3B](https://huggingface.co/ornith-ai/Ornith-1.5-35B-A3B),
`qwen3_5_moe`: 40 layers (30 Gated DeltaNet + 10 gated full attention),
256 routed experts (8 active, moe_intermediate 512) plus a shared expert per
layer, untied 248 320-token embeddings, Qwen3-VL vision tower, one MTP
layer; 72 GB bf16, MIT (metadata only: the repo ships no LICENSE file).
By weights it is a fine-tune of Qwen3.6-35B-A3B: on four tensors sampled by
HTTP range requests it is 4-6x closer to Qwen3.6-35B-A3B than to
Qwen3.5-35B-A3B (layernorm relative difference 1.1 % vs 6.5 %, router
9.5 % vs 15.9 %), and identical to neither.

Released: `roman220220/Ornith-1.5-35B-A3B-gptq-mlx-jang` (8/6/6/3) and
`roman220220/Ornith-1.5-35B-A3B-gptq-mlx-jang-small` (experts' gate / up 2,
down 3).

## Results

Perplexity, 40 x 512 tokens: text = wikitext-2 test; code = the Python
standard library (`code.test.txt`, held out: the calibration's code half is
the transformers and torch sources). bf16 from HF transformers on the pod,
the builds from MLX.

| | Size | PPL text | vs bf16 | PPL code | vs bf16 |
|---|---|---|---|---|---|
| bf16 | 72 GB | 9.711 | — | 2.166 | — |
| 8/6/6/3 | 17.05 GB (3.88 bpw) | 10.368 | +6.8 % | 2.507 | +15.7 % |
| 8/6/6, experts 2/2/3 | 14.36 GB (3.27 bpw) | 11.761 | +21.1 % | 3.122 | +44.1 % |

Code loses more than prose in relative terms (its bf16 perplexity is 2.2
against 9.7): for a coding model the code column is the one to watch.
Vision: tower features vs HF at cosine 0.994 mean (0.781 min), and both
builds describe the test image correctly.

MacBook Air M5, 26 GB (8/6/6/3): 43 tok/s decoding, 788 tok/s prefill at a
2K-token prompt; peak 17.2 GB short, 18.8 GB at 2K context, i.e. at the
~19 GB default GPU limit of a 26 GB Mac.

Same code test on FrogNano-4B-2609 (MLX, same tokenizer, so absolute values
compare): bf16 3.167, our 8/6/4 3.277 (+3.5 %). The 2-bit Ornith build
(3.122) still predicts code slightly better than bf16 FrogNano.

## Exact GPTQ codes on the real model

MLX's own re-quantization of the calibrated weights would have changed
8.31 % of the packed words of the 3-bit build (4.85 % of the 2/3-bit one);
`convert_mlx.py --gptq-work` writes GPTQ's codes instead (section 0).

## Pipeline notes (A100 SXM 80 GB, 250 GB RAM, RunPod secure cloud)

- Layer-by-layer calibration with the model in CPU memory: 21.6 min per
  layer at first, all of it CPU (19 cores busy, GPU idle): every chunk's
  module inputs went to the CPU, and the experts hook gathered rows per
  expert per chunk with a device sync each. Keeping everything on the GPU
  and gathering expert rows once per layer: ~30 s per layer (capture 1 s,
  linear modules 3 s, 256 experts 15 s); the codes were identical in the
  dry run. Whole calibration ~25 min.
- Writing the exact codes with MLX's CPU backend on Linux took an hour
  (single-threaded); on the GPU stream (mlx[cuda]) minutes.
- mlx_lm.convert shards by size: a tensor's weight, scales and biases can
  land in different shards (the 2/3-bit build), so the codes are written
  with a key -> shard index over all shards.
- The runpod/pytorch torch 2.9.1 image: Ubuntu 24.04 refuses pip into the
  system Python (PEP 668): a venv on the volume with system site packages.
  The latest torchvision doesn't match torch 2.9.1 (`torchvision::nms does
  not exist`): torchvision 0.24 from the cu128 index. mlx[cuda] installs its
  own NCCL, and torch then fails to load (`undefined symbol:
  ncclCommWindowRegister`) wherever transformers imports it, which mlx_lm
  does: the MLX steps preload the NCCL next to torch.
- The network volume (MooseFS) makes pip and Python imports slow (minutes);
  big sequential reads are fine.

## GGUF: don't quantize ssm_alpha/ssm_beta (delta-rule gates)

`poc/frognano_gguf_pipeline.sh` (imatrix + JANG tensor-type recipe,
RTX 4090). First release force-quantized every `ssm_*` tensor to `Q6_K`,
including `ssm_alpha.weight` / `ssm_beta.weight` (the gated-DeltaNet's
input gates -- the GGUF-side equivalent of the MLX pipeline's
`linear_attn.in_proj_[ab]`, which `DEFAULT_KEEP_FLOAT` already knew to
leave unquantized, see above). Live user report on a Vulkan backend (old
GCN GPU): 7.4 tok/s generation vs bartowski's reference quant's 25.7
tok/s, with *normal* prompt-processing throughput -- the asymmetry
(prefill fine, decode wrecked) is the signature of a GPU/CPU graph split:
the SSM-scan op expects these gates in float, and when a backend's fused
kernel doesn't accept a quantized input it falls back to splitting the
graph and round-tripping through the CPU for that subgraph. Prefill
batches many tokens so the round-trip amortizes; decode pays it on every
single token.

Confirmed via a GGUF header dump (no download needed -- `gguf`'s
`GGUFReader` on a `Range:` HTTP fetch of the first ~20 MB covers the KV
metadata + all 441 tensor-info entries; only the tensor *data* needs the
full file):

| Tensor | Shape | What happened |
|---|---|---|
| `ssm_alpha.weight`, `ssm_beta.weight` (24 layers) | (2560, 32) | force-quantized to `Q6_K` by this pipeline's override -- the bug |
| `ssm_a`, `ssm_conv1d.weight`, `ssm_dt.bias`, `ssm_norm.weight` | small | `llama-quantize` already keeps these F32 regardless of `--tensor-type` -- the override was a harmless no-op |
| `ssm_out.weight` (24 layers) | (4096, 2560) | the real linear-attention output projection; correctly quantized, left alone |

Fix: drop `ssm_a/ssm_conv1d/ssm_dt/ssm_norm/ssm_alpha/ssm_beta` from the
`--tensor-type` overrides entirely (only `ssm_out` needs one). Cost of
leaving alpha/beta unquantized: 48 tensors x (327,680 - 67,200) bytes =
~12.5 MB, 0.36 % of the 3.48 GB file. Perplexity on the republished build
was *slightly better* (8.7031 -> 8.6972), not worse -- this tradeoff has
no downside.

## GGUF: Q8_0 is ~3x slower than Q6_K on Vulkan hardware without int8 dot

Same pipeline, after the ssm_alpha/ssm_beta fix above. Live user report on
dual AMD FirePro D700 (Tahiti, GCN1) via Vulkan: the first republish
(`Q8_0` on attention/embeddings, `Q6_K` on `ssm_out`, matching this
account's usual GGUF convention) ran at 7-9 tok/s generation, vs ~26 tok/s
for a reference `Q4_K_M` build at the same nominal quant level, with
*normal* prompt-processing throughput. Investigated live via SSH onto the
box (`roman220@10.0.0.11`, both D700s, user granted explicit access):

1. **Vulkan device capability** (`vulkaninfo`/`llama-bench`): `int dot: 0,
   matrix cores: none, fp16: 0` -- Tahiti (2012-era GCN1) has no hardware
   int8 dot-product instruction, no fp16, no tensor cores at all.
2. **`GGML_SCHED_DEBUG=2`** (needs `-v`/`--verbose` too, or the debug-level
   log lines are silently dropped) on `llama-cli --single-turn`: every
   node across 35k lines of graph-split output scheduled cleanly to
   `Vulkan0`/`Vulkan1` (one real multi-GPU split at the model's midpoint
   layer) -- no surprise CPU fallback for `GATED_DELTA_NET`/`SSM_SCAN`/
   `SSM_CONV` at all. Ruled out an op-support gap.
3. **Single-GPU `llama-bench --device Vulkan0`**: our jang build 9.19
   tok/s vs the reference `Q4_K_M` 25.97 tok/s, same box, same build --
   ruled out multi-GPU cross-device sync overhead (identical gap on one
   GPU alone).
4. **Isolated the tensor type** with `llama-quantize --allow-requantize
   --tensor-type attn_*=<type> ...` directly on the already-quantized
   file (no imatrix needed for a pure speed A/B): replacing only the
   `Q8_0` tensors (attn_q/k/v/output/gate, token_embd) --
   - `-> Q5_K`: 24.74 tok/s (recovered)
   - `-> Q6_K`: 24.97 tok/s (recovered, better quality, chosen)
   `ssm_out` was untouched in every variant (stayed `Q6_K`) -- confirms
   it was never part of the problem.

Conclusion: `Q8_0`'s Vulkan compute-shader path apparently assumes/
benefits enormously from a hardware int8 dot product; without it (common
on anything older than ~2016-era GPUs, and on several current low-power/
integrated Vulkan devices too) it falls back to a much slower generic
path. K-quants (`Q4_K`/`Q5_K`/`Q6_K`) were never built around that
instruction and don't pay the penalty. **Default recipe changed: `Q6_K`
for attention/embeddings instead of `Q8_0`, every GGUF JANG release
going forward should use a K-quant for "keep this precise", not `Q8_0`,
unless there's a specific reason to need `Q8_0`'s slightly higher
precision and the target hardware is confirmed modern enough.**

Debugging tools worth keeping: `GGML_SCHED_DEBUG=2` + `-v` for an exact
per-node backend assignment dump; `llama-quantize --allow-requantize
--tensor-type X=Y` to A/B one tensor group's type on an already-built
GGUF without re-running imatrix/conversion; `llama-bench --device
VulkanN` to isolate one GPU in a multi-GPU box; `vulkaninfo --summary`
for `int dot`/`fp16`/`matrix cores` device capability flags.

## Code perplexity (added for this model specifically)

FrogNano is a coding agent; wikitext PPL alone doesn't speak to its job.
Added a `ppl_code` step (held-out Python stdlib, no test suites -- the
same corpus `qwen35-quant/poc/make_data.py` uses for the MLX pipeline).
Results tracked about the same as wikitext: F16 2.915, this quant 2.947
(+1.10%, vs +1.15% on wikitext) -- no extra quality cost on the domain
that actually matters here.
