# MoE experts on the SSD: findings (2026-10-19)

**Question.** Can a MoE model that doesn't fit this Mac's memory run with its
routed experts left on disk, read on demand, and how fast? The trigger was
MiniMax M2.7 2-bit (47 GB, 256 experts, top-8): it doesn't fit 24 GB.

**Short answer.** Yes, and not slowly. On this Mac, with a cache of a quarter
of the experts, routing reuse plus the SSD give roughly 5–9 tokens/s on the
models measured. The price is not the disk but **one CPU round trip per MoE
layer** (the router's choice must reach the CPU before the experts can be
read). For MiniMax-class models the estimate is about 4–8 tokens/s in
~16 GB. That needs a real implementation in the mlx-lm fork, not this
prototype.

Machine: MacBook Air M5, 24 GB, macOS 27; SSD internal. Runtime: LLMTray's
mlx_lm venv (MLX 0.32.4, ipsupport-llc/mlx-lm pinned).

## 1. Expert sizes (`scripts/expert_sizes.py`)

| model | experts / top-k | MoE layers | routed experts | one expert, one layer | per token, no cache |
|---|---|---|---|---|---|
| LFM2.5-8B-A1B (gptq jang) | 32 / 4 | 22 | 3.39 GB | 4.8 MB | 424 MB |
| Gemma 4 26B-A4B (gptq jang) | 128 / 8 | 30 | 12.85 GB | 3.3 MB | 803 MB |
| Ornith-1.5-35B-A3B (small) | 256 / 8 | 40 | 11.86 GB | 1.2 MB | 371 MB |
| Nemotron 30B-A3B (jang) | 128 / 6 | 23 | 14.69 GB | 5.0 MB | 689 MB |
| Qwen3.8-Whittle-MoE-27B 4bit | 64 / 16 | 64 | 6.79 GB | 1.7 MB | 1.7 GB |

Experts are stacked on disk (`switch_mlp.*` with a leading `[E, ...]`), so
each expert is one contiguous slice per tensor: one `pread` each. That is
what "pre-stacked" layouts like JANGTQ-PRESTACK buy: no per-expert files,
no repacking at load.

## 2. The SSD (`scripts/ssd_read.py`)

Random reads of expert-sized chunks, page cache bypassed (`F_NOCACHE`), files
not read for weeks:

| chunk | threads | GB/s |
|---|---|---|
| 5 MB | 1 | 4.65 |
| 5 MB | 8 | 7.71 |
| 1.7 MB | 8 | 6.90 |
| 3.4 MB | 16 | 4.94 |

About **7–8 GB/s with 8 threads**. (A first run on a recently used file read
up to 14.6 GB/s: page cache. A run on the `*CRACK*` models read 0.06 GB/s:
those are symlinks to the NAS.)

## 3. How often experts repeat (`scripts/route_trace.py`, `scripts/cache_sim.py`)

Routing traced while generating ~400 tokens for a coding prompt; a per-layer
LRU cache simulated. Ornith generates with its MTP head (two tokens checked
per step), so its steps need the union of both tokens' experts.

**LFM2.5-8B-A1B** (32 experts, top-4): next-token reuse 45%.

| cache (share of experts) | in memory | hit rate | SSD per token |
|---|---|---|---|
| 12.5% | 0.42 GB | 37% | 267 MB |
| 25% | 0.85 GB | 61% | 166 MB |
| 50% | 1.7 GB | 86% | 60 MB |
| 75% | 2.5 GB | 96% | 18 MB |

**Ornith-1.5-35B-A3B** (256 experts, top-8, as MiniMax): next-step reuse 58%.

| cache | in memory | hit rate | SSD per step |
|---|---|---|---|
| 12.5% | 1.5 GB | 53% | 286 MB |
| 25% | 3.0 GB | 71% | 175 MB |
| 37.5% | 4.6 GB | 80% | 121 MB |
| 50% | 6.1 GB | 86% | 87 MB |
| 75% | 9.1 GB | 92% | 50 MB |

At 7.5 GB/s, a quarter of the experts in memory costs ~20–25 ms of reading
per token. Prefill touches 50–85% of all experts: a prompt is read layer by
layer, once per chunk.

## 4. The prototype (`scripts/offload_proto.py`)

The model is loaded lazily and its routed experts never materialized. Each
`SwitchGLU` keeps an LRU cache of experts, each with its own arrays. A
decoding step waits for the router (`np.array(indices)`), reads the missing
experts with `pread` (8 threads, `F_NOCACHE`) and computes each needed expert
with `mx.quantized_matmul` on its own arrays. A prompt uses a stack of just
the experts it needs, once.

Dead ends on the way, kept for the record:
- A stacked slot array per layer, updated with `w[slot] = expert`: MLX copied
  the whole stack on every write, about 6 GB a step on Ornith, giving 1.2 tok/s.
- Building a fresh stack of the step's experts each layer: much slower than
  the per-expert path.

**LFM2.5-8B-A1B**, greedy decode, 120 tokens (4.06 GB model):

| run | tok/s | peak memory |
|---|---|---|
| all in memory | 37.8 | 4.23 GB |
| all in memory, router synced each layer | 25.1 | 4.10 GB |
| experts on disk, cache 100% | 24.7 | 3.80 GB |
| experts on disk, cache 50% | 11.4 | 2.38 GB |
| experts on disk, cache 25% | 8.7 | 1.54 GB |

**Ornith-1.5-35B-A3B**, 100 tokens (14.8 GB model):

| run | tok/s | peak memory |
|---|---|---|
| experts on disk, cache 50% | 4.4 | 8.7 GB |
| experts on disk, cache 25% | 5.0 | 5.9 GB |
| experts on disk, cache 12.5% | 5.3 | 4.5 GB |

**Caveat: these Ornith numbers are not clean.** The Mac had 14 GB of swap
in use (other apps), checked only after the runs. The all-in-memory Ornith
runs (8.5 tok/s, and 2.1 with a synced router) were swapping and are left out.
That 12.5% beats 50% says the bigger caches were hurt by memory pressure too.
To be measured again with swap clear.

What the numbers say:
- **The floor is the per-layer round trip, not the disk.** On LFM a synced
  router alone costs 34% (37.8 to 25.1). The offload path at 100% cache
  matches it (24.7): the mechanism itself costs nothing more.
- **Reading is latency-bound.** A layer misses only a few experts, read
  tensor by tensor: ~2 GB/s effective against the SSD's 7.5. That's 30–45%
  of the time at 25–50% cache. Batching all of a step's reads, or reading a
  layer ahead, is the next win.
- **Memory goes down a lot.** The LFM model runs in 1.5 GB instead of 4.2;
  Ornith in 4.5 GB instead of 15.

## 5. MiniMax M2.x on this Mac (estimate)

Assumed: 62 layers, 256 experts, top-8, ~44 GB of experts and ~3 GB dense
weights at 2 bits (47 GB total), so ~2.8 MB per expert per layer and 1.4 GB
per token uncached.

- **Memory:** about 19 GB for Metal. Dense weights 3 GB and KV/activations
  ~2 GB leave room for **~12–13 GB of experts, about 28%**.
- **Hit rate:** about 72% (Ornith's curve at 25–37.5%), so ~390 MB per token
  from the SSD, about 50 ms.
- **Round trips:** 62 layers at ~0.6 ms each, about 37 ms.
- **Compute:** ~10B active at ~2.2 bits, about 25–35 ms.
- **Total:** ~110–120 ms per token, **~8 tok/s**, with reads overlapped
  across layers. Without overlap (the prototype's reads block), ~5 tok/s.
  Prefill streams every layer's experts once per chunk: about 6 s per 2k
  tokens.
- Unknown: whether MiniMax routes with Ornith's locality. A one-time trace
  on a pod would settle it.

## 6. What a real implementation needs (the fork)

1. An `OffloadedSwitchGLU` in mlx-lm, chosen by a load option (`--experts-on-disk SHARE`): per-layer LRU of per-expert arrays, reads from the checkpoint's stacked slices.
2. **Fewer round trips:** read a layer ahead. Predict layer L+1's experts from layer L's hidden state (pre-gating) and start the reads during L's compute. The real router still decides; a wrong guess only costs a read.
3. Batched reads per step (all missing slices in one parallel burst), a pinned read pool.
4. Prefill: stream each layer's experts in chunks with double buffering.
5. Works with batching and MTP (a step's union of experts), with the KV cache and prompt cache unchanged.
6. In LLMTray: a model that doesn't fit gets "Run with experts on disk" in its card, with the expected speed.

## Files

- `scripts/expert_sizes.py`, `scripts/ssd_read.py`, `scripts/route_trace.py`, `scripts/cache_sim.py`, `scripts/offload_proto.py`, `scripts/prompt_code.txt`
- `docs/data/trace_*.json`: routing traces; `cache_sim_*.jsonl`; `proto_ornith.jsonl` (with the swap caveat).
