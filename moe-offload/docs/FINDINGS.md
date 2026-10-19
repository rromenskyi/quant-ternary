# MoE experts on the SSD: findings (2026-10-19)

**Question.** Can a MoE model that doesn't fit this Mac's memory run with its
routed experts left on disk, read as needed, and how fast? The trigger was
MiniMax M2.7 2-bit (47 GB, 256 experts, top-8), which doesn't fit 24 GB.

**Short answer.** Yes. A first prototype runs Ornith-1.5-35B-A3B (same 256 ×
top-8 routing as MiniMax) at **7.4 tok/s with a quarter of its experts in
memory (5.9 GB MLX peak)**, against 20.6 tok/s all in memory (15.2 GB).
LFM2.5-8B-A1B runs at 18.0 tok/s in 1.5 GB, against 45.2 tok/s in 4.2 GB.

Most of the lost time is reading, and the reads are latency-bound: the
prototype gets ~1.6 GB/s from an SSD that does 7–8 GB/s. Waiting for the
router at every layer costs little (3% on LFM, 15% on Ornith).

For MiniMax on this Mac the estimate is **~2.5 tok/s with the prototype as
it is, ~4–5 tok/s with reads at full SSD speed overlapped with compute**,
in ~17–18 GB (section 5). Unmeasured: MiniMax's own routing locality.

Machine: MacBook Air M5, 24 GB, macOS 27, internal SSD. Runtime: LLMTray's
mlx_lm venv (MLX 0.32.4, the pinned ipsupport-llc/mlx-lm). Greedy decoding,
one token per step, no MTP, in all prototype runs.

## 1. Expert sizes (`scripts/expert_sizes.py` → `data/expert_sizes.jsonl`)

| model | experts / top-k | MoE layers | routed experts | one expert, one layer | per token, no cache |
|---|---|---|---|---|---|
| LFM2.5-8B-A1B (gptq jang) | 32 / 4 | 22 | 3.39 GB | 4.82 MB | 424 MB |
| Gemma 4 26B-A4B (gptq jang) | 128 / 8 | 30 | 12.85 GB | 3.35 MB | 803 MB |
| Ornith-1.5-35B-A3B (small) | 256 / 8 | 41 (40 + MTP head) | 11.86 GB | 1.13 MB | 371 MB |
| Nemotron 30B-A3B (jang) | 128 / 6 | 23 | 14.69 GB | 4.99 MB | 689 MB |
| Qwen3.8-Whittle-MoE-27B 4bit | 64 / 16 | 64 | 6.79 GB | 1.66 MB | 1.7 GB |

Experts are stacked on disk (`switch_mlp.*` with a leading `[E, ...]`), so
each expert is one contiguous slice per tensor, one `pread`. That is what
"pre-stacked" layouts like JANGTQ-PRESTACK buy: no per-expert files and no
repacking at load.

## 2. The SSD (`scripts/ssd_read.py` → `data/ssd_read.jsonl`)

Random reads of expert-sized chunks, page cache bypassed (`F_NOCACHE`), from
local files not read for weeks:

| chunk | threads | GB/s |
|---|---|---|
| 1.16 MB | 1 | 3.21 |
| 1.16 MB | 8 | 6.74 |
| 1.66 MB | 8 | 6.98 |
| 3.35 MB | 8 | 7.28 |
| 4.99 MB | 8 | 7.74 |
| 4.99 MB | 16 | 8.01 |

About **7–8 GB/s with 8+ threads**, 3.2 GB/s with one. Two traps on the way:
- A file read recently came back at up to 14.6 GB/s: page cache.
- The `*CRACK*` models gave 0.06 GB/s: they are symlinks to the NAS.

## 3. How often experts repeat (`scripts/route_trace.py`, `scripts/cache_sim.py`)

Routing traced while generating ~400 tokens for a coding prompt
(`data/trace_*.json`); a per-layer LRU cache simulated, a step's whole set of
experts looked up before any is evicted (as the prototype does).

Ornith's trace was taken with `mlx_lm.generate`, which drafts with its MTP
head: a step checks ~1.93 tokens, so its numbers are **per step** (the union
of both tokens' experts), and its MTP head is one of the 41 layers. Reuse
is the share of a step's experts that the step before also used.

**LFM2.5-8B-A1B** (32 experts, top-4): next-token reuse 45%.

| cache (share of experts) | in memory | hit rate | SSD per token |
|---|---|---|---|
| 12.5% | 0.42 GB | 45% | 234 MB |
| 25% | 0.85 GB | 63% | 156 MB |
| 50% | 1.7 GB | 87% | 57 MB |
| 75% | 2.5 GB | 96% | 18 MB |

**Ornith-1.5-35B-A3B** (256 experts, top-8): next-step reuse 37%.

| cache | in memory | hit rate | SSD per step |
|---|---|---|---|
| 12.5% | 1.5 GB | 56% | 264 MB |
| 25% | 3.0 GB | 72% | 166 MB |
| 37.5% | 4.5 GB | 81% | 116 MB |
| 50% | 5.9 GB | 86% | 84 MB |
| 75% | 8.9 GB | 92% | 48 MB |

A prompt touches 49% (Ornith) to 87% (LFM) of all experts. Prefill reads a
layer's experts once per chunk.

## 4. The prototype (`scripts/offload_proto.py` → `data/proto_*.jsonl`)

**How it works:**
- The model is loaded lazily, and its routed experts are never materialized.
- Each `SwitchGLU` keeps an LRU cache of experts, each expert with its own arrays.
- A decoding step waits for the router (`np.array(indices)`), reads the missing experts with `pread` (8 threads, `F_NOCACHE`), and computes each needed expert with `mx.quantized_matmul` on its own arrays.
- A prompt runs `gather_qmm` on a stack of just the experts it uses, once.

Dead ends, kept for the record:
- A stacked slot array per layer, updated with `w[slot] = expert`: MLX copied the whole stack on every write.
- Building a fresh stack of a step's experts at every layer.

Memory is MLX's peak (`mx.get_peak_memory`), not the process's RSS: Python
and the read buffers come on top.

Each data file starts with the machine state. Both sets were taken with 70–80%
memory free; pageouts rose by 24 across the Ornith set.

**LFM2.5-8B-A1B**, 120 tokens (4.06 GB model):

| run | tok/s | MLX peak | hit rate | read time |
|---|---|---|---|---|
| all in memory | 45.2 | 4.23 GB | – | – |
| all in memory, router waited for at each layer | 43.9 | 4.10 GB | – | – |
| experts on disk, cache 100% | 36.3 | 3.80 GB | 100% | 4% |
| experts on disk, cache 50% | 23.6 | 2.38 GB | 85% | 31% |
| experts on disk, cache 25% | 18.0 | 1.54 GB | 62% | 48% |

**Ornith-1.5-35B-A3B**, 100 tokens (14.8 GB model):

| run | tok/s | MLX peak | hit rate | read time |
|---|---|---|---|---|
| all in memory | 20.6 | 15.17 GB | – | – |
| all in memory, router waited for at each layer | 17.5 | 15.02 GB | – | – |
| experts on disk, cache 50% | 9.7 | 8.74 GB | 88% | 40% |
| experts on disk, cache 25% | 7.4 | 5.90 GB | 70% | 49% |
| experts on disk, cache 12.5% | 6.6 | 4.48 GB | 53% | 52% |

(An earlier set of runs, with other apps holding memory, measured roughly
half these speeds for the same code: it was dropped. Speed numbers on this
Mac need memory pressure checked first.)

What the numbers say:
1. **Waiting for the router is cheap:** 3% on LFM (22 layers), 15% on Ornith (41).
2. **The per-expert path costs something even with everything cached:** LFM 36.3 against 43.9 with the same waits. That is 24 separate small matmuls per layer instead of one fused `gather_qmm`.
3. **Reading is the main cost:** 31–52% of the time.
   - It is latency-bound: Ornith at 25% reads 106 MB per token in ~66 ms, about 1.6 GB/s against the SSD's 7–8.
   - A layer misses a few experts, read tensor by tensor in one burst, then waited for.
   - Reading all of a step's misses at once, or a layer ahead, is the next win.
4. **Memory goes down by a lot:** LFM in 1.5 GB instead of 4.2; Ornith in 5.9 GB at 7.4 tok/s, or 4.5 GB at 6.6 tok/s, instead of 15.2 GB.

## 5. MiniMax M2.x on this Mac (estimate)

**Assumed (not measured here):**
- 62 layers, 256 experts, top-8.
- About 44 GB of experts and 3 GB of other weights at 2 bits (47 GB in all).
- So ~2.8 MB per expert per layer, and 1.4 GB per token uncached.
- About 10B parameters active per token.

**Memory:**
- macOS lets Metal use ~19 GB of this 24 GB Mac by default (the `iogpu.wired_limit_mb` default).
- 3 GB of other weights and ~2 GB of KV cache and activations leave ~13–14 GB for experts, about **30%**.
- Total ~17–18 GB.

**Reads:**
- At Ornith's hit rate for 25–37.5% (72–81%), a token misses ~25% of 1.4 GB, about **350 MB**.
- At the prototype's 1.6 GB/s that is ~220 ms.
- At the SSD's 7.5 GB/s it is ~47 ms.

**Compute, scaled from Ornith:**
- Ornith all in memory: 49 ms per token at ~3B active.
- At ~10B active (more layers, bigger experts) and 2 bits instead of ~3–4: about **100–170 ms**.
- Waiting for the router at 62 layers: ~15 ms.

**Together:**

| reads | per token | tok/s |
|---|---|---|
| as the prototype reads, nothing overlapped | 220 + 100–170 + 15 ms | ~2.5–3 |
| at full SSD speed, overlapped with compute | max(47, 100–170) + 15 ms | ~5–8 |

Prefill reads every layer's experts once per chunk, 44 GB at 7.5 GB/s: about
6 s of reading per chunk before compute.

**Unknown:**
- MiniMax's routing locality. Ornith's curve is used, and it was measured on MTP steps. One trace on a pod would settle it.
- The real active compute of a 2-bit MiniMax on this GPU.

## 6. What a real implementation needs (the fork)

1. An `OffloadedSwitchGLU` in mlx-lm, chosen at load (`--experts-on-disk SHARE`).
   - Per-layer LRU of per-expert arrays, read from the checkpoint's stacked slices.
   - Cached experts computed through one `gather_qmm` (a slot array MLX can update in place, or a custom kernel) instead of per-expert matmuls.
2. **Reads at SSD speed:**
   - All of a step's misses in one parallel burst, with a pinned buffer pool.
   - Reading layer L+1's likely experts during layer L. Predict them from L's hidden state (pre-gating); the router still decides, and a wrong guess only costs a read.
3. Prefill: stream each layer's experts in chunks, with double buffering.
4. Compatible with batching and MTP (a step's union of experts), the KV cache and the prompt cache.
5. In LLMTray: a model that doesn't fit gets "Run with experts on disk" in its card, with the expected speed.

## Files

- **Scripts:**
  - `scripts/expert_sizes.py`
  - `scripts/ssd_read.py`
  - `scripts/route_trace.py`
  - `scripts/cache_sim.py` (`TRACE EXPERTS EXPERT_MB GBPS TOP_K`)
  - `scripts/offload_proto.py` (`MODEL SHARE PROMPT TOKENS [--full|--sync-only]`)
  - `scripts/prompt_code.txt`
- **Data:**
  - `data/expert_sizes.jsonl`, `data/ssd_read.jsonl`
  - `data/trace_*.json`, `data/cache_sim_*.jsonl`
  - `data/proto_lfm.jsonl`, `data/proto_ornith.jsonl` (each with the machine state first)
