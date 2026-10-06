# FrogNano-4B-2609: findings

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

The MTP layer isn't carried: mlx-lm's qwen3_5 drops `mtp.*` (no MTP decoding
for qwen3_5 in the fork yet).

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
