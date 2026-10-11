# LFM2 / LFM2.5 (lfm2, lfm2_moe): findings

## Architecture, as it matters for quantization

- Hybrid decoder: gated short convolutions (`conv.in_proj` -> B, C, x;
  `y = C * conv(B * x)`, kernel 3) in most layers, full attention with GQA
  and QK-layernorm in a few (6 of 16 in the 1.2B, 8 of 30 in the 2.6B, 6 of
  24 in the 8B). Tied embeddings, vocab 128,000 (the 2.6B / 8B: 0.26 B
  parameters, ~10 % of the 2.6B).
- `lfm2_moe` (8B-A1B): 32 experts, top-4, `moe_intermediate_size` 1792,
  sigmoid router with `expert_bias`; the first 2 layers are dense MLPs.
  Experts are 92 % of the weights.
- transformers 5.x fuses the experts (`Lfm2MoeExperts.gate_up_proj`
  [E, 2I, H], w1 rows first, then w3; `down_proj` [E, H, I]); the checkpoint
  stores them per expert (`experts.{e}.w1/w3/w2`). `assemble_lfm2.py` splits
  them back. mlx-lm's `lfm2_moe` renames w1/w3/w2 to gate/up/down_proj and
  stacks experts into `switch_mlp`; dense `lfm2` keeps w1/w3/w2.
- The MoE model's rotary module is `pos_emb`, the dense one's `rotary_emb`;
  both have `embedding_norm`, which runs before the first layer and has to
  be on the GPU when capturing layer kwargs.

## Results (2026-10-11)

MLX on the pod (CUDA), wikitext-2 test and Python stdlib, 40 x 512 tokens;
`results/*.json` has every entry with its greedy chat answer.

| Model | Variant | Recipe | Size GB | PPL text | PPL code |
|---|---|---|---|---|---|
| 8B-A1B | bf16 | | 16.94 | 91.76 | 13.128 |
| | Liquid AI MLX-4bit | 4 everywhere | 4.83 | +9.2 % | +6.7 % |
| | q4 (published) | 8/6/6/4 | 5.03 | +2.2 % | +2.6 % |
| | main (published) | 8/6/6/3 | 4.06 | +9.0 % | +7.8 % |
| | small (not published) | 8/6/6, experts gate-up 2 / down 3 | 3.41 | +33.1 % | +25.7 % |
| 2.6B | bf16 | | 5.39 | 51.77 | 5.007 |
| | published | 8/6/4 | 1.78 | −0.5 % | +1.7 % |
| 1.2B-Instruct | bf16 | | 2.34 | 143.74 | 14.710 |
| | Liquid AI MLX-4bit | 4 everywhere | 0.66 | +18.9 % | +16.3 % |
| | published | 8/6/4 | 0.80 | +4.5 % | +5.7 % |
| 1.2B-Thinking | bf16 | | 2.34 | 157.24 | 13.700 |
| | Liquid AI MLX-4bit | 4 everywhere | 0.66 | +14.8 % | +11.6 % |
| | published | 8/6/4 | 0.80 | +3.8 % | +2.5 % |

- The small models are where round-to-nearest 4-bit hurts most: for 0.14 GB
  more (attention 8, convolutions 6, embeddings 8) and GPTQ on the 4-bit MLP,
  the loss falls to a quarter or a fifth of the vendor build's.
- 8B: experts at 3 bits reach the vendor 4-bit quality in 16 % less space;
  at 4 bits, a quarter to a third of its loss for 0.19 GB more. Experts at
  2 bits (gate/up) cost +33 %: not released (the user's call).
- The high raw perplexities are the chat models on plain wikitext, not a
  bug: HF transformers gives the 8B bf16 89 on the first 10 chunks (MLX 91.8
  on 40). Compare ratios only.
- Sizes are decimal GB of the safetensors (an early report mixed GiB from
  `du` with GB).

## Exact GPTQ codes

As in qwen35-quant (docs/GPTQ_EXACT_CODES.md): `mlx_lm.convert` re-rounds the
on-grid weights and differs from GPTQ's codes in 8.01 % of packed words on
the 8B; `convert_lfm2.py` overwrites them with the calibrated codes
(`convert_mlx.write_gptq_codes`, given this model's tensor names).

## LFM2.5-VL-3B (2026-10-11)

mlx-lm could not load `lfm2_vl` at all (no module for the model type; its
`lfm2-vl.py` was unreachable and dropped the vision weights). The fork now
has it (ipsupport-llc/mlx-lm#36): SigLIP2 NaFlex tower (position grid resized
per crop, bilinear with antialias), pixel-unshuffle projector, and a
torch-free tiling processor (512 tiles on the grid closest to the aspect
ratio, up to 10, plus a thumbnail; tile / thumbnail marker tokens).

Port check against HF transformers 5.19 (two tiled photos, one small image):
token ids identical; pixels within 1-2 levels (PIL vs torchvision bicubic);
image features **fp32 vs fp32 cosine 1.00000, min 1.0000**. In bf16 a few
tokens of the 1024-patch tiles fall to cosine 0.5-0.8 -- HF's own bf16 run
does the same on the same tokens (bf16 vs fp32 min 0.61 / 0.84), so it is
bf16, not the port; the mean stays 0.998.

Quantized: the decoder as the 2.6B (8/6/4, GPTQ; it is the same decoder),
vision tower and projector RTN 8-bit except the tower's fc2 (input 4304,
not a multiple of 64: bf16) and the position grid (float, resized not looked
up).

| | Size GB | PPL text | PPL code | image feature cosine vs HF fp32 (3 images) |
|---|---|---|---|---|
| bf16 (MLX) | 6.25 | 51.77 | 13.451 | 0.998 / 0.998 / 0.9995 |
| published | 2.36 | +2.3 % | +1.5 % | 0.997 / 0.997 / 0.998 |

Greedy answers keep their content (e.g. "an insect on the flower" -> "a bee
on a flower"). The older mlx-community/LFM2-VL-1.6B-6bit (model_type
`lfm2-vl`, `image_token_index`, `vision_feature_layer -2` with 26 of 27 tower
layers stored) also loads with vision in the fork.

## Not done
- Speed on a Mac was not measured (pod MLX/CUDA numbers in the results are
  not representative).
