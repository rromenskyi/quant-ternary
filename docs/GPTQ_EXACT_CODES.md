# GPTQ codes and `mlx_lm.convert`: matching the grid is not enough

Applies to every GPTQ -> MLX pipeline in this repository (nemotron-extreme-quant,
gemma4-quant, qwen35-quant). Found 2026-10-06 on the qwen35-quant pipeline.

## The assumption

The pipelines calibrate with GPTQ on MLX's affine grid (`gptq_nbit(...,
scheme="affine")`, whose scale / bias formula reproduces MLX's quantize kernel
bit for bit: nemotron-extreme-quant FINDINGS §1.2), write the on-grid weights
in bf16, and let stock `mlx_lm.convert` quantize them again. The assumption
was that with the same formula MLX re-derives the same codes.

## Why it fails

MLX derives each group's scale and bias from the group's **min and max**.
GPTQ derives them from the group's weights when it starts the group, then
quantizes the group column by column, pushing each column's rounding error
onto the later columns of the same group. After that error feedback the
group's quantized values often don't use the extreme codes 0 and 2^b - 1, so
the min / max MLX sees is narrower than GPTQ's range: MLX picks a different
scale and bias, and some codes move by one step. The calibration is partly
undone, silently (the model still works, just worse than calibrated).

Measured:

| Model, bits | Packed words that MLX's re-quantization changed |
|---|---|
| tiny random qwen3_5_moe, 3-bit experts (dry run) | 12.5 % (7.8 % of a 3-bit tensor's codes) |
| Ornith-1.5-35B-A3B, 8/6/6/3 | 8.31 % |
| Ornith-1.5-35B-A3B, experts 2/2/3 | 4.85 % |

At 8 and 6 bits the differences in the dry run were bf16 rounding only; the
damage concentrates at 2-3 bits, where one step is large.

## The fix (qwen35-quant)

`gptq_qwen35.py` saves GPTQ's own scale and bias per group next to the
weights; `convert_mlx.py --gptq-work` runs `mlx_lm.convert` for the model's
structure, then rewrites every calibrated tensor with GPTQ's codes, scales
and biases. The packing is MLX's (a little-endian bit stream of b-bit codes
in uint32 words, verified bit for bit against `mx.quantize` for 2, 3, 4, 5,
6 and 8 bits). A tensor's weight, scales and biases can sit in different
shards, so the rewrite indexes all shards. After the fix MLX's dequantized
weights equal GPTQ's to bf16 rounding.

## Releases made before the fix

Nemotron (nemotron-extreme-quant), Gemma 4 (gemma4-quant) and
FrogNano-4B-2609 (qwen35-quant) were converted by re-quantization, so some of
their codes are MLX's, not GPTQ's. Their measured numbers stand as published
(they were measured on the released files); a rebuild with exact codes can
only improve them. FrogNano lost 1.5 % on text and 3.5 % on code, so its
rebuild isn't urgent; the 3-bit expert builds would gain most.
