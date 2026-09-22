# zimage-quant

GPTQ (Hessian-corrected) quantization for [Z-Image-Turbo](https://huggingface.co/Tongyi-MAI/Z-Image-Turbo)
(Tongyi-MAI, 6.15B diffusion transformer), targeting
[mflux](https://github.com/filipstrand/mflux)'s native MLX checkpoint format.

Sibling project to [`nemotron-extreme-quant`](../nemotron-extreme-quant),
reusing its `gptq_nbit` implementation (`poc/gptq.py`) unchanged — the same
Hessian-based error compensation that project applies to LLM linears applies
here to a diffusion transformer's attention/feed-forward linears, just with
different calibration data (real denoising-loop activations instead of token
sequences).

## Why this exists

mflux's `--quantize N` does naive round-to-nearest (RTN) quantization on
load. This project instead:

1. Runs `ZImagePipeline`'s own denoising loop for a handful of real prompts,
   capturing each target Linear's input activation via a forward-pre-hook
   at every diffusion step.
2. Runs GPTQ (`gptq_nbit`, `scheme="affine"` — matches `mx.quantize`'s exact
   grid) on the pooled activations to correct each weight's quantization
   error using the layer's Hessian, instead of rounding each weight
   independently.
3. Re-quantizes the corrected (still-bf16, now "on-grid") weights with
   `mx.quantize` and splices them into a copy of mflux's own safetensors
   shards — same on-disk format, same key names (mflux is a line-by-line
   port of the real diffusers model), zero format conversion.

Result: same on-disk size as plain `mflux-save --quantize N`, better
quality at that size (see the model cards on HF for the actual comparison).

## Two-stage design (`poc/zimage_gptq_calibrate.py` + `poc/zimage_gptq_splice.py`)

Calibration (`zimage_gptq_calibrate.py`) needs `torch`+`diffusers`, runs
fine on CUDA/CPU/MPS, and has no `mlx` dependency — it's meant to run
wherever is fastest (a rented CUDA GPU pod, in practice; mlx is Apple-only
so it can't accelerate this half at all). It writes GPTQ-corrected bf16
weights as `batch_*.safetensors` files plus a `gptq_progress.json` so a
killed/interrupted run resumes instead of restarting.

Splicing (`zimage_gptq_splice.py`) needs `mlx` (Apple Silicon only) and
does the final `mx.quantize` + safetensors-shard patching into an
`mflux-save`d template. Run this on a Mac after copying the calibrate
stage's output over.

`poc/zimage_gptq.py` is the original monolithic (calibrate+splice in one
process, MLX-only) version, kept for small-scale local runs where renting a
GPU isn't worth it.

## Usage

```bash
# 1. One-time RTN template (any bit width mflux supports) -- this project's
#    splice step patches GPTQ-corrected weights into a copy of this.
mflux-save --path /tmp/zimage-8bit-rtn --model z-image-turbo --quantize 8

# 2. Calibrate + GPTQ-correct (fast on a CUDA pod, slow-but-working on CPU/MPS)
python poc/zimage_gptq_calibrate.py \
    --hf-pipeline-dir ~/.cache/huggingface/hub/models--Tongyi-MAI--Z-Image-Turbo/snapshots/<sha> \
    --output-dir /tmp/zimage-corrected \
    --layers all --device cuda --prompts 4 --steps 9 --bits 8 --group-size 64

# 3. Splice into the final MLX checkpoint (needs mlx -- run on a Mac)
python poc/zimage_gptq_splice.py \
    --corrected-dir /tmp/zimage-corrected \
    --mflux-saved-dir /tmp/zimage-8bit-rtn \
    --output-dir /tmp/zimage-8bit-gptq-full \
    --bits 8 --group-size 64

# 4. Use it
mflux-generate-z-image-turbo --model /tmp/zimage-8bit-gptq-full \
    --base-model z-image-turbo --prompt "..." --steps 9
```

Published checkpoints: [z-image-turbo-gptq-mlx-8bit](https://huggingface.co/roman220220/z-image-turbo-gptq-mlx-8bit),
[z-image-turbo-gptq-mlx-4bit](https://huggingface.co/roman220220/z-image-turbo-gptq-mlx-4bit),
[z-image-turbo-gptq-mlx-mixed](https://huggingface.co/roman220220/z-image-turbo-gptq-mlx-mixed)
(attention 8-bit + feed-forward 4-bit — a `jang-dense`-style component-type
bit allocation, adapted from `nemotron-extreme-quant`'s LLM recipe of the
same name; validated to beat uniform 4-bit on both PSNR and generation
composition stability for +0.8GB, via `--attn-bits`/`--ffn-bits` on
`zimage_gptq_calibrate.py`).

Use these models directly from [LLMTray](https://github.com/ipsupport-llc/llmtray)'s
built-in image generation (tool-calling `generate_image`), which drives
mflux under the hood.
