# FLUX.2 klein — findings

Research log for quantizing `black-forest-labs/FLUX.2-klein-4B` (Apache 2.0)
for mflux / LLMTray. Every claim here was checked against the real
checkpoint or code, and is marked where it is still an estimate.

## Why this model

The goal is image **editing** in LLMTray chat, not just generation (LLMTray
only has `generate_image` today). klein 4B does text-to-image and editing
(`mflux-generate-flux2-edit`) in one small model, and it is Apache 2.0, so
our quants can be republished. The alternatives that are also Apache
(Qwen-Image-2512 / Qwen-Image-Edit-2509, 20B) don't fit next to a loaded
Gemma 4 26B on a 26GB Mac.

Excluded on licence: FLUX.1-dev / Krea / Kontext, FLUX.2-klein-9B, FIBO,
Ideogram 4 and Krea-2 are non-commercial or custom-licence and gated.
Running them yourself is fine; republishing quants of them isn't.

## Anatomy (from the checkpoint's configs)

| Component | Class | Size (bf16) | Notes |
|---|---|---|---|
| transformer | `Flux2Transformer2DModel` | 7.75GB, ~3.9B params | 5 double + 20 single blocks, 24 heads × 128 (hidden 3072), `joint_attention_dim 7680` |
| text_encoder | `Qwen3ForCausalLM` | 8.05GB, ~4B params | 36 layers, hidden 2560 |
| vae | `AutoencoderKLFlux2` | 0.17GB | 32 latent channels |

- It's distilled (`is_distilled: true`): mflux's default is 4 steps.
- The repo root also has `flux-2-klein-4b.safetensors` (7.75GB). It's the
  transformer again in single-file form, so a diffusers-layout download
  can skip it.

## The text encoder is stock Qwen/Qwen3-4B, byte for byte

Tensors sampled across the model (`embed_tokens`, `layers.0` norm,
`layers.20.q_proj`, `layers.35.down_proj`) are byte-identical to
`Qwen/Qwen3-4B` (the hybrid-thinking release) and differ from
`Qwen3-4B-Base` and `Qwen3-4B-Instruct-2507`. The files differ only because
BFL re-sharded them. BFL did not fine-tune it.

Consequences:
- Anything that works for quantizing Qwen3-4B as an LLM applies.
- An MLX Qwen3-4B already on disk is literally the same weights.

## A quarter of the text encoder is dead weight

The pipeline takes **hidden states 9, 18 and 27** and concatenates them
(3 × 2560 = 7680 = `joint_attention_dim`). This is the same in the
reference diffusers pipeline (`pipeline_flux2_klein.py`:
`hidden_states_layers=(9, 18, 27)`, `output_hidden_states=True`) and in
mflux (`qwen3_text_encoder.get_prompt_embeds`).

- Index 0 of the hidden-state list is the embedding output, so index 27 is
  the output of layer 26.
- **Layers 27–35 (9 of 36) and the final norm never affect the image.**
  Yet both implementations run them on every prompt, and their weights
  stay resident.
- The tied LM head is unused too, but it's the same tensor as
  `embed_tokens`, which is needed, so there's nothing to save there.

This is a convenience trade-off, not an oversight. The encoder is the
stock LLM class unchanged, and it runs once per image rather than per
denoising step. On a datacenter GPU the extra layers disappear in the
noise. On a Mac the memory matters: ~2GB in bf16, ~1GB at 8-bit.

**Measured: truncating to 27 layers gives bit-identical prompt embeddings**
(`mx.array_equal`, shape `[1, 512, 7680]`) and frees 0.51GB at 4-bit.

It saves **no time in MLX**: encoding took 336ms either way. MLX is lazy
and never evaluates layers the requested outputs don't depend on, so only
PyTorch/diffusers (eager) actually burns compute on them. In mflux the waste
is memory only.

## Memory budget (estimate until measured)

Weights only. Activations at 1024² come on top, measured below.

| | transformer | text encoder (36 → 27 layers) | total |
|---|---|---|---|
| bf16 | 7.75GB | 8.05 → ~6.1GB | ~15.9 → ~14GB |
| 8-bit | ~4.1GB | ~4.3 → ~3.3GB | ~8.6 → ~7.6GB |
| 4-bit | ~2.2GB | ~2.3 → ~1.8GB | ~4.7 → ~4.2GB |

Target: fit next to Gemma 4 26B JANG (~15GB + 0.45GB drafter) on a 26GB
Mac. That points to 4-bit or a JANG mix (attention 8 / FFN 4) plus the
truncated encoder.

## Measurements (M-series, 26GB, mflux 0.20, 1024², 4 steps)

Per-stage peak memory (reset before each stage):

| weights | resident | encode | 4 transformer steps | VAE decode (untiled) | total time |
|---|---|---|---|---|---|
| bf16 | 15.96GB | 16.7GB | 17.5GB | — | 26.8s |
| 8-bit (mflux RTN) | 8.56GB | 9.4GB | 10.1GB | — | 19.9s |
| 4-bit (mflux RTN) | 4.61GB | 5.6GB | 6.2GB | **10.9–14.7GB** | 19.8s |

- **The VAE decode is the memory peak, not the transformer.** At 1024²
  it adds ~6–10GB on top of the weights in one go (varied 10.9–14.7GB
  between runs). The bf16 and 8-bit rows were measured with 256-px tiles.
- **Tiled decode (`TilingConfig(vae_decode_tile_size=256)`, CLI
  `--vae-tile-size 256`):** the 4-bit decode peak drops to 6.3GB at the same
  speed (3.6s).
  - 512-px tiles were consistently *slower*: 11.8–16.8s decode vs 3.6s,
    reproduced twice. Use 256.
  - Pixels change (mean |diff| ~7–9/255 vs untiled; the diff map shows the
    tile grid), but 1:1 crops across a tile boundary show no visible seam.
    Generation itself is deterministic: two untiled runs are identical.
- **Speed is compute-bound:** 8-bit and 4-bit both take ~20s. bf16 is
  slower (26.8s), because it has more weight bytes to read each step. 4-bit
  saves memory only.
- **Quality vs bf16** (same seed, one prompt):
  - 8-bit RTN: PSNR 27.3dB, same composition, detail-level differences.
  - 4-bit RTN: PSNR 18.7dB. Still a clean image, but the composition
    drifts (fox pose, tree layout). The model is no longer doing what bf16
    does. This is where GPTQ helped on Z-Image, and it is the thing to fix.

## Architecture detail that shapes the quant recipe

In the 20 single-stream blocks, attention and MLP are **fused into single
Linears**:
- `attn.to_qkv_mlp_proj` (3072 → 27648 = q, k, v 9216 + MLP gate/up
  18432) is 43.8% of all transformer weights;
- `attn.to_out` (12288 → 3072 = attention-out 3072 + MLP-out 9216) is
  19.5%.

MLX gives one bit width per module, so a classic JANG split (attention 8 /
FFN 4) inside single blocks needs the model code to split these Linears
first. The 5 double-stream blocks have separate attention and FFN Linears.
Modulation / embedder / norm_out Linears are ~4% combined.

## Per-component quantization mixes (vs bf16, 3 prompts, seed 7)

Script `poc/klein_quant_mix.py`, raw numbers in `docs/mix_results_2026-09-23.json`.
Plain RTN, group 64, text encoder truncated to 27 layers, 256-px tiled
decode. PSNR vs bf16 (higher = closer):

| mix | fox | neon sign | portrait | TE | transformer |
|---|---|---|---|---|---|
| A: all 8-bit | 21.6 | 20.0 | 22.9 | 3.31GB | 4.12GB |
| B: all 4-bit | 17.8 | 16.9 | 14.7 | 1.75GB | 2.18GB |
| **C: TE 8 / transformer 4** | 18.9 | **20.9** | 17.1 | 3.31GB | 2.18GB |
| D: TE 4 / transformer 8 | 19.4 | 16.4 | 18.1 | 1.75GB | 4.12GB |
| E: TE 8 / transformer 4, modulation+embedders+double-block attention 8 | 18.9 | 18.9 | 18.1 | 3.31GB | 2.45GB |

Visual check ([docs/klein_mix_grid.jpg](klein_mix_grid.jpg); rows are the mixes, columns fox / sign / portrait):
- Every mix spells "OPEN 24/7" correctly.
- **A 4-bit text encoder (B, D) shifts the text layout**: the "24/7" line
  slides right of "OPEN". B also changes the portrait's face.
- **C matches bf16's layout, face and fox pose closely.** The text encoder
  is the conditioning signal, and its precision matters more per byte than
  the transformer's.
- E (raising the small "sensitive" transformer modules to 8-bit) buys
  nothing over C.
- A single seed and PSNR on diffusion output are noisy. Treat the ranking
  as directional, not final.

Note: this run's peak-memory numbers are not usable. Each config loaded
bf16 and quantized in place, which inflates the peak. Use the per-stage
table above.

## Recipe so far

- **Text encoder: 8-bit, truncated to 27 layers** (3.31GB). Keep it
  high-precision; it's cheap relative to its effect.
- **Transformer: 4-bit** (2.18GB). Next step: GPTQ calibration on real
  denoising activations (same method as zimage-quant) to remove the
  remaining composition drift vs bf16.
- **VAE: bf16, tiled decode at 256px.**
- Total ≈ 5.7GB weights and ~7GB peak at 1024². That fits next to Gemma 4
  26B JANG + drafter (~15.5GB) on a 26GB Mac.

Open questions:
- GPTQ vs RTN for the transformer: calibration pipeline to port from
  zimage-quant.
- The editing variant (`flux2_klein_edit`): same weights, different
  pipeline. Measure its memory with a reference image.
- Unloading the text encoder between the prompt encode and denoising
  (3.3GB freed during the transformer steps, at a reload cost per image).
