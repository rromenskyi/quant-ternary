# Nemotron Extreme Quantization

A research pipeline for post-training mixed-precision quantization of
NVIDIA's Nemotron hybrid (Mamba-2 + Attention + MoE) models, plus a LoRA
fine-tuning pipeline for adapting them to tool-calling coding-agent use —
targeting Apple Silicon (MLX) as the deployment platform.

## What this actually does

Given a Nemotron checkpoint (dense or MoE), this pipeline:

1. **Quantizes it with GPTQ-style Hessian-calibrated correction**, at a
   bit-width assigned **per component type** (attention/Mamba/MoE-expert/
   embeddings/lm_head) rather than uniformly — see
   [`poc/gptq_stock_convert.py`](poc/gptq_stock_convert.py) and
   [`poc/mlx_convert_recipe.py`](poc/mlx_convert_recipe.py).
2. Optionally **fine-tunes it first with LoRA/QLoRA** (via
   [Axolotl](https://github.com/axolotl-ai-cloud/axolotl)) for a specific
   downstream task, then merges before quantizing — see
   [`poc/run_4b_lora_retrain.sh`](poc/run_4b_lora_retrain.sh).
3. Converts the result to **MLX** for Apple Silicon inference (LM Studio,
   `mlx_lm`), or, for the custom binary/ternary research path, a
   from-scratch packed format (see [`docs/RUNBOOK.md`](docs/RUNBOOK.md)).

Both the quantization methodology and the fine-tuning pipeline surfaced
several non-obvious, sometimes counter-intuitive findings along the way —
GPTQ's calibration-based correction actually *underperforming* naive
round-to-nearest quantization for out-of-calibration-domain capabilities,
a LoRA interference effect between simultaneously-adapted attention and
MLP modules, and a chat-template formatting bug that silently breaks a
fine-tuned model's ability to answer plain questions. All of it is
written up in [`docs/FINDINGS.md`](docs/FINDINGS.md), with the exact
numbers and the code that implements each fix.

## Results produced by this pipeline

- `roman220220/nemotron-30b-a3b-gptq-jang-component` — 30B-A3B MoE model,
  component-type mixed-precision GPTQ
- `roman220220/nemotron-30b-a3b-gptq3bit-g64` — 30B-A3B MoE model,
  uniform 3-bit GPTQ
- `roman220220/Nemotron-3.5-Lightning-30B-A3B-JANG-GPTQ-ipsupport-code-lora`
  — the above, LoRA fine-tuned for tool-calling reliability first
- `roman220220/NVIDIA-Nemotron-3-Nano-4B-JANG-GPTQ-ipsupport-code-lora` —
  dense 4B model, attention-only LoRA fine-tuned + 8-bit quantized (see
  `docs/FINDINGS.md` for why this recipe, specifically, was the one that
  worked)

## Layout

```
nemotron-extreme-quant/
├── poc/          # every quantization/fine-tuning/eval script this
│                 # project actually uses (despite the name -- this is
│                 # where the real, working pipeline lives)
├── docs/
│   ├── RUNBOOK.md        # step-by-step: pod setup through a quantized MLX model
│   ├── FINETUNING.md     # the ipsupport-code LoRA fine-tuning pipeline
│   ├── FINDINGS.md       # research findings, organized by topic
│   ├── patches/          # upstream patches (e.g. llama.cpp Nemotron fix)
│   └── archive/          # superseded plans + the raw chronological research log
├── models/       # downloaded/quantized checkpoints (gitignored)
├── cache/        # calibration data, etc. (gitignored)
└── output/       # run artifacts (gitignored)
```

`src/`, `scripts/`, `configs/`, and `tests/` existed early on as scaffolding
for a more general policy-driven quantization framework; that approach was
abandoned in favor of the direct, pragmatic scripts in `poc/` once those
started producing real results, and the scaffolding was removed. History
is in `git log` if it's ever useful again.

## Quick start

See [`docs/RUNBOOK.md`](docs/RUNBOOK.md) for the full walkthrough —
RunPod GPU pod setup, fetching the source model and calibration corpus,
running the quantization, and converting to MLX. For the LoRA fine-tuning
pipeline, see [`docs/FINETUNING.md`](docs/FINETUNING.md).

## Research philosophy

Only publicly documented quantization techniques are used or adapted —
GPTQ/OBQ-style Hessian correction, QuIP-style incoherence processing,
activation-aware scaling, salient-weight preservation. The project's
earlier exploration of pushing all the way to 1.1–2.0 effective bits/weight
via custom binary/ternary packing is preserved in
[`docs/archive/`](docs/archive/) — it produced real, instructive negative
results (documented in `docs/archive/stage-a-b-poc-validation.md`) before
the project settled on the mixed-precision GPTQ approach that actually
ships working models today.

## License

Code in this repository is licensed under [Apache 2.0](LICENSE). It does
not include any NVIDIA model weights. Any Nemotron checkpoint you download
or quantize with these tools remains subject to the
[NVIDIA Nemotron Open Model License](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-nemotron-open-model-license/) —
see [`NOTICE`](NOTICE).
