# Nemotron Extreme Quantization

Research pipeline for post-training compression of NVIDIA's Nemotron hybrid
(Mamba + Attention + MoE) models down to **1.1–2.0 effective bits per
weight**, while keeping reasoning, coding, and agentic behavior intact.

Status: **pre-alpha, PoC in progress.** Nothing here should be treated as a
finished result yet — see [`docs/poc.md`](docs/poc.md) for the current
go/no-go gate.

## Why

Standard 4-bit/8-bit quantization is well understood. This project explores
whether *binary and ternary* weight representations — optionally corrected
with a small sparse residual — can go far past that, using ideas from Bonsai,
GPTQ/OBQ, QuIP-style incoherence processing, and activation-aware scaling.
Only publicly documented techniques are used; see
[`docs/spec.md`](docs/spec.md#research-philosophy) for the constraint this
places on the design.

## Layout

```
nemotron-extreme-quant/
├── src/nemotron_quant/   # library: backends, model inspection, quantizers
├── scripts/              # CLI entry points
├── configs/              # quantization policy YAMLs
├── tests/                # unit tests
├── docs/                 # spec, roadmap, PoC results
├── cache/                # downloaded models & calibration data (gitignored)
├── output/               # quantized checkpoints (gitignored)
└── artifacts/            # reports, metrics, logs (gitignored)
```

## Quick start

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

# Phase 0 — what does this machine actually support?
python scripts/benchmark_backend.py --device cpu

# Phase 1 — inspect a checkpoint without loading its weights
python scripts/inspect_model.py --model nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16
```

## Docs

- [`docs/spec.md`](docs/spec.md) — goals, target architecture, hardware/backend constraints, deployment requirements
- [`docs/roadmap.md`](docs/roadmap.md) — phased implementation plan
- [`docs/poc.md`](docs/poc.md) — the proof-of-concept gate this project has to clear before the full pipeline is worth building

## License

Code in this repository is licensed under [Apache 2.0](LICENSE). It does not
include any NVIDIA model weights. Any Nemotron checkpoint you download or
quantize with these tools remains subject to the
[NVIDIA Nemotron Open Model License](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-nemotron-open-model-license/) —
see [`NOTICE`](NOTICE).
