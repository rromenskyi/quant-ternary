# Nemotron Extreme Quantization

Research pipeline for extreme post-training quantization of NVIDIA Nemotron 3.5 Lightning (30B-A3B) to 1.1–2.0 effective bpw.

## Project Structure

```
nemotron-extreme-quant/
├── configs/           # YAML configurations
├── nemotron_quant/    # Core library
├── scripts/           # CLI entry points
├── tests/             # Unit tests
├── cache/             # Calibration/cache data
├── output/            # Quantized models
└── artifacts/         # Reports, metrics, logs
```

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Phase 0: Backend capability detection
python scripts/benchmark_backend.py --device cpu

# Phase 1: Model inspection (requires HF access)
python scripts/inspect_model.py --model nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16

# Phase 3-6: Binary quantization test on single expert
python scripts/quantize_tensor.py --model ... --tensor <expert_tensor_name>
```

## Phases

See [SPEC.md](SPEC.md) for full specification.

- Phase 0: Backend discovery
- Phase 1: Model inspection & tensor inventory
- Phase 2: Quantization policy system
- Phase 3: Naive binary quantization
- Phase 4: Ternary baseline
- Phase 5: Single tensor validation
- Phase 6: Single MoE expert
- Phase 7+: Full model (later)
