# Nemotron Extreme Quantization — Full Engineering & Research Specification

## Reference implementation to study

Before starting, clone and examine the reference reproduction:

```bash
git clone https://github.com/ThakiCloud/bonsai-1bit-repro
cd bonsai-1bit-repro

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

This repo contains a public reproduction of Bonsai-style 1-bit quantization techniques. Study its:
- Binary packing/unpacking implementation
- Group-wise quantization approach
- Quantization policy/config system
- Evaluation methodology

**Do not copy proprietary PrismML code.** Use only as a reference for public techniques.

---

## Project goal

*(Copied from the original spec for reproducibility)*

---

## Project goal

Build a reproducible, backend-neutral post-training extreme-quantization pipeline for:

```
nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16
```

Target: **1.1–2.0 effective bits per weight** preserving reasoning, coding, agentic behavior, instruction following, factuality.

Primary practical target: **1.5–1.8 bpw**
Stretch target: **~1.125 bpw** (binary + group scales)

---

## 1. Research philosophy

Not a conventional Q4/Q8 conversion. Explore extreme compression inspired by:
- Bonsai-style binary weight compression
- GPTQ / OBQ
- QuIP / incoherence processing
- Ternary quantization
- Activation-aware quantization
- Salient/outlier preservation
- Sparse residual correction

Only public techniques. No proprietary PrismML IP reproduction.

---

## 2. Target model

Primary: `nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16` (BF16 checkpoint)
Compare baseline: `nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4`

Architecture (discovered from checkpoint, not hardcoded):
- ~30B total params, ~3B active/token
- Mixture-of-Experts
- Mamba-2 components
- Transformer attention
- Multi-Token Prediction (MTP)
- Very long context

---

## 3. Hardware assumptions

Local: x86-64 CPU, Intel Arc Pro B50 16GB, Intel XPU-capable PyTorch
Cloud (optional): A100/H100/H200

Pipeline must work without CUDA.

---

## 4. Backend requirements

Core algorithms backend-neutral.
Required: CPU
Preferred: Intel XPU
Optional: CUDA

Every compute command: `--device cpu|xpu|cuda`
Backend abstraction in `nemotron_quant/backends/`

---

## 5. No silent fallback

Explicit logging when operations move between backends.

---

## 6. Resource constraints

No assumption that full BF16 model fits in RAM/VRAM.
Streaming: disk → CPU staging → compute → quantized output → disk → free original.

---

## 7. Disk planning

Estimate before full-model ops. Fail early if insufficient.

---

## 8. Repository structure

See SPEC.md sections 8–76 for full structure and phases.

---

## 9–76. Phases & Details

(Full phase breakdown in original spec — see sections 9 through 76)

Key phases for FIRST MILESTONE (Phases 0–6):

### PHASE 0 — Environment & backend discovery
`scripts/benchmark_backend.py` → `artifacts/backend_capabilities.json`

### PHASE 1 — Inspect checkpoint
`scripts/inspect_model.py` → `artifacts/model_inventory.json/csv`, `model_summary.md`

### PHASE 2 — Quantization policy system
YAML configs with tensor-class rules

### PHASE 3 — Naive binary quantization
Group-wise sign + scale, group sizes 32/64/128/256

### PHASE 4 — Ternary baseline
{-scale, 0, +scale} with configurable threshold

### PHASE 5 — Reference reconstruction tests
Single tensor: weight MSE, output MSE, cosine similarity

### PHASE 6 — One MoE expert
Quantize gate/up/down projections, measure with real activations

---

## 74. First Assignment (Current)

Implement ONLY Phases 0–3, 5–6.

## 75. Expected First Milestone Output

- Repository tree
- Backend capability report
- Model inventory summary
- Tensor classification summary
- Selected MoE expert details
- Binary pack/unpack test results
- Quantization metrics
- Actual bpw
- CPU/XPU runtime & peak memory
- Observed issues
- Recommendation: YES/NO to proceed

---

## 76. Principle

> Measure first. Compress second. Optimize third.

---

## 77. Hard Requirements: Deployment Artifacts

The end goal is not just a quantized checkpoint — it is **runnable artifacts** on target inference engines.

### 77.1 GGUF / llama.cpp (Hard Requirement)

Produce a **valid GGUF file** that loads and runs in **stock/current llama.cpp** (no forks, no patches).

**Targets (in priority order):**

1. **Q1_0** — native llama.cpp 1-bit binary quantization (sign + block scale)
2. **TQ1_0** — native llama.cpp 1-bit ternary quantization (if merged/available)
3. **Custom type** — ONLY if representation exactly matches GGUF spec and a minimal llama.cpp PR is acceptable

**Forbidden:** Lying to llama.cpp about tensor layout, embedding fake Q4_K_S, shipping broken GGUF that crashes/segfaults llama.cpp.

**Must pass:** `llama-cli -m model.gguf -p "test" -n 10` on clean llama.cpp main branch.

### 77.2 MLX / Apple Silicon (Hard Requirement)

Produce an **MLX-compatible artifact** (safetensors + config or `.mlxmodel`) that runs on **Apple Silicon** via `mlx-lm` or native MLX.

**Targets:**

1. **Native 2-bit (MLX `quantize` API)** — if quality/size tradeoff acceptable
2. **Custom 1-bit backend** — ONLY if 1-bit **materially** outperforms 2-bit on coding/agent benchmarks at similar effective bpw

**Must pass:** `python -m mlx_lm.generate --model ./mlx_model --prompt "test" --max-tokens 10` on Apple Silicon.

### 77.3 Unified Quantization Recipe (Preferred)

**Same quantization recipe** (policy, group sizes, residual %, calibration, rotation, GPTQ params) should produce both artifacts.

Weights should be **maximally close** — only packing/format differences allowed.

If GGUF Q1_0 and MLX 1-bit require different recipes to hit quality, document the divergence and justify.

### 77.4 Acceptance Criteria

| Artifact | Test Command | Must Pass |
|----------|--------------|-----------|
| GGUF Q1_0/TQ1_0 | `llama-cli -m out.gguf -p "def fib(n):" -n 50` | ✓ Clean generation, no NaN, no crash |
| MLX 2-bit/1-bit | `mlx_lm.generate --model ./mlx --prompt "def fib(n):"` | ✓ Clean generation, no NaN, no crash |
| Parity | Same prompts, same sampling | Outputs functionally equivalent (not bit-identical) |

---

## 78. Integration Points in Pipeline

Add two export phases (run AFTER Phase 19 full-model quantization, BEFORE Phase 21 optimized inference):

### Phase 19.5 — GGUF Export

```
scripts/export_gguf.py
  --input output/nemotron35-binary-r3/
  --output artifacts/nemotron35-1.5bpw.gguf
  --quant-type Q1_0|TQ1_0
  --llama-cpp-dir /path/to/llama.cpp  # for validation
```

- Map research format → GGUF tensor layout
- Pack binary/ternary weights per GGUF spec
- Write metadata: arch, quantization_version, block_size, etc.
- **Validate** by loading in llama.cpp

### Phase 19.6 — MLX Export

```
scripts/export_mlx.py
  --input output/nemotron35-binary-r3/
  --output artifacts/nemotron35-mlx/
  --bits 1|2
```

- Convert to MLX array format
- Use `mlx.nn.quantize` for 2-bit
- For 1-bit: implement custom `mlx.nn.QuantizedLinear` if needed
- Write `config.json` + safetensors
- **Validate** by loading in `mlx-lm`

---

## 79. GGUF/MLX Specific Constraints

### GGUF Q1_0 Layout
- Block size: 32 weights per scale (fixed in GGUF)
- Scale: FP16 per block
- Weights: 1-bit packed (32 weights = 4 bytes)
- Our group_size must be **32 or multiple** for direct mapping

### GGUF TQ1_0 Layout (if available)
- 2-bit trits per weight, block scale
- Check llama.cpp PR status before committing

### MLX Quantization
- `mlx.nn.quantize(model, bits=2, group_size=64)` — native
- For 1-bit: no native API, need custom kernel + weight format
- MLX prefers row-major, contiguous arrays

### Architecture Mismatches to Resolve
- Nemotron = MoE + Mamba + MTP + Attention
- llama.cpp / MLX may not support all components natively
- **Strategy:** Export main-model transformer + MoE first; MTP as separate tensors; document unsupported ops

---

## 80. Revised Phase Order (with Export)

```
...
PHASE 19    full-model quantization
PHASE 19.5  GGUF export + validation
PHASE 19.6  MLX export + validation
PHASE 20    reference inference (research format)
PHASE 21    optimized inference backends
...
```

Export validation is a **quality gate** — if GGUF/MLX export fails, the quantization recipe is not complete.