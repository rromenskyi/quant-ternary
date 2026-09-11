# PoC Validation Plan — Nemotron Extreme Quantization

## Purpose

**Before** investing weeks in full pipeline implementation, prove that the core methodology (binary/ternary quantization + sparse residual) can achieve **<2.0 effective bpw with acceptable quality degradation** on representative Nemotron 3.5 tensors.

This is a **go/no-go gate**. If PoC fails, we pivot or stop.

---

## PoC Scope (Minimal Viable Experiment)

| Component | Scope |
|-----------|-------|
| **Model** | Single MoE expert (3 linear layers: gate, up, down) from Nemotron 3.5 30B |
| **Data** | 128 calibration sequences (coding + reasoning mix), seq_len=1024 |
| **Methods** | 1. Naive binary (group 128)<br>2. Activation-aware binary<br>3. Binary + 1% residual (magnitude)<br>4. Binary + 3% residual (activation-weighted)<br>5. Ternary (optimized threshold)<br>6. GPTQ-binary (if public impl integrates cleanly) |
| **Metrics** | Weight reconstruction (MSE, cosine)<br>Activation reconstruction (output MSE, cosine on real calibration activations)<br>Effective bpw (packed + scales + residual) |
| **Hardware** | CPU (reference), Intel Arc Pro B50 / XPU (if available) |
| **Time budget** | 1–2 days of compute |

---

## Success Criteria (Go/No-Go)

| Metric | Threshold | Rationale |
|--------|-----------|-----------|
| **Activation output cosine** (real calibrations) | ≥ 0.95 | Below this → catastrophic error accumulation across layers |
| **Weight cosine** | ≥ 0.98 | Sanity check — if weight recon is bad, activation recon is meaningless |
| **Effective bpw** | ≤ 1.8 | Must beat 2.0 bpw target with residual overhead included |
| **No NaN/Inf** | Zero | Numerical stability gate |

**All 4 must pass** on at least **2 of 3 expert matrices** (gate, up, down) for **Go**.

---

## PoC Implementation Steps (Minimal Code)

### 1. Model Access & Single Expert Extraction (30 min)
```bash
# Download only config + 1 shard containing expert 0 of layer 10
# Use safetensors header/index — no full model load
python scripts/poc_extract_expert.py --layer 10 --expert 0 --output poc/expert_L10_E0/
```

### 2. Calibration Activations Capture (1–2 hrs CPU)
```bash
# Run 128 sequences through original model up to layer 10 expert input
# Save input activations X (float16) for gate/up/down
python scripts/poc_collect_acts.py --model nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16 \
    --layer 10 --expert 0 --sequences 128 --seqlen 1024 \
    --calib coding --output poc/calib_L10_E0/
```

### 3. Quantization Methods (core PoC logic)
```python
# poc/quantize.py — standalone, no framework deps beyond torch/numpy
def naive_binary(W, group_size=128):
    scale = W.abs().mean(dim=-1, keepdim=True)
    B = W.sign()
    return scale, B

def activation_aware_binary(W, X, group_size=128):
    # X: [n_samples, in_features] calibration activations
    # Optimal scale per group: minimize ||XW - X(scale*sign(W))||
    pass

def binary_with_residual(W, X, residual_pct, criterion="magnitude"):
    Wq = naive_binary(W)
    R = select_residual(W, Wq, X, residual_pct, criterion)
    return Wq, R

def ternary_optimized(W, X, group_size=128):
    # Search threshold per group minimizing weight MSE (or activation MSE)
    pass
```

### 4. Evaluation
```python
# poc/evaluate.py
def evaluate(W_orig, W_quant, X_calib):
    Y_orig = X_calib @ W_orig.T
    Y_quant = X_calib @ W_quant.T
    return {
        "weight_cosine": cosine(W_orig, W_quant),
        "weight_mse": mse(W_orig, W_quant),
        "act_cosine": cosine(Y_orig, Y_quant),
        "act_mse": mse(Y_orig, Y_quant),
        "effective_bpw": compute_bpw(W_quant),
    }
```

### 5. Report
```markdown
# PoC Results — Nemotron L10 E0 Expert

| Method | Weight Cos | Act Cos | Eff BPW | Verdict |
|--------|-----------|---------|---------|---------|
| Naive binary g128 | 0.991 | 0.912 | 1.125 | ❌ Act cos < 0.95 |
| Act-aware binary g128 | 0.993 | 0.958 | 1.125 | ✅ |
| Binary + 1% residual (mag) | 0.996 | 0.967 | 1.28 | ✅ |
| Binary + 3% residual (act-w) | 0.998 | 0.981 | 1.52 | ✅ |
| Ternary g128 | 0.995 | 0.970 | 1.33 | ✅ |
| GPTQ-binary | 0.997 | 0.985 | 1.125 | ✅ |

**Decision: GO** — Activation-aware binary + residual achieves target.
```

---

## What PoC Does NOT Need

- Full model inspection / tensor inventory
- Policy system / YAML configs
- Multi-block / multi-expert / full model
- Storage format / manifest / resumable quantization
- Optimized kernels / inference
- Evaluation suite (coding, agent, reasoning)
- Cloud packaging / GGUF

---

## PoC Deliverables

1. **`poc/quantize.py`** — ~200 lines, standalone quantization methods
2. **`poc/evaluate.py`** — ~100 lines, metrics
3. **`poc/extract_expert.py`** — safetensors shard reading
4. **`poc/collect_acts.py`** — activation capture (uses HF transformers)
4. **`poc/results.md`** — the decision table above

---

## If PoC Fails (No-Go Scenarios)

| Failure Mode | Pivot |
|--------------|-------|
| All methods: act_cosine < 0.90 | Nemotron 3.5 architecture fundamentally incompatible with extreme quantization → try higher bpw target (2.5–3.0) or different model |
| Only GPTQ works but too slow/unstable | Invest in GPTQ optimization / use 4-bit as floor |
| Residual >5% needed for 0.95 | Effective bpw > 2.0 → target unachievable, report negative result |
| XPU/CPU numerical divergence | Block on backend parity before proceeding |

---

## Integration with Main Plan

| PoC Outcome | Main Plan Action |
|-------------|------------------|
| **GO** | Proceed with Phase 0–6 implementation (full pipeline) |
| **PARTIAL** (some methods work) | Implement only working methods in Phase 3–6, defer others |
| **NO-GO** | Document findings, archive repo, pivot to new approach |

---

## Quick Start (When Ready)

```bash
cd nemotron-extreme-quant
python3 -m venv .venv
source .venv/bin/activate
pip install torch transformers safetensors numpy pyyaml psutil

# Run PoC sequence
python scripts/poc_extract_expert.py --layer 10 --expert 0
python scripts/poc_collect_acts.py --layer 10 --expert 0 --sequences 128
python scripts/poc_run.py  # runs all methods, emits poc/results.md
```

---

## Reference: Bonsai-1bit-repro

The reference repo `https://github.com/ThakiCloud/bonsai-1bit-repro` contains:
- Binary quantization kernels (CPU/CUDA)
- Packing utilities
- Example quantization scripts

**Use for:** API patterns, packing logic, kernel structure.
**Do NOT:** Copy blindly — their target/model differs. Validate every assumption on Nemotron tensors.

---

## Decision Authority

**You** decide Go/No-Go based on `poc/results.md`.

I provide the numbers. You decide if they're good enough to invest the next 2–4 weeks.