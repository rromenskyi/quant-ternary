# Nemotron Extreme Quantization — Implementation Plan

This document breaks the SPEC into concrete, trackable tasks with ownership and acceptance criteria.

---

## Phase 0: Environment & Backend Discovery

| Task | Description | Acceptance | Owner |
|------|-------------|------------|-------|
| 0.1 | Create `scripts/benchmark_backend.py` | Runs on CPU/XPU/CUDA, prints capabilities JSON | bonsai |
| 0.2 | Implement `backends/base.py` + `cpu.py` | Abstract device ops: matmul, cholesky, qr, eig, hadamard, topk | bonsai |
| 0.3 | Implement `backends/xpu.py` | XPU backend with explicit fallback logging | bonsai |
| 0.4 | Implement `backends/cuda.py` | CUDA backend (stub for cloud) | bonsai |
| 0.5 | Run benchmark, emit `artifacts/backend_capabilities.json` | JSON matches spec format | user |

---

## Phase 1: Model Inspection & Tensor Inventory

| Task | Description | Acceptance | Owner |
|------|-------------|------------|-------|
| 1.1 | `model/inspector.py` — read HF config + safetensors index | No model weights loaded | bonsai |
| 1.2 | `model/tensor_inventory.py` — build full inventory | CSV/JSON with: name, shape, dtype, params, bytes, shard, block, expert, class, eligibility | bonsai |
| 1.3 | `model/tensor_classifier.py` — semantic classification | Rules-based (regex + architecture heuristics) | bonsai |
| 1.4 | `scripts/inspect_model.py` — CLI entry point | Produces `artifacts/model_inventory.{json,csv}` + `model_summary.md` | bonsai |
| 1.5 | Run inspection on Nemotron 3.5 BF16 | Summary shows: total params, by-class breakdown, hypothetical bpw table | user |

---

## Phase 2: Quantization Policy System

| Task | Description | Acceptance | Owner |
|------|-------------|------------|-------|
| 2.1 | Config schema: `configs/quant/*.yaml` examples | Binary, ternary, GPTQ, residual variants | bonsai |
| 2.2 | Policy loader: `quantizers/policy.py` | Resolves default + ordered rules (class, regex, layer range, expert range, size threshold) | bonsai |
| 2.3 | Protected tensor list (norm, router, mamba state, dt, A, D, embeddings, lm_head) | These default to BF16 unless overridden | bonsai |
| 2.4 | `scripts/quantize_tensor.py` — test single tensor with policy | Dry-run: shows what method would apply | bonsai |

---

## Phase 3: Naive Binary Quantization

| Task | Description | Acceptance | Owner |
|------|-------------|------------|-------|
| 3.1 | `quantizers/binary.py` — group-wise `scale=mean(|W|), B=sign(W)` | Group sizes 32/64/128/256 | bonsai |
| 3.2 | `storage/binary_pack.py` — pack/unpack 8 weights/byte | Round-trip exact on random + adversarial tensors | bonsai |
| 3.3 | Unit tests: `tests/test_binary_pack.py` | 100% pass | bonsai |
| 3.4 | `quantizers/base.py` — common interface | `quantize(tensor, config) -> QuantizedTensor` | bonsai |
| 3.5 | Actual bpw calculator (scales + padding + metadata) | Matches spec accounting formula | bonsai |

---

## Phase 4: Ternary Quantization

| Task | Description | Acceptance | Owner |
|------|-------------|------------|-------|
| 4.1 | `quantizers/ternary.py` — threshold-based {-s, 0, +s} | Fixed / per-tensor / per-group threshold search | bonsai |
| 4.2 | `storage/ternary_pack.py` — 2-bit packing (4 values/byte) | Round-trip exact | bonsai |
| 4.3 | Unit tests: `tests/test_ternary_pack.py` | 100% pass | bonsai |

---

## Phase 5: Single Tensor Validation

| Task | Description | Acceptance | Owner |
|------|-------------|------------|-------|
| 5.1 | `scripts/quantize_tensor.py` — full validation mode | Weight MSE/MAE/max/cos, output MSE/cos on random X + calibration X | bonsai |
| 5.2 | Run on representative MoE expert matrices (gate, up, down) | Metrics table per tensor | user |
| 5.3 | CPU vs XPU numerical comparison | Bitwise identical or documented diff | user |

---

## Phase 6: Single MoE Expert

| Task | Description | Acceptance | Owner |
|------|-------------|------------|-------|
| 6.1 | `scripts/quantize_expert.py` — quantize gate/up/down together | Policy-driven, produces expert-level metrics | bonsai |
| 6.2 | Test binary/ternary/diff group sizes | Table: method vs bpw vs error | user |
| 6.3 | Activation reconstruction with real calibration data | Output cosine > 0.95 (target) | user |

---

## Phase 7: Single Block

| Task | Description | Acceptance | Owner |
|------|-------------|------------|-------|
| 7.1 | `scripts/quantize_block.py` — full block (attn + Mamba + MoE) | Input/output cosine, activation MSE, runtime, peak mem | bonsai |
| 7.2 | Run on 3–5 consecutive blocks | Error accumulation measured | user |

---

## Phase 8: Calibration Pipeline

| Task | Description | Acceptance | Owner |
|------|-------------|------------|-------|
| 8.1 | `calibration/datasets.py` — coding/reasoning/agent/mixed sources | 128 seq × 512–2048 tokens default | bonsai |
| 8.2 | `calibration/collector.py` + `hooks.py` — activation capture | Streaming, sharded cache to `cache/calibration/` | bonsai |
| 8.3 | `scripts/collect_calibration.py` — CLI | Manifest with dataset hash, tokenizer hash, model rev, seed | bonsai |
| 8.4 | Run calibration collection | Cache populated, manifest valid | user |

---

## Phase 9: Activation-Aware Scaling

| Task | Description | Acceptance | Owner |
|------|-------------|------------|-------|
| 9.1 | `calibration/statistics.py` — compute E[xx^T] per group | Streaming covariance | bonsai |
| 9.2 | Optimal scale: `scale = sqrt(diag(X^TX)) / mean(|W|)` or similar | Beats naive scaling on activation MSE | bonsai |
| 9.3 | Ablation: naive vs activation-aware on expert/block | Measured improvement | user |

---

## Phase 10: GPTQ / OBQ-style Compensation

| Task | Description | Acceptance | Owner |
|------|-------------|------------|-------|
| 10.1 | `quantizers/gptq_binary.py` — second-order binary quantization | Cholesky on Hessian, damping, block size configurable | bonsai /
| 10.2 | Integrate public impl (e.g., `auto_gptq`, `llm-compressor`) behind interface | Wrapper, record upstream commit | bonsai /
| 10.3 | Numerical stability: NaN/Inf/ill-conditioned detection + explicit error | Never silent fallback | bonsai |
| 10.4 | Compare GPTQ-binary vs naive binary on block | Activation MSE improvement | user |

---

## Phase 11: Rotation / Incoherence

| Task | Description | Acceptance | Owner |
|------|-------------|------------|-------|
| 11.1 | `transforms/hadamard.py` — fast Walsh-Hadamard | In-place, batched | bonsai |
| 11.2 | `transforms/rotations.py` — random orthogonal / QuIP-style | Configurable | bonsai |
| 11.3 | Experiment pairs: binary vs rot+binary, GPTQ vs rot+GPTQ | Measured delta | user |

---

## Phase 12: Salient / Sparse Residual

| Task | Description | Acceptance | Owner |
|------|-------------|------------|-------|
| 12.1 | `quantizers/residual.py` — sparse correction `W ≈ Wq + R` | Magnitude / activation-weighted / Hessian / reconstruction criteria | bonsai |
| 12.2 | `storage/residual_pack.py` — COO/CSR sparse storage | Indices + FP16 values, bpw accounted | bonsai |
| 12.3 | Residual fractions: 0.1% / 0.25% / 0.5% / 1% / 2% / 3% / 5% | Sweep table | user |
| 12.4 | Reference inference: `Y = X@Wq + X@R` | Numerically matches dense | bonsai |

---

## Phase 13: Sensitivity Scan

| Task | Description | Acceptance | Owner |
|------|-------------|------------|-------|
| 13.1 | `scripts/sensitivity_scan.py` — per-class/layer quantization | Leave rest BF16, measure perplexity delta + activation error | bonsai /
| 13.2 | Generate ranking + suggested policy | Auto-suggested YAML | user |

---

## Phase 14: Mamba-Specific Study

| Task | Description | Acceptance | Owner |
|------|-------------|------------|-------|
| 14.1 | Separate Mamba param categories (projections vs state vs conv vs dt vs A/D) | Classifier extended | bonsai |
| 14.2 | Quantize only large projections first | Preserve recurrence dynamics | user |
| 14.3 | Ablate: FP8/Q8/Q4 on state params | Find tolerance threshold | user |

---

## Phase 15: MoE-Specific Study

| Task | Description | Acceptance | Owner |
|------|-------------|------------|-------|
| 15.1 | Router statistics from calibration | Activation freq, token count, prob stats per expert | bonsai /
| 15.2 | Expert-specific quantization policy | Per-expert method/scales/residual% | bonsai /
| 15.3 | Usage-weighted bit allocation optimizer | Maximize quality s.t. total bpw ≤ target | user /

---

## Phase 16: MTP Handling

| Task | Description | Acceptance | Owner |
|------|-------------|------------|-------|
| 16.1 | Identify MTP tensors in classifier | Separate from main blocks | bonsai |
| 16.2 | Keep MTP high-precision initially | Verify main model unaffected | user |

---

## Phase 17: Multi-Block Validation

| Task | Description | Acceptance | Owner |
|------|-------------|------------|-------|
| 17.1 | Quantize 3–5 consecutive blocks end-to-end | Accumulated error measured | user |
| 17.2 | Gate: proceed to full model only if error manageable | Cosine > threshold | user |

---

## Phase 18: Research Storage Format

| Task | Description | Acceptance | Owner |
|------|-------------|------------|-------|
| 18.1 | `storage/manifest.py` + `reader.py` | Config + quant_manifest + source_manifest + tensor shards | bonsai |
| 18.2 | Reproducibility metadata (source rev, policy hash, calib hash, seed, versions) | Embedded in manifest | bonsai |

---

## Phase 19: Full Model Quantization

| Task | Description | Acceptance | Owner |
|------|-------------|------------|-------|
| 19.1 | `scripts/quantize_model.py` — streaming, resumable | Progress state every block, atomic writes | bonsai /
| 19.2 | Cloud package scripts | `package_cloud_job.sh`, `run_cloud_job.sh` | bonsai |
| 19.3 | Full run on cloud GPU (A100/H100) | Produces quantized model at target bpw | user |

---

## Phase 20: Reference Inference

| Task | Description | Acceptance | Owner |
|------|-------------|------------|-------|
| 20.1 | `inference/reference_linear.py` — unpack → matmul | Correctness only, slow OK | bonsai |
| 20.2 | Full model forward pass correctness test | Matches original logits within tolerance | user |

---

## Phase 21: Optimized Inference (Optional, Parallelizable)

| Task | Description | Acceptance | Owner |
|------|-------------|------------|-------|
| 21.1 | CPU: AVX2/AVX-512/VNNI kernels | `inference/binary_linear.py` | codex |
| 21.2 | XPU: SYCL/oneAPI kernels | `inference/xpu_binary.py` | codex |
| 21.3 | CUDA: Triton/CUTLASS kernels | `inference/cuda_binary.py` | codex |

---

## Phase 22: Evaluation Suite

| Task | Description | Acceptance | Owner |
|------|-------------|------------|-------|
| 22.1 | `eval/perplexity.py` — fixed corpus | Deterministic, versioned | bonsai |
| 22.2 | `eval/coding.py` — HumanEval + repo editing | Executable scoring | bonsai /
| 22.3 | `eval/agent.py` — multi-turn tool-use tasks | Success rate, tool calls, tokens | bonsai /
| 22.4 | `eval/reasoning.py` — logic/math deterministic | Auto-checked | bonsai |
| 22.5 | `eval/factuality.py` — trap questions | Hallucination rate | bonsai |
| 22.6 | `scripts/evaluate.py` — unified runner | CSV results DB | bonsai |
| 22.7 | Run evaluation matrix (spec §62) | `artifacts/results.csv` + `report.md` + plots | user |

---

## Phase 23: GGUF / llama.cpp Integration (Optional)

| Task | Description | Acceptance | Owner |
|------|-------------|------------|-------|
| 23.1 | Map research format → GGUF quant type | Or propose new type | codex |
| 23.2 | llama.cpp load + inference test | No misrepresentation | codex |

---

## Execution Order & Dependencies

```
Phase 0 → Phase 1 → Phase 2 → Phase 3 → Phase 4
                                    ↓
                              Phase 5 → Phase 6 → Phase 7
                                    ↓
                              Phase 8 → Phase 9 → Phase 10
                                    ↓              ↓
                              Phase 11 ← Phase 12 ←
                                    ↓
                              Phase 13 → Phase 14 → Phase 15 → Phase 16 → Phase 17
                                    ↓
                              Phase 18 → Phase 19 → Phase 20 → Phase 22
                                    ↓              ↓          ↓
                              (Phase 21)    (Phase 21)  (Phase 23)
```

---

## Current Sprint: Phases 0–6 (First Milestone)

**Goal:** Reproduce SPEC §75 output.

**Immediate next steps:**
1. `benchmark_backend.py` + backend abstractions (Phase 0)
2. `inspect_model.py` + inventory + classifier (Phase 1)
3. Policy system + config examples (Phase 2)
4. Binary quantizer + pack/unpack + tests (Phase 3)
5. Single tensor validation (Phase 5)
6. Single MoE expert (Phase 6)

**Estimated effort:** ~20–30 files, mostly in `nemotron_quant/` and `scripts/`.

**Claude Code can parallelize:**
- Backend abstraction (Phase 0) — independent
- Model inspection (Phase 1) — independent
- Binary/ternary quantizers (Phases 3–4) — independent once base interface exists
- Storage/packing (Phases 3–4) — independent
- Calibration datasets (Phase 8) — can start early

**User must run:**
- `benchmark_backend.py` (local)
- `inspect_model.py` (needs HF access to Nemotron)
- `collect_calibration.py` (needs model + data)
- `quantize_expert.py` / `quantize_block.py` (needs calibration cache)
- Full quantization + evaluation (cloud)

---

## Tracking

Update this file with `[x]` when tasks complete. Add notes on blockers, actual vs expected results, and decisions made.

---

## Phase 24: GGUF Export & Validation (Hard Requirement)

| Task | Description | Acceptance | Owner |
|------|-------------|------------|-------|
| 24.1 | `scripts/export_gguf.py` — research format → GGUF | Maps tensors, packs Q1_0/TQ1_0, writes metadata | codex |
| 24.2 | GGUF Q1_0 layout compliance | Block size 32, FP16 scales, 1-bit packed | codex |
| 24.3 | llama.cpp validation | `llama-cli -m out.gguf -p "def fib(n):" -n 50` clean run | user |
| 24.4 | GGUF metadata: arch, quantization_version, block_size, expert info | Loadable in stock llama.cpp | codex |

## Phase 25: MLX Export & Validation (Hard Requirement)

| Task | Description | Acceptance | Owner |
|------|-------------|------------|-------|
| 25.1 | `scripts/export_mlx.py` — research format → MLX | Safetensors + config, native 2-bit or custom 1-bit | codex |
| 25.2 | MLX 2-bit via `mlx.nn.quantize` | Group size 64, runs on Apple Silicon | codex |
| 25.3 | MLX 1-bit custom backend (if justified) | Custom `QuantizedLinear`, kernel + weight format | codex |
| 25.4 | mlx-lm validation | `mlx_lm.generate --model ./mlx --prompt "def fib(n):"` clean run | user |
| 25.5 | Parity check: GGUF vs MLX on same prompts | Functionally equivalent outputs | user |

## Updated Execution Order

```
...
PHASE 19    full-model quantization
PHASE 19.5  GGUF export + validation  (Phase 24)
PHASE 19.6  MLX export + validation   (Phase 25)
PHASE 20    reference inference (research format)
PHASE 21    optimized inference backends
PHASE 22    evaluation suite
```

**Quality Gate:** Phases 24 & 25 must pass before Phase 22 evaluation on exported artifacts.

---

## Notes on GGUF/MLX for Nemotron Architecture

- **MoE**: llama.cpp supports MoE (e.g., Mixtral). Map expert tensors to `blk.<layer>.ffn_experts.<expert>.*`
- **Mamba**: No native support in llama.cpp/MLX yet. Export as custom tensors; document limitation.
- **MTP**: Export separately; not used in standard inference.
- **Tokenizer**: Must export tokenizer config + vocabulary for both targets.
- **Chat template**: Include in GGUF metadata / MLX config.

---

## Revised First Milestone (Phases 0–6 + Export Prep)

**Goal:** PoC validation + export readiness check.

**Added to sprint:**
- Research storage format design (Phase 18) — must support lossless round-trip to GGUF/MLX
- Binary pack format compatible with GGUF Q1_0 (group_size=32 multiple)
- Tensor naming convention aligned with GGUF expectations

This ensures no "rewrite everything" at export time.