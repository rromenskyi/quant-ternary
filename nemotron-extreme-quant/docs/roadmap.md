# Roadmap

Owner key: **agent** = can be implemented by a coding agent unattended,
**user** = needs a human at a keyboard with hardware/model access (running a
benchmark, downloading a gated model, kicking off a cloud job), **kernel** =
specialized low-level kernel work (AVX/SYCL/CUDA/Triton), usually a separate
track from the rest of the pipeline.

Update the checkboxes as work lands. Note blockers and actual-vs-expected
results inline rather than in a separate log.

## Current status

- [x] Phase 0 — backend abstraction (`base.py`, `cpu.py`, `xpu.py`,
      `cuda.py`) and `scripts/benchmark_backend.py`
- [~] Phase 1 — `model/inspector.py` and `model/tensor_classifier.py` exist;
      `tensor_inventory.py` and the `inspect_model.py` CLI entry point are
      not written yet
- [ ] Everything from Phase 2 onward

The PoC described in [`poc.md`](poc.md) runs independently of this roadmap,
against a small dense model, to validate the quantization math before more
of this plan gets built out.

---

## Phase 0 — Environment & backend discovery

| Task | Description | Acceptance | Owner |
|---|---|---|---|
| 0.1 | `scripts/benchmark_backend.py` | Runs on CPU/XPU/CUDA, prints a capabilities JSON | agent |
| 0.2 | `backends/base.py` + `cpu.py` | Abstract device ops: matmul, cholesky, qr, eig, hadamard, topk | agent |
| 0.3 | `backends/xpu.py` | XPU backend with explicit fallback logging | agent |
| 0.4 | `backends/cuda.py` | CUDA backend (stub until a cloud run needs it) | agent |
| 0.5 | Run the benchmark, emit `artifacts/backend_capabilities.json` | JSON matches the spec format | user |

## Phase 1 — Model inspection & tensor inventory

| Task | Description | Acceptance | Owner |
|---|---|---|---|
| 1.1 | `model/inspector.py` — read HF config + safetensors index | No model weights loaded | agent |
| 1.2 | `model/tensor_inventory.py` — full inventory | CSV/JSON: name, shape, dtype, params, bytes, shard, block, expert, class, eligibility | agent |
| 1.3 | `model/tensor_classifier.py` — semantic classification | Rules-based (regex + architecture heuristics) | agent |
| 1.4 | `scripts/inspect_model.py` — CLI | Produces `artifacts/model_inventory.{json,csv}` + `model_summary.md` | agent |
| 1.5 | Run inspection on the target checkpoint | Summary shows total params, by-class breakdown, hypothetical bpw table | user |

## Phase 2 — Quantization policy system

| Task | Description | Acceptance | Owner |
|---|---|---|---|
| 2.1 | Config schema: `configs/quant/*.yaml` | Binary, ternary, GPTQ, residual variants | agent |
| 2.2 | Policy loader: `quantizers/policy.py` | Resolves default + ordered rules (class, regex, layer range, expert range, size threshold) | agent |
| 2.3 | Protected tensor list (norms, router, Mamba state/dt/A/D, embeddings, lm_head) | Defaults to BF16 unless explicitly overridden | agent |
| 2.4 | `scripts/quantize_tensor.py` — dry-run mode | Shows which method a given tensor would resolve to, without quantizing | agent |

## Phase 3 — Naive binary quantization

| Task | Description | Acceptance | Owner |
|---|---|---|---|
| 3.1 | `quantizers/binary.py` — group-wise `scale = mean(\|W\|)`, `B = sign(W)` | Group sizes 32/64/128/256 | agent |
| 3.2 | `storage/binary_pack.py` — pack/unpack 8 weights/byte | Exact round-trip on random + adversarial tensors | agent |
| 3.3 | `tests/test_binary_pack.py` | 100% pass | agent |
| 3.4 | `quantizers/base.py` — common interface | `quantize(tensor, config) -> QuantizedTensor` | agent |
| 3.5 | Effective-bpw calculator (scales + padding + metadata) | Matches the spec's accounting formula | agent |

## Phase 4 — Ternary quantization

| Task | Description | Acceptance | Owner |
|---|---|---|---|
| 4.1 | `quantizers/ternary.py` — threshold-based `{-s, 0, +s}` | Fixed / per-tensor / per-group threshold search | agent |
| 4.2 | `storage/ternary_pack.py` — 2-bit packing (4 values/byte) | Exact round-trip | agent |
| 4.3 | `tests/test_ternary_pack.py` | 100% pass | agent |

## Phase 5 — Single-tensor validation

| Task | Description | Acceptance | Owner |
|---|---|---|---|
| 5.1 | `scripts/quantize_tensor.py` — full validation mode | Weight MSE/MAE/max/cosine; output MSE/cosine on random *and* calibration input | agent |
| 5.2 | Run on representative MoE expert matrices (gate/up/down) | Per-tensor metrics table | user |
| 5.3 | CPU vs XPU numerical comparison | Bitwise identical, or the diff is documented | user |

## Phase 6 — Single MoE expert

| Task | Description | Acceptance | Owner |
|---|---|---|---|
| 6.1 | `scripts/quantize_expert.py` — gate/up/down together | Policy-driven, expert-level metrics | agent |
| 6.2 | Sweep binary/ternary/residual × group size | Table: method × bpw × error | user |
| 6.3 | Activation reconstruction on real calibration data | Output cosine > 0.95 (target) | user |

## Phase 7 — Single block

| Task | Description | Acceptance | Owner |
|---|---|---|---|
| 7.1 | `scripts/quantize_block.py` — full block (attention + Mamba + MoE) | Input/output cosine, activation MSE, runtime, peak memory | agent |
| 7.2 | Run on 3–5 consecutive blocks | Error accumulation measured | user |

## Phase 8 — Calibration pipeline

| Task | Description | Acceptance | Owner |
|---|---|---|---|
| 8.1 | `calibration/datasets.py` — coding/reasoning/agent/mixed sources | 128 sequences × 512–2048 tokens by default | agent |
| 8.2 | `calibration/collector.py` + `hooks.py` — activation capture | Streaming, sharded cache under `cache/calibration/` | agent |
| 8.3 | `scripts/collect_calibration.py` | Manifest records dataset hash, tokenizer hash, model revision, seed | agent |
| 8.4 | Run calibration collection | Cache populated, manifest valid | user |

## Phase 9 — Activation-aware scaling

| Task | Description | Acceptance | Owner |
|---|---|---|---|
| 9.1 | `calibration/statistics.py` — streaming `E[xx^T]` per group | Numerically stable over the full calibration set | agent |
| 9.2 | Optimal scale, e.g. `scale = sqrt(diag(X^TX)) / mean(\|W\|)` | Beats naive scaling on activation MSE | agent |
| 9.3 | Ablation: naive vs. activation-aware, expert- and block-level | Measured improvement | user |

## Phase 10 — GPTQ / OBQ-style compensation

| Task | Description | Acceptance | Owner |
|---|---|---|---|
| 10.1 | `quantizers/gptq_binary.py` — second-order binary quantization | Cholesky on the Hessian, configurable damping and block size | agent |
| 10.2 | Wrap a public implementation (`auto_gptq`, `llm-compressor`) behind our interface | Upstream commit recorded | agent |
| 10.3 | NaN/Inf/ill-conditioning detection | Explicit error, never a silent fallback | agent |
| 10.4 | Compare GPTQ-binary vs. naive binary on a block | Activation-MSE improvement measured | user |

## Phase 11 — Rotation / incoherence processing

| Task | Description | Acceptance | Owner |
|---|---|---|---|
| 11.1 | `transforms/hadamard.py` — fast Walsh–Hadamard transform | In-place, batched | agent |
| 11.2 | `transforms/rotations.py` — random orthogonal / QuIP-style rotation | Configurable | agent |
| 11.3 | Paired experiments: binary vs. rotated+binary, GPTQ vs. rotated+GPTQ | Measured delta | user |

## Phase 12 — Salient / sparse residual correction

| Task | Description | Acceptance | Owner |
|---|---|---|---|
| 12.1 | `quantizers/residual.py` — `W ≈ Wq + R` | Magnitude / activation-weighted / Hessian / reconstruction-based selection | agent |
| 12.2 | `storage/residual_pack.py` — COO/CSR sparse storage | Indices + FP16 values, bpw accounted for | agent |
| 12.3 | Sweep residual fraction: 0.1% / 0.25% / 0.5% / 1% / 2% / 3% / 5% | Table | user |
| 12.4 | Reference inference `Y = X@Wq + X@R` | Matches dense numerically | agent |

## Phase 13 — Sensitivity scan

| Task | Description | Acceptance | Owner |
|---|---|---|---|
| 13.1 | `scripts/sensitivity_scan.py` — quantize one class/layer at a time, rest stays BF16 | Perplexity delta + activation error per tensor class | agent |
| 13.2 | Rank sensitivity, auto-suggest a policy | Suggested YAML | user |

## Phase 14 — Mamba-specific study

| Task | Description | Acceptance | Owner |
|---|---|---|---|
| 14.1 | Separate Mamba parameter categories (projections vs. state vs. conv vs. dt vs. A/D) | Classifier extended | agent |
| 14.2 | Quantize only the large projections first | Recurrence dynamics preserved | user |
| 14.3 | Ablate FP8/Q8/Q4 on state parameters | Find the tolerance threshold | user |

## Phase 15 — MoE-specific study

| Task | Description | Acceptance | Owner |
|---|---|---|---|
| 15.1 | Router statistics from calibration | Per-expert activation frequency, token count, probability stats | agent |
| 15.2 | Per-expert quantization policy | Method/scales/residual % can vary by expert | agent |
| 15.3 | Usage-weighted bit-allocation optimizer | Maximizes quality subject to a total-bpw budget | user |

## Phase 16 — MTP handling

| Task | Description | Acceptance | Owner |
|---|---|---|---|
| 16.1 | Identify MTP tensors in the classifier | Kept separate from the main blocks | agent |
| 16.2 | Keep MTP high-precision initially | Main model unaffected, verified | user |

## Phase 17 — Multi-block validation

| Task | Description | Acceptance | Owner |
|---|---|---|---|
| 17.1 | Quantize 3–5 consecutive blocks end-to-end | Accumulated error measured | user |
| 17.2 | Gate: only proceed to the full model if error is manageable | Cosine above threshold | user |

## Phase 18 — Research storage format

| Task | Description | Acceptance | Owner |
|---|---|---|---|
| 18.1 | `storage/manifest.py` + `reader.py` | Config + quant manifest + source manifest + tensor shards | agent |
| 18.2 | Reproducibility metadata (source revision, policy hash, calibration hash, seed, versions) | Embedded in the manifest | agent |

This format must support a lossless round-trip into both export targets
below — see Phases 19.5/19.6. In particular, the binary pack format from
Phase 3 should already be compatible with GGUF's `Q1_0` layout
(group size a multiple of 32), and tensor naming should be chosen with
GGUF's naming convention in mind, so export doesn't require a rewrite.

## Phase 19 — Full model quantization

| Task | Description | Acceptance | Owner |
|---|---|---|---|
| 19.1 | `scripts/quantize_model.py` — streaming, resumable | Progress checkpointed per block, atomic writes | agent |
| 19.2 | Cloud packaging scripts | `package_cloud_job.sh`, `run_cloud_job.sh` | agent |
| 19.3 | Full run on cloud GPU (A100/H100) | Produces a quantized model at the target bpw | user |

## Phase 19.5 — GGUF export & validation (hard requirement)

| Task | Description | Acceptance | Owner |
|---|---|---|---|
| 19.5.1 | `scripts/export_gguf.py` | Maps tensors, packs Q1_0/TQ1_0, writes metadata | kernel |
| 19.5.2 | GGUF Q1_0 layout compliance | Block size 32, FP16 scales, 1-bit packed | kernel |
| 19.5.3 | llama.cpp validation | `llama-cli -m out.gguf -p "def fib(n):" -n 50` runs clean | user |
| 19.5.4 | GGUF metadata: arch, quantization_version, block_size, expert info | Loadable in stock llama.cpp | kernel |

## Phase 19.6 — MLX export & validation (hard requirement)

| Task | Description | Acceptance | Owner |
|---|---|---|---|
| 19.6.1 | `scripts/export_mlx.py` | Safetensors + config, native 2-bit or custom 1-bit | kernel |
| 19.6.2 | MLX 2-bit via `mlx.nn.quantize` | Group size 64, runs on Apple Silicon | kernel |
| 19.6.3 | MLX 1-bit custom backend, if justified | Custom `QuantizedLinear` + kernel | kernel |
| 19.6.4 | `mlx_lm` validation | `mlx_lm.generate --model ./mlx --prompt "def fib(n):"` runs clean | user |
| 19.6.5 | GGUF vs. MLX parity check | Functionally equivalent outputs on the same prompts | user |

**Quality gate:** Phases 19.5 and 19.6 must pass before Phase 22 evaluates
the exported artifacts.

## Phase 20 — Reference inference

| Task | Description | Acceptance | Owner |
|---|---|---|---|
| 20.1 | `inference/reference_linear.py` — unpack → matmul | Correctness only; slow is fine | agent |
| 20.2 | Full-model forward-pass correctness test | Matches original logits within tolerance | user |

## Phase 21 — Optimized inference (optional, parallelizable)

| Task | Description | Acceptance | Owner |
|---|---|---|---|
| 21.1 | CPU: AVX2/AVX-512/VNNI kernels | `inference/binary_linear.py` | kernel |
| 21.2 | XPU: SYCL/oneAPI kernels | `inference/xpu_binary.py` | kernel |
| 21.3 | CUDA: Triton/CUTLASS kernels | `inference/cuda_binary.py` | kernel |

## Phase 22 — Evaluation suite

| Task | Description | Acceptance | Owner |
|---|---|---|---|
| 22.1 | `eval/perplexity.py` — fixed corpus | Deterministic, versioned | agent |
| 22.2 | `eval/coding.py` — HumanEval + repo editing | Executable scoring | agent |
| 22.3 | `eval/agent.py` — multi-turn tool-use tasks | Success rate, tool calls, tokens | agent |
| 22.4 | `eval/reasoning.py` — logic/math | Auto-checked, deterministic | agent |
| 22.5 | `eval/factuality.py` — trap questions | Hallucination rate | agent |
| 22.6 | `scripts/evaluate.py` — unified runner | CSV results database | agent |
| 22.7 | Run the full evaluation matrix | `artifacts/results.csv` + `report.md` + plots | user |

## Phase 23 — GGUF / llama.cpp integration follow-ups (optional)

| Task | Description | Acceptance | Owner |
|---|---|---|---|
| 23.1 | Map the research format to a GGUF quant type, or propose a new one | — | kernel |
| 23.2 | llama.cpp load + inference test | No misrepresentation of the quant type | kernel |

---

## Dependency graph

```
Phase 0 → Phase 1 → Phase 2 → Phase 3 → Phase 4
                                    │
                              Phase 5 → Phase 6 → Phase 7
                                    │
                              Phase 8 → Phase 9 → Phase 10
                                    │              │
                              Phase 11 ←───────────┘
                                    │
                              Phase 12
                                    │
                              Phase 13 → Phase 14 → Phase 15 → Phase 16 → Phase 17
                                    │
                              Phase 18 → Phase 19 → Phase 19.5 → Phase 19.6
                                                          │            │
                                                     Phase 20      Phase 22
                                                          │
                                                    (Phase 21, optional)
```

## First milestone: Phases 0–6

**Goal:** reproduce the metrics table in [`poc.md`](poc.md) against a real
checkpoint, end to end.

Parallelizable independently once the base interfaces exist:
- Backend abstraction (Phase 0)
- Model inspection (Phase 1)
- Binary/ternary quantizers + packing (Phases 3–4)
- Calibration datasets (Phase 8) can start early even though it's used later

Steps that need a human:
- `benchmark_backend.py` (local)
- `inspect_model.py` (needs model access)
- `collect_calibration.py` (needs model + data)
- `quantize_expert.py` / `quantize_block.py` (needs the calibration cache)
- Full quantization + evaluation (cloud)
