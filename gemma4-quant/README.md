# gemma4-quant

Quantization of Google's Gemma 4 for Apple Silicon (MLX) and for
llama.cpp / ollama (GGUF), plus Gemma 4's MTP drafter for speculative
decoding. Sibling of [`nemotron-extreme-quant`](../nemotron-extreme-quant)
and [`zimage-quant`](../zimage-quant); reuses their GPTQ implementation
(`poc/gptq.py`).

## Published

| HF repo | What | Size |
|---|---|---|
| [`roman220220/gemma-4-26B-A4B-it-gptq-mlx-jang`](https://huggingface.co/roman220220/gemma-4-26B-A4B-it-gptq-mlx-jang) | 26B-A4B MoE, text + vision, MLX | ~15GB (from 51.6GB) |
| [`roman220220/gemma-4-E4B-it-gptq-mlx-jang`](https://huggingface.co/roman220220/gemma-4-E4B-it-gptq-mlx-jang) | E4B, text + vision + audio, MLX | ~6.3GB |
| [`roman220220/gemma-4-26B-A4B-it-GGUF-jang-imatrix`](https://huggingface.co/roman220220/gemma-4-26B-A4B-it-GGUF-jang-imatrix) | 26B-A4B GGUF + mmproj + ollama Modelfile + MTP drafter | 12.5GB + 0.8GB + 0.46GB |
| [`roman220220/gemma-4-26B-A4B-it-assistant-mlx-8bit`](https://huggingface.co/roman220220/gemma-4-26B-A4B-it-assistant-mlx-8bit) | 26B MTP drafter, MLX 8-bit | 446MB |

Model cards live in [`cards/`](cards). The pipelines copy them in, so edit
them here, not on HF.

The two MLX repos predate the leftover-RTN fix (FINDINGS: "uncalibrated
Linears shipped in bf16"). The E4B carries ~30MB of bf16 audio projections
plus ~110MB of never-loaded weights, and the 26B carries 6.5MB. Republishing
was deliberately deferred: the gain is negligible. The next real re-release
picks the fix up automatically.

## Method, in one paragraph each

**MLX: GPTQ with JANG bit allocation.** GPTQ (Hessian-corrected)
quantization calibrated on real data: diverse text prompts, COCO photos,
LibriSpeech clips.
- Bits are spent by role. Attention is 8-bit, the FFN / 128 routed experts
  4-bit, the router stays untouched, and embeddings plus every other Linear
  get 8-bit RTN, so nothing large is left in bf16.
- Calibration runs on a CUDA pod with torch. The splice runs on the same
  pod with `mlx[cuda]`, so the corrected weights never round-trip through a
  Mac.

**GGUF: imatrix + JANG.** llama.cpp's k-quant grid doesn't match MLX's, so
GPTQ corrections don't transfer. Instead:
- the importance matrix is computed on a broad real corpus;
- JANG is expressed as `llama-quantize` per-tensor overrides: attention
  Q5_K, experts IQ3_S.

Plus the fixes needed for the result to actually chat: a tokenizer fix, a
control-token `token_type` fix, and an ollama `RENDERER`/`PARSER` Modelfile.

**MTP drafter.** Google's 4-layer `gemma-4-26B-A4B-it-assistant` reads the
main model's hidden state and KV cache and guesses a few tokens ahead. The
main model verifies them all in one pass, so the output stays the main
model's own.
- The MLX support is ours: `gemma4_assistant` in
  [ipsupport-llc/mlx-lm](https://github.com/ipsupport-llc/mlx-lm).
- The GGUF is ollama's own draft layer.

Measured on the 26B JANG on a 26GB Mac, with an empty prompt cache: 35.8 →
52.9 tok/s. The benchmark also covers the case where the server already has
the prompt start cached.

## Layout

```
poc/
  gemma4_mlx_pipeline.sh          MLX pipeline (VARIANT=26b|e4b)
  gemma4_26b_gguf_pipeline.sh     GGUF pipeline
  gemma4_mtp_drafter_pipeline.sh  MTP drafter pipeline
  pipeline_dashboard.py           live dashboard for all three (stdlib only)
  pipeline_lib.sh, pipeline_status.py   shared step/status plumbing
  gemma4{,_26b}_gptq_calibrate.py GPTQ calibration (torch)
  gemma4{,_26b}_gptq_splice.py    quantize + re-shard into MLX format (mlx)
  splice_common.py                leftover RTN / keep-float / dead-weight rules
  gemma4_smoke_test.py            release gate: text, vision, audio, no float leftovers
  gemma4_26b_fix_tokenizer.py, gemma4_gguf_fix_token_types.py   GGUF fixes
  gemma4_mtp_drafter_parity.py, gemma4_mtp_bench.py            drafter checks
  gptq.py, quantize.py            GPTQ implementation, byte-identical copies of nemotron-extreme-quant/poc (so the folder rsyncs to a pod self-contained)
  test_pipeline_tools.py, test_26b_moe_calibrate.py            tests
cards/                            HF model cards (source of truth)
docs/
  RUNBOOK.md                      exact commands, pod to published repo
  FINDINGS.md                     every bug and gotcha, and why the scripts look the way they do
```

Start with [docs/RUNBOOK.md](docs/RUNBOOK.md).
