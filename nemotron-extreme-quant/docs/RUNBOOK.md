# Runbook: from zero to a quantized Nemotron-3.5-Lightning-30B-A3B MLX model

This covers two paths:
- **A. Stock-format GPTQ** (`poc/gptq_stock_convert.py` + `poc/run_pipeline.sh`)
  -- GPTQ-calibrated N-bit quantization (no rotation, no salient overlay),
  packed by *stock* `mlx_lm.convert`. Same size/speed as a naive RTN
  conversion at the same bit-width, better quality. No custom loader code
  needed -- loads with plain `mlx_lm`/LM Studio.
- **B. Custom ternary+rotation+salient** (`poc/pack_mlx.py`) -- this
  project's research method for going below what a standard N-bit format
  can do (sub-3-bit routed experts), at the cost of a custom Metal kernel
  and custom `mlx_lm` model file. Slower generation (see docs/FINDINGS.md's
  benchmarks) and needs `trust_remote_code=True`.

Both start from the same source checkpoint and the same calibration corpus.

## 1. Prerequisites

- A RunPod (or similar) GPU pod for the PyTorch/CUDA-side GPTQ work --
  this project used a single A100. Mamba's `causal_conv1d` kernel has no
  CPU fallback, so the model must run on GPU for calibration.
- The Mac that will actually run the resulting MLX model (Apple Silicon).
- `huggingface-cli`/`hf` authenticated on the pod (`hf auth login`) if you
  want to push results to HF.
- This repo's `poc/` directory present on both the pod (for GPTQ) and
  locally (for `run_pipeline.sh`, which syncs `poc/*.py` to the pod itself
  -- you don't need to manually scp anything).

RunPod reassigns a new public host/port every time a stopped pod is
started. Check the current values (`runpodctl pod list` or the RunPod
dashboard) and export them before running anything below if they've
changed since your last session:

```bash
export POD_HOST=1.2.3.4
export POD_PORT=12345
export POD_SSH_KEY=~/.runpod/ssh/runpodctl-ssh-key   # or wherever yours lives
export HF_USER=your-hf-username
```

## 2. Get the source model + calibration corpus onto the pod

```bash
ssh -i $POD_SSH_KEY -p $POD_PORT root@$POD_HOST

# Source checkpoint (bf16, full precision) -- ~62GB
hf download nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16 \
    --local-dir /root/nemotron30b-bf16-src

# Calibration corpus (wikitext-2-raw) -- llama.cpp ships a fetch script,
# or grab it directly:
mkdir -p /root/llama.cpp/wikitext-2-raw
cd /root/llama.cpp/wikitext-2-raw
curl -L -o wikitext-2-raw-v1.zip \
    https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-2-raw-v1.zip
unzip wikitext-2-raw-v1.zip && mv wikitext-2-raw/* . && rmdir wikitext-2-raw
```

Install deps on the pod (PyTorch + transformers + `mlx` with the CUDA
backend, for the stock-conversion path's `mlx_lm.convert` step to run
*on the pod* rather than needing a round-trip to a Mac):

```bash
pip install torch transformers accelerate
pip install "mlx[cuda]" mlx-lm   # NOT "mlx[cuda12]" -- that extra doesn't exist;
                                  # the base "mlx" wheel alone has no CUDA runtime
                                  # on Linux at all (see docs/FINDINGS.md)
```

## 3A. Path A: stock-format GPTQ (recommended default)

One command from your Mac, no manual SSH needed for the actual run:

```bash
cd poc
./run_pipeline.sh --bits 3 --group-size 64 \
    --calib-chunks 24 --calib-chunk-tokens 512 --moe-subbatch 12
```

This: syncs `poc/*.py` to the pod, runs `gptq_stock_convert.py` (GPTQ,
no rotation/salient, uniform `--bits`/`--group-size`), then stock
`mlx_lm.convert -q` on the result, uploads to
`$HF_USER/nemotron-30b-a3b-gptq3bit-g64`, and rsyncs the finished MLX
model to `models/gptq3bit-g64/` in this repo.

Flags worth knowing:
- `--sequential` -- capture each block's calibration activations with all
  *prior* blocks already quantized in place (more faithful, ~1.5x slower).
  See docs/FINDINGS.md (sequential vs. one-shot calibration) for why this
  mattered less than expected for the ternary path; untested for this
  GPTQ-only path.
- `--run-name my-label` -- override the auto-generated name (otherwise
  derived from bits/group-size/sequential).
- `--no-upload` / `--no-download` -- skip either side-effect for a quick
  local-only test.

To try a different bit-width (e.g. 4-bit, likely better quality at ~1GB
more size than 3-bit):

```bash
./run_pipeline.sh --bits 4 --group-size 64 --calib-chunks 24 --calib-chunk-tokens 512 --moe-subbatch 12
```

Loading the result needs nothing special -- it's a stock MLX model:

```bash
mlx_lm.generate --model models/gptq3bit-g64 --prompt "Hello"
# or point mlx_lm.server / LM Studio at models/gptq3bit-g64 directly
```

## 3B. Path B: custom ternary+rotation+salient (this project's research method)

Manual (not yet wrapped by `run_pipeline.sh` -- this path has more moving
parts: a custom Metal kernel and a custom `mlx_lm` model file that must
ship *inside* the model directory). On the pod:

```bash
cd /root/poc
python3 pack_mlx.py \
    --model /root/nemotron30b-bf16-src \
    --output /root/lightning30b-ternary-src \
    --wikitext /root/llama.cpp/wikitext-2-raw/wiki.train.raw \
    --calib-chunks 24 --calib-chunk-tokens 512 --moe-subbatch 12 \
    --salient-fraction 0.06 [--sequential]
```

Patch the output's `config.json` and copy the custom loader files in
(pack_mlx.py's `main()` does the tensor packing but not this config step
-- see docs/FINDINGS.md for why):

```bash
python3 -c "
import json
p='/root/lightning30b-ternary-src/config.json'
cfg = json.load(open(p))
cfg['model_file'] = 'mlx_model_ternary.py'
cfg['quantization'] = {'group_size': 32, 'bits': 8, 'mode': 'affine'}  # matches NBIT_BITS/NBIT_GROUP_SIZE in quantize_full_moe_model.py
json.dump(cfg, open(p, 'w'), indent=2)
"
cp mlx_model_ternary.py rotated_switch_linear.py rotation_mlx.py sparse_salient_mlx.py \
    /root/lightning30b-ternary-src/
```

Loading REQUIRES `trust_remote_code=True` (this is a custom architecture,
not stock `mlx_lm`):

```bash
mlx_lm.generate --model /root/lightning30b-ternary-src --trust-remote-code --prompt "Hello"
```

**Known limitations of this path** (see docs/FINDINGS.md for the full
story): generation throughput measured at ~7 tokens/sec
regardless of internal chunk-size tuning (vs ~30-70 tok/s for stock
3-bit on the same hardware) -- the custom salient-correction kernel's
resolve+gather+scatter path is the bottleneck, not something fixed by a
quick parameter change. Peak memory sits close to this project's 24GB Mac
target's ceiling; `sudo sysctl iogpu.wired_limit_mb=<N>` (see MLX's own
`mx.set_wired_limit` docs) may be needed, at the cost of leaving the OS
less headroom.

## 4. LoRA-merged, MTP-preserving pipeline (self-speculative decoding)

Same Path A pipeline (`run_pipeline.sh`), plus two flags. Neither the LoRA
merge nor GPTQ can see `mtp.*` weights -- HF transformers drops them on
load before either PyTorch step runs (see docs/FINDINGS.md section 4.1) --
so `--mlx-lm-git` triggers extracting them from the untouched bf16 source
up front and splicing them back into the finished MLX model afterward,
around the merge/GPTQ/convert chain rather than by patching any single
step in it:

```bash
cd poc
./run_pipeline.sh --quant-recipe-mode component --component-recipe jang --group-size 64 \
    --lora-adapter roman220220/ipsupport-code-nemotron-lora \
    --mlx-lm-git "git+https://github.com/ipsupport-llc/mlx-lm.git@nemotron-h-mtp"
```

- `--lora-adapter <hf-repo-or-local-dir>`: merges this PEFT adapter into
  the base checkpoint (`poc/merge_lora.py`, via `peft.PeftModel.
  merge_and_unload()`) before GPTQ. Idempotent -- skips the merge if a
  prior run already left a complete merged checkpoint at
  `/root/nemotron30b-bf16-<run-name>-merged`.
- `--mlx-lm-git <git+https url>`: installs this mlx-lm fork/branch instead
  of stock PyPI `mlx-lm` before the conversion step (needs a NemotronH MTP
  head + `sanitize()` that keeps `mtp.*` -- e.g. `ipsupport-llc/mlx-lm@
  nemotron-h-mtp`), AND triggers the extract/inject steps around it
  (`poc/extract_mtp_weights.py` before the merge/GPTQ chain,
  `poc/inject_mtp_weights.py` after `mlx_lm.convert`/`mlx_convert_recipe.
  py`). Using either flag auto-suffixes the run name (`-lora`/`-mtp`), so
  the result is always a separate HF repo from a plain run -- it never
  overwrites an existing production model.
- `--mtp-bits N` (default 8): flat bit-width for the injected head when
  NOT using `--component-recipe`. With `--component-recipe jang`, the
  head instead gets that recipe's own `mtp_attention`/`mtp_moe_shared`/
  `mtp_moe_routed_up`/`mtp_moe_routed_down`/`mtp_fusion` tier (8/8/6/6/8
  bit) -- deliberately higher than the backbone's most aggressive
  `moe_routed_up`/`down` (4/3-bit), since the head is ~4% of total size,
  isn't GPTQ-calibrated (plain RTN), and a degraded draft only costs
  *speedup* (self-speculative accept rate), never correctness.

Verify the head actually made it into the result and the model still
generates correctly:

```bash
python3 -c "
from mlx_lm.utils import load
model, tok = load('models/<run-name>')
print('has mtp:', model.mtp is not None)
"
mlx_lm.generate --model models/<run-name> --prompt "Hello" --max-tokens 50
```

Measure how much speedup the head actually buys (draft accept rate and
tok/s of self-speculative decoding on held-out text) -- the number to
compare MTP-head quantization variants by (RTN-affine vs. GPTQ-affine vs.
RTN-nvfp4); it never affects correctness:

```bash
python poc/eval_mtp_accept_rate.py --model models/<run-name> \
  --text /path/to/wikitext-2-raw/wiki.test.raw \
  --prompts 8 --prompt-tokens 64 --gen-tokens 200
```

It drives `nemotron_h_mtp_generate_step` directly and prints per-prompt
and overall `accept_rate` / `tok/s` (an accepted draft yields two tokens,
the draft plus a free bonus token, which the script accounts for).

There is no CLI flag yet to actually *use* the MTP head for generation
(the self-speculative driver, `nemotron_h_mtp_generate_step` in
`ipsupport-llc/mlx-lm@nemotron-h-mtp`'s `mlx_lm/generate.py`, isn't wired
into `mlx_lm.server`/`mlx_lm.generate`'s CLI yet) -- call it directly, or
from a driver script, the same way its own tests do.

## 5. Comparing results

Perplexity, same corpus, either path's output:

```bash
python3 poc/ppl_wikitext.py --model <hf-checkpoint-dir> --text /root/llama.cpp/wikitext-2-raw/wiki.test.raw \
    --chunks 40 --chunk-tokens 512
```

(For an already-MLX-converted model rather than an HF checkpoint, use a
plain-MLX perplexity script instead -- `poc/gptq_stock_convert.py`'s
output stage, before `mlx_lm.convert`, is still an HF checkpoint and works
directly with `ppl_wikitext.py`.)

Generation speed: `mlx_lm.generate --model <dir> --prompt "..." --max-tokens 100`
prints `tokens-per-sec` for both prompt and generation phases.
