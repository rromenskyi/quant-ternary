#!/usr/bin/env bash
# The pipeline's scripts on a tiny random checkpoint, on a CPU (a Mac is
# fine: seconds, no download but the real model's config and tokenizer):
#   1. the mlx-lm port against transformers (logits, fp32)
#   2. gptq_inkling2.py's driver against the transformers forward
#   3. calibrate -> assemble: the assembled checkpoint, MLX vs transformers
#   4. convert: every quantized tensor on the GPTQ grid, at the recipe's bits
#   5. reference + check on the converted model (vision, answers, perplexity)
#
#   LIKE=<snapshot with config.json + tokenizer> ./smoke_test.sh
# Needs torch, torchvision, transformers>=5.24, mlx and the ipsupport-llc/mlx-lm fork.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LIKE="${LIKE:?a snapshot dir with the real config.json and tokenizer files}"
T="${T:-$(mktemp -d)}"
PY="${PY:-python3}"
export MLX_ENABLE_TF32=0   # exact fp32 on M5 (TF32 broke a 1e-4 comparison)
cd "$HERE"

mkdir -p "$T/data"
"$PY" -c "
import transformers, pathlib
fs = sorted(pathlib.Path(transformers.__path__[0]).rglob('*.py'))
pathlib.Path('$T/data/a.txt').write_text('\n\n'.join(p.read_text() for p in fs[:300]))
pathlib.Path('$T/data/b.txt').write_text('\n\n'.join(p.read_text() for p in fs[300:600]))"
"$PY" make_tiny_checkpoint.py --like "$LIKE" --out "$T/tiny"
"$PY" compare_hf_mlx.py --hf "$T/tiny" --mlx "$T/tiny"
"$PY" test_driver.py --model "$T/tiny"
"$PY" gptq_inkling2.py --model "$T/tiny" --work "$T/gptq" --recipe attn=8,experts=3 --device cpu \
  --calib "a:$T/data/a.txt" --calib "b:$T/data/b.txt" --calib-chunks 8 --calib-chunk-tokens 128
"$PY" assemble_checkpoint.py --model "$T/tiny" --work "$T/gptq" --out "$T/ongrid"
"$PY" convert_mlx.py --hf "$T/ongrid" --out "$T/mlx" --gptq-work "$T/gptq"
"$PY" compare_hf_mlx.py --hf "$T/ongrid" --mlx "$T/ongrid" --quantized "$T/mlx"
"$PY" hf_reference.py --model "$T/ongrid" --out "$T/ref" --wikitext "$T/data/a.txt" --code "$T/data/b.txt" \
  --chunks 2 --chunk-tokens 128
# A random model: its answers are noise and only the machinery is checked.
"$PY" check_mlx.py --model "$T/mlx" --ref "$T/ref" --wikitext "$T/data/a.txt" --code "$T/data/b.txt" \
  --out "$T/check.json" --max-tokens 8 --max-ppl-ratio 1.05
echo "SMOKE_TEST_DONE $T"
