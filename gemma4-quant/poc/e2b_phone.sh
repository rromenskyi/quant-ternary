#!/bin/bash
# The phone E2B (PLE 4-bit, token embeddings 6-bit, the +100 MB 8-bit Linears,
# towers 4-bit) in MLX and GGUF, each scored against its bf16 master.
set -uo pipefail
QAT="${QAT:-/workspace/qat}"; PY="${PY:-/workspace/venv/bin/python}"; HF="${HF:-/workspace/venv/bin/hf}"
HERE="$(cd "$(dirname "$0")" && pwd)"; cd "$HERE"
GGUF="$(ls "$QAT"/E2B-gguf/*q4_0*.gguf | grep -v mmproj | head -1)"
echo "=== MLX"
OUT="$QAT/E2B-phone-mlx"
[ -f "$OUT/config.json" ] || "$PY" qat_aligned_convert.py --hf-checkpoint-dir "$QAT/E2B-unq" --output-dir "$OUT" --gguf "$GGUF" \
    --ple-bits 4 --embed-bits 6 --other-bits 4 --raise-json "$QAT/E2B-sens.json" --raise-budget-mb 100 | tail -2
du -s --block-size=1M "$OUT" | cut -f1 | sed 's/$/ MB MLX/'
"$PY" qat_eval.py --text "$QAT/wiki.test.raw" --ref "$QAT/E2B-unq" --windows 128 --chat --json "$QAT/E2B-phone-eval.json" \
    "$QAT/E2B-qat-mlx-r100" "$OUT"
echo "=== GGUF: llama.cpp"
L=/root/llama.cpp
if [ ! -x "$L/build/bin/llama-quantize" ]; then
  git clone -q --depth 1 https://github.com/ggml-org/llama.cpp "$L"
  cmake -S "$L" -B "$L/build" -DGGML_CUDA=OFF -DLLAMA_CURL=OFF -DCMAKE_BUILD_TYPE=Release > /root/cmake.log 2>&1
  cmake --build "$L/build" -j16 --target llama-quantize llama-perplexity > /root/build.log 2>&1
fi
"$PY" -c "import sentencepiece" 2>/dev/null || /workspace/venv/bin/pip install -q sentencepiece
BF="$QAT/E2B-bf16.gguf"
[ -f "$BF" ] || PYTHONPATH="$L/gguf-py" "$PY" "$L/convert_hf_to_gguf.py" "$QAT/E2B-unq" --outtype bf16 --outfile "$BF" 2>&1 | tail -3
TYPES="$("$PY" gguf_types.py "$QAT/E2B-sens.json" 100)"
OG="$QAT/E2B-phone-Q4_0.gguf"
# shellcheck disable=SC2086
"$L/build/bin/llama-quantize" --tensor-type per_layer_token_embd=q4_k --tensor-type token_embd=q6_k $TYPES "$BF" "$OG" Q4_0 16 2>&1 | tail -3
ls -la "$OG" "$GGUF" "$BF" | awk '{print $5, $9}'
echo "=== GGUF: the q4_0 tensors against Google's"
PYTHONPATH="$L/gguf-py" "$PY" - "$OG" "$GGUF" <<'PYEOF'
import sys
from gguf import GGUFReader
a, b = GGUFReader(sys.argv[1]), GGUFReader(sys.argv[2])
tb = {t.name: t for t in b.tensors}
same = diff = 0
for t in a.tensors:
    o = tb.get(t.name)
    if o is None or t.tensor_type != o.tensor_type or t.tensor_type.name != "Q4_0":
        continue
    if bytes(t.data) == bytes(o.data): same += 1
    else: diff += 1
print(f"Q4_0 tensors identical to Google's: {same}, different: {diff}")
PYEOF
echo "=== GGUF: KL to the bf16 GGUF (raw text, 32 chunks of 512)"
P="$L/build/bin/llama-perplexity"
"$P" -m "$BF" -f "$QAT/wiki.test.raw" -c 512 --chunks 32 -t 16 --kl-divergence-base "$QAT/E2B-bf16.kld" 2>&1 | grep -E "Final estimate"
for m in "$GGUF" "$OG"; do
  echo "-- $(basename "$m")"
  "$P" -m "$m" -f "$QAT/wiki.test.raw" -c 512 --chunks 32 -t 16 --kl-divergence-base "$QAT/E2B-bf16.kld" --kl-divergence 2>&1 \
    | grep -E "Mean PPL\(Q\)|Mean    KLD|Same top p" | head -4
done
echo E2B_PHONE_DONE
