#!/usr/bin/env bash
# GGUF pipeline for Liquid AI LFM2.5 dense checkpoints (1.2B-Instruct,
# 1.2B-Thinking, 2.6B) on a CUDA pod: imatrix on a broad corpus, plus a
# JANG-style per-tensor recipe mirroring the MLX GPTQ releases' bit split
# (attn=8, shortconv=6, mlp=4 -- see lfm2-quant/docs/FINDINGS.md) expressed
# as llama-quantize --tensor-type overrides instead of GPTQ codes.
#
# LFM2's hybrid decoder has no "attention" in most layers -- it's gated
# short convolutions (blk.N.shortconv.{in_proj,conv,out_proj}) with full
# GQA attention in only a handful of layers. The conv kernel itself
# (shortconv.conv, 3 taps) is left to llama-quantize's own default instead
# of forced, same as norms: too small to matter for size, and llama.cpp
# falls back to a compatible type on its own when a tensor's dims don't
# divide the requested block size.
#
#   VARIANT=lfm2-2.6b ./lfm2_gguf_pipeline.sh                # build + ppl + smoke
#   VARIANT=lfm2-1.2b-instruct PUBLISH=1 ./lfm2_gguf_pipeline.sh
#
# RUNTIME NOTE: launch with `llama-server --jinja` (or the shipped
# Modelfile) so the chat template's <think> handling applies correctly on
# the Thinking variant.
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/../../gemma4-quant/poc/pipeline_lib.sh"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

VARIANT="${VARIANT:-lfm2-2.6b}"
LLAMA="${LLAMA:-$WORK/llama.cpp}"
BIN="$LLAMA/build/bin"
CALIB="${CALIB:-$WORK/calibration_datav3.txt}"
CALIB_URL="https://gist.githubusercontent.com/bartowski1182/eb213dccb3571f863da82e99418f81e8/raw/calibration_datav3.txt"
NGL="${NGL:-99}"

case "$VARIANT" in
  lfm2-2.6b)
    MODEL_ID="${MODEL_ID:-LiquidAI/LFM2.5-2.6B}"
    HF_REPO="${HF_REPO:-roman220220/LFM2.5-2.6B-GGUF-jang-imatrix}"
    OLLAMA_PARAMS="temperature 0.1
PARAMETER top_k 50
PARAMETER repeat_penalty 1.1"
    ;;
  lfm2-1.2b-instruct)
    MODEL_ID="${MODEL_ID:-LiquidAI/LFM2.5-1.2B-Instruct}"
    HF_REPO="${HF_REPO:-roman220220/LFM2.5-1.2B-Instruct-GGUF-jang-imatrix}"
    OLLAMA_PARAMS="temperature 0.1
PARAMETER top_k 50
PARAMETER top_p 0.1
PARAMETER repeat_penalty 1.05"
    ;;
  lfm2-1.2b-thinking)
    MODEL_ID="${MODEL_ID:-LiquidAI/LFM2.5-1.2B-Thinking}"
    HF_REPO="${HF_REPO:-roman220220/LFM2.5-1.2B-Thinking-GGUF-jang-imatrix}"
    OLLAMA_PARAMS="temperature 0.05
PARAMETER top_k 50
PARAMETER repeat_penalty 1.05"
    ;;
  *) echo "VARIANT must be lfm2-2.6b, lfm2-1.2b-instruct or lfm2-1.2b-thinking" >&2; exit 2 ;;
esac
NAME="${HF_REPO#*/}"
WORK_NAME="${MODEL_ID##*/}"
SNAPSHOT_FIXED="${SNAPSHOT_FIXED:-}"   # filled in after download
F16="$WORK/${WORK_NAME}-f16.gguf"
IMATRIX="$WORK/${WORK_NAME}.imatrix"
OUT_DIR="${OUT_DIR:-$WORK/$NAME}"
CARD="$HERE/../cards/$NAME.md"
PPL_TEST="${PPL_TEST:-$WORK/wikitext-2-raw/wiki.test.raw}"

# --- recipe (env-driven for PPL sweeps) --------------------------------------
BASE_QUANT="${BASE_QUANT:-Q4_K_M}"   # llama-quantize base preset (MLP/FFN)
ATTN_TYPE="${ATTN_TYPE:-Q8_0}"       # full-attention layers + embeddings: few, sensitive
CONV_TYPE="${CONV_TYPE:-Q6_K}"       # gated short convolution in/out projections: most layers
RUN_TAG="${RUN_TAG:-jang}"
TEXT_GGUF="$OUT_DIR/${WORK_NAME,,}-${RUN_TAG}.gguf"

pipeline_init "gguf-lfm2-$VARIANT" "VARIANT=$VARIANT MODEL_ID=$MODEL_ID OUT_DIR=$OUT_DIR RECIPE=mlp=$BASE_QUANT/attn=$ATTN_TYPE/shortconv=$CONV_TYPE"
mkdir -p "$OUT_DIR"

# --- llama.cpp (CUDA, current source; LFM2 needs the conversion/lfm2.py port) -
if [ -x "$BIN/llama-quantize" ] && [ -x "$BIN/llama-imatrix" ] && [ -x "$BIN/llama-perplexity" ]; then
  skip_step llama_build "binaries present in $BIN"
else
  run_step llama_build "clone + build llama.cpp (CUDA)" bash -c "
    [ -d '$LLAMA' ] || git clone --depth 1 https://github.com/ggml-org/llama.cpp '$LLAMA'
    cd '$LLAMA' && pip install --quiet --break-system-packages -r requirements/requirements-convert_hf_to_gguf.txt \
      huggingface_hub hf_transfer
    # llama.cpp's pinned requirements bring an old transformers; LFM2.5's
    # tokenizer_config.json needs transformers>=5.8 (TokenizersBackend).
    pip install --quiet --break-system-packages --upgrade 'transformers>=5.8' tokenizers
    export PATH=/usr/local/cuda/bin:\$PATH CUDACXX=/usr/local/cuda/bin/nvcc
    cmake -B build -DGGML_CUDA=ON -DLLAMA_CURL=OFF && cmake --build build --config Release -j \$(nproc)"
fi

# --- source + calibration data -------------------------------------------------
_PIPE_CUR=download; _pipe_mark START download "$MODEL_ID"
SNAPSHOT="$(hf_snapshot "$MODEL_ID")"
_pipe_mark DONE download "$SNAPSHOT"; _PIPE_CUR=""
if [ -s "$CALIB" ]; then skip_step calib_data "$CALIB exists"; else
  run_step calib_data "fetch bartowski calibration_datav3" curl -sfL -o "$CALIB" "$CALIB_URL"
fi

# --- convert to F16 GGUF (conversion/lfm2.py: Lfm2ForCausalLM is registered) --
if [ -f "$F16" ]; then skip_step convert_f16 "$F16 exists"; else
  run_step convert_f16 "bf16 -> f16 GGUF" \
    python3 "$LLAMA/convert_hf_to_gguf.py" "$SNAPSHOT" --outfile "$F16" --outtype f16
fi

# --- imatrix (GPU) ------------------------------------------------------------
if [ -f "$IMATRIX" ]; then skip_step imatrix "$IMATRIX exists"; else
  run_step imatrix "llama-imatrix, 200 chunks" \
    "$BIN/llama-imatrix" -m "$F16" -f "$CALIB" -o "$IMATRIX" -ngl "$NGL" --chunks 200
fi

# --- quantize (JANG recipe: attn+embeddings high, shortconv mid, mlp base) ---
if [ -f "$TEXT_GGUF" ]; then skip_step quantize "$TEXT_GGUF exists"; else
  TT=(--tensor-type "attn_q=$ATTN_TYPE" --tensor-type "attn_k=$ATTN_TYPE"
      --tensor-type "attn_v=$ATTN_TYPE" --tensor-type "attn_output=$ATTN_TYPE"
      --tensor-type "token_embd=$ATTN_TYPE" --tensor-type "output\\.weight=$ATTN_TYPE"
      --tensor-type "shortconv\\.in_proj=$CONV_TYPE" --tensor-type "shortconv\\.out_proj=$CONV_TYPE")
  run_step quantize "llama-quantize $BASE_QUANT (attn/embed $ATTN_TYPE, shortconv $CONV_TYPE)" \
    "$BIN/llama-quantize" --imatrix "$IMATRIX" "${TT[@]}" "$F16" "$TEXT_GGUF" "$BASE_QUANT" 32
fi

# --- perplexity (quality record: quant vs F16) -------------------------------
# The classic research.metamind.io S3 mirror (used by the gemma4 GGUF
# pipelines) now 403s / redirects without a Location header; pull the same
# text from the HF-hosted parquet instead (same source make_data.py uses).
if [ -s "$PPL_TEST" ]; then skip_step ppl_data "$PPL_TEST exists"; else
  run_step ppl_data "fetch wikitext-2-raw test (HF parquet)" bash -c "
    mkdir -p '$(dirname "$PPL_TEST")'
    pip install --quiet --break-system-packages pandas pyarrow
    python3 -c \"
import io, pathlib, urllib.request
import pandas as pd
url = 'https://huggingface.co/datasets/Salesforce/wikitext/resolve/main/wikitext-2-raw-v1/test-00000-of-00001.parquet'
df = pd.read_parquet(io.BytesIO(urllib.request.urlopen(url).read()))
pathlib.Path('$PPL_TEST').write_text(''.join(df['text']), encoding='utf-8')
\""
fi
run_step ppl "wikitext PPL: F16 vs $BASE_QUANT ($RUN_TAG)" bash -c "
  echo -n 'F16          : '; '$BIN/llama-perplexity' -m '$F16' -f '$PPL_TEST' -ngl '$NGL' --chunks ${PPL_CHUNKS:-40} 2>&1 | grep -E 'Final estimate' | tail -1
  echo -n '$BASE_QUANT ($RUN_TAG): '; '$BIN/llama-perplexity' -m '$TEXT_GGUF' -f '$PPL_TEST' -ngl '$NGL' --chunks ${PPL_CHUNKS:-40} 2>&1 | grep -E 'Final estimate' | tail -1"

# --- ollama Modelfile ----------------------------------------------------------
# No dedicated LFM2 RENDERER/PARSER is known to ship in Ollama (unlike the
# gemma4 pipelines); the GGUF carries LFM2's own jinja chat template via
# convert_hf_to_gguf.py, which recent Ollama applies directly. If thinking
# or tool-call tokens ever leak as raw text, fall back to llama.cpp --jinja.
write_ollama_files () {
  cat > "$OUT_DIR/Modelfile" << EOF
# Ollama Modelfile for $HF_REPO
#   curl -L -o Modelfile https://huggingface.co/$HF_REPO/resolve/main/Modelfile
#   ollama create ${NAME,,} -f Modelfile && ollama run ${NAME,,}
FROM hf.co/$HF_REPO
PARAMETER $OLLAMA_PARAMS
EOF
  cat "$OUT_DIR/Modelfile"
}
run_step ollama_files "Modelfile (FROM hf.co + Liquid AI's sampling defaults)" write_ollama_files

# --- smoke: llama-server --jinja ----------------------------------------------
gguf_smoke () {
  local port=18092
  "$BIN/llama-server" -m "$TEXT_GGUF" -ngl "$NGL" --jinja --port "$port" \
    > "$LOG_DIR/gguf_lfm2_smoke_server.log" 2>&1 &
  local pid=$! rc=0
  for _ in $(seq 1 180); do curl -sf "localhost:$port/health" >/dev/null && break; sleep 2; done
  if python3 - "$port" "$VARIANT" <<'PY'
import json, sys, urllib.request
port, variant = sys.argv[1], sys.argv[2]
body = {"messages": [{"role": "user", "content": "List three facts about the Roman Empire, one line each."}], "max_tokens": 2048, "temperature": 0}
req = urllib.request.Request(f"http://localhost:{port}/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
msg = json.load(urllib.request.urlopen(req, timeout=600))["choices"][0]["message"]
# the Thinking variant's <think> block is split into reasoning_content by
# llama-server's --jinja reasoning parser; fall back to it if content is empty
# (budget spent entirely on reasoning), it's still evidence of a coherent model.
text = msg.get("content") or msg.get("reasoning_content") or ""
print("text:", text[:300])
fail = []
if len(text.strip()) < 20: fail.append("reply too short")
# <think>...</think> is expected on the Thinking variant, not a leak
leaks = [m for m in ("<|im_start|>", "<|im_end|>") if m in text]
if leaks: fail.append(f"control tokens leaked: {leaks}")
for f in fail: print("SMOKE_TEST_FAILED:", f)
print("SMOKE_TEST_PASSED" if not fail else "")
sys.exit(1 if fail else 0)
PY
  then rc=0; else rc=$?; fi
  kill "$pid" 2>/dev/null; wait "$pid" 2>/dev/null || true
  return "$rc"
}
run_step smoke "llama-server --jinja: chat reply, no control-token leak" gguf_smoke

# --- card + publish ----------------------------------------------------------
if [ -f "$CARD" ]; then
  run_step card "copy $(basename "$CARD") -> README.md + banner + LICENSE" bash -c '
    cp "$1" "$2/README.md"
    cp "$3/llmtray-banner.png" "$2/llmtray-banner.png"
    [ -f "$4/LICENSE" ] && cp "$4/LICENSE" "$2/LICENSE"' _ "$CARD" "$OUT_DIR" "$HERE/../../cards_assets" "$SNAPSHOT"
else
  skip_step card "no $CARD yet"
fi
if [ "${PUBLISH:-0}" = 1 ]; then
  [ -f "$OUT_DIR/README.md" ] || { echo "no model card in $OUT_DIR" >&2; exit 1; }
  run_step publish "hf upload $HF_REPO" \
    hf upload "$HF_REPO" "$OUT_DIR" . --exclude '*.tmp' --commit-message "lfm2_gguf_pipeline.sh VARIANT=$VARIANT"
else
  skip_step publish "PUBLISH!=1"
fi

echo "LFM2_GGUF_PIPELINE_DONE $OUT_DIR"
ls -la "$OUT_DIR"
