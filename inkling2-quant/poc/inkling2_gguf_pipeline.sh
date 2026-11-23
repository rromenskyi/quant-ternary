#!/usr/bin/env bash
# GGUF pipeline for Inkling 2 Mini (thinkingmachines/Inkling-2-Mini,
# inkling2_loop). llama.cpp has it in master (#32217 inkling2, #32694 the
# loop); its converter dequantizes the NVFP4 experts itself
# (conversion/inkling2.py), so the published snapshot converts as it is.
# Vision goes to a separate mmproj GGUF (F16).
#
#   ./inkling2_gguf_pipeline.sh                # build + ppl + smoke
#   PUBLISH=1 ./inkling2_gguf_pipeline.sh      # ... and upload
#
# Steps (-> $LOG_DIR/inkling2gguf_<step>.log, status lines in $LOG_DIR/pipeline.log):
#   llama_build   clone + build llama.cpp (CUDA)
#   download      HF snapshot
#   convert_f16   text GGUF (F16) + mmproj (F16)
#   imatrix       llama-imatrix on a broad corpus
#   quantize      llama-quantize, JANG tensor-type recipe
#   ppl, ppl_code wikitext / Python stdlib perplexity: F16 vs quant
#   smoke         llama-server --jinja: chat reply, no control-token leak
#   card          model card copied in LAST
#   publish       hf upload (PUBLISH=1 only)
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/../../gemma4-quant/poc/pipeline_lib.sh"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

MODEL_ID="${MODEL_ID:-thinkingmachines/Inkling-2-Mini}"
LLAMA="${LLAMA:-$WORK/llama.cpp}"
LLAMA_REF="${LLAMA_REF:-master}"
BIN="$LLAMA/build/bin"
CALIB="${CALIB:-$WORK/calibration_datav3.txt}"
CALIB_URL="https://gist.githubusercontent.com/bartowski1182/eb213dccb3571f863da82e99418f81e8/raw/calibration_datav3.txt"
F16="$WORK/inkling2-mini-f16.gguf"
IMATRIX="$WORK/inkling2-mini.imatrix"
OUT_DIR="${OUT_DIR:-$WORK/gguf-inkling2-mini-out}"
NGL="${NGL:-99}"
HF_REPO="${HF_REPO:-roman220220/Inkling-2-Mini-GGUF-jang-imatrix}"
CARD="$HERE/../cards/${HF_REPO#*/}.md"
PPL_TEST="${PPL_TEST:-$WORK/data/wiki.test.raw}"
CODE_TEST="${CODE_TEST:-$WORK/data/code.test.txt}"   # inkling2_mlx_pipeline.sh's data step

# --- recipe: the routed experts at the base preset, the rest higher ---------
BASE_QUANT="${BASE_QUANT:-Q4_K_M}"
# Q6_K, not Q8_0: Q8_0's Vulkan kernel is slow on cards without int8 dot
# (frognano_gguf_pipeline.sh has the numbers).
ATTN_TYPE="${ATTN_TYPE:-Q6_K}"
RUN_TAG="${RUN_TAG:-jang}"
TEXT_GGUF="$OUT_DIR/inkling-2-mini-${RUN_TAG}.gguf"
MMPROJ="$OUT_DIR/mmproj-inkling-2-mini-f16.gguf"

pipeline_init inkling2gguf "MODEL_ID=$MODEL_ID OUT_DIR=$OUT_DIR RECIPE=experts=$BASE_QUANT/attn=$ATTN_TYPE"
mkdir -p "$OUT_DIR"

# --- llama.cpp (CUDA) ----------------------------------------------------------
if [ -x "$BIN/llama-quantize" ] && [ -x "$BIN/llama-imatrix" ] && [ -x "$BIN/llama-perplexity" ]; then
  skip_step llama_build "binaries present in $BIN"
else
  run_step llama_build "clone + build llama.cpp $LLAMA_REF (CUDA)" bash -c "
    [ -d '$LLAMA' ] || git clone https://github.com/ggml-org/llama.cpp '$LLAMA'
    cd '$LLAMA' && git fetch -q origin '$LLAMA_REF' && git checkout -q FETCH_HEAD && git log -1 --format='llama.cpp %h %cd'
    pip install --quiet -r requirements/requirements-convert_hf_to_gguf.txt huggingface_hub hf_transfer
    pip install --quiet --upgrade 'transformers>=5.24' tokenizers tiktoken
    export PATH=/usr/local/cuda/bin:\$PATH CUDACXX=/usr/local/cuda/bin/nvcc
    cmake -B build -DGGML_CUDA=ON -DLLAMA_CURL=OFF && cmake --build build --config Release -j \$(nproc)"
fi

# --- source + calibration data -------------------------------------------------
_PIPE_CUR=download; _pipe_mark START download "$MODEL_ID"
SNAPSHOT="$(HF_HUB_ENABLE_HF_TRANSFER=1 hf_snapshot "$MODEL_ID")"
_pipe_mark DONE download "$SNAPSHOT"; _PIPE_CUR=""
if [ -s "$CALIB" ]; then skip_step calib_data "$CALIB exists"; else
  run_step calib_data "fetch bartowski calibration_datav3" curl -sfL -o "$CALIB" "$CALIB_URL"
fi

# --- convert to F16 GGUF (text) + mmproj -----------------------------------------
if [ -f "$F16" ]; then skip_step convert_f16 "$F16 exists"; else
  run_step convert_f16 "NVFP4 experts dequantized -> F16 GGUF" \
    python3 "$LLAMA/convert_hf_to_gguf.py" "$SNAPSHOT" --outfile "$F16" --outtype f16
fi
if [ -f "$MMPROJ" ]; then skip_step convert_mmproj "$MMPROJ exists"; else
  run_step convert_mmproj "vision tower -> mmproj F16" \
    python3 "$LLAMA/convert_hf_to_gguf.py" "$SNAPSHOT" --mmproj --outfile "$MMPROJ" --outtype f16
fi

# --- imatrix (GPU) ------------------------------------------------------------
if [ -f "$IMATRIX" ]; then skip_step imatrix "$IMATRIX exists"; else
  run_step imatrix "llama-imatrix, 200 chunks" \
    "$BIN/llama-imatrix" -m "$F16" -f "$CALIB" -o "$IMATRIX" -ngl "$NGL" --chunks 200
fi

# --- quantize (JANG: attention, embeddings, head high; experts at the base) ----
# The loop's message gates and the attention sinks stay as the converter
# writes them (it keeps the loop tensors unquantized, as the reference).
if [ -f "$TEXT_GGUF" ]; then skip_step quantize "$TEXT_GGUF exists"; else
  TT=()
  for t in attn_q attn_k attn_v attn_output attn_gate attn_rel token_embd 'output\.weight'; do
    TT+=(--tensor-type "$t=$ATTN_TYPE")
  done
  run_step quantize "llama-quantize $BASE_QUANT (attn/embed/head $ATTN_TYPE)" \
    "$BIN/llama-quantize" --imatrix "$IMATRIX" "${TT[@]}" "$F16" "$TEXT_GGUF" "$BASE_QUANT" 32
fi

# --- perplexity (quality record: quant vs F16) -------------------------------
for set in "ppl:$PPL_TEST" "ppl_code:$CODE_TEST"; do
  step="${set%%:*}" file="${set#*:}"
  [ -s "$file" ] || { echo "no $file: run inkling2_mlx_pipeline.sh's data step first" >&2; exit 1; }
  run_step "$step" "$step: F16 vs $BASE_QUANT ($RUN_TAG)" bash -c "
    echo -n 'F16          : '; '$BIN/llama-perplexity' -m '$F16' -f '$file' -ngl '$NGL' --chunks ${PPL_CHUNKS:-40} 2>&1 | grep -E 'Final estimate' | tail -1
    echo -n '$BASE_QUANT ($RUN_TAG): '; '$BIN/llama-perplexity' -m '$TEXT_GGUF' -f '$file' -ngl '$NGL' --chunks ${PPL_CHUNKS:-40} 2>&1 | grep -E 'Final estimate' | tail -1"
done

# --- smoke: llama-server --jinja -----------------------------------------------
gguf_smoke () {
  local port=18094
  "$BIN/llama-server" -m "$TEXT_GGUF" --mmproj "$MMPROJ" -ngl "$NGL" --jinja --port "$port" \
    > "$LOG_DIR/inkling2gguf_smoke_server.log" 2>&1 &
  local pid=$! rc=0
  for _ in $(seq 1 180); do curl -sf "localhost:$port/health" >/dev/null && break; sleep 2; done
  if python3 - "$port" <<'PY'
import json, sys, urllib.request
port = sys.argv[1]
fail = []
# Longer than ~96 tokens: llama.cpp #32778 broke those to '?' once.
prompt = ("You are helping a developer. " * 12) + "List three facts about the Roman Empire, one line each."
for content in (prompt, "Write a Python function that returns the n-th Fibonacci number."):
    body = {"messages": [{"role": "user", "content": content}], "max_tokens": 1024, "temperature": 0}
    req = urllib.request.Request(f"http://localhost:{port}/v1/chat/completions", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    msg = json.load(urllib.request.urlopen(req, timeout=600))["choices"][0]["message"]
    text = msg.get("content") or msg.get("reasoning_content") or ""
    print("text:", text[:300])
    if len(text.strip()) < 20 or set(text.strip()) <= {"?"}:
        fail.append("reply too short or all '?'")
    leaks = [m for m in ("<|end_message|>", "<|message_model|>", "<|content_text|>", "<|message_user|>") if m in text]
    if leaks:
        fail.append(f"control tokens leaked: {leaks}")
for f in fail:
    print("SMOKE_TEST_FAILED:", f)
print("SMOKE_TEST_PASSED" if not fail else "")
sys.exit(1 if fail else 0)
PY
  then rc=0; else rc=$?; fi
  kill "$pid" 2>/dev/null; wait "$pid" 2>/dev/null || true
  return "$rc"
}
run_step smoke "llama-server --jinja: chat replies, no control-token leak" gguf_smoke

# --- card + publish ----------------------------------------------------------
if [ -f "$CARD" ]; then
  run_step card "copy $(basename "$CARD") -> README.md" bash -c '
    cp "$1" "$2/README.md"
    for img in $(grep -o "[a-z-]*banner\.png" "$1" | sort -u); do cp "$3/$img" "$2/$img"; done
    lic="$(dirname "$1")/licenses/$4-LICENSE"
    if [ -f "$lic" ]; then cp "$lic" "$2/LICENSE"; fi' \
    _ "$CARD" "$OUT_DIR" "$HERE/../../cards_assets" "${MODEL_ID##*/}"
else
  skip_step card "no $CARD yet"
fi
if [ "${PUBLISH:-0}" = 1 ]; then
  [ -f "$OUT_DIR/README.md" ] || { echo "no model card in $OUT_DIR" >&2; exit 1; }
  run_step publish "hf upload $HF_REPO" \
    hf upload "$HF_REPO" "$OUT_DIR" . --exclude '*.tmp' --commit-message "inkling2_gguf_pipeline.sh"
else
  skip_step publish "PUBLISH!=1"
fi

echo "INKLING2_GGUF_PIPELINE_DONE $OUT_DIR"
ls -la "$OUT_DIR"
