#!/usr/bin/env bash
# GGUF pipeline for FrogNano-4B-2609 (microsoft/FrogNano-4B-2609, qwen3_5
# dense: linear/gated-delta-net attention in most layers, full GQA attention
# every full_attention_interval-th layer, with an extra attention output
# gate). llama.cpp has mainline QWEN35 support (conversion/qwen.py,
# gated-delta-net V-head reordering, MRoPE) -- no architecture port needed,
# unlike this project's lfm2 / muse_glimmer GGUF work.
#
#   ./frognano_gguf_pipeline.sh                # build + ppl + smoke
#   PUBLISH=1 ./frognano_gguf_pipeline.sh       # ... and upload
#
# Steps (-> $LOG_DIR/frognano_<step>.log, status lines in $LOG_DIR/pipeline.log):
#   llama_build   clone + build llama.cpp (CUDA)
#   download      HF snapshot of the bf16 checkpoint
#   convert_f16   bf16 -> f16 GGUF
#   imatrix       llama-imatrix on a broad corpus
#   quantize      llama-quantize, JANG tensor-type recipe
#   ppl           wikitext PPL: F16 vs quant
#   ollama_files  Modelfile (FROM hf.co + sampling defaults)
#   smoke         llama-server --jinja: chat reply, no control-token leak
#   card          model card copied in LAST
#   publish       hf upload (PUBLISH=1 only; HF_TOKEN in the environment)
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/../../gemma4-quant/poc/pipeline_lib.sh"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

MODEL_ID="${MODEL_ID:-microsoft/FrogNano-4B-2609}"
LLAMA="${LLAMA:-$WORK/llama.cpp}"
BIN="$LLAMA/build/bin"
CALIB="${CALIB:-$WORK/calibration_datav3.txt}"
CALIB_URL="https://gist.githubusercontent.com/bartowski1182/eb213dccb3571f863da82e99418f81e8/raw/calibration_datav3.txt"
F16="$WORK/frognano-f16.gguf"
IMATRIX="$WORK/frognano.imatrix"
OUT_DIR="${OUT_DIR:-$WORK/gguf-frognano-out}"
NGL="${NGL:-99}"
HF_REPO="${HF_REPO:-roman220220/FrogNano-4B-2609-GGUF-jang-imatrix}"
CARD="$HERE/../cards/${HF_REPO#*/}.md"
PPL_TEST="${PPL_TEST:-$WORK/wikitext-2-raw/wiki.test.raw}"

# --- recipe (env-driven for PPL sweeps; matches the MLX GPTQ split) ----------
BASE_QUANT="${BASE_QUANT:-Q4_K_M}"   # llama-quantize base preset (feed-forward)
ATTN_TYPE="${ATTN_TYPE:-Q8_0}"       # full-attention layers + embeddings: few, sensitive
LINEAR_TYPE="${LINEAR_TYPE:-Q6_K}"   # gated-delta-net (ssm_*) layers: most layers
RUN_TAG="${RUN_TAG:-jang}"
TEXT_GGUF="$OUT_DIR/frognano-4b-${RUN_TAG}.gguf"

pipeline_init frognano "MODEL_ID=$MODEL_ID OUT_DIR=$OUT_DIR RECIPE=mlp=$BASE_QUANT/attn=$ATTN_TYPE/linear=$LINEAR_TYPE"
mkdir -p "$OUT_DIR"

# --- llama.cpp (CUDA, current source) ----------------------------------------
if [ -x "$BIN/llama-quantize" ] && [ -x "$BIN/llama-imatrix" ] && [ -x "$BIN/llama-perplexity" ]; then
  skip_step llama_build "binaries present in $BIN"
else
  run_step llama_build "clone + build llama.cpp (CUDA)" bash -c "
    [ -d '$LLAMA' ] || git clone --depth 1 https://github.com/ggml-org/llama.cpp '$LLAMA'
    cd '$LLAMA' && pip install --quiet --break-system-packages -r requirements/requirements-convert_hf_to_gguf.txt \
      huggingface_hub hf_transfer
    pip install --quiet --break-system-packages --upgrade 'transformers>=5.15' tokenizers
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

# --- convert to F16 GGUF ------------------------------------------------------
if [ -f "$F16" ]; then skip_step convert_f16 "$F16 exists"; else
  run_step convert_f16 "bf16 -> f16 GGUF" \
    python3 "$LLAMA/convert_hf_to_gguf.py" "$SNAPSHOT" --outfile "$F16" --outtype f16
fi

# --- imatrix (GPU) ------------------------------------------------------------
if [ -f "$IMATRIX" ]; then skip_step imatrix "$IMATRIX exists"; else
  run_step imatrix "llama-imatrix, 200 chunks" \
    "$BIN/llama-imatrix" -m "$F16" -f "$CALIB" -o "$IMATRIX" -ngl "$NGL" --chunks 200
fi

# --- quantize (JANG recipe: attn+embeddings high, linear mid, ffn base) ------
# Dense qwen3_5 tensor names: attn_q/k/v/output/gate (full-attention layers),
# ssm_* (gated-delta-net layers), ffn_gate/up/down.
if [ -f "$TEXT_GGUF" ]; then skip_step quantize "$TEXT_GGUF exists"; else
  # ssm_a / ssm_conv1d / ssm_dt / ssm_norm are small gate/norm tensors that
  # llama-quantize already force-keeps at F32 regardless of --tensor-type
  # (confirmed empirically: overriding them is a no-op). ssm_alpha/ssm_beta
  # are the delta rule's input gates (gptq_qwen35.py's DEFAULT_KEEP_FLOAT:
  # "linear_attn.in_proj_[ab]$") -- small (2560x32) matrices where
  # quantizing them saves ~12.5 MB total (0.36% of the file) but forced a
  # Vulkan GPU/CPU graph split on every decode step (reported live: 7.4
  # tok/s vs 25.7 tok/s on an old-GCN card), since the SSM-scan kernel
  # expects these in float. Left at the base preset's float handling.
  TT=(--tensor-type "attn_q=$ATTN_TYPE" --tensor-type "attn_k=$ATTN_TYPE"
      --tensor-type "attn_v=$ATTN_TYPE" --tensor-type "attn_output=$ATTN_TYPE"
      --tensor-type "attn_gate=$ATTN_TYPE" --tensor-type "token_embd=$ATTN_TYPE"
      --tensor-type "output\\.weight=$ATTN_TYPE"
      --tensor-type "ssm_out=$LINEAR_TYPE")
  run_step quantize "llama-quantize $BASE_QUANT (attn/embed $ATTN_TYPE, linear $LINEAR_TYPE)" \
    "$BIN/llama-quantize" --imatrix "$IMATRIX" "${TT[@]}" "$F16" "$TEXT_GGUF" "$BASE_QUANT" 32
fi

# --- perplexity (quality record: quant vs F16) -------------------------------
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
write_ollama_files () {
  cat > "$OUT_DIR/Modelfile" << EOF
# Ollama Modelfile for $HF_REPO
#   curl -L -o Modelfile https://huggingface.co/$HF_REPO/resolve/main/Modelfile
#   ollama create frognano-4b-jang -f Modelfile && ollama run frognano-4b-jang
FROM hf.co/$HF_REPO
EOF
  cat "$OUT_DIR/Modelfile"
}
run_step ollama_files "Modelfile (FROM hf.co)" write_ollama_files

# --- smoke: llama-server --jinja -----------------------------------------------
gguf_smoke () {
  local port=18093
  "$BIN/llama-server" -m "$TEXT_GGUF" -ngl "$NGL" --jinja --port "$port" \
    > "$LOG_DIR/frognano_smoke_server.log" 2>&1 &
  local pid=$! rc=0
  for _ in $(seq 1 180); do curl -sf "localhost:$port/health" >/dev/null && break; sleep 2; done
  if python3 - "$port" <<'PY'
import json, sys, urllib.request
port = sys.argv[1]
body = {"messages": [{"role": "user", "content": "List three facts about the Roman Empire, one line each."}], "max_tokens": 1024, "temperature": 0}
req = urllib.request.Request(f"http://localhost:{port}/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
msg = json.load(urllib.request.urlopen(req, timeout=600))["choices"][0]["message"]
text = msg.get("content") or msg.get("reasoning_content") or ""
print("text:", text[:300])
fail = []
if len(text.strip()) < 20: fail.append("reply too short")
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
  run_step card "copy $(basename "$CARD") -> README.md" bash -c '
    cp "$1" "$2/README.md"
    for img in $(grep -o "[a-z-]*banner\.png" "$1" | sort -u); do cp "$3/$img" "$2/$img"; done
    if [ -f "$4/LICENSE" ]; then cp "$4/LICENSE" "$2/LICENSE"; fi' _ "$CARD" "$OUT_DIR" "$HERE/../../cards_assets" "$SNAPSHOT"
else
  skip_step card "no $CARD yet"
fi
if [ "${PUBLISH:-0}" = 1 ]; then
  [ -f "$OUT_DIR/README.md" ] || { echo "no model card in $OUT_DIR" >&2; exit 1; }
  run_step publish "hf upload $HF_REPO" \
    hf upload "$HF_REPO" "$OUT_DIR" . --exclude '*.tmp' --commit-message "frognano_gguf_pipeline.sh"
else
  skip_step publish "PUBLISH!=1"
fi

echo "FROGNANO_GGUF_PIPELINE_DONE $OUT_DIR"
ls -la "$OUT_DIR"
