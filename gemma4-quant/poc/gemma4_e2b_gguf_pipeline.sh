#!/usr/bin/env bash
# End-to-end GGUF pipeline for Gemma 4 E2B (dense, text + vision) on a CUDA pod.
#
# Same two contributions as the 26B GGUF pipeline: imatrix on a broad corpus,
# and a JANG-style per-tensor recipe (attention kept HIGH, feed-forward pushed
# LOW). Two things differ from 26B:
#   - E2B is DENSE (no MoE experts, no MTP drafter), so the recipe targets
#     ffn_gate/up/down (not *_exps) and there is no drafter/DRAFT step.
#   - E2B's hidden 1536 (=6x256) and intermediate 6144 (=24x256) are BOTH
#     divisible by 256, so llama.cpp K-quants / IQ2 / IQ3 are eligible on
#     every tensor (unlike 26B=2688 / 4B=3136, which hit the 256-alignment
#     floor and fell back to legacy formats). We can push smaller, cleanly.
#
# Recipe is env-driven so a PPL sweep is one variable change:
#   BASE_QUANT (default Q4_K_M), ATTN_TYPE (default Q6_K), FFN_TYPE (optional).
# A `ppl` step reports wikitext perplexity for the result vs the F16 GGUF.
#
#   ./gemma4_e2b_gguf_pipeline.sh                        # build + ppl + smoke
#   BASE_QUANT=IQ3_M ATTN_TYPE=Q5_K RUN_TAG=iq3m ./gemma4_e2b_gguf_pipeline.sh   # a smaller variant
#   PUBLISH=1 ./gemma4_e2b_gguf_pipeline.sh              # ... and upload
#
# RUNTIME NOTE: always launch with `--jinja` (llama.cpp) or the shipped
# Modelfile (ollama) -- see docs/FINDINGS.md "GGUF gotchas".
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/pipeline_lib.sh"

MODEL_ID="${MODEL_ID:-google/gemma-4-E2B-it}"
LLAMA="${LLAMA:-$WORK/llama.cpp}"
BIN="$LLAMA/build/bin"
MODEL_FIXED="${MODEL_FIXED:-$WORK/gemma4-e2b-src-fixed}"
CALIB="${CALIB:-$WORK/calibration_datav3.txt}"
CALIB_URL="https://gist.githubusercontent.com/bartowski1182/eb213dccb3571f863da82e99418f81e8/raw/calibration_datav3.txt"
F16="$WORK/gemma4-e2b-f16.gguf"
IMATRIX="$WORK/gemma4-e2b.imatrix"
OUT_DIR="${OUT_DIR:-$WORK/gguf-e2b-out}"
NGL="${NGL:-99}"
HF_REPO="${HF_REPO:-roman220220/gemma-4-E2B-it-GGUF-jang-imatrix}"
CARD="$HERE/../cards/${HF_REPO#*/}.md"

# --- recipe (env-driven for PPL sweeps) --------------------------------------
BASE_QUANT="${BASE_QUANT:-Q4_K_M}"   # llama-quantize base preset
ATTN_TYPE="${ATTN_TYPE:-Q6_K}"       # attention kept high (small, sensitive)
FFN_TYPE="${FFN_TYPE:-}"             # optional ffn override; empty = base preset decides
RUN_TAG="${RUN_TAG:-jang}"
TEXT_GGUF="$OUT_DIR/gemma4-e2b-${RUN_TAG}.gguf"
MMPROJ_F16="$OUT_DIR/mmproj-gemma4-e2b-f16.gguf"
MMPROJ_Q8="$OUT_DIR/mmproj-gemma4-e2b-q8.gguf"
PPL_TEST="${PPL_TEST:-$WORK/wikitext-2-raw/wiki.test.raw}"

pipeline_init gguf-e2b "MODEL_ID=$MODEL_ID OUT_DIR=$OUT_DIR RECIPE=$BASE_QUANT/attn=$ATTN_TYPE/ffn=${FFN_TYPE:-base}"
mkdir -p "$OUT_DIR"

# --- llama.cpp (CUDA, current source) ----------------------------------------
if [ -x "$BIN/llama-quantize" ] && [ -x "$BIN/llama-imatrix" ] && [ -x "$BIN/llama-perplexity" ]; then
  skip_step llama_build "binaries present in $BIN"
else
  run_step llama_build "clone + build llama.cpp (CUDA)" bash -c "
    [ -d '$LLAMA' ] || git clone --depth 1 https://github.com/ggml-org/llama.cpp '$LLAMA'
    cd '$LLAMA' && pip install --quiet -r requirements/requirements-convert_hf_to_gguf.txt
    export PATH=/usr/local/cuda/bin:\$PATH CUDACXX=/usr/local/cuda/bin/nvcc
    cmake -B build -DGGML_CUDA=ON -DLLAMA_CURL=OFF && cmake --build build --config Release -j \$(nproc)"
fi

# --- source + tokenizer fix (dict-form extra_special_tokens) ------------------
_PIPE_CUR=download; _pipe_mark START download "$MODEL_ID"
SNAPSHOT="$(hf_snapshot "$MODEL_ID")"
_pipe_mark DONE download "$SNAPSHOT"; _PIPE_CUR=""
if [ -f "$MODEL_FIXED/tokenizer_config.json" ]; then
  skip_step fix_tokenizer "$MODEL_FIXED exists"
else
  run_step fix_tokenizer "dict-form extra_special_tokens" \
    python3 "$HERE/gemma4_26b_fix_tokenizer.py" "$SNAPSHOT" "$MODEL_FIXED"
fi
if [ -s "$CALIB" ]; then skip_step calib_data "$CALIB exists"; else
  run_step calib_data "fetch bartowski calibration_datav3" curl -sfL -o "$CALIB" "$CALIB_URL"
fi

# --- convert to F16 GGUF ------------------------------------------------------
if [ -f "$F16" ]; then skip_step convert_f16 "$F16 exists"; else
  run_step convert_f16 "bf16 -> f16 GGUF" \
    python3 "$LLAMA/convert_hf_to_gguf.py" "$MODEL_FIXED" --outfile "$F16" --outtype f16
fi

# --- imatrix (GPU) ------------------------------------------------------------
if [ -f "$IMATRIX" ]; then skip_step imatrix "$IMATRIX exists"; else
  run_step imatrix "llama-imatrix, 200 chunks" \
    "$BIN/llama-imatrix" -m "$F16" -f "$CALIB" -o "$IMATRIX" -ngl "$NGL" --chunks 200
fi

# --- quantize (JANG dense recipe) + token_type CONTROL fix --------------------
# Dense tensor names: attn_q/attn_k/attn_v/attn_output, ffn_gate/ffn_up/ffn_down.
if [ -f "$TEXT_GGUF" ]; then skip_step quantize "$TEXT_GGUF exists"; else
  TT=(--tensor-type "attn_q=$ATTN_TYPE" --tensor-type "attn_k=$ATTN_TYPE"
      --tensor-type "attn_v=$ATTN_TYPE" --tensor-type "attn_output=$ATTN_TYPE")
  [ -n "$FFN_TYPE" ] && TT+=(--tensor-type "ffn_gate=$FFN_TYPE" --tensor-type "ffn_up=$FFN_TYPE" --tensor-type "ffn_down=$FFN_TYPE")
  run_step quantize "llama-quantize $BASE_QUANT (attn $ATTN_TYPE${FFN_TYPE:+, ffn $FFN_TYPE}) + token_type fix" bash -c "
    '$BIN/llama-quantize' --imatrix '$IMATRIX' ${TT[*]} '$F16' '$TEXT_GGUF.tmp' $BASE_QUANT 32 &&
    python3 '$HERE/gemma4_gguf_fix_token_types.py' --file '$TEXT_GGUF.tmp' --from-tokenizer '$MODEL_FIXED/tokenizer.json' &&
    mv '$TEXT_GGUF.tmp' '$TEXT_GGUF'"
fi

# --- perplexity (quality record: quant vs F16) -------------------------------
if [ ! -f "$PPL_TEST" ]; then
  run_step ppl_data "fetch wikitext-2-raw test" bash -c "
    cd '$WORK' && curl -sfL -o wikitext-2-raw-v1.zip https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-2-raw-v1.zip &&
    unzip -o wikitext-2-raw-v1.zip >/dev/null"
fi
run_step ppl "wikitext PPL: F16 vs $BASE_QUANT ($RUN_TAG)" bash -c "
  echo -n 'F16          : '; '$BIN/llama-perplexity' -m '$F16' -f '$PPL_TEST' -ngl '$NGL' --chunks ${PPL_CHUNKS:-40} 2>&1 | grep -E 'Final estimate' | tail -1
  echo -n '$BASE_QUANT ($RUN_TAG): '; '$BIN/llama-perplexity' -m '$TEXT_GGUF' -f '$PPL_TEST' -ngl '$NGL' --chunks ${PPL_CHUNKS:-40} 2>&1 | grep -E 'Final estimate' | tail -1"

# --- vision mmproj -----------------------------------------------------------
if [ -f "$MMPROJ_F16" ]; then skip_step mmproj_convert "$MMPROJ_F16 exists"; else
  run_step mmproj_convert "vision tower -> mmproj GGUF" \
    python3 "$LLAMA/convert_hf_to_gguf.py" "$MODEL_FIXED" --mmproj --outfile "$MMPROJ_F16"
fi
if [ -f "$MMPROJ_Q8" ]; then skip_step mmproj_quantize "$MMPROJ_Q8 exists"; else
  run_step mmproj_quantize "mmproj -> Q8_0" "$BIN/llama-quantize" "$MMPROJ_F16" "$MMPROJ_Q8" Q8_0 16
fi

# --- ollama Modelfile (no drafter; dense model) ------------------------------
write_ollama_files () {
  cat > "$OUT_DIR/Modelfile" << EOF
# Ollama Modelfile for $HF_REPO
#   curl -L -o Modelfile https://huggingface.co/$HF_REPO/resolve/main/Modelfile
#   ollama create gemma4-e2b-jang -f Modelfile && ollama run gemma4-e2b-jang
# Requires Ollama >= 0.30.0.
FROM hf.co/$HF_REPO
RENDERER gemma4
PARSER gemma4
PARAMETER temperature 1
PARAMETER top_k 64
PARAMETER top_p 0.95
EOF
  echo '{"temperature":1,"top_k":64,"top_p":0.95}' > "$OUT_DIR/params"
  cat "$OUT_DIR/Modelfile"
}
run_step ollama_files "Modelfile (FROM hf.co + RENDERER/PARSER) + params" write_ollama_files

# --- smoke: llama-server --jinja + mmproj ------------------------------------
gguf_smoke () {
  local port=18091 img="$WORK/smoke-assets/cats.jpg"
  mkdir -p "$(dirname "$img")"
  [ -f "$img" ] || curl -sfL -o "$img" http://images.cocodataset.org/val2017/000000039769.jpg
  "$BIN/llama-server" -m "$TEXT_GGUF" --mmproj "$MMPROJ_Q8" -ngl "$NGL" --jinja --port "$port" \
    > "$LOG_DIR/gguf_e2b_smoke_server.log" 2>&1 &
  local pid=$! rc=0
  for _ in $(seq 1 180); do curl -sf "localhost:$port/health" >/dev/null && break; sleep 2; done
  if python3 - "$port" "$img" <<'PY'
import base64, json, sys, urllib.request
port, img = sys.argv[1], sys.argv[2]
def chat(content):
    body = {"messages": [{"role": "user", "content": content}], "max_tokens": 600, "temperature": 0}
    req = urllib.request.Request(f"http://localhost:{port}/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=600))["choices"][0]["message"].get("content") or ""
fail = []
text = chat("List three facts about the Roman Empire, one line each.")
print("text:", text[:300])
if len(text.strip()) < 20: fail.append("text reply too short")
leaks = [m for m in ("<|channel>", "<channel|>", "<|turn>", "<turn|>", "<|think|>") if m in text]
if leaks: fail.append(f"control tokens leaked: {leaks}")
uri = "data:image/jpeg;base64," + base64.b64encode(open(img, "rb").read()).decode()
vis = chat([{"type": "image_url", "image_url": {"url": uri}}, {"type": "text", "text": "What animals are in this picture? One sentence."}])
print("vision:", vis[:300])
if "cat" not in vis.lower(): fail.append("vision reply lacks 'cat'")
for f in fail: print("SMOKE_TEST_FAILED:", f)
print("SMOKE_TEST_PASSED" if not fail else "")
sys.exit(1 if fail else 0)
PY
  then rc=0; else rc=$?; fi
  kill "$pid" 2>/dev/null; wait "$pid" 2>/dev/null || true
  return "$rc"
}
run_step smoke "llama-server --jinja + mmproj: text (no control-token leak) + vision" gguf_smoke

# --- card + publish ----------------------------------------------------------
[ -f "$CARD" ] && run_step card "copy $(basename "$CARD") -> README.md" cp "$CARD" "$OUT_DIR/README.md" || skip_step card "no card yet"
if [ "${PUBLISH:-0}" = 1 ]; then
  run_step publish "hf upload $HF_REPO" \
    hf upload "$HF_REPO" "$OUT_DIR" . --exclude '*-f16.gguf' --exclude '*.tmp' --commit-message "gemma4_e2b_gguf_pipeline.sh"
else
  skip_step publish "PUBLISH!=1"
fi

echo "GEMMA4_E2B_GGUF_PIPELINE_DONE"
ls -la "$OUT_DIR"
