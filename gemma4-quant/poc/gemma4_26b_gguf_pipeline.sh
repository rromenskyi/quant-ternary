#!/usr/bin/env bash
# End-to-end GGUF pipeline for Gemma 4 26B-A4B (MoE), with our two
# contributions layered on top of stock llama.cpp:
#   1. imatrix computed on a broad, deliberately-chosen calibration corpus
#      (llama.cpp k-quant grid != our MLX affine grid, so the GPTQ *weights*
#      don't transfer -- imatrix is the GGUF-world equivalent of "calibrate
#      on real data"). Our 8 GPTQ prompts are too narrow for imatrix (it
#      would overfit and HURT off-topic quality), so we use bartowski's
#      calibration_datav3 (broad: code + prose + facts + multilingual).
#   2. JANG-mixed precision via llama-quantize per-tensor overrides:
#      attention kept HIGH (it's small and error-sensitive), the 128 routed
#      experts pushed LOW (they're ~90% of the weights) -- same "spend bits
#      where they matter" principle as the MLX JANG recipe.
#
# Runs entirely on a CUDA pod. Every step is resumable (skipped when its
# output exists). Steps -> $LOG_DIR/gguf_<step>.log, status lines in
# $LOG_DIR/pipeline.log (watch with pipeline_dashboard.py).
#
#   ./gemma4_26b_gguf_pipeline.sh               # build + smoke test
#   PUBLISH=1 ./gemma4_26b_gguf_pipeline.sh     # ... and upload to HF
#
# RUNTIME NOTE: always launch the result with `--jinja` (llama.cpp) or the
# shipped Modelfile (ollama) -- see docs/FINDINGS.md "GGUF gotchas".
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/pipeline_lib.sh"

MODEL_ID="${MODEL_ID:-google/gemma-4-26B-A4B-it}"
LLAMA="${LLAMA:-$WORK/llama.cpp}"
BIN="$LLAMA/build/bin"
MODEL_FIXED="${MODEL_FIXED:-$WORK/gemma4-26b-src-fixed}"
CALIB="${CALIB:-$WORK/calibration_datav3.txt}"
CALIB_URL="https://gist.githubusercontent.com/bartowski1182/eb213dccb3571f863da82e99418f81e8/raw/calibration_datav3.txt"
F16="$WORK/gemma4-26b-f16.gguf"
IMATRIX="$WORK/gemma4-26b.imatrix"
OUT_DIR="${OUT_DIR:-$WORK/gguf-out}"
NGL="${NGL:-99}"   # offload all layers to GPU for imatrix / smoke test
HF_REPO="${HF_REPO:-roman220220/gemma-4-26B-A4B-it-GGUF-jang-imatrix}"
CARD="$HERE/../cards/${HF_REPO#*/}.md"
TEXT_GGUF="$OUT_DIR/gemma4-26b-a4b-jang-iq3s.gguf"
MMPROJ_F16="$OUT_DIR/mmproj-gemma4-26b-f16.gguf"
MMPROJ_Q8="$OUT_DIR/mmproj-gemma4-26b-q8.gguf"

# MTP drafter (speculative decoding): Google's gemma-4-26B-A4B-it-assistant,
# taken as the exact GGUF the official ollama gemma4:26b ships as its
# `draft` layer (arch gemma4-assistant, Q8_0, 462MB) and rehosted in the HF
# repo. Using ollama's own file rather than a llama.cpp conversion
# guarantees it loads in ollama's gemma4 engine. Ollama's DRAFT directive
# only takes a LOCAL path, so users download it next to the Modelfile.
DRAFT_FILE="gemma4-26b-assistant-q8_0.gguf"
DRAFT_DIGEST="sha256:6326fb9f5e487aa8dcdd313a091e3c67724cb2a666ec3b7d2895b5b26d93ed1b"

pipeline_init gguf "MODEL_ID=$MODEL_ID OUT_DIR=$OUT_DIR"
mkdir -p "$OUT_DIR"

# --- llama.cpp from CURRENT source with CUDA ---------------------------------
# Prebuilt release binaries do NOT have Gemma 4 MoE + vision (it landed only
# in the conversion/gemma.py + GEMMA4 arch refactor).
if [ -x "$BIN/llama-quantize" ] && [ -x "$BIN/llama-imatrix" ]; then
  skip_step llama_build "binaries present in $BIN"
else
  run_step llama_build "clone + build llama.cpp (CUDA)" bash -c "
    [ -d '$LLAMA' ] || git clone --depth 1 https://github.com/ggml-org/llama.cpp '$LLAMA'
    cd '$LLAMA' && pip install --quiet -r requirements/requirements-convert_hf_to_gguf.txt
    export PATH=/usr/local/cuda/bin:\$PATH CUDACXX=/usr/local/cuda/bin/nvcc
    cmake -B build -DGGML_CUDA=ON -DLLAMA_CURL=OFF && cmake --build build --config Release -j \$(nproc)"
fi

# --- source checkpoint + tokenizer fix -----------------------------------------
_PIPE_CUR=download; _pipe_mark START download "$MODEL_ID"
SNAPSHOT="$(hf_snapshot "$MODEL_ID")"
_pipe_mark DONE download "$SNAPSHOT"; _PIPE_CUR=""

# transformers 5.17 crashes on the checkpoint's list-form extra_special_tokens;
# converting it to the dict form (NOT deleting it) is what keeps Gemma 4's
# control tokens registered -- see gemma4_26b_fix_tokenizer.py.
if [ -f "$MODEL_FIXED/tokenizer_config.json" ]; then
  skip_step fix_tokenizer "$MODEL_FIXED exists"
else
  run_step fix_tokenizer "symlinked copy with dict-form extra_special_tokens" \
    python3 "$HERE/gemma4_26b_fix_tokenizer.py" "$SNAPSHOT" "$MODEL_FIXED"
fi

if [ -s "$CALIB" ]; then
  skip_step calib_data "$CALIB exists"
else
  run_step calib_data "fetch bartowski calibration_datav3" curl -sfL -o "$CALIB" "$CALIB_URL"
fi

# --- Step 1: HF bf16 -> GGUF F16 ---------------------------------------------------
if [ -f "$F16" ]; then skip_step convert_f16 "$F16 exists"; else
  run_step convert_f16 "bf16 -> f16 GGUF" \
    python3 "$LLAMA/convert_hf_to_gguf.py" "$MODEL_FIXED" --outfile "$F16" --outtype f16
fi

# --- Step 2: imatrix on the broad corpus (GPU) --------------------------------------
if [ -f "$IMATRIX" ]; then skip_step imatrix "$IMATRIX exists"; else
  run_step imatrix "llama-imatrix, 200 chunks" \
    "$BIN/llama-imatrix" -m "$F16" -f "$CALIB" -o "$IMATRIX" -ngl "$NGL" --chunks 200
fi

# --- Step 3: JANG-mixed quantize ----------------------------------------------------
# Base Q3_K_M, attention UP to Q5_K, the 128 routed experts DOWN to IQ3_S
# (~90% of the weights, where the size drop comes from). Real GGUF tensor
# names: attn_output (NOT attn_o), ffn_*_exps for experts. ~12.5GB, smaller
# than a stock ~13GB Q3_K_M.
#
# After every quantize, token_type MUST be fixed: convert_hf_to_gguf.py tags
# Gemma 4's control tokens (<|turn>, <|channel>, <|think|>, <|tool*>,
# image/audio/video markers) NORMAL/USER_DEFINED, so they decode as literal
# text ("<|channel>thought..."). gemma4_gguf_fix_token_types.py flips them to
# CONTROL in place; --from-tokenizer derives the ids, nothing hardcoded.
if [ -f "$TEXT_GGUF" ]; then skip_step quantize "$TEXT_GGUF exists"; else
  run_step quantize "llama-quantize JANG (attn Q5_K, experts IQ3_S) + token_type fix" bash -c "
    '$BIN/llama-quantize' --imatrix '$IMATRIX' \
      --tensor-type attn_q=Q5_K --tensor-type attn_k=Q5_K --tensor-type attn_v=Q5_K \
      --tensor-type attn_output=Q5_K \
      --tensor-type ffn_gate_up_exps=IQ3_S --tensor-type ffn_down_exps=IQ3_S \
      '$F16' '$TEXT_GGUF.tmp' Q3_K_M 32 &&
    python3 '$HERE/gemma4_gguf_fix_token_types.py' --file '$TEXT_GGUF.tmp' \
      --from-tokenizer '$MODEL_FIXED/tokenizer.json' &&
    mv '$TEXT_GGUF.tmp' '$TEXT_GGUF'"
fi

# --- Step 4: vision tower (mmproj) -> GGUF, then Q8_0 -------------------------------
# Vision is a SEPARATE mmproj file in llama.cpp. 27 vision ffn_down tensors
# have a 4304-wide dim divisible by no GGUF block size -> llama.cpp falls back
# to F16 for those. Expected.
if [ -f "$MMPROJ_F16" ]; then skip_step mmproj_convert "$MMPROJ_F16 exists"; else
  run_step mmproj_convert "vision tower -> mmproj GGUF" \
    python3 "$LLAMA/convert_hf_to_gguf.py" "$MODEL_FIXED" --mmproj --outfile "$MMPROJ_F16"
fi
if [ -f "$MMPROJ_Q8" ]; then skip_step mmproj_quantize "$MMPROJ_Q8 exists"; else
  run_step mmproj_quantize "mmproj -> Q8_0" "$BIN/llama-quantize" "$MMPROJ_F16" "$MMPROJ_Q8" Q8_0 16
fi

# --- Step 5: MTP drafter + Ollama packaging ------------------------------------------
if [ -f "$OUT_DIR/$DRAFT_FILE" ]; then skip_step drafter "$DRAFT_FILE exists"; else
  run_step drafter "fetch ollama's gemma4:26b draft layer (checksum-verified)" bash -c "
    curl -sfL 'https://registry.ollama.ai/v2/library/gemma4/blobs/$DRAFT_DIGEST' -o '$OUT_DIR/$DRAFT_FILE.tmp' &&
    echo '${DRAFT_DIGEST#sha256:}  $OUT_DIR/$DRAFT_FILE.tmp' | { command -v sha256sum >/dev/null && sha256sum -c - || shasum -a 256 -c -; } &&
    mv '$OUT_DIR/$DRAFT_FILE.tmp' '$OUT_DIR/$DRAFT_FILE'"
fi

# `ollama run hf.co/<repo>` only reads a Go `template` / `params` / `system`
# from a HF repo; it can NOT set Ollama's native Gemma 4 RENDERER/PARSER.
# Without those, the thinking block and Gemma 4's native tool-call syntax
# leak into chat as raw text (verified on a real ollama install -- the
# token_type fix is necessary but not sufficient for ollama). The official
# ollama gemma4 manifest sets renderer=gemma4 + parser=gemma4, so we ship a
# Modelfile that does the same and pulls the weights straight from HF.
# Sampling params copied from that manifest. RENDERER/PARSER are accepted by
# ollama's Modelfile parser though docs/modelfile.mdx doesn't list them yet.
write_ollama_files () {
  cat > "$OUT_DIR/Modelfile" << EOF
# Ollama Modelfile for $HF_REPO
# Usage:
#   curl -L -o Modelfile https://huggingface.co/$HF_REPO/resolve/main/Modelfile
#   curl -L -o $DRAFT_FILE https://huggingface.co/$HF_REPO/resolve/main/$DRAFT_FILE
#   ollama create gemma4-26b-jang -f Modelfile
#   ollama run gemma4-26b-jang
# Requires Ollama >= 0.30.0. Drop the DRAFT/draft_num_predict lines if memory is tight.
FROM hf.co/$HF_REPO
DRAFT ./$DRAFT_FILE

RENDERER gemma4
PARSER gemma4

PARAMETER temperature 1
PARAMETER top_k 64
PARAMETER top_p 0.95
PARAMETER draft_num_predict 3
EOF
  echo '{"temperature":1,"top_k":64,"top_p":0.95}' > "$OUT_DIR/params"
  cat "$OUT_DIR/Modelfile"
}
run_step ollama_files "Modelfile (FROM hf.co + DRAFT + RENDERER/PARSER) + params" write_ollama_files

# --- Step 6: smoke test -----------------------------------------------------------------
# Through llama-server's OpenAI API with --jinja and the mmproj -- the way
# the model is actually run. Fails on any control token leaking into the
# reply (the token_type / template regressions we shipped fixes for), and
# checks vision on the COCO cats photo.
gguf_smoke () {
  local port=18089 img="$WORK/smoke-assets/cats.jpg"
  mkdir -p "$(dirname "$img")"
  [ -f "$img" ] || curl -sfL -o "$img" http://images.cocodataset.org/val2017/000000039769.jpg
  "$BIN/llama-server" -m "$TEXT_GGUF" --mmproj "$MMPROJ_Q8" -ngl "$NGL" --jinja --port "$port" \
    > "$LOG_DIR/gguf_smoke_server.log" 2>&1 &
  local pid=$! rc=0
  for _ in $(seq 1 180); do curl -sf "localhost:$port/health" >/dev/null && break; sleep 2; done
  # Inside `if` so errexit can't skip stopping the server on failure.
  if python3 - "$port" "$img" <<'PY'
import base64, json, sys, urllib.request
port, img = sys.argv[1], sys.argv[2]
def chat(content):
    body = {"messages": [{"role": "user", "content": content}], "max_tokens": 600, "temperature": 0}
    req = urllib.request.Request(f"http://localhost:{port}/v1/chat/completions", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
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

# --- Step 7: model card + publish ---------------------------------------------------------
run_step card "copy $(basename "$CARD") -> README.md" cp "$CARD" "$OUT_DIR/README.md"
if [ "${PUBLISH:-0}" = 1 ]; then
  run_step publish "hf upload $HF_REPO (text GGUF, mmproj Q8, drafter, Modelfile, params, README)" \
    hf upload "$HF_REPO" "$OUT_DIR" . --exclude '*-f16.gguf' --exclude '*.tmp' --commit-message "gemma4_26b_gguf_pipeline.sh"
else
  skip_step publish "PUBLISH!=1"
fi

echo "GEMMA4_26B_GGUF_PIPELINE_DONE"
ls -la "$OUT_DIR"
