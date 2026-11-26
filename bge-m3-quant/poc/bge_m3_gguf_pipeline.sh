#!/usr/bin/env bash
# GGUF builds of BAAI/bge-m3 (XLM-RoBERTa, 568 M, dense embeddings; CLS
# pooling, 8192 context), measured against the transformers reference, on a
# CPU (a prebuilt llama.cpp, no GPU). Every step is resumable.
#
#   WORK=~/bge-m3-gguf ./bge_m3_gguf_pipeline.sh            # build + measure
#   WORK=~/bge-m3-gguf PUBLISH=1 ./bge_m3_gguf_pipeline.sh  # ... and upload
#
# Steps (-> $LOG_DIR/bgem3_<step>.log):
#   tools       llama.cpp release binaries + the same tag's source (converter), a venv
#   download    BAAI/bge-m3 at a pinned revision (pytorch_model.bin, fp32)
#   convert     F16 GGUF
#   baseline    ggml-org/bge-m3-Q8_0-GGUF (what LLMTray for Windows uses)
#   quantize    QUANTS from the F16
#   evalset     Tatoeba bitext (5 languages), stdlib code pairs, long modules
#   reference   transformers fp32 dense embeddings
#   eval        every GGUF through llama-server --embedding (eval_gguf.py)
#   card, publish (PUBLISH=1; HF token in the default HF location)
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/../../gemma4-quant/poc/pipeline_lib.sh"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

LLAMA_TAG="${LLAMA_TAG:-b12751}"
MODEL_ID="${MODEL_ID:-BAAI/bge-m3}"
QUANTS="${QUANTS:-Q8_0 Q6_K Q5_K_M Q4_K_M}"
PUBLISH_QUANTS="${PUBLISH_QUANTS:-$QUANTS}"
HF_REPO="${HF_REPO:-roman220220/bge-m3-GGUF}"
THREADS="${THREADS:-8}"   # a shared node: not every core
BIN="$WORK/bin/llama-$LLAMA_TAG"
SRC="$WORK/llama.cpp-$LLAMA_TAG"
PY="$WORK/venv/bin/python"
SNAP="$WORK/src"
OUT="$WORK/out"
F16="$OUT/bge-m3-f16.gguf"
BASE="$WORK/ref/bge-m3-q8_0.gguf"
EVAL="$WORK/eval.jsonl"
REF="$WORK/reference"
RESULTS="$WORK/results.json"
CARD="${CARD:-$HERE/../cards/${HF_REPO#*/}.md}"

pipeline_init bgem3 "MODEL_ID=$MODEL_ID QUANTS=$QUANTS LLAMA_TAG=$LLAMA_TAG"
mkdir -p "$OUT" "$WORK/ref"

if [ -x "$BIN/llama-quantize" ] && [ -x "$BIN/llama-server" ] && [ -f "$SRC/convert_hf_to_gguf.py" ] && [ -x "$PY" ]; then
  skip_step tools "llama.cpp $LLAMA_TAG and the venv in $WORK"
else
  run_step tools "llama.cpp $LLAMA_TAG binaries + source, venv" bash -c "
    cd '$WORK' && mkdir -p bin
    curl -sfL https://github.com/ggml-org/llama.cpp/releases/download/$LLAMA_TAG/llama-$LLAMA_TAG-bin-ubuntu-x64.tar.gz | tar xzf - -C bin
    curl -sfL https://github.com/ggml-org/llama.cpp/archive/refs/tags/$LLAMA_TAG.tar.gz | tar xzf -
    [ -x ~/.local/bin/uv ] || curl -LsSf https://astral.sh/uv/install.sh | sh
    ~/.local/bin/uv venv -q -p 3.12 venv
    VIRTUAL_ENV=venv ~/.local/bin/uv pip install -q --index-strategy unsafe-best-match \
      --extra-index-url https://download.pytorch.org/whl/cpu torch transformers sentencepiece protobuf numpy \
      huggingface_hub ./llama.cpp-$LLAMA_TAG/gguf-py"
fi

if [ -s "$SNAP/pytorch_model.bin" ] && [ -s "$SNAP/REVISION" ]; then
  skip_step download "$SNAP at $(cat "$SNAP/REVISION")"
else
  run_step download "$MODEL_ID" env HF_HUB_DISABLE_XET=1 HF_HUB_DOWNLOAD_TIMEOUT=300 "$PY" -c "
from huggingface_hub import HfApi, snapshot_download
rev = HfApi().model_info('$MODEL_ID').sha
snapshot_download('$MODEL_ID', revision=rev, local_dir='$SNAP', allow_patterns=['config.json', 'tokenizer*',
    'special_tokens_map.json', 'sentencepiece.bpe.model', 'pytorch_model.bin', '1_Pooling/*', 'modules.json',
    'config_sentence_transformers.json', 'sentence_bert_config.json', 'README.md'])
open('$SNAP/REVISION', 'w').write(rev)
print('revision', rev)"
fi

if [ -s "$F16" ]; then skip_step convert "$F16 exists"; else
  run_step convert "F16 GGUF" nice "$PY" "$SRC/convert_hf_to_gguf.py" "$SNAP" --outfile "$F16" --outtype f16
fi

if [ -s "$BASE" ]; then skip_step baseline "$BASE exists"; else
  run_step baseline "ggml-org/bge-m3-Q8_0-GGUF" env HF_HUB_DISABLE_XET=1 "$PY" -c "
from huggingface_hub import hf_hub_download
hf_hub_download('ggml-org/bge-m3-Q8_0-GGUF', 'bge-m3-q8_0.gguf', local_dir='$WORK/ref')"
fi

for q in $QUANTS; do
  f="$OUT/bge-m3-${q,,}.gguf"
  if [ -s "$f" ]; then skip_step "quantize_$q" "$f exists"; else
    run_step "quantize_$q" "llama-quantize $q" nice "$BIN/llama-quantize" "$F16" "$f" "$q" "$THREADS"
  fi
done

if [ -s "$EVAL" ]; then skip_step evalset "$EVAL exists"; else
  run_step evalset "Tatoeba bitext + stdlib code + long modules" "$PY" "$HERE/make_eval_set.py" --out "$EVAL"
fi

if [ -s "$REF/dense.npy" ] && [ "$REF/dense.npy" -nt "$EVAL" ]; then skip_step reference "$REF newer than $EVAL"; else
  run_step reference "transformers fp32 dense embeddings" nice "$PY" "$HERE/reference.py" --model "$SNAP" \
    --eval "$EVAL" --out "$REF" --threads "$THREADS"
fi

ARGS=(--gguf "$F16")
for q in $QUANTS; do ARGS+=(--gguf "$OUT/bge-m3-${q,,}.gguf"); done
run_step eval "every GGUF through llama-server" nice "$PY" "$HERE/eval_gguf.py" --server "$BIN/llama-server" \
  --eval "$EVAL" --ref "$REF" "${ARGS[@]}" --baseline "$BASE" --out "$RESULTS" --threads "$THREADS"

if [ "${PUBLISH:-0}" = 1 ]; then
  [ -f "$CARD" ] || { echo "no $CARD" >&2; exit 1; }
  PUB="$WORK/publish"
  rm -rf "$PUB" && mkdir -p "$PUB"
  cp "$CARD" "$PUB/README.md"
  cp "$HERE/../cards/licenses/bge-m3-LICENSE" "$PUB/LICENSE"
  for img in $(grep -o "[a-z-]*banner\.png" "$CARD" | sort -u); do cp "$HERE/../../cards_assets/$img" "$PUB/"; done
  for q in $PUBLISH_QUANTS; do ln "$OUT/bge-m3-${q,,}.gguf" "$PUB/"; done
  run_step publish "hf upload $HF_REPO" "$WORK/venv/bin/hf" upload "$HF_REPO" "$PUB" . \
    --commit-message "bge_m3_gguf_pipeline.sh (llama.cpp $LLAMA_TAG, $MODEL_ID@$(cat "$SNAP/REVISION"))"
else
  skip_step publish "PUBLISH!=1"
fi
echo "BGEM3_GGUF_PIPELINE_DONE $RESULTS"
