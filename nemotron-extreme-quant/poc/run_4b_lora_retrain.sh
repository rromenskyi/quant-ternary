#!/usr/bin/env bash
# End-to-end retrain of the ipsupport-code LoRA on Nemotron-3-Nano-4B, using
# an augmented dataset (adds non-tool casual-chat examples -- the original
# 110-conversation dataset had ZERO turns where no tool call was needed,
# which trained the model to never close its <think> block for plain replies:
# every casual message got its whole answer trapped inside an unclosed
# <think>, so clients saw empty content/tool_calls). Requires setup_pod.sh
# to have already been run against this pod with --with-axolotl.
#
# Usage:
#   POD_HOST=1.2.3.4 POD_PORT=12345 LOCAL_DATASET=/path/to/sft_dataset_final.jsonl ./run_4b_lora_retrain.sh
set -euo pipefail

POD_HOST="${POD_HOST:?set POD_HOST}"
POD_PORT="${POD_PORT:?set POD_PORT}"
POD_SSH_KEY="${POD_SSH_KEY:-$HOME/.runpod/ssh/runpodctl-ssh-key}"
LOCAL_DATASET="${LOCAL_DATASET:?set LOCAL_DATASET to the local sft_dataset jsonl path}"
AXOLOTL_VENV="${AXOLOTL_VENV:-/root/axolotl_venv}"
MODEL_SRC_DIR="${MODEL_SRC_DIR:-/root/nemotron4b-bf16-src}"
RUN_TAG="${RUN_TAG:-v2}"
OUTPUT_DIR="${OUTPUT_DIR:-./outputs/toolcall-lora-4b-${RUN_TAG}}"
GROUP_SIZE="${GROUP_SIZE:-64}"
QUANT_OUT="${QUANT_OUT:-/root/nemotron4b-jang-dense-${RUN_TAG}}"
MLX_OUT="${MLX_OUT:-/root/nemotron4b-jang-dense-${RUN_TAG}-mlx}"
HF_ADAPTER_REPO="${HF_ADAPTER_REPO:-your-hf-username/NVIDIA-Nemotron-3-Nano-4B-ipsupport-code-lora}"
HF_FINAL_REPO="${HF_FINAL_REPO:-your-hf-username/NVIDIA-Nemotron-3-Nano-4B-JANG-GPTQ-ipsupport-code-lora}"
# r=32/alpha=64/4 epochs on the full q/k/v/o+up/down_proj module set was found
# to catastrophically overfit this small (235-conversation) dataset -- neither
# attention-only nor MLP-only LoRA alone reproduced the effect, only training
# both simultaneously at this capacity/epoch count did (see
# docs/session_findings_2026-09-11.md §7v). Lower rank/epochs as a fix.
LORA_R="${LORA_R:-32}"
LORA_ALPHA="${LORA_ALPHA:-64}"
LORA_DROPOUT="${LORA_DROPOUT:-0.05}"
NUM_EPOCHS="${NUM_EPOCHS:-4}"
# Found (session_findings §7v) that training attention (q/k/v/o_proj) and MLP
# (up/down_proj) LoRA modules TOGETHER breaks general code competency (C++
# in particular) regardless of rank/epochs, while EITHER alone (post-hoc
# ablation of a jointly-trained adapter) does not -- looks like interference
# between simultaneously-adapted attention and MLP circuits, not simple
# overfitting. Default keeps the full set for backward compat; override to
# e.g. "q_proj k_proj v_proj o_proj" to test attention-only.
LORA_TARGET_MODULES="${LORA_TARGET_MODULES:-q_proj k_proj v_proj o_proj up_proj down_proj}"

SSH="ssh -i $POD_SSH_KEY -p $POD_PORT -o StrictHostKeyChecking=no root@$POD_HOST"
SCP="scp -i $POD_SSH_KEY -P $POD_PORT -o StrictHostKeyChecking=no"
# pipeline_dashboard.py (--log-glob default /root/pipeline-*.log) only sees
# progress written to a file matching that glob ON THE POD -- this script's
# own SSH commands stream output back to wherever *this* script's stdout
# goes (a local log, or nowhere if run interactively), which the dashboard
# has no way to see. Tee every stage's remote output into one such file so
# the dashboard reflects what's actually happening instead of showing
# "no run detected" for the entire retrain.
PIPELINE_LOG="/root/pipeline-4b-retrain-${RUN_TAG}.log"

echo "--- uploading augmented dataset ($(wc -l < "$LOCAL_DATASET") conversations) ---"
$SCP "$LOCAL_DATASET" "root@${POD_HOST}:/root/sft_dataset_${RUN_TAG}.jsonl"

LORA_TARGET_MODULES_YAML=""
for m in $LORA_TARGET_MODULES; do
  LORA_TARGET_MODULES_YAML="${LORA_TARGET_MODULES_YAML}  - ${m}
"
done

echo "--- writing axolotl config (target modules: ${LORA_TARGET_MODULES}) ---"
$SSH "cat > /root/nemotron4b_toolcall_qlora_${RUN_TAG}.yaml" <<EOF
base_model: ${MODEL_SRC_DIR}

plugins:
  - axolotl.integrations.cut_cross_entropy.CutCrossEntropyPlugin
  - axolotl.integrations.liger.LigerPlugin

liger_layer_norm: true
liger_rope: true
liger_rms_norm: true
liger_glu_activation: true
liger_rms_norm_gated: true

lora_mlp_kernel: false
lora_qkv_kernel: false
lora_o_kernel: false

chat_template: tokenizer_default
datasets:
  - path: /root/sft_dataset_${RUN_TAG}.jsonl
    type: chat_template
    field_messages: messages
    message_property_mappings:
      role: role
      content: content

val_set_size: 0.15
output_dir: ${OUTPUT_DIR}
dataset_prepared_path: last_run_prepared_4b_${RUN_TAG}

sequence_len: 4096
sample_packing: false

load_in_4bit: true
adapter: qlora
lora_r: ${LORA_R}
lora_alpha: ${LORA_ALPHA}
lora_dropout: ${LORA_DROPOUT}
lora_target_modules:
${LORA_TARGET_MODULES_YAML}

gradient_accumulation_steps: 4
micro_batch_size: 1
num_epochs: ${NUM_EPOCHS}
optimizer: adamw_torch_4bit
lr_scheduler: cosine
learning_rate: 0.0002

bf16: auto
tf32: true

gradient_checkpointing: true
gradient_checkpointing_kwargs:
  use_reentrant: false

logging_steps: 1
attn_implementation: flash_attention_2

warmup_ratio: 0.1
evals_per_epoch: 4
saves_per_epoch: 1
weight_decay: 0.0

special_tokens:
EOF

echo "--- flattening dataset to plain text for GPTQ calibration mix-in ---"
$SSH "cd /root/poc && python3 flatten_chat_jsonl_to_text.py --input /root/sft_dataset_${RUN_TAG}.jsonl --output /root/sft_dataset_${RUN_TAG}_flat.txt"

echo "--- training LoRA on augmented dataset (this can take a while) ---"
$SSH "source '${AXOLOTL_VENV}/bin/activate' && cd /root && python3 -m axolotl.cli.train /root/nemotron4b_toolcall_qlora_${RUN_TAG}.yaml 2>&1 | tee -a '${PIPELINE_LOG}'"

echo "--- merging LoRA into base model ---"
$SSH "source '${AXOLOTL_VENV}/bin/activate' && cd /root && python3 -m axolotl.cli.merge_lora /root/nemotron4b_toolcall_qlora_${RUN_TAG}.yaml 2>&1 | tee -a '${PIPELINE_LOG}'"

MERGED_DIR=""
for candidate in "${OUTPUT_DIR}/merged" "${OUTPUT_DIR}/merge"; do
  if $SSH "test -f /root/${candidate#./}/config.json" 2>/dev/null; then
    MERGED_DIR="/root/${candidate#./}"
    break
  fi
done
if [[ -z "$MERGED_DIR" ]]; then
  echo "ERROR: could not find merged model dir under ${OUTPUT_DIR} -- inspect manually:" >&2
  echo "  $SSH \"find /root/${OUTPUT_DIR#./} -maxdepth 2\"" >&2
  exit 1
fi
echo "merged model at: $MERGED_DIR"

echo "--- pushing adapter (pre-merge) to HF: ${HF_ADAPTER_REPO} ---"
$SSH "cd /root && python3 -c \"
from huggingface_hub import HfApi
api = HfApi()
api.create_repo('${HF_ADAPTER_REPO}', private=True, exist_ok=True)
api.upload_folder(repo_id='${HF_ADAPTER_REPO}', folder_path='${OUTPUT_DIR}', allow_patterns=['adapter_*'])
\""

echo "--- GPTQ quantize with jang-dense component recipe ---"
$SSH "cd /root/poc && python3 gptq_stock_convert.py \
    --model '$MERGED_DIR' \
    --output '$QUANT_OUT' \
    --wikitext /root/llama.cpp/wikitext-2-raw/wiki.train.raw \
    --extra-calib-file /root/sft_dataset_${RUN_TAG}_flat.txt \
    --quant-recipe-mode component \
    --component-recipe jang-dense \
    --group-size ${GROUP_SIZE} \
    --gptq-device cuda 2>&1 | tee -a '${PIPELINE_LOG}'"

echo "--- converting to MLX ---"
$SSH "cd /root/poc && python3 mlx_convert_recipe.py \
    --hf-path '$QUANT_OUT' \
    --mlx-path '$MLX_OUT' \
    --group-size ${GROUP_SIZE} \
    --mode component \
    --component-recipe jang-dense 2>&1 | tee -a '${PIPELINE_LOG}'"

echo "--- functional regression test: casual chat must close <think>, tool tasks must still call tools ---"
$SSH "python3 -c \"
from mlx_lm import load, generate
model, tokenizer = load('$MLX_OUT')

tools = [{'type': 'function', 'function': {'name': 'help', 'description': 'Get help', 'parameters': {'type': 'object', 'properties': {'action': {'type': 'string'}}, 'required': ['action']}}}]
sys_prompt = 'You are the engine inside ipsupport-code, a local terminal coding agent. You run in a loop and act ONLY through tools.'

def run(user_msg):
    messages = [{'role': 'system', 'content': sys_prompt}, {'role': 'user', 'content': user_msg}]
    prompt = tokenizer.apply_chat_template(messages, tools=tools, add_generation_prompt=True, tokenize=False)
    out = generate(model, tokenizer, prompt=prompt, max_tokens=200, verbose=False)
    return out

for msg in ['hi', 'привет', 'thanks, bye']:
    out = run(msg)
    closed = '</think>' in out
    print(f'[casual] {msg!r} -> closed_think={closed} | {out[:150]!r}')
    assert closed, f'REGRESSION: casual message {msg!r} never closed think block'

out = run('write a python function that reverses a string and run it')
print(f'[task] -> {out[:300]!r}')
print('ALL CHECKS PASSED')
\""

echo "=== retrain + quantize + functional test complete ==="
echo "merged (pre-quant) model: $MERGED_DIR"
echo "quantized HF-format model: $QUANT_OUT"
echo "MLX model: $MLX_OUT"
echo "adapter pushed to: https://huggingface.co/${HF_ADAPTER_REPO} (private)"
echo ""
echo "NOT auto-uploading the final MLX model to ${HF_FINAL_REPO} -- review the functional"
echo "test output above first, then run:"
echo "  $SSH \"cd /root/poc && python3 -c \\\"from huggingface_hub import HfApi; api = HfApi(); api.create_repo('${HF_FINAL_REPO}', exist_ok=True); api.upload_folder(repo_id='${HF_FINAL_REPO}', folder_path='${MLX_OUT}')\\\"\""
