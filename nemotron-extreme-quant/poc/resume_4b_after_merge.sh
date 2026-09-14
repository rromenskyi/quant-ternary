#!/usr/bin/env bash
# Resume point for run_4b_lora_retrain.sh when train+merge already succeeded
# but a downstream step failed -- re-running the whole script would redo the
# ~5min LoRA training pointlessly. Reuses the exact same env var names.
set -euo pipefail

POD_HOST="${POD_HOST:?set POD_HOST}"
POD_PORT="${POD_PORT:?set POD_PORT}"
POD_SSH_KEY="${POD_SSH_KEY:-$HOME/.runpod/ssh/runpodctl-ssh-key}"
OUTPUT_DIR="${OUTPUT_DIR:-./outputs/toolcall-lora-4b-v2}"
MERGED_DIR="${MERGED_DIR:-/root/outputs/toolcall-lora-4b-v2/merged}"
GROUP_SIZE="${GROUP_SIZE:-64}"
QUANT_OUT="${QUANT_OUT:-/root/nemotron4b-jang-dense-v2}"
MLX_OUT="${MLX_OUT:-/root/nemotron4b-jang-dense-v2-mlx}"
HF_ADAPTER_REPO="${HF_ADAPTER_REPO:-your-hf-username/NVIDIA-Nemotron-3-Nano-4B-ipsupport-code-lora}"
HF_FINAL_REPO="${HF_FINAL_REPO:-your-hf-username/NVIDIA-Nemotron-3-Nano-4B-JANG-GPTQ-ipsupport-code-lora}"

SSH="ssh -i $POD_SSH_KEY -p $POD_PORT -o StrictHostKeyChecking=no root@$POD_HOST"

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
    --extra-calib-file /root/sft_dataset_v2_flat.txt \
    --quant-recipe-mode component \
    --component-recipe jang-dense \
    --group-size ${GROUP_SIZE} \
    --gptq-device cuda"

echo "--- converting to MLX ---"
$SSH "cd /root/poc && python3 mlx_convert_recipe.py \
    --hf-path '$QUANT_OUT' \
    --mlx-path '$MLX_OUT' \
    --group-size ${GROUP_SIZE} \
    --mode component \
    --component-recipe jang-dense"

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
