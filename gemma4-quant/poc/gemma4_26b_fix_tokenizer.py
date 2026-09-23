"""Prepare a llama.cpp-convertible copy of the Gemma 4 26B checkpoint.

transformers 5.17.0's tokenizer init crashes on this checkpoint's
`extra_special_tokens` field: the config stores it as a LIST
(`["<|video|>"]`), but `_set_model_specific_special_tokens` does
`list(special_tokens.keys())` on it -> "'list' object has no attribute
'keys'". convert_hf_to_gguf.py loads the tokenizer via AutoTokenizer, so
this crash blocks GGUF conversion.

DO NOT just delete the field (an earlier version of this script did): with
AutoTokenizer failing to construct, the converter fell back to a path that
mis-registered Gemma 4's control tokens (`<|turn>`, `<|channel>`,
`<|think|>`, ...), producing a GGUF whose chat/thinking mode was broken
(model emitted the literal text "<thought" then stopped) even though raw
completion worked perfectly. Instead, convert the field to the DICT form
transformers expects (`{"video_token": "<|video|>"}`) so AutoTokenizer
constructs fully and every special/control token is registered correctly.

Weights are symlinked (not copied) so this costs ~no disk; only the small
config/tokenizer files are copied and patched.
"""

import glob
import json
import os
import shutil
import sys

MODEL = sys.argv[1] if len(sys.argv) > 1 else (
    "/root/.cache/huggingface/hub/models--google--gemma-4-26B-A4B-it/"
    "snapshots/4d7ae4984b7db7de8f8457170b3f1a419ee76d52"
)
FIXED = sys.argv[2] if len(sys.argv) > 2 else "/workspace/gemma4-26b-src-fixed"

shutil.rmtree(FIXED, ignore_errors=True)
os.makedirs(FIXED)

for f in glob.glob(MODEL + "/*"):
    bn = os.path.basename(f)
    dst = os.path.join(FIXED, bn)
    if bn.endswith(".safetensors"):
        os.symlink(f, dst)
    else:
        shutil.copy2(f, dst)

p = os.path.join(FIXED, "tokenizer_config.json")
tc = json.load(open(p))
est = tc.get("extra_special_tokens")
if isinstance(est, list):
    # Convert list -> dict form transformers 5.17 expects. Name each entry
    # by stripping the <| |> / <| |> markers; falls back to a positional
    # key if that yields an empty name.
    as_dict = {}
    for i, tok in enumerate(est):
        name = tok.strip("<|>") or f"extra_token_{i}"
        name = name.replace("|", "").strip() or f"extra_token_{i}"
        as_dict[f"{name}_token"] = tok
    tc["extra_special_tokens"] = as_dict
    json.dump(tc, open(p, "w"))
    print("converted extra_special_tokens list -> dict:", as_dict)
else:
    print("extra_special_tokens already dict/absent:", est)

print("files:", sorted(os.listdir(FIXED)))
