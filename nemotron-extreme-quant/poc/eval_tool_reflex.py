"""Regression eval for the ipsupport-code LoRA's tool reflex (FINDINGS 2.5):
on a non-task prompt ("привет"), count
  - double_close: the special </think> generated more than once (no tools in prompt)
  - tool_call_with_tools: a <tool_call> emitted although nothing was asked
    (tools in prompt, LLMTray's generate_image schema)
  - said_no_tool_but_called: ... while the reasoning says "no tool needed"
temperature 1.0 / top_p 0.95, seeds 0..N-1, in-process mlx_lm. A release
should score 0 on all three (stock and no-LoRA models do).

    python eval_tool_reflex.py <mlx model dir> [N=20]
"""
import json, sys
from pathlib import Path
import mlx.core as mx
from mlx_lm import load
from mlx_lm.generate import stream_generate
from mlx_lm.sample_utils import make_sampler
TOOL_JSON = '{"type": "function", "function": {"name": "generate_image", "description": "Generate an image from a text description using a local diffusion model running on this Mac. Call this whenever the user asks to draw, create, generate, sketch, or make a picture, image, illustration, artwork, or photo.", "parameters": {"type": "object", "properties": {"prompt": {"type": "string", "description": "A detailed visual description of the image to generate."}, "width": {"type": "integer", "description": "Image width in pixels. Defaults to 1024."}, "height": {"type": "integer", "description": "Image height in pixels. Defaults to 1024."}}, "required": ["prompt"]}}}'
path = sys.argv[1]; N = int(sys.argv[2]) if len(sys.argv) > 2 else 20
TOOL = json.loads(TOOL_JSON)
model, tok = load(path)
end_id = tok.convert_tokens_to_ids("</think>")
res = {"double_close": 0, "tool_call_with_tools": 0, "said_no_tool_but_called": 0}
for tools in (False, True):
    ids = tok.apply_chat_template([{"role": "user", "content": "привет"}], tools=[TOOL] if tools else None, add_generation_prompt=True)
    for seed in range(N):
        mx.random.seed(seed)
        toks = [r.token for r in stream_generate(model, tok, ids, max_tokens=600, sampler=make_sampler(temp=1.0, top_p=0.95), prefill_step_size=128)]
        text = tok.decode(toks)
        if not tools and toks.count(end_id) > 1: res["double_close"] += 1
        called = "<tool_call>" in text
        if tools:
            res["tool_call_with_tools"] += called
            res["said_no_tool_but_called"] += called and ("no tool" in text.lower() or "no need to call" in text.lower())
print(Path(path).name, json.dumps(res), f"(N={N} each)", flush=True)
