"""The scored windows qat_eval.py and qat_sensitivity.py share:
(tokens, first scored position) per window.

- Plain: BOS + text, as llama.cpp's perplexity (Gemma without BOS is off the
  rails).
- --chat: the text as the model's reply to "Continue this text." -- for the
  big Gemma 4 -it (12B, 26B, 31B), which score raw text in the thousands
  (llama.cpp on Google's own GGUF too). With the thinking channel closed
  (enable_thinking=False: `<|channel>thought\\n<channel|>` after the model's
  turn). Left open, the model expects its reasoning first and reads the text
  as that: the 26B scored 82867 that way, 30.4 with it closed, on the same
  windows. Only the text's own tokens are scored.
"""


def windows(tok, text_path: str, ctx: int, n: int, chat: bool = False) -> list[tuple[list[int], int]]:
    ids = tok.encode(open(text_path).read(), add_special_tokens=False)
    prefix = [tok.bos_token_id]
    if chat:
        prompt = tok.apply_chat_template([{"role": "user", "content": "Continue this text."}],
                                         add_generation_prompt=True, tokenize=False, enable_thinking=False)
        prefix = tok.encode(prompt, add_special_tokens=False)
        if prefix[0] != tok.bos_token_id:
            prefix = [tok.bos_token_id] + prefix
    step = ctx - len(prefix)
    return [(prefix + ids[i: i + step], len(prefix) - 1) for i in range(0, len(ids) - step, step)][:n]
