"""ACE-Step 1.5's 5 Hz LM planner, prompted the way the official pipeline
(github.com/ace-step/ACE-Step-1.5, acestep/llm_inference.py) does -- which
is what the LM was trained on. mlx-audio's port (branch pc/add-ace) sends
one user message with "# Instruction ... # Lyrics ... # Metas - duration",
a layout the LM never saw: it ignores the duration (plans a ~3-minute song,
of which a 30 s clip is only the intro) and the lyrics barely come through
(Whisper WER ~1.0 vs 0.07-0.46 for the official pipeline, same lyrics).

Official layout, reproduced here:
- system "# Instruction\\n<instruction>\\n\\n", user "# Caption\\n..\\n\\n# Lyric\\n..\\n";
- phase 1: the LM writes its metadata as YAML in <think> ... </think>; the
  duration (and language) asked for replace what it wrote -- the official
  code forces them with a constrained decoder -- and the YAML is re-dumped
  with sorted keys, as in training;
- phase 2: the assistant turn continues after "</think>\\n\\n" with audio
  codes only, with classifier-free guidance (scale 2.0) against the
  unconditional prompt "NO USER INPUT" + "<think>\\n\\n</think>\\n\\n",
  stopping at duration x 5 codes (5 Hz).
Sampling as the official defaults: temperature 0.85, top-p 0.9.

    codes, metadata = plan(model, tokenizer, caption, lyrics, duration=30, language="en", seed=1)
"""
from __future__ import annotations

import re

import mlx.core as mx
import yaml

INSTRUCTION = "Generate audio semantic tokens based on the given conditions:"
NEGATIVE = "NO USER INPUT"


def _template(tokenizer, user: str) -> str:
    return tokenizer.apply_chat_template(
        [{"role": "system", "content": f"# Instruction\n{INSTRUCTION}\n\n"}, {"role": "user", "content": user}],
        tokenize=False, add_generation_prompt=True,
    )


def _sample(logits: mx.array, temperature: float, top_p: float) -> int:
    logits = logits.astype(mx.float32) / temperature
    probs = mx.softmax(logits, axis=-1)
    order = mx.argsort(-probs)
    sorted_p = probs[order]
    keep = (mx.cumsum(sorted_p) - sorted_p) < top_p
    sorted_p = mx.where(keep, sorted_p, 0)
    choice = mx.random.categorical(mx.log(sorted_p + 1e-20))
    return int(order[choice].item())


class _Stream:
    """One prompt's KV cache, fed a token at a time."""

    def __init__(self, model, tokens: list[int]):
        from mlx_lm.models.cache import make_prompt_cache

        self.model = model
        self.cache = make_prompt_cache(model)
        self.logits = self._feed(tokens)

    def _feed(self, tokens: list[int]) -> mx.array:
        out = self.model(mx.array(tokens)[None], cache=self.cache)
        return out[0, -1]

    def push(self, token: int) -> None:
        self.logits = self._feed([token])


def _code_mask(tokenizer) -> mx.array:
    vocab = tokenizer.get_vocab()
    size = max(vocab.values()) + 1
    mask = [False] * size
    for text, i in vocab.items():
        if (m := re.fullmatch(r"<\|audio_code_(\d+)\|>", text)) and int(m.group(1)) < 64000:
            mask[i] = True
    return mx.array(mask)


def parse_cot(body):
    """The LM's <think> YAML; when that doesn't parse (a caption with an
    unquoted colon, say), the known "key: value" lines one by one -- the
    official constrained decoder never lets it be malformed."""
    keys = ("bpm", "caption", "duration", "keyscale", "language", "timesignature")
    try:
        meta = yaml.safe_load(body)
        if isinstance(meta, dict):
            return {k: v for k, v in meta.items() if k in keys}
    except yaml.YAMLError:
        pass
    meta, current = {}, None
    for line in body.splitlines():
        m = re.match(r"^(\w+):\s*(.*)$", line)
        if m and m.group(1) in keys:
            current = m.group(1)
            value = m.group(2).strip()
            meta[current] = int(value) if value.isdigit() else value
        elif current == "caption" and line.startswith(" "):
            meta["caption"] = f"{meta['caption']} {line.strip()}".strip()   # a wrapped caption
    return meta


def plan(model, tokenizer, caption: str, lyrics: str, duration: float, language: str = "en", seed: int | None = None,
         temperature: float = 0.85, top_p: float = 0.9, cfg: float = 2.0, max_cot_tokens: int = 512):
    if seed is not None:
        mx.random.seed(seed)
    lyrics = lyrics.strip()   # empty as is, as the official pipeline sends it
    user = f"# Caption\n{caption}\n\n# Lyric\n{lyrics}\n"
    enc = lambda s: tokenizer.encode(s, add_special_tokens=False)

    # Phase 1: the LM's metadata (unguided: short, and only a plan).
    prompt = _template(tokenizer, user)
    stream = _Stream(model, enc(prompt))
    text = ""
    eos = set(getattr(tokenizer, "eos_token_ids", None) or [tokenizer.eos_token_id])
    for _ in range(max_cot_tokens):
        token = _sample(stream.logits, temperature, top_p)
        if token in eos:
            break
        text += tokenizer.decode([token])
        if "</think>" in text:
            break
        stream.push(token)
    body = text.split("<think>", 1)[-1].split("</think>", 1)[0]
    metadata = parse_cot(body)
    metadata["duration"] = int(round(duration))
    if language and language != "unknown":
        metadata["language"] = language
    ts = metadata.get("timesignature")
    if isinstance(ts, str) and ts.endswith("/4"):
        metadata["timesignature"] = int(ts.split("/")[0]) if ts.split("/")[0].isdigit() else ts
    cot = "<think>\n" + yaml.dump(metadata, allow_unicode=True, sort_keys=True).strip() + "\n</think>"

    # Phase 2: codes only, guided.
    cond = _Stream(model, enc(prompt + cot + "\n\n"))
    uncond = _Stream(model, enc(_template(tokenizer, NEGATIVE) + "<think>\n\n</think>\n\n")) if cfg != 1.0 else None
    mask = _code_mask(tokenizer)
    need = int(round(duration * 5))
    codes = []
    for _ in range(need):
        logits = cond.logits
        if uncond is not None:
            logits = uncond.logits + cfg * (logits - uncond.logits)
        n = min(logits.shape[-1], mask.shape[0])
        logits = mx.where(mask[:n], logits[:n], -mx.inf)
        token = _sample(logits, temperature, top_p)
        codes.append(token)
        cond.push(token)
        if uncond is not None:
            uncond.push(token)
    return "".join(tokenizer.decode([t]) for t in codes), metadata


def load_lm(path: str):
    """mlx_lm.load, or -- for the official checkpoints, which store their
    keys without the "model." prefix mlx-lm's Qwen3 expects -- the same with
    the keys remapped (as the official pipeline does)."""
    import glob
    from pathlib import Path

    import mlx_lm
    from mlx_lm.utils import _get_classes, load_config, load_tokenizer

    try:
        return mlx_lm.load(path)
    except ValueError:
        local = Path(path)
        config = load_config(local)
        weights = {}
        for f in glob.glob(str(local / "model*.safetensors")):
            weights.update(mx.load(f))
        if not next(iter(weights)).startswith("model."):
            weights = {f"model.{k}": v for k, v in weights.items()}
        model_class, args_class = _get_classes(config=config)
        model = model_class(args_class.from_dict(config))
        if hasattr(model, "sanitize"):
            weights = model.sanitize(weights)
        model.load_weights(list(weights.items()), strict=True)
        mx.eval(model.parameters())
        model.eval()
        return model, load_tokenizer(local)


def install(lm_path: str, dit_metadata: bool = True) -> None:
    """Makes mlx-audio's ACE-Step plan with this planner and the LM at
    lm_path (an HF repo id or a local Qwen3 checkpoint). With
    `dit_metadata`, the DiT is also conditioned the official way: the LM's
    caption and its bpm / keyscale / timesignature in the DiT prompt (mlx-audio
    sends the user's caption and "N/A" for all three). The LM is loaded for
    each generation and freed after planning, like mlx-audio's own."""
    from mlx_audio.tts.models.ace_step import ace_step as ace_module
    from mlx_audio.tts.models.ace_step import lm as lm_module

    planned: dict = {}

    def generate_audio_codes(self, caption, lyrics="", duration=30, language="en", seed=None):
        return planned["codes"], planned["metadata"]

    lm_module.ACEStepLM.generate_audio_codes = generate_audio_codes
    lm_module.ACEStepLM.load = lambda self: setattr(self, "_loaded", True)

    model_cls = ace_module.Model
    original_generate = model_cls.generate
    original_text = model_cls._prepare_text_embeddings

    def generate(self, text, lyrics="", duration=30.0, seed=None, vocal_language="en", **kwargs):
        lm, tokenizer = load_lm(lm_path)
        planned["codes"], planned["metadata"] = plan(lm, tokenizer, text, lyrics, duration, vocal_language, seed)
        del lm, tokenizer
        mx.clear_cache()
        yield from original_generate(self, text=text, lyrics=lyrics, duration=duration, seed=seed,
                                     vocal_language=vocal_language, **kwargs)

    def prepare_text(self, text, max_length=256, duration=30.0, bpm=None, keyscale=None, timesignature=None):
        meta = planned.get("metadata") or {}
        if dit_metadata and meta:
            text = str(meta.get("caption") or text)
            bpm = bpm if bpm is not None else meta.get("bpm")
            keyscale = keyscale or meta.get("keyscale")
            timesignature = timesignature or (str(meta["timesignature"]) if meta.get("timesignature") else None)
        return original_text(self, text, max_length=max_length, duration=duration, bpm=bpm, keyscale=keyscale,
                             timesignature=timesignature)

    model_cls.generate = generate
    model_cls._prepare_text_embeddings = prepare_text
