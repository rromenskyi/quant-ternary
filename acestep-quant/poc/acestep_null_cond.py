"""CFG for ACE-Step's non-turbo DiTs (sft, base) in mlx-audio: its
unconditional branch runs the condition encoder on all-zero text and lyric
embeddings, while the official pipeline (acestep/models/mlx/dit_generate.py)
uses the trained `null_condition_emb` (in the checkpoint) broadcast over the
condition sequence. With mlx-audio's version the sft model sings gibberish
(Whisper WER ~1-1.9); the turbo DiT runs without CFG, so it never showed.

    install(model)   # after mlx_audio's load(), before generate()
"""
from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn


class NullAwareEncoder(nn.Module):
    """The condition encoder, returning the trained null condition for the
    all-zero call mlx-audio makes for CFG's unconditional branch."""

    def __init__(self, inner, null_emb):
        super().__init__()
        self.inner = inner
        self._null = null_emb

    def __call__(self, text_hidden_states=None, lyric_hidden_states=None, **kwargs):
        out, mask = self.inner(text_hidden_states=text_hidden_states, lyric_hidden_states=lyric_hidden_states, **kwargs)
        unconditional = (not mx.any(text_hidden_states).item()) and (not mx.any(lyric_hidden_states).item())
        if unconditional:
            out = mx.broadcast_to(self._null.astype(out.dtype), out.shape)
        return out, mask


def install(model) -> None:
    if not isinstance(model.encoder, NullAwareEncoder):
        model.encoder = NullAwareEncoder(model.encoder, model.null_condition_emb)
