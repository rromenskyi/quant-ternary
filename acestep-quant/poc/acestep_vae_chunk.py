"""Decodes ACE-Step 1.5 latents in time windows: the Oobleck VAE's full
decode peaks at ~9 GB above the weights for 30 s of audio, and grows
linearly with the length. Windows of `chunk` latent frames (25 Hz) are
decoded with `overlap` extra frames each side, which are cut off again; the
decoder is convolutional with a small receptive field, so the cut edges
match the full decode.

    decode = chunked(vae.decode, chunk=250, overlap=32)
"""
from __future__ import annotations

import mlx.core as mx

SAMPLES_PER_FRAME = 1920  # 48 kHz / 25 Hz


def chunked(decode, chunk: int = 250, overlap: int = 32):
    def run(latents: mx.array) -> mx.array:
        # latents: [batch, time, dim]
        total = latents.shape[1]
        if total <= chunk + 2 * overlap:
            return decode(latents)
        pieces = []
        for start in range(0, total, chunk):
            end = min(start + chunk, total)
            lo, hi = max(0, start - overlap), min(total, end + overlap)
            audio = decode(latents[:, lo:hi, :])
            first = (start - lo) * SAMPLES_PER_FRAME
            audio = audio[:, :, first : first + (end - start) * SAMPLES_PER_FRAME]
            mx.eval(audio)
            pieces.append(audio)
            mx.clear_cache()
        return mx.concatenate(pieces, axis=-1)

    return run
