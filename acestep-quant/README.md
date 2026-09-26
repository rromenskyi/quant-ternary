# acestep-quant

Running and quantizing [ACE-Step 1.5](https://huggingface.co/ACE-Step/Ace-Step1.5)
(MIT; text/lyrics → music with vocals, 48 kHz stereo) on Apple Silicon through
MLX, for LLMTray's chat (`generate_music`, planned). Sibling of `flux2-quant`
and `zimage-quant`.

Pipeline:
1. 5 Hz LM planner (Qwen3 0.6B / 4B, mlx_lm) turns the caption and lyrics
   into audio codes.
2. DiT (2.4B, flow matching, 8 turbo steps) makes 25 Hz latents from them.
3. Oobleck VAE decodes those to 48 kHz audio.

Code: [mlx-audio](https://github.com/Blaizzy/mlx-audio), branch `pc/add-ace`
(commit `1e8264a`). ACE-Step is not on its main branch or in its PyPI
releases. Weights: `mlx-community/ACE-Step1.5-MLX` (fp32) and `-MLX-4bit`.

Scripts (`poc/`):
- `acestep_bench.py`: one generation, with per-stage time and peak memory,
  the wav and a spectrogram. `--cap-codes` stops the LM once the requested
  duration is planned.
- `acestep_vae_chunk.py`: VAE decode in time windows, bit-identical to the
  full decode, about half the peak.
- `acestep_lyrics_wer.py`: vocal intelligibility, as Whisper's transcript
  vs the lyrics (WER).

Findings: [docs/FINDINGS.md](docs/FINDINGS.md).
