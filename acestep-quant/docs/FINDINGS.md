# ACE-Step 1.5 on MLX — findings

Machine: M5, 26 GB (Metal limit ~19 GB). mlx 0.32.2, mlx-lm 0.31.1,
mlx-audio at `pc/add-ace` 1e8264a. Default: 30 s tracks, turbo DiT, seed 7.

## Baseline (mlx-community 4-bit DiT, LM 0.6B)

Instrumental, 8 steps, "upbeat synthwave …", warm (weights cached):

| | total | LM | diffusion | VAE decode | peak |
|---|---|---|---|---|---|
| as shipped | 22.9 s | 15.5 s (1085 codes) | 2.6 s | 3.8 s | 11.9 GB |
| LM capped at duration×5 codes | **10.8 s** (RTF 0.36) | 4.0 s (150 codes) | 2.6 s | 3.8 s | 11.9 GB |

- **The LM overruns the requested length ~7×.** The 0.6B planner ignores
  the duration in its prompt; its metadata says 218 s for a 30 s request.
  It writes codes up to `max_new_tokens` (3000), and everything past the
  duration is cut off afterwards. Stopping the stream at duration × 5
  codes (5 Hz) gives the same track (same seed) in 4 s instead of 15.5 s.
- Weights resident after load: 2.9 GB (4-bit DiT, fp32 Qwen3-Embedding
  0.6B text encoder, fp32 VAE). LM peak 4.3 GB (loaded and offloaded
  around planning).
- **The peak is the VAE decode**: 11.9 GB for 30 s, 9.6 GB of it
  activations, growing linearly with the length. A 60 s track would not
  fit beside the weights.

## Chunked VAE decode (`poc/acestep_vae_chunk.py`)

Decode in windows of `chunk` latent frames (25 Hz), with `overlap` frames
on each side that are cut off again:

| decode (30 s) | peak | time |
|---|---|---|
| full | 9.55 GB | 3.5 s |
| chunk 250 (10 s), overlap 32 | 5.65 GB | 3.3 s |
| chunk 250, overlap 16 | 5.35 GB | 3.0 s |
| chunk 125 (5 s), overlap 32 | 4.83 GB | 4.0 s |

**Bit-identical** to the full decode on a real 20 s track (RMS 0.19, max
|diff| 0): the decoder's receptive field fits inside 16 frames. The peak
stops depending on the track length.

## Vocals: measuring them without listening (`poc/acestep_lyrics_wer.py`)

Whisper large-v3-turbo (mlx-whisper) transcribes the track, and the score
is the word error rate against the lyrics that were asked for (section
markers dropped). The metric itself was checked: the same lyrics spoken by
macOS `say` over the synthwave instrumental score WER 0.0 at equal level
and at −6 dB, so music doesn't hide intelligible words from it. Whisper
hallucinates on unintelligible singing ("To be continued…", "pi pi pi…"),
which scores ≥ 1.

## mlx-audio's LM prompt is not the one the LM was trained on

Pop song, 2 verses + chorus, "upbeat pop song with female vocals…", 30 s:

| pipeline | WER per seed | per-song time |
|---|---|---|
| mlx-audio as is, 4-bit DiT, LM 0.6B (with and without the code cap) | 0.96 / 0.98 / 0.98 / 1.34 / 0.96 / 0.96 | 11–13 s |
| mlx-audio, full-precision DiT, LM 0.6B | 1.0 / 1.0 / 1.0 | 14 s (peak 20 GB) |
| mlx-audio, 4-bit DiT, LM 4B | 5.07 (hallucination) / 0.80 | 35 s |
| **official ACE-Step-1.5 pipeline** (its MLX backends, LM 1.7B, bf16 DiT) | **0.09 / 0.46 / 0.07** | 68–85 s |
| 4-bit DiT + `poc/acestep_planner.py` (official LM prompt), LM 1.7B | 0.75 / 0.34 / 0.11 | 22 s |

The model sings the lyrics almost word for word through the official
pipeline, so the fault is in the port, not in the weights. mlx-audio's
planner (`lm.py`):
- sends one user message with `# Instruction … # Lyrics … # Metas -
  duration`;
- the official layout (and the training one) is a system message
  `# Instruction\n…\n\n` plus a user message `# Caption\n…\n\n# Lyric\n…\n`
  (singular "Lyric"), with the metadata as YAML inside the assistant's
  `<think>` block.

The LM ignores the duration it was given: its CoT says 191–218 s for a
30 s request. It plans a 3-minute song, and the 30 s clip is the
instrumental intro, which is why WER is ~1.

`poc/acestep_planner.py` reproduces the official way:
- **phase 1**: CoT until `</think>`; the requested duration and language
  replace the LM's own (the official code forces them with a constrained
  decoder), and the YAML is re-dumped with sorted keys;
- **phase 2**: codes only (every non-code token masked), CFG 2.0 against
  "NO USER INPUT" + an empty `<think>`, stop at duration × 5 codes;
- temperature 0.85, top-p 0.9 (official defaults);
- LM 1.7B from the official checkpoint. Its keys lack mlx-lm's `model.`
  prefix, so they are remapped, as the official loader does.

Also official: the DiT gets the LM's CoT caption and bpm / keyscale /
timesignature in its prompt; mlx-audio sends the user's caption with
"N/A" for all three. `install(dit_metadata=True)` does it the official way.
3 seeds weren't enough to tell whether that helps (1.0 / 0.27 / 0.30 vs
0.75 / 0.34 / 0.11); a 3-song × 4-seed run follows.

## Vocal test set: 3 songs × 4 seeds (2026-09-26)

Songs in `docs/vocal_eval/`: pop (female), rock (male), ballad (female),
each 2 verses + a chorus, 30 s, English. DiT 8 steps. Raw per-track WER in
`docs/vocal_eval/wer_*.jsonl`. WER is capped at 1 per track for the means.

| config | mean WER | pop | rock | ballad | tracks < 0.3 | time / track |
|---|---|---|---|---|---|---|
| official pipeline (LM 1.7B, bf16 DiT) | 0.66 | 0.63 | 0.58 | 0.77 | 2 / 12 | 68–85 s |
| **4-bit DiT + our planner, LM 1.7B** | **0.56** | 0.35 | 0.70 | 0.63 | 3 / 12 | ~22 s |
| same + LM metadata in the DiT prompt | 0.65 | 0.40 | 0.83 | 0.71 | 3 / 12 | ~22 s |
| fp32 DiT + our planner + metadata | 0.68 | 0.51 | 0.83 | 0.70 | 0 / 12 | ~36 s, peak 15.4 GB |
| mlx-audio as is (LM 0.6B, code cap) | 0.94 | 1.0 | 1.0 | 0.82 | 0 / 12 | ~11 s |

- **The planner is what matters.** With it, the MLX port sings as
  intelligibly as the official pipeline, 3× faster; without it, almost
  nothing is intelligible.
- **The 4-bit DiT is not the bottleneck for vocals.** The full-precision DiT
  scores no better, and is slower and heavier.
- The LM metadata in the DiT prompt doesn't help. The difference is within
  noise; the official pipeline itself isn't deterministic: pop seed 1 got
  0.09 in one run and 0.41 in another.
- WER only measures intelligibility, not timbre or mix quality. The spread
  per seed is large (0.1–1.0 within one song), so the practical advice is
  the same as upstream's: generate a couple of seeds.

Default for LLMTray, from this: mlx-community 4-bit DiT + this planner with
LM 1.7B, no DiT metadata, chunked VAE decode.
