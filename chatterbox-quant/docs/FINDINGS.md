# Chatterbox Multilingual V3 on MLX: findings

For LLMTray's spoken answers (llmtray adr/0016). Measured 2026-11-06 on the
base M5 Mac (26 GB). The Mac was under memory pressure during the runs
(10 GB swap in use from iOS simulators), so single timings vary up to 2x.
The ratios between variants held across repeats.

## What runs

- Model: `mlx-community/chatterbox-multilingual-v3`, converted from
  `ResembleAI/chatterbox` (`t3_mtl23ls_v3`, MIT) with mlx-audio 0.4.7. Every
  weight is fp32: 693M parameters, T3 540M, S3Gen 152M, 2.71 GB.
- Runtime: the ipsupport-llc/mlx-audio fork, commit 67f89de (chatterbox
  multilingual v2/v3, `lang_code`).
- The speech tokenizer `mlx-community/S3TokenizerV2` (Apache-2.0, from
  CosyVoice2-0.5B) is fetched by name at load. LLMTray points it at a local
  folder.
- The multilingual repo has no `conds.safetensors`, so there is no built-in
  voice. `mlx-community/chatterbox-fp16`'s file (Resemble's `conds.pt`,
  which their multilingual model loads too) works as the built-in voice.

## Speed (one 3.5–4 s English sentence, after a warm-up)

| variant | T3 tokens/s | T3 time | S3Gen time | total | peak MLX memory |
|---|---|---|---|---|---|
| as converted, fp32 | 14–26 | 3.9–7.1 s | 3.3–4.0 s | 8.0–10.9 s | 4.1 GB |
| **T3 8-bit (g64), rest fp32** | **38–51** | **1.8–2.4 s** | 3.2–3.6 s | **5.0–6.0 s** | **3.5 GB** |
| T3 4-bit (g64), rest fp32 | 36–49 | 1.9–2.5 s | 3.2–3.5 s | 5.1–6.1 s | — |
| T3 and S3Gen bf16 | 9 | slower | — | — | 4.0 GB |

- **T3 at 8 bits is the win**: 2–3x faster tokens. With speech tokens at
  25/s, T3 is then 1.5–2x faster than real time.
- **4 bits gives nothing more** than 8, so 8 bits it is.
- **bf16 is slower here.** T3 in bf16 ran at 9 tok/s. S3Gen in bf16, on
  identical tokens, was slower than fp32 and audibly different (SNR 11 dB
  against fp32, spectral correlation 0.995). S3Gen stays fp32.
  - That comparison ran S3Gen alone, on one sentence's captured tokens (4 s
    of speech): fp32 1.15 s, bf16 3.05 s.
  - Inside `generate()`, right after T3, the same fp32 S3Gen took 3.2–4.0 s
    in the table above. That gap isn't explained yet. Suspects: memory
    pressure (swap), and `mx.clear_cache()` between T3 and S3Gen, which
    makes it allocate again.
- **S3Gen is now the bottleneck**: about 1 s of compute per 1 s of speech
  (flow matching, 10 steps with CFG, then HiFT). A sentence takes a little
  longer to make than to play, so read-as-it-streams has gaps. Next to try:
  fewer flow steps, and Chatterbox Turbo's distilled `s3gen_meanflow`
  (2 steps, no CFG; it takes the same S3 v2 tokens).
- **Measure honestly**: MLX is lazy. A timer around `generate()` that
  doesn't force the audio misses S3Gen. One early run showed RTF 0.59 that
  way; it was wrong.

## Output check

The 8-bit folder (built by `scripts/quantize_t3.py`) loads with mlx-audio's
own loader (`load_model`, `config.json` "quantization"), and its audio
matches quantizing T3 on the fly: same length, every sample within 1e-4,
for an English and a Russian sentence with the same seeds.

- **Size**: `model.safetensors` is 1.31 GB; the repo is 1.32 GB with the
  config, tokenizers, built-in voice and LICENSE.
- **Built-in voice**: every run here used it (no clip). Cloning from a clip
  wasn't tested.
- **Through LLMTray's runner** (`llmtray_tts_runner.py`, five sentences ru
  and en): RTF 1.36–1.56 against 2–3 for fp32, peak footprint 4.5 GB
  against 6.1 GB.

By ear: fp32 vs T3 8-bit samples were left for the user's comparison.

## Russian stress

- Chatterbox reads Russian from stressed text (U+0301 after the stressed
  vowel).
- LLMTray stresses it itself with ruaccent in tiny mode (MIT) and a hashed
  accents dictionary (llmtray `runtime/llmtray_ru_stress.py`); load 0.9 s,
  0.1–11 ms a sentence.
- mlx-audio's own stresser, `russian_text_stresser`, is AGPL-3.0 and is
  never installed.
- Checked: замо́к (lock) and плачу́ (pay) by context, Интегра́ция, атла́сную.
  ё gets no mark (it is always stressed): "идё́т" was a bug, fixed.

## The decoder: Turbo's meanflow S3Gen (2026-11-07)

S3Gen turned out to be the slow half, so three decoders were tried on the
same speech tokens:

Times are the second call on the same tokens (after the warm-up described
below), one Russian phrase:

| decoder | S3Gen time, 3.7 s phrase |
|---|---|
| multilingual S3Gen, 10 steps + CFG (as shipped) | 2.1 s |
| same, 4 steps | 0.9 s |
| **Chatterbox Turbo's meanflow S3Gen, 2 steps, no CFG** | **0.5 s** |

- **Tokens fit as they are.** Turbo was trained English-only, but its
  S3Gen takes the same S3 v2 tokens (25 Hz, 0..6560) and conditions on the
  same reference dict. Its output follows the 10-step one on one Russian
  and one English phrase (same tokens): correlation of the energy
  envelopes 0.975 and 0.987, of the log spectrograms 0.937 and 0.935.
- **By ear: no difference.** Blind A/B/C test on 2 Russian and 2 English
  phrases, shuffled order, judged by the user: "разницы не слышу" (I don't
  hear a difference). The fastest decoder wins.
- **How it ships.** `scripts/extract_meanflow.py` takes Turbo's S3Gen
  without its speech tokenizer: 570 MB fp32. It is checked to be equal,
  parameter by parameter, to the full Turbo model's S3Gen. It sits at
  `meanflow/s3gen_meanflow.safetensors`: in a subfolder, because mlx-audio
  loads every top-level `*.safetensors` of a model folder, and a second
  top-level file breaks `load_model` with "Unrecognized weight keys". With
  the subfolder, `load_model` on the folder works as before (every runner
  test here loaded it that way). The script reads the file back and
  checks every tensor against the source; it also matched the full Turbo
  model's S3Gen, loaded by mlx-audio, parameter for parameter.
- **The first-call timing is a warm-up.** S3Gen's first call after load
  took 9 s, the second 2 s on the same length; a new length costs +0.5–1 s
  once. LLMTray's runner speaks a short phrase before it reports ready.
- **The earlier T3 numbers were low.** The 38–51 tok/s above was measured
  under swap. On an unloaded Mac, 8-bit T3 runs at 120 tok/s. A sync-free
  loop (sampling, the repetition penalty and the next embedding on the
  GPU; the stop token looked for every 16 tokens) gives 134 tok/s with the
  same tokens.

**End to end through LLMTray's runner** (five sentences, ru and en, base M5):

| setup | RTF | peak footprint |
|---|---|---|
| fp32 | 2–3 | 6.1 GB |
| T3 8-bit | 1.4–1.6 | 4.5 GB |
| **T3 8-bit + sync-free T3 + meanflow decoder + warm-up** | **0.24–0.31** | 4.6 GB |

The runner is ready 2.5 s after launch, and the first sentence plays about
3 s in.
