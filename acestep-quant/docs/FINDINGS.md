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

Planner fixes after review (in `acestep_planner.py` and LLMTray's runner):
- only codes 0–63999 are allowed. The tokenizer also has
  `<|audio_code_64000…65534|>`, which the FSQ can't represent; mlx-audio
  would clamp them to 63999;
- a malformed CoT YAML (an unquoted colon in the caption) falls back to
  parsing the known `key: value` lines, instead of dropping all metadata;
- empty lyrics are sent empty, as the official pipeline does, not as
  `[Instrumental]`.

The eval above ran before these fixes. They change only edge cases
(sampling one of the 1,535 out-of-range codes, a malformed CoT, the
instrumental prompt).

## In LLMTray

`generate_music` (LLMTray PR #79) runs this recipe:
- `runtime/llmtray_music_runner.py`, with the planner inlined;
- mlx-audio pinned to `1e8264a` and installed from a GitHub tarball;
- mlx-community 4-bit DiT + the official LM 1.7B folder, ~9 GB;
- 30 s of music in ~23 s through the app's Swift path.

## ACE-Step variants on the vocal test set (2026-09-26)

Same 3 songs × 4 seeds, 30 s, Whisper WER (capped at 1 per track). Times
on the M5 unless noted.

| config | mean WER | pop | rock | ballad | < 0.3 | s / track |
|---|---|---|---|---|---|---|
| turbo 4-bit, LM 1.7B, 8 steps (LLMTray today) | 0.56 | 0.35 | 0.70 | 0.63 | 3 | 22 |
| turbo 4-bit, LM 1.7B, 20 steps | 0.55 | 0.33 | 0.69 | 0.63 | 2 | 33 |
| turbo 4-bit, LM 4B, 8 steps | 0.51 | 0.14 | 0.80 | 0.59 | 5 | 48 |
| turbo 4-bit, LM 4B, 20 steps | 0.45 | 0.14 | 0.65 | 0.55 | 5 | 56 |
| official pipeline, turbo | 0.66 | 0.63 | 0.58 | 0.77 | 2 | 83 |
| official pipeline, **sft** (50 steps, CFG 7) | **0.17** | 0.11 | 0.21 | 0.18 | 9 | 114 |
| **MLX sft** (bf16, ours, no LM, 50 steps, CFG 7) | **0.22** | 0.02 | 0.22 | 0.43 | 6 | **36** (peak 10.6 GB) |

Listening (the user): the turbo tracks sound fuller and more like a
finished song. sft puts the voice upfront with less music. The two are
different trade-offs, not a strict ranking, and WER only measures the
vocals.

### Making sft work in mlx-audio

- mlx-audio's DiT architecture is the same for sft and turbo. The official
  `modeling_acestep_v15_base.py` and `_turbo.py` differ only in sampling:
  linspace schedule with shift vs turbo's fixed 8-step table, ODE to t=0,
  and CFG with APG and a momentum buffer.
- `poc/acestep_convert_mlx.py` converts any official DiT. mlx-audio's
  `convert.py` is hard-wired to turbo. The DiT is stored bf16.
- **The sft DiT makes cacophony (no music, no words) when fed mlx-audio's
  5 Hz LM hints.** Without the LM (`use_lm=False`), with CFG 7, APG,
  shift 1, CFG on every step, it sings: pop WER 0.07 on the first probe.
  sft doesn't need the planner, so there are ~15 s and ~4 GB less.
- mlx-audio's unconditional CFG branch runs the encoder on zeros.
  Officially it is the trained `null_condition_emb`
  (`poc/acestep_null_cond.py` patches it). It matters a little: mean WER
  0.22 with `null_condition_emb` vs 0.28 with zeros (rock 0.22 vs 0.32,
  ballad 0.43 vs 0.49, pop 0.02 both). LLMTray uses it.
- Red herring for the record: the official `silence_latent.pt` is
  [1, 64, T], and mlx-audio's loader transposes it itself. The
  mlx-community turbo repo's copy is missing from the snapshot, and mlx-audio
  then uses zeros, which turbo tolerates.

### sft quantization: 8-bit and GPTQ 4-bit (2026-09-26)

`poc/acestep_sft_gptq.py` quantizes the DiT's decoder and condition
encoders (the Linears mlx-community's turbo 4-bit covers; the encoders'
embeddings round-to-nearest):
- GPTQ calibrates on the real sft sampling loop: 50 steps, both CFG
  branches, 8 calibration songs outside the eval set;
- Linears reading the same input share a Hessian (q/k/v, cross-attn k/v,
  gate/up);
- `--rtn-only` gives plain round-to-nearest.

The output loads like mlx-community's turbo 4-bit (config `quantization`,
the loader quantizes what has `.scales`). Two save bugs, both caught by
listening (tracks with no music and no words):
- `copytree` dropped the text encoder's `model.safetensors`;
- the null-cond wrapper renamed the encoder to `encoder.inner.*`.

`save_quantized` now remaps the names and asserts every DiT key is present.
Under Python 3.14, `gc.collect()` mid-GPTQ segfaulted (visit_decref on mlx
objects). It is gone, and `--hessians` keeps the calibration across a
crash.

Teacher-forced error against bf16 (`poc/acestep_tf_metric.py`, the
klein/MiniMax method): the bf16 model samples 3 songs, every decoder and
encoder call is recorded, and each variant gets exactly those inputs, so
there is no trajectory drift or seed luck
(`docs/sft_tf_results_2026-09-26.json`).

| DiT | velocity error | condition error | DiT size |
|---|---|---|---|
| 8-bit RTN | 2.1% | 0.46% | 2.7 GB |
| 4-bit RTN | 13.9% | 5.0% | 1.65 GB |
| **4-bit GPTQ** | **10.4%** (−25%) | **2.65%** (−47%) | 1.65 GB |

Lyrics (the vocal eval above: 3 songs × 4 seeds, Whisper WER; 12 tracks,
so a few points is noise), M5:

| DiT | mean WER | pop | rock | ballad | < 0.3 | s / track | peak |
|---|---|---|---|---|---|---|---|
| bf16 | 0.22 | 0.02 | 0.22 | 0.43 | 6 | 36 | 10.6 GB |
| 8-bit RTN | 0.24 | 0.02 | 0.26 | 0.45 | 6 | 49 | 8.6 GB |
| 4-bit GPTQ | 0.29 | 0.06 | 0.35 | 0.45 | 5 | 36 | 7.5 GB |

8-bit is slower than bf16 here (the 8-bit quantized matmul), 4-bit the
same speed. The 8-bit teacher-forced error is 2%: practically bf16.

Published as `roman220220/ACE-Step1.5-sft-MLX-{bf16,8bit,gptq-4bit}`. In
LLMTray, the user picks among them.

## MiniMax Music 3 (MiniMaxAI/MiniMax-Music3), on a rented A100 (2026-09-26)

Community license: commercial use is allowed. The UI must show
"MiniMax-Music3"; above US$20M/yr revenue a separate authorization is
needed; safeguards against rights violations are required.

Components the diffusers pipeline actually loads, ~28.5 GB bf16 (the repo is
57 GB):
- global LM, Qwen3-8B: 17.2 GB;
- flow-matching DiT, 2.4B: 9.7 GB;
- RVQ depth decoder, 0.6B: 1.3 GB;
- vocoder and condition encoder: 0.3 GB.

`qwen_7B/` (18.5 GB, MiniMax "abab", Mixtral-style) and the `.pth` files are
not used by the diffusers pipeline.

**The LM's last hidden state of every frame is part of what the synthesis
stage decodes** (`frame_hiddens` = LM hidden ⊕ depth hiddens,
`encoders.py`). So LM quantization error reaches the audio directly, not
only through token choices.

Reference, bf16 on the A100 (diffusers 0.40.0, 30 s):
- 54 s a track, 24.5 GB peak;
- vocal WER: 0.76 with our one-line captions, 0.63 with detailed
  structured captions (the format MiniMax recommends);
- listening: music, clearly AI, decent. It adds lyric-driven sound design,
  e.g. rain in the song whose lyrics mention rain.

Quantization, LM only, other parts bf16. WER on 12 sampled tracks is **not
usable** for this: RTN-4 scored 0.32 against bf16's 0.63 (seed luck, since
AR sampling diverges at the first different token). Instead,
`poc/minimax_tf_metric.py` does teacher forcing:
- bf16 generates 3 songs and every LM input is recorded;
- the whole sequence goes through bf16 and each variant;
- the metrics are taken at the audio frames.

| LM 8B | hidden-state rel. error | top-1 agreement | KL |
|---|---|---|---|
| RTN 8-bit g64 | 1.5% | 97.2% | 0.0018 |
| RTN 4-bit g64 | 15.0% | 85.5% | 0.056 |
| **GPTQ 4-bit g64** (`poc/minimax_gptq_lm.py`) | **9.9% (−34%)** | **90.3%** | **0.027 (−52%)** |

GPTQ calibration:
- 8 songs outside the eval set, including Russian and Spanish;
- Hessians summed over the AR generation, q/k/v and gate/up sharing one;
- 6 min of GPTQ on the A100.

The packed 4-bit LM (3.9 GB) is kept for an MLX port. Estimated size of a
Mac recipe (LM GPTQ-4, DiT and depth 8-bit): ~8 GB. Estimated speed on an
M5: 1–2 min per 30 s, which is not measured.

Head to head on one electronic track (dubstep synth-pop, 60 s, listening):
none of the three really does dubstep. ACE-Step **sft** sounded best on
electronic music, turbo was "a little electronic", and MiniMax was fine but
no better. **Decision: MiniMax parked; ACE-Step sft goes into LLMTray.**

Pod pitfalls:
- On a community A100 node (load average ~205), long processes were
  SIGKILLed from outside the container: downloads and generation, with no
  cgroup OOM and no killer inside. A secure-cloud pod ran clean.
- `hf download` of many large files in parallel silently failed there.
  Per-file `curl -C -` in a resume loop worked.
- The pipeline's `modular_model_index.json` points every component at the
  hub repo: `load_components` re-downloads even from a local directory.
  Point it at the local path and set `HF_HUB_OFFLINE=1`.
- A secure host failed to start the container at all
  (`/dev/dri/card7` missing). Its logs showed it; deleting and re-creating
  the pod fixed it.
