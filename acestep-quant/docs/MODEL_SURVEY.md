# Open-weight song generators for a local Mac app (checked 2026-09-26)

Every figure comes from pages fetched on that date: the HF API
(`?blobs=true`; disk size = sum of file sizes), model cards, GitHub
README/LICENSE files and arXiv. Quality and speed numbers are what each team
reports about its own model; none were reproduced here, except where
`FINDINGS.md` measures them. **UNVERIFIED** marks what couldn't be checked.

| model | released | weights license → commercial? | vocals / max length / rate | size on disk | Mac path |
|---|---|---|---|---|---|
| **ACE-Step 1.5** (`ACE-Step/Ace-Step1.5`, `acestep-v15-{sft,base,xl-*}`) | 2026-01; XL 2026-04 | **MIT → yes**; the card says outputs can be used commercially | yes, 50+ languages, 10 s – 10 min, 48 kHz | bundle 10.1 GB; XL DiT repos 20 GB each | official MLX backends; mlx-audio (`pc/add-ace`); acestep.cpp (Metal) |
| **MiniMax Music 3** (`MiniMaxAI/MiniMax-Music3`) | 2026-08 | community license: commercial OK with attribution; > $20M/yr needs authorization | yes, up to 5 min, 44.1 kHz | 57 GB repo, ~28.5 GB used | diffusers supports `mps`; no MLX port. See FINDINGS |
| YuE2-3B (`m-a-p/YuE2-3B` + `YuE2-Vae`) | 2026-09-09 | CC BY-NC 4.0 + waiver for individual creators → **companies need a license** | yes, zh/en, 5 min demos, 48 kHz | 7.3 + 0.5 GB | community `vanch007/mlx-Yue2-3B`; audio.cpp GGUF |
| HeartMuLa-oss-3B (+ HeartCodec) | 2026-01/02 | **Apache-2.0 → yes** (MuLaCover is CC BY-NC) | yes, zh/en/ja/ko/es, 240 s, 48 kHz | 15.8 + 6.6 GB (fp32) | no official MPS; audio.cpp; Q8 GGUF 7.7 GB |
| SongGeneration 2 / LeVo 2 | 2026-02/03 | **academic only** | yes, 4.5 min, 48 kHz | 7.3–12.9 GB + 15 GB runtime | community MLX; official repos withdrawn |
| DiffRhythm 2 (`ASLP-lab/DiffRhythm2`) | 2025-10 | Apache-2.0 → yes | yes (needs reference audio / MuQ prompt), 48 kHz | 5.1 GB | UNVERIFIED |
| SongBloom | 2025-06 | UNVERIFIED (official repos gone) | yes, needs a 10 s audio prompt, 150 s | 8.5 GB (mirror) | none found |
| Muse (`bolshyC/Muse-*`) | 2026-01 | Apache-2.0, but trained on Suno-V5 output | yes | 1.3–16.7 GB | none |
| JAM-0.5 | 2025-07 | non-commercial | yes, 3:50, 44.1 kHz | 2.2 GB | none |
| Stable Audio 3 small/medium | 2026-05 | Stability Community (< $1M revenue free) | **instrumental only** | 3.5 / 10.5 GB | CoreML official; MLX port |
| MusicGen | 2023 | CC-BY-NC | no realistic vocals | up to 20 GB | MLX ports |
| Magenta RealTime 2 | 2026-05 | CC-BY-4.0 | streaming instrumental | 15.6 GB | ships `.mlxfn` |

Reported quality, from each team's own tables:
- YuE2 claims WildSongBench 6.73 against Suno v5's 6.87.
- HeartMuLa's paper: musicality 69.55 vs ACE-Step 1.0 67.42.
- ACE-Step 1.5's paper, SongEval coherence: 4.72 vs HeartMuLa 4.68 vs
  DiffRhythm 2 3.99.

For a commercial Mac app, only **ACE-Step** (MIT, licensed data, fastest,
MLX paths) and **HeartMuLa** (Apache-2.0, slower, no MLX port) are clean.
MiniMax Music 3 is also usable, with its attribution clause.

Sources: HF API/model cards and GitHub READMEs/LICENSEs of every repo above;
arXiv 2602.00744 (ACE-Step 1.5), 2601.10547 (HeartMuLa), 2510.22950
(DiffRhythm 2), 2503.01183, 2506.07520 (LeVo), 2506.07634 (SongBloom),
2507.20880 (JAM), 2601.03973 (Muse), 2503.08638 (YuE), 2605.17991
(Stable Audio 3).
