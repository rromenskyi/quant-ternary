# Runbook: LFM2 / LFM2.5 -> MLX (GPTQ JANG)

`poc/lfm2_mlx_pipeline.sh`, resumable (finished steps are skipped,
calibration resumes at the next layer). Status lines go to
`$WORK/logs/pipeline.log`, step output to `$WORK/logs/lfm2_<step>.log`.
Nothing is uploaded unless `PUBLISH=1`. Why: [FINDINGS.md](FINDINGS.md).

## 1. Pod

What all five releases used, 2026-10-11: **one L40S (48 GB)**, image
`runpod/pytorch:1.0.3-cu1281-torch291-ubuntu2404`, **200 GB volume**. The
8B MoE calibrates in ~4.5 min (24 layers, ~10 s each).

A 200 GB volume holds the snapshots plus two or three 8B builds, not more:
the 8B's `gptq-<variant>` and `ongrid-<variant>` are ~17 GB each. It filled
during the fourth model; delete those two directories once a variant has
passed its check (the pipeline rebuilds them if needed; note the convert
step then needs them again).

## 2. Run

Copy the repo to the pod (`qwen35-quant/` and `nemotron-extreme-quant/` are
imported, `cards_assets/` is used by the card step), then:

```bash
cd lfm2-quant/poc
VARIANT=lfm2-8b SETUP=1 ./lfm2_mlx_pipeline.sh       # once per pod: deps
for v in lfm2-8b lfm2-8b-q4 lfm2-2.6b lfm2-1.2b-instruct lfm2-1.2b-thinking; do
  VARIANT=$v ./lfm2_mlx_pipeline.sh
done
```

Vision-language: `VARIANT=lfm2-vl-3b` (VISION=1). The environment needs
the fork with LFM2-VL vision (ipsupport-llc/mlx-lm main after #36),
`torchvision` matching the image's torch (0.24.1 for torch 2.9.1) and
`pandas`; on an image whose system Python is PEP 668-locked, use a venv
with `--system-site-packages` and put it first on PATH.

Results: `$WORK/results-<model>.json` (bf16, `baseline` = the vendor's MLX
build if there is one, every variant). Read them before publishing.

## 3. Publish

The card is copied in last (`cards/<repo>.md` -> README.md, banner, the
snapshot's LICENSE). With the gptq/ongrid directories deleted, don't rerun
the pipeline for the upload (it would recalibrate): copy the card the same
way and `hf upload roman220220/<repo> /workspace/<repo> .`, HF_TOKEN passed
over ssh stdin, never written on the pod.

Kaggle: append to `tools/kaggle/releases.json` / `cards.json` and run
`kaggle_publish.py`, `kaggle_cards.py`, `make_public.py` on a spec holding
only the new entries (KAGGLE_API_TOKEN over stdin). Then terminate the pod.
