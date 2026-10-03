"""Figures for the IPSupport quantization whitepaper.

Every number here is copied from the projects' docs/FINDINGS.md (cited per
figure); nothing is recomputed or estimated. Palette: the validated default
categorical order (blue, orange, aqua, yellow), at most three series on any
scatter/line chart; static print, so legends + direct labels, no hover layer.
"""
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import font_manager

HERE = Path(__file__).parent
OUT = HERE / "figures"
OUT.mkdir(exist_ok=True)

for f in sorted((HERE / "fonts").glob("Inter-*.otf")):
    font_manager.fontManager.addfont(str(f))

BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#ffffff"
GRAY = "#b9b8b2"

mpl.rcParams.update({
    "font.family": "Inter",
    "font.size": 8.5,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "axes.linewidth": 0.8,
    "xtick.color": INK2, "ytick.color": INK2,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "legend.frameon": False, "savefig.bbox": "tight", "savefig.dpi": 300,
})


def save(fig, name):
    fig.savefig(OUT / f"{name}.pdf")
    plt.close(fig)


# --- Fig 1: Gemma 4 size, ours vs mlx-community qat-4bit -------------------
# gemma4-quant/docs/FINDINGS.md: E2B/E4B 128x512 raw windows vs the QAT
# master; 12B 128 chat windows; 31B sizes only (mlx-community's 31B KL was
# not measured, so it carries no label).
models = ["Gemma 4 E2B", "Gemma 4 E4B", "Gemma 4 12B", "Gemma 4 31B"]
ours = [4.04, 5.95, 7.89, 20.65]
mlxc = [4.33, 6.80, 10.99, 28.8]
kl_ours = ["KL 0.030", "KL 0.026", "KL 0.025", "KL 0.021"]
kl_mlxc = ["KL 0.067", "KL 0.055", "KL 0.026", "KL not measured"]
fig, ax = plt.subplots(figsize=(6.4, 2.7))
y = range(len(models))
h = 0.36
ax.barh([i + h / 2 for i in y], mlxc, height=h, color=ORANGE, label="mlx-community qat-4bit")
ax.barh([i - h / 2 for i in y], ours, height=h, color=BLUE, label="IPSupport (q4_0 grid + scanned 8-bit)")
for i in y:
    ax.text(ours[i] + 0.25, i - h / 2, f"{ours[i]:.2f} GB · {kl_ours[i]}", va="center", fontsize=7.5, color=INK)
    ax.text(mlxc[i] + 0.25, i + h / 2, f"{mlxc[i]:.2f} GB · {kl_mlxc[i]}", va="center", fontsize=7.5, color=INK2)
ax.set_yticks(list(y), models)
ax.invert_yaxis()
ax.set_xlabel("weights on disk (GB) — lower is better")
ax.set_xlim(0, 36)
ax.grid(axis="y", visible=False)
ax.legend(loc="lower center", bbox_to_anchor=(0.45, 1.0), ncol=2, fontsize=7.5)
save(fig, "fig_gemma_size")

# --- Fig 2: E2B sensitivity budget ------------------------------------------
# FINDINGS "QAT + a few Linears at 8-bit": all-q4_0 = 3.94 GB; mlx-community
# qat-4bit = 4.33 GB, i.e. +390 MB over it, KL 0.067 on the same 128 windows.
budget = [0, 50, 100, 200, 400, 800]
kl = [0.054, 0.034, 0.030, 0.025, 0.018, 0.012]
fig, ax = plt.subplots(figsize=(6.4, 2.6))
ax.plot(budget, kl, color=BLUE, lw=2, marker="o", ms=5, label="IPSupport: greedy ΔKL/MB scan")
ax.scatter([390], [0.067], s=46, color=ORANGE, zorder=3, edgecolor=SURFACE, linewidth=1.5,
           label="mlx-community qat-4bit (all MLP at 8-bit)")
ax.annotate("+50 MB already beats it\n(KL 0.034 vs 0.067, top-1 91.3% vs 87.6%)",
            xy=(50, 0.034), xytext=(150, 0.058), fontsize=7.5, color=INK,
            arrowprops=dict(arrowstyle="-", color=INK2, lw=0.8))
ax.set_xlabel("MB added over the all-q4_0 build (3.94 GB)")
ax.set_ylabel("KL to the QAT master")
ax.set_ylim(0, 0.075)
ax.legend(loc="center right", bbox_to_anchor=(1.0, 0.62), fontsize=7.5)
save(fig, "fig_e2b_budget")

# --- Fig 3: VoiceChat 11B latency per 80 ms frame ----------------------------
# voicechat-quant/docs/FINDINGS.md: base = mlx-community 4-bit (185.2 ms);
# final = GPTQ-3 + fork 2dce524 on an idle Mac (87.7 ms). "other" = the
# measured total minus the four parts.
parts = ["perception", "LLM", "TTS", "codec", "other"]
colors = [BLUE, ORANGE, AQUA, YELLOW, GRAY]
rows = {
    "mlx-community 4-bit": [62.7, 62.3, 47.8, 9.5, 185.2 - (62.7 + 62.3 + 47.8 + 9.5)],
    "IPSupport GPTQ-3 + fork": [16.4, 39.5, 24.4, 5.9, 87.7 - (16.4 + 39.5 + 24.4 + 5.9)],
}
fig, ax = plt.subplots(figsize=(6.4, 1.9))
for r, (name, vals) in enumerate(rows.items()):
    left = 0
    for v, c, p in zip(vals, colors, parts):
        ax.barh(r, v, left=left, color=c, height=0.55, edgecolor=SURFACE, linewidth=1.5,
                label=p if r == 0 else None)
        left += v
    ax.text(left + 2, r, f"{sum(vals):.1f} ms", va="center", fontsize=8, color=INK, fontweight="bold")
ax.axvline(80, color=INK2, lw=1, ls="--")
ax.text(80, -0.62, "80 ms = real time", fontsize=7, color=INK2, ha="center")
ax.set_yticks([0, 1], list(rows))
ax.invert_yaxis()
ax.set_xlim(0, 200)
ax.set_xlabel("ms per 80 ms duplex frame, base M5 (lower is better)")
ax.grid(axis="y", visible=False)
ax.legend(ncol=5, loc="upper center", bbox_to_anchor=(0.5, -0.42), fontsize=7.5)
save(fig, "fig_voicechat_latency")

# --- Fig 4: Nemotron 30B-A3B, PPL vs bits/weight -----------------------------
# nemotron-extreme-quant/docs/FINDINGS.md §1.4 (wikitext-2, 20x512).
fig, ax = plt.subplots(figsize=(6.4, 2.7))
ours_pts = [(3.5, 6.24, "uniform 3-bit GPTQ"), (4.215, 5.81, "positional mix"),
            (4.338, 5.90, "sensitivity mix"), (4.237, 5.24, "JANG map + our GPTQ")]
ax.scatter([p[0] for p in ours_pts], [p[1] for p in ours_pts], s=46, color=BLUE, zorder=3,
           edgecolor=SURFACE, linewidth=1.5, label="IPSupport (GPTQ, MLX-grid exact)")
ax.scatter([3.73], [5.43], s=46, color=ORANGE, zorder=3, edgecolor=SURFACE, linewidth=1.5,
           label="JANG_2L-CRACK (third party, claimed 3.73 bpw)")
ax.scatter([3.5], [6.54], s=46, color=AQUA, zorder=3, edgecolor=SURFACE, linewidth=1.5,
           label="naive RTN 3-bit")
ax.axhline(5.11, color=INK2, lw=1, ls="--")
ax.text(3.37, 5.13, "bf16 5.11", fontsize=7, color=INK2, va="bottom", ha="left")
offs = {"uniform 3-bit GPTQ": (0.02, 0.0, "left"), "positional mix": (-0.02, 0.0, "right"),
        "sensitivity mix": (0.0, 0.07, "center"), "JANG map + our GPTQ": (-0.02, 0.0, "right")}
for x, yv, lab in ours_pts:
    dx, dy, ha = offs[lab]
    ax.text(x + dx, yv + dy, lab, fontsize=7.2, color=INK, ha=ha, va="center")
ax.set_xlabel("average bits per weight")
ax.set_ylabel("wikitext-2 PPL (lower is better)")
ax.set_xlim(3.35, 4.5)
ax.set_ylim(5.0, 6.75)
ax.legend(loc="upper right", fontsize=7.2)
save(fig, "fig_nemotron_ppl")

print("figures:", sorted(p.name for p in OUT.glob("*.pdf")))
