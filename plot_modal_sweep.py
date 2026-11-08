"""Plot the Modal code-language scaling sweep and the adaptive exit curve."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
HAIRLINE = "#e1e0d9"
BASELINE = "#c3c2b7"
COLORS = {"dense8": "#2a78d6", "hybrid_e4": "#eb6834", "hybrid_e8": "#1baf7a"}
LABELS = {
    "dense8": "Dense 8 layers",
    "hybrid_e4": "Hybrid, 4 experts",
    "hybrid_e8": "Hybrid, 8 experts",
}

final = json.loads(Path("results/final_eval.json").read_text(encoding="utf-8"))
modal = {m["name"]: m for m in json.loads(Path("results/modal_results.json").read_text(encoding="utf-8"))}

plt.rcParams.update(
    {"font.family": ["Segoe UI", "DejaVu Sans", "sans-serif"], "font.size": 9.5}
)
fig, (ax_scale, ax_exit) = plt.subplots(
    1, 2, figsize=(12.4, 4.9), dpi=200, gridspec_kw={"width_ratios": [1.15, 1]}
)
fig.patch.set_facecolor(SURFACE)

for ax in (ax_scale, ax_exit):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(BASELINE)
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(colors=MUTED, labelsize=9, length=3)
    ax.grid(axis="y", color=HAIRLINE, linewidth=0.7)
    ax.set_axisbelow(True)

# Panel A: full-depth quality against active compute.
ax_scale.set_title("MoE capacity per active compute", loc="left", color=INK, fontsize=12,
                   fontweight="600", pad=10)
for name in ("dense8", "hybrid_e4", "hybrid_e8"):
    x = modal[name]["params_applied_per_token"] / 1e6
    y = final[name]["val_loss"]
    ax_scale.scatter(
        [x], [y], s=90, color=COLORS[name], edgecolors=SURFACE,
        linewidths=2.0, zorder=3,
    )
    offsets = {"dense8": (8, 8), "hybrid_e4": (8, 8), "hybrid_e8": (8, -18)}
    align = {"dense8": "left", "hybrid_e4": "left", "hybrid_e8": "left"}
    ax_scale.annotate(
        f"{LABELS[name]}\n{modal[name]['params_total'] / 1e6:.2f}M params",
        xy=(x, y), xytext=offsets[name], textcoords="offset points",
        ha=align[name], va="bottom", fontsize=8.5, color=INK_2,
    )
ax_scale.set_xlim(1.35, 1.80)
ax_scale.set_ylim(1.055, 1.205)
ax_scale.set_xlabel("parameter applications per token (millions)", color=INK_2)
ax_scale.set_ylabel("validation loss", color=INK_2)
ax_scale.text(
    0.01, 0.02, "lower is better", transform=ax_scale.transAxes,
    fontsize=8, color=MUTED, ha="left", va="bottom",
)

# Panel B: exit curve for the best model.
rows = final["hybrid_e8"]["exit_curve"]
xs = [r["avg_loops"] for r in rows]
ys = [r["val_loss"] for r in rows]
ax_exit.plot(
    xs, ys, color=COLORS["hybrid_e8"], linewidth=2, marker="o",
    markersize=6, markeredgecolor=SURFACE, markeredgewidth=1.5, zorder=3,
)
for r in rows:
    if r["threshold"] in (0.2, 0.7):
        ax_exit.annotate(
            f"threshold {r['threshold']:.1f}\n{r['early_exit_fraction']:.0%} exit early",
            xy=(r["avg_loops"], r["val_loss"]), xytext=(8, -28),
            textcoords="offset points", fontsize=8, color=INK_2,
        )
ax_exit.set_title("Adaptive exits, 8-expert hybrid", loc="left", color=INK,
                  fontsize=12, fontweight="600", pad=10)
ax_exit.set_xticks([1.0, 1.25, 1.5])
ax_exit.set_xlim(0.95, 1.68)
ax_exit.set_ylim(1.108, 1.126)
ax_exit.set_xlabel("average loops used per sequence", color=INK_2)
ax_exit.set_ylabel("validation loss", color=INK_2)
ax_exit.text(
    0.01, 0.02, "exit threshold rises left to right", transform=ax_exit.transAxes,
    fontsize=8, color=MUTED, ha="left", va="bottom",
)

fig.text(
    0.055, 0.96, "More MoE capacity beats a deeper dense model at fixed active compute",
    ha="left", va="top", fontsize=16, fontweight="700", color=INK,
)
fig.text(
    0.055, 0.90,
    "Python stdlib code, 11.1M training chars, 1,500 steps, one seed. "
    "Hybrid: 4 unique blocks × 2 loops, shared depth, top-1 experts, per-loop adapters, GQA.",
    ha="left", va="top", fontsize=9.5, color=INK_2,
)
fig.text(
    0.055, 0.035,
    "Evaluated on 1,600 validation windows. Exit curve uses the same windows as the full-depth reference. "
    "Memory: dense 1.62M params · e4 2.38M · e8 4.49M. One seed; differences below ~0.01 are noise.",
    ha="left", va="bottom", fontsize=8, color=MUTED,
)

fig.subplots_adjust(left=0.07, right=0.97, top=0.77, bottom=0.17, wspace=0.30)
fig.savefig("results/modal_sweep.png", dpi=200, facecolor=SURFACE)
fig.savefig("results/modal_sweep.svg", facecolor=SURFACE)
print("wrote results/modal_sweep.png")
