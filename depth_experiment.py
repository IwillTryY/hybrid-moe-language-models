"""Measure what adding hidden layers changes: params, accuracy, training time."""

import time
from statistics import mean, stdev

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from matplotlib.ticker import FuncFormatter
from torch.utils.data import DataLoader, TensorDataset

from mlp import MLP, evaluate, make_spiral, train

DEPTHS = [1, 2, 3, 4, 5]
SEEDS = range(5)
WIDTH = 64
EPOCHS = 100

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
HAIRLINE = "#e1e0d9"
BASELINE = "#c3c2b7"
BLUE = "#2a78d6"
ORANGE = "#eb6834"


def run_experiment():
    results = []
    for depth in DEPTHS:
        accs, times = [], []
        params = 0
        for seed in SEEDS:
            torch.manual_seed(seed)
            x, y = make_spiral(num_classes=3, seed=seed)
            perm = torch.randperm(x.size(0), generator=torch.Generator().manual_seed(seed))
            x, y = x[perm], y[perm]
            split = int(0.8 * x.size(0))
            train_ds = TensorDataset(x[:split], y[:split])
            val_ds = TensorDataset(x[split:], y[split:])

            model = MLP(
                in_dim=2,
                hidden_dims=tuple([WIDTH] * depth),
                num_classes=3,
                activation="gelu",
                dropout=0.0,
            )
            params = sum(p.numel() for p in model.parameters())

            start = time.perf_counter()
            train(
                model, train_ds, val_ds,
                epochs=EPOCHS, batch_size=64, lr=3e-3, device="cpu", verbose=False,
            )
            times.append(time.perf_counter() - start)

            _, acc = evaluate(model, DataLoader(val_ds), nn.CrossEntropyLoss(), "cpu")
            accs.append(acc)

        results.append(
            {
                "depth": depth,
                "params": params,
                "acc_mean": mean(accs),
                "acc_std": stdev(accs),
                "time_mean": mean(times),
            }
        )
    return results


def draw_chart(results):
    plt.rcParams.update(
        {
            "font.family": ["Segoe UI", "DejaVu Sans", "sans-serif"],
            "font.size": 9.5,
        }
    )
    fig, (ax_acc, ax_params) = plt.subplots(
        1, 2, figsize=(11.5, 4.6), dpi=200, gridspec_kw={"width_ratios": [1.15, 1]}
    )
    fig.patch.set_facecolor(SURFACE)

    depths = [r["depth"] for r in results]
    accs = [r["acc_mean"] for r in results]
    stds = [r["acc_std"] for r in results]
    params = [r["params"] for r in results]
    current = 3  # depth used in mlp.py's demo

    for ax in (ax_acc, ax_params):
        ax.set_facecolor(SURFACE)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(BASELINE)
            ax.spines[side].set_linewidth(0.8)
        ax.tick_params(colors=MUTED, labelsize=9, length=3, width=0.8)
        ax.grid(axis="y", color=HAIRLINE, linewidth=0.7)
        ax.set_axisbelow(True)
        ax.set_xticks(depths)
        ax.set_xlim(0.6, max(depths) + 0.4)
        ax.set_xlabel("hidden layers (64 units each)", color=INK_2, fontsize=9.5)

    ax_acc.errorbar(
        depths, accs, yerr=stds, color=BLUE, linewidth=2,
        marker="o", markersize=7, markeredgecolor=SURFACE, markeredgewidth=1.5,
        capsize=0, zorder=3,
    )
    low = min(a - s for a, s in zip(accs, stds)) - 0.015
    ax_acc.set_ylim(max(0.0, min(low, 0.98)), 1.003)
    ax_acc.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax_acc.set_ylabel("validation accuracy (5 seeds)", color=INK_2, fontsize=9.5)
    ax_acc.set_title("Accuracy plateaus after two layers", loc="left", color=INK, fontsize=12, fontweight="600", pad=10)

    ax_params.plot(
        depths, params, color=BLUE, linewidth=2, zorder=2,
        marker="o", markersize=7, markeredgecolor=SURFACE, markeredgewidth=1.5,
    )
    for depth, count in zip(depths, params):
        ax_params.annotate(
            f"{count:,}", xy=(depth, count), xytext=(0, 9),
            textcoords="offset points", ha="center", fontsize=8.5, color=INK_2,
        )
    ax_params.set_ylim(0, max(params) * 1.18)
    ax_params.set_yticks([0, 5000, 10000, 15000, 20000])
    ax_params.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v / 1000:.0f}k"))
    ax_params.set_ylabel("trainable parameters", color=INK_2, fontsize=9.5)
    ax_params.set_title("Each layer adds 4,160 parameters", loc="left", color=INK, fontsize=12, fontweight="600", pad=10)

    for ax, y in ((ax_acc, accs), (ax_params, params)):
        ax.plot(
            [current], [y[depths.index(current)]],
            marker="o", markersize=10, markerfacecolor="none",
            markeredgecolor=ORANGE, markeredgewidth=2, zorder=5,
        )
    ax_acc.annotate(
        "current demo", xy=(current, accs[depths.index(current)]),
        xytext=(10, -18), textcoords="offset points",
        fontsize=8.5, color=ORANGE,
    )

    fig.text(
        0.06, 0.95, "One extra layer unlocks the task; more layers only add weights",
        ha="left", va="top", fontsize=16, color=INK, fontweight="700",
    )
    fig.text(
        0.06, 0.885,
        f"3-class spiral, width {WIDTH}, {EPOCHS} epochs, AdamW, {len(list(SEEDS))} seeds. "
        "Depth 1 underfits; depth 2 and beyond tie the accuracy ceiling.",
        ha="left", va="top", fontsize=9.5, color=INK_2,
    )
    fig.text(
        0.06, 0.03,
        "Each added 64-wide layer adds 64 × 64 weights + 64 biases = 4,160 params "
        "and one more GELU nonlinearity. Accuracy error bars: ±1 std over seeds.",
        ha="left", va="bottom", fontsize=8, color=MUTED,
    )

    fig.subplots_adjust(left=0.08, right=0.97, top=0.75, bottom=0.17, wspace=0.32)
    fig.savefig("depth_effect.png", dpi=200, facecolor=SURFACE)
    fig.savefig("depth_effect.svg", facecolor=SURFACE)
    plt.close(fig)
    print("wrote depth_effect.png")


def main():
    results = run_experiment()
    print(f"{'depth':>5} {'params':>8} {'val acc':>10} {'std':>7} {'sec':>6}")
    for r in results:
        print(
            f"{r['depth']:>5} {r['params']:>8,} {r['acc_mean']:>9.2%} "
            f"{r['acc_std']:>7.4f} {r['time_mean']:>6.2f}"
        )
    draw_chart(results)


if __name__ == "__main__":
    main()
