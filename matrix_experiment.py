"""Compare flat dense layers with two 8x8 matrix-layout layers."""

import time
from statistics import mean, stdev

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from matplotlib.patches import Rectangle
from torch.utils.data import DataLoader, TensorDataset

from matrix_mlp import GRID, LocalMatrixMLP, RowColMatrixMLP
from mlp import MLP, evaluate, make_spiral, train

SEEDS = [0, 1, 2]
EPOCHS = 80
BATCH_SIZE = 128
LR = 3e-3

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
HAIRLINE = "#e1e0d9"

VARIANTS = [
    {
        "name": "Dense flat",
        "subtitle": "every cell connects to every cell",
        "color": "#2a78d6",
        "kind": "dense",
        "build": lambda: MLP(2, (64, 64, 64), 3, dropout=0.0),
    },
    {
        "name": "Local matrix",
        "subtitle": "3×3 neighborhood, one shared kernel",
        "color": "#eb6834",
        "kind": "local",
        "build": lambda: LocalMatrixMLP(blocks=3),
    },
    {
        "name": "Row + column matrix",
        "subtitle": "R · H · Cᵀ, two 8×8 factors per block",
        "color": "#1baf7a",
        "kind": "rowcol",
        "build": lambda: RowColMatrixMLP(blocks=3),
    },
]


def make_data(seed):
    torch.manual_seed(seed)
    x, y = make_spiral(num_classes=3, seed=seed)
    perm = torch.randperm(x.size(0), generator=torch.Generator().manual_seed(seed))
    x, y = x[perm], y[perm]
    split = int(0.8 * x.size(0))
    return TensorDataset(x[:split], y[:split]), TensorDataset(x[split:], y[split:])


def run_variant(variant):
    accs, times = [], []
    params = 0
    for seed in SEEDS:
        train_ds, val_ds = make_data(seed)
        torch.manual_seed(seed)
        model = variant["build"]()
        params = sum(p.numel() for p in model.parameters())

        start = time.perf_counter()
        train(
            model, train_ds, val_ds,
            epochs=EPOCHS, batch_size=BATCH_SIZE, lr=LR, device="cpu", verbose=False,
        )
        times.append(time.perf_counter() - start)

        _, acc = evaluate(model, DataLoader(val_ds), nn.CrossEntropyLoss(), "cpu")
        accs.append(acc)
        print(
            f"  {variant['name']:20s} seed {seed}: {acc:.2%} ({times[-1]:.1f}s)",
            flush=True,
        )

    return {
        **variant,
        "params": params,
        "acc": mean(accs),
        "acc_std": stdev(accs),
        "time": mean(times),
    }


def draw_edges(ax, kind, color):
    cells = [(c, 7 - r) for r in range(GRID) for c in range(GRID)]
    cx, cy = 3, 4  # center cell (column 3, row 3)

    if kind == "dense":
        for x, y in cells:
            if (x, y) != (cx, cy):
                ax.plot([cx, x], [cy, y], color=color, lw=0.55, alpha=0.17, zorder=1)
    elif kind == "local":
        for x, y in cells:
            if max(abs(x - cx), abs(y - cy)) <= 1 and (x, y) != (cx, cy):
                ax.plot([cx, x], [cy, y], color=color, lw=2.0, alpha=0.85, zorder=1)
        ax.add_patch(
            Rectangle(
                (cx - 1.5, cy - 1.5), 3, 3, fill=False, edgecolor=color,
                linewidth=1.0, linestyle=(0, (2, 2)), alpha=0.8, zorder=2,
            )
        )
    else:  # rowcol
        ax.plot([-0.35, 7.35], [cy, cy], color=color, lw=2.4, alpha=0.75, zorder=1)
        ax.plot([cx, cx], [0.65, 7.35], color=color, lw=2.4, alpha=0.75, zorder=1)

    ax.scatter(
        [x for x, _ in cells], [y for _, y in cells],
        s=26, c=INK_2, edgecolors=SURFACE, linewidths=1.0, zorder=3,
    )
    ax.scatter(
        [cx], [cy], s=78, c=color, edgecolors=SURFACE, linewidths=2.0, zorder=4,
    )


def draw_figure(results):
    plt.rcParams.update(
        {"font.family": ["Segoe UI", "DejaVu Sans", "sans-serif"], "font.size": 9.5}
    )
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 5.5), dpi=200)
    fig.patch.set_facecolor(SURFACE)

    for ax, result in zip(axes, results):
        ax.set_facecolor(SURFACE)
        ax.set_xlim(-1.0, 8.0)
        ax.set_ylim(-2.7, 7.9)
        ax.set_aspect("equal")
        ax.axis("off")

        draw_edges(ax, result["kind"], result["color"])

        ax.text(
            0.5, 1.09, result["name"], transform=ax.transAxes,
            ha="center", va="bottom", fontsize=12, fontweight="600", color=INK,
        )
        ax.text(
            0.5, 1.025, result["subtitle"], transform=ax.transAxes,
            ha="center", va="bottom", fontsize=8.5, color=MUTED,
        )
        ax.text(
            3.5, -0.9, f"{result['acc']:.1%}", transform=ax.transData,
            ha="center", va="top", fontsize=17, fontweight="700", color=INK,
        )
        ax.text(
            3.5, -1.65, f"val accuracy  ·  {result['params']:,} params",
            transform=ax.transData, ha="center", va="top",
            fontsize=8.5, color=INK_2,
        )

    fig.text(
        0.045, 0.965, "Matrix of neurons: same 64 units, different edges",
        ha="left", va="top", fontsize=16, fontweight="700", color=INK,
    )
    fig.text(
        0.045, 0.90,
        "Reshaping a 64-vector to 8×8 changes nothing by itself. "
        "Changing which cells connect, and which weights they share, changes the layer.",
        ha="left", va="top", fontsize=9.5, color=INK_2,
    )
    fig.text(
        0.045, 0.035,
        f"8×8 hidden grid, 3-class spiral, {len(SEEDS)} seeds, {EPOCHS} epochs, "
        f"batch {BATCH_SIZE}. Validation accuracy: mean of seeds "
        f"(spread ≤ {max(r['acc_std'] for r in results):.1%}). "
        "Dense and a reshaped dense layer are mathematically identical.",
        ha="left", va="bottom", fontsize=8, color=MUTED,
    )

    fig.subplots_adjust(left=0.03, right=0.97, top=0.76, bottom=0.15, wspace=0.06)
    fig.savefig("matrix_mlp.png", dpi=200, facecolor=SURFACE)
    fig.savefig("matrix_mlp.svg", facecolor=SURFACE)
    plt.close(fig)
    print("wrote matrix_mlp.png", flush=True)


def main():
    results = []
    for variant in VARIANTS:
        print(variant["name"], flush=True)
        results.append(run_variant(variant))
        r = results[-1]
        print(
            f"  -> {r['acc']:.2%} ± {r['acc_std']:.2%}, {r['params']:,} params, "
            f"{r['time']:.1f}s/run",
            flush=True,
        )
    draw_figure(results)


if __name__ == "__main__":
    main()
