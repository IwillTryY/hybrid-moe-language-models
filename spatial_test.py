"""Does a matrix/grid layer help when the data actually has 2D structure?

Task: 8x8 grayscale images, either a horizontal or a vertical 1x3 bar, random
position, Gaussian pixel noise. Train on bars whose center lies in the top half.
Test on bars in the top half (seen positions) and the bottom half (unseen).

Dense baseline sees the 64 pixels as a flat vector. Local model keeps the 8x8
grid, uses shared 3x3 conv kernels, then pools over space, so its filters do not
care where the bar sits.
"""

import argparse
from statistics import mean, stdev

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib.ticker import FuncFormatter

SIZE = 8
BAR = 3
NOISE = 0.12
STEPS = 400
BATCH = 64
LR = 3e-3
SEEDS = [0, 1, 2]
TRAIN_SIZES = [48, 192, 768]

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
HAIRLINE = "#e1e0d9"
BASELINE = "#c3c2b7"
DENSE_COLOR = "#2a78d6"
LOCAL_COLOR = "#eb6834"


class DenseBarNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(SIZE * SIZE, 64), nn.GELU(),
            nn.Linear(64, 64), nn.GELU(),
            nn.Linear(64, 2),
        )

    def forward(self, x):
        return self.net(x)


class LocalBarNet(nn.Module):
    def __init__(self, channels=4):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, channels, 3, padding=1), nn.GELU(),
            nn.Conv2d(channels, channels, 3, padding=1), nn.GELU(),
        )
        self.head = nn.Linear(2 * channels, 2)

    def forward(self, x):
        f = self.features(x)
        pooled = torch.cat([f.amax(dim=(2, 3)), f.mean(dim=(2, 3))], dim=1)
        return self.head(pooled)


def make_bars(n, region, seed, noise=NOISE):
    """region: 'top' or 'bottom' half of the image for the bar center."""
    g = torch.Generator().manual_seed(seed)
    images = torch.zeros(n, 1, SIZE, SIZE)
    labels = torch.randint(0, 2, (n,), generator=g)
    top = region == "top"
    for i in range(n):
        horizontal = labels[i].item() == 0
        if horizontal:
            row = int(torch.randint(0, 4, (1,), generator=g)) if top else int(
                torch.randint(4, 8, (1,), generator=g)
            )
            start = int(torch.randint(0, SIZE - BAR + 1, (1,), generator=g))
            images[i, 0, row, start:start + BAR] = 1.0
        else:
            col = int(torch.randint(0, SIZE, (1,), generator=g))
            start = int(torch.randint(0, 3, (1,), generator=g)) if top else int(
                torch.randint(3, 6, (1,), generator=g)
            )
            images[i, 0, start:start + BAR, col] = 1.0
    images += noise * torch.randn(images.shape, generator=g)
    return images, labels


def param_count(model):
    return sum(p.numel() for p in model.parameters())


def train_model(model, images, labels, seed):
    torch.manual_seed(seed)
    g = torch.Generator().manual_seed(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=STEPS)
    loss_fn = nn.CrossEntropyLoss()

    model.train()
    step = 0
    while step < STEPS:
        perm = torch.randperm(images.size(0), generator=g)
        for start in range(0, images.size(0), BATCH):
            idx = perm[start:start + BATCH]
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(model(images[idx]), labels[idx])
            loss.backward()
            opt.step()
            sched.step()
            step += 1
            if step >= STEPS:
                break
    return model


@torch.no_grad()
def accuracy(model, images, labels):
    model.eval()
    correct = 0
    for start in range(0, images.size(0), 512):
        logits = model(images[start:start + 512])
        correct += (logits.argmax(1) == labels[start:start + 512]).sum().item()
    return correct / images.size(0)


def run_experiment():
    test_seen = make_bars(2000, "top", seed=9991)
    test_shift = make_bars(2000, "bottom", seed=9992)

    results = {}
    for size in TRAIN_SIZES:
        for name, builder in (("Dense", DenseBarNet), ("Local grid", LocalBarNet)):
            seen, shifted, params = [], [], 0
            for seed in SEEDS:
                images, labels = make_bars(size, "top", seed=5000 + size * 10 + seed)
                torch.manual_seed(seed)
                model = builder()
                params = param_count(model)
                train_model(model, images, labels, seed)
                seen.append(accuracy(model, *test_seen))
                shifted.append(accuracy(model, *test_shift))
                print(
                    f"size {size:4d}  {name:10s} seed {seed}: "
                    f"seen {seen[-1]:.1%}  shifted {shifted[-1]:.1%}",
                    flush=True,
                )
            results[(size, name)] = {
                "seen": seen,
                "shifted": shifted,
                "params": params,
            }
    return results


def draw_chart(results):
    plt.rcParams.update(
        {"font.family": ["Segoe UI", "DejaVu Sans", "sans-serif"], "font.size": 9.5}
    )
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.9), dpi=200, sharey=True)
    fig.patch.set_facecolor(SURFACE)

    series = [
        ("Dense", DENSE_COLOR),
        ("Local grid", LOCAL_COLOR),
    ]
    panels = [
        ("seen", "Seen positions (top half)", axes[0]),
        ("shifted", "Held-out positions (bottom half)", axes[1]),
    ]

    for metric, title, ax in panels:
        ax.set_facecolor(SURFACE)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(BASELINE)
            ax.spines[side].set_linewidth(0.8)
        ax.tick_params(colors=MUTED, labelsize=9, length=3)
        ax.grid(axis="y", color=HAIRLINE, linewidth=0.7)
        ax.set_axisbelow(True)

        for name, color in series:
            vals = [mean(results[(size, name)][metric]) for size in TRAIN_SIZES]
            errs = [stdev(results[(size, name)][metric]) for size in TRAIN_SIZES]
            ax.errorbar(
                TRAIN_SIZES, vals, yerr=errs, color=color, linewidth=2,
                label=name, marker="o", markersize=7, markeredgecolor=SURFACE,
                markeredgewidth=1.5, capsize=0, zorder=3,
            )

        ax.axhline(0.5, color=BASELINE, linewidth=0.8, linestyle=(0, (3, 3)), zorder=1)
        ax.text(
            TRAIN_SIZES[-1], 0.462, "chance", ha="right", va="bottom",
            fontsize=8, color=MUTED,
        )
        ax.set_xscale("log")
        ax.set_xticks(TRAIN_SIZES)
        ax.set_xticklabels([str(s) for s in TRAIN_SIZES])
        ax.xaxis.set_minor_formatter(FuncFormatter(lambda v, _: ""))
        ax.set_xlabel("training examples (log scale)", color=INK_2, fontsize=9.5)
        ax.set_ylim(0.45, 1.01)
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0%}"))
        ax.set_title(title, loc="left", color=INK, fontsize=12, fontweight="600", pad=10)

    axes[0].set_ylabel("test accuracy (3 seeds)", color=INK_2, fontsize=9.5)
    axes[0].set_xlim(TRAIN_SIZES[0] * 0.8, TRAIN_SIZES[-1] * 1.15)
    axes[1].set_xlim(TRAIN_SIZES[0] * 0.8, TRAIN_SIZES[-1] * 1.15)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="upper right", bbox_to_anchor=(0.965, 0.925),
        ncol=2, frameon=False, fontsize=9, labelcolor=INK_2,
        handlelength=1.8, columnspacing=1.6,
    )

    dense_params = results[(TRAIN_SIZES[0], "Dense")]["params"]
    local_params = results[(TRAIN_SIZES[0], "Local grid")]["params"]
    fig.text(
        0.055, 0.96, "A matrix layer pays off when the data has 2D structure",
        ha="left", va="top", fontsize=16, fontweight="700", color=INK,
    )
    fig.text(
        0.055, 0.90,
        "8×8 horizontal-vs-vertical bar images, pixel noise 0.12. "
        "Train only on top-half positions. Dense flattens pixels; local grid shares 3×3 filters and pools.",
        ha="left", va="top", fontsize=9.5, color=INK_2,
    )
    fig.text(
        0.055, 0.035,
        f"Dense: {dense_params:,} params. Local grid: {local_params:,} params. "
        f"{len(SEEDS)} seeds, {STEPS} AdamW steps, batch {BATCH}. "
        "Error bars: ±1 std. Fixed 2,000-example test sets.",
        ha="left", va="bottom", fontsize=8, color=MUTED,
    )

    fig.subplots_adjust(left=0.075, right=0.965, top=0.79, bottom=0.17, wspace=0.18)
    fig.savefig("spatial_test.png", dpi=200, facecolor=SURFACE)
    fig.savefig("spatial_test.svg", facecolor=SURFACE)
    plt.close(fig)
    print("wrote spatial_test.png", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true", help="single setting, no chart")
    args = parser.parse_args()

    if args.smoke:
        test_seen = make_bars(1000, "top", seed=9991)
        test_shift = make_bars(1000, "bottom", seed=9992)
        for name, builder in (("Dense", DenseBarNet), ("Local grid", LocalBarNet)):
            images, labels = make_bars(192, "top", seed=501)
            torch.manual_seed(0)
            model = builder()
            train_model(model, images, labels, seed=0)
            print(
                f"{name:10s}: seen {accuracy(model, *test_seen):.1%}  "
                f"shifted {accuracy(model, *test_shift):.1%}  "
                f"params {param_count(model):,}"
            )
        return

    results = run_experiment()
    print()
    for size in TRAIN_SIZES:
        for name in ("Dense", "Local grid"):
            r = results[(size, name)]
            print(
                f"size {size:4d}  {name:10s}  seen {mean(r['seen']):.1%}  "
                f"shifted {mean(r['shifted']):.1%}  params {r['params']:,}"
            )
    draw_chart(results)


if __name__ == "__main__":
    main()
