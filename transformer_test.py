"""Transformer variant of the bar-shift test.

Per-pixel tokens, two attention blocks, mean pooling. Same train/test split as
spatial_test.py: train on top-half bar positions, test seen top positions and
held-out bottom positions.

Position handling is the variable:
  absolute - learned embedding per pixel position (standard ViT-style)
  relative - bias indexed by the 2D offset between tokens, no absolute position
  none     - no position information; tokens form an unordered bag
"""

import argparse
from statistics import mean, stdev

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from matplotlib.ticker import FuncFormatter

from spatial_test import LocalBarNet, accuracy, make_bars, param_count

SIZE = 8
TRAIN_SIZE = 768
STEPS = 500
BATCH = 256
LR = 3e-3
SEEDS = [0, 1, 2]

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
HAIRLINE = "#e1e0d9"
BASELINE = "#c3c2b7"

VARIANTS = [
    {"name": "Local CNN (reference)", "color": "#2a78d6", "pos": None},
    {"name": "Transformer, relative pos", "color": "#eb6834", "pos": "relative"},
    {"name": "Transformer, absolute pos", "color": "#1baf7a", "pos": "absolute"},
    {"name": "Transformer, no position", "color": "#eda100", "pos": "none"},
]


class AttentionBlock(nn.Module):
    def __init__(self, d_model, heads, mlp_ratio=2, use_rel_bias=False):
        super().__init__()
        self.heads = heads
        self.head_dim = d_model // heads
        self.use_rel_bias = use_rel_bias
        self.norm1 = nn.LayerNorm(d_model)
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, mlp_ratio * d_model),
            nn.GELU(),
            nn.Linear(mlp_ratio * d_model, d_model),
        )
        if use_rel_bias:
            self.rel_bias = nn.Parameter(torch.zeros(heads, 15 * 15))

    def forward(self, x, rel_index):
        b, t, d = x.shape
        h = self.norm1(x)
        qkv = self.qkv(h).reshape(b, t, 3, self.heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        att = (q @ k.transpose(-2, -1)) * self.head_dim ** -0.5
        if self.use_rel_bias:
            att = att + self.rel_bias[:, rel_index].unsqueeze(0)
        att = att.softmax(dim=-1)
        out = (att @ v).transpose(1, 2).reshape(b, t, d)
        x = x + self.proj(out)
        x = x + self.mlp(self.norm2(x))
        return x


class TinyTransformer(nn.Module):
    def __init__(self, d_model=16, heads=2, layers=2, pos="relative", mlp_ratio=2):
        super().__init__()
        self.pos = pos
        tokens = SIZE * SIZE
        self.embed = nn.Linear(1, d_model)
        if pos == "absolute":
            self.pos_embed = nn.Parameter(torch.zeros(1, tokens, d_model))
            nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.blocks = nn.ModuleList(
            [
                AttentionBlock(d_model, heads, mlp_ratio, use_rel_bias=(pos == "relative"))
                for _ in range(layers)
            ]
        )
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, 2)

        rows = torch.arange(SIZE).repeat_interleave(SIZE)
        cols = torch.arange(SIZE).repeat(SIZE)
        rel = (rows[:, None] - rows[None, :] + SIZE - 1) * (2 * SIZE - 1)
        rel = rel + (cols[:, None] - cols[None, :] + SIZE - 1)
        self.register_buffer("rel_index", rel, persistent=False)

    def forward(self, x):
        tokens = x.flatten(2).transpose(1, 2)  # B, 64, 1
        h = self.embed(tokens)
        if self.pos == "absolute":
            h = h + self.pos_embed
        for block in self.blocks:
            h = block(h, self.rel_index)
        h = self.norm(h).mean(dim=1)
        return self.head(h)


def build(variant):
    if variant["pos"] is None:
        return LocalBarNet()
    return TinyTransformer(pos=variant["pos"])


def fit_model(model, images, labels, seed):
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


def run(variant, seed, train_images, train_labels, test_seen, test_shift):
    torch.manual_seed(seed)
    model = build(variant)
    fit_model(model, train_images, train_labels, seed)
    return {
        "seen": accuracy(model, *test_seen),
        "shifted": accuracy(model, *test_shift),
        "params": param_count(model),
    }


def run_experiment():
    test_seen = make_bars(2000, "top", seed=9991)
    test_shift = make_bars(2000, "bottom", seed=9992)
    results = []
    for variant in VARIANTS:
        runs = []
        for seed in SEEDS:
            images, labels = make_bars(
                TRAIN_SIZE, "top", seed=5000 + TRAIN_SIZE * 10 + seed
            )
            r = run(variant, seed, images, labels, test_seen, test_shift)
            runs.append(r)
            print(
                f"{variant['name']:30s} seed {seed}: "
                f"seen {r['seen']:.1%}  shifted {r['shifted']:.1%}  "
                f"params {r['params']:,}",
                flush=True,
            )
        results.append(
            {
                **variant,
                "params": runs[0]["params"],
                "seen": mean(r["seen"] for r in runs),
                "seen_std": stdev(r["seen"] for r in runs),
                "shifted": mean(r["shifted"] for r in runs),
                "shifted_std": stdev(r["shifted"] for r in runs),
            }
        )
    return results


def draw_chart(results):
    plt.rcParams.update(
        {"font.family": ["Segoe UI", "DejaVu Sans", "sans-serif"], "font.size": 9.5}
    )
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 5.0), dpi=200)
    fig.patch.set_facecolor(SURFACE)

    names = [r["name"] for r in results]
    ys = list(range(len(results)))[::-1]
    panels = [
        ("seen", "seen_std", "Seen positions (top half)"),
        ("shifted", "shifted_std", "Held-out positions (bottom half)"),
    ]

    for ax, (metric, std_key, title) in zip(axes, panels):
        ax.set_facecolor(SURFACE)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(BASELINE)
        ax.spines["bottom"].set_linewidth(0.8)
        ax.tick_params(colors=MUTED, labelsize=9, length=3)
        ax.grid(axis="x", color=HAIRLINE, linewidth=0.7)
        ax.set_axisbelow(True)
        ax.set_xlim(0, 1.13)

        vals = [r[metric] for r in results]
        errs = [r[std_key] for r in results]
        colors = [r["color"] for r in results]
        ax.barh(ys, vals, height=0.56, color=colors, edgecolor=SURFACE, linewidth=2)
        ax.errorbar(
            vals, ys, xerr=errs, fmt="none", ecolor=INK_2,
            elinewidth=1, capsize=0, alpha=0.7,
        )
        for y, v in zip(ys, vals):
            ax.text(
                v + 0.018, y, f"{v:.0%}", va="center", ha="left",
                fontsize=9.5, color=INK, fontweight="600",
            )

        if metric == "shifted":
            ax.axvline(0.5, color=BASELINE, linewidth=0.8, linestyle=(0, (3, 3)), zorder=0)
            ax.text(0.5, -0.56, "chance", ha="center", va="bottom",
                    fontsize=8, color=MUTED)

        ax.set_ylim(-0.65, 3.45)
        if ax is axes[0]:
            ax.set_yticks(ys, labels=names)
            ax.tick_params(axis="y", colors=MUTED, labelsize=9)
            for label in ax.get_yticklabels():
                label.set_color(INK_2)
        else:
            ax.set_yticks(ys)
            ax.set_yticklabels([])
            ax.tick_params(axis="y", length=0)
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0%}"))
        ax.set_title(title, loc="left", color=INK, fontsize=12, fontweight="600", pad=10)
        ax.set_xlabel("test accuracy", color=INK_2, fontsize=9.5)

    fig.text(
        0.055, 0.96, "Relative positions transfer; absolute positions memorize",
        ha="left", va="top", fontsize=16, fontweight="700", color=INK,
    )
    fig.text(
        0.055, 0.90,
        "8×8 images as 64 tokens, two attention blocks, mean pooling. "
        "Train on top-half bar positions only. Relative 2D bias preserves geometry; "
        "a learned position table is tied to the training rows.",
        ha="left", va="top", fontsize=9.5, color=INK_2,
    )
    params = "  ·  ".join(f"{r['name']}: {r['params']:,}" for r in results)
    fig.text(
        0.055, 0.035,
        f"{len(SEEDS)} seeds, {TRAIN_SIZE} training examples, {STEPS} AdamW steps, batch {BATCH}. "
        f"Error bars: ±1 std. Params: {params}.",
        ha="left", va="bottom", fontsize=8, color=MUTED,
    )

    fig.subplots_adjust(left=0.265, right=0.97, top=0.76, bottom=0.17, wspace=0.08)
    fig.savefig("transformer_test.png", dpi=200, facecolor=SURFACE)
    fig.savefig("transformer_test.svg", facecolor=SURFACE)
    plt.close(fig)
    print("wrote transformer_test.png", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    if args.smoke:
        test_seen = make_bars(1000, "top", seed=9991)
        test_shift = make_bars(1000, "bottom", seed=9992)
        images, labels = make_bars(TRAIN_SIZE, "top", seed=5000 + TRAIN_SIZE * 10)
        for variant in VARIANTS:
            r = run(variant, 0, images, labels, test_seen, test_shift)
            print(
                f"{variant['name']:30s} seen {r['seen']:.1%}  "
                f"shifted {r['shifted']:.1%}  params {r['params']:,}"
            )
        return

    results = run_experiment()
    print()
    for r in results:
        print(
            f"{r['name']:30s} seen {r['seen']:.1%}  shifted {r['shifted']:.1%}  "
            f"params {r['params']:,}"
        )
    draw_chart(results)


if __name__ == "__main__":
    main()
