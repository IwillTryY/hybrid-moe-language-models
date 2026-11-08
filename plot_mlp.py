"""Render an architecture diagram of the MLP from mlp.py."""

import argparse
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch.nn as nn
from matplotlib.collections import LineCollection

from mlp import MLP

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
EDGE = "#d6d5ce"
HIDDEN_COLORS = [  # sequential blue, shallow to deep
    "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7",
    "#3987e5", "#2a78d6", "#256abf", "#1c5cab",
]
INPUT_COLOR = "#898781"
OUTPUT_COLOR = "#eb6834"

SHOW_MAX = 8
R_HIDDEN = 0.055
R_IO = 0.075
S_HIDDEN = 110
S_IO = 250


def layer_info(model):
    """Flatten model.net into linear layers plus the ops that follow each one."""
    linears, ops = [], []
    for module in model.net:
        if isinstance(module, nn.Linear):
            linears.append(module)
            ops.append([])
        elif isinstance(module, (nn.ReLU, nn.GELU, nn.SiLU, nn.Tanh)):
            ops[-1].append(type(module).__name__)
        elif isinstance(module, nn.LayerNorm):
            ops[-1].append("LayerNorm")
        elif isinstance(module, nn.Dropout) and module.p > 0:
            ops[-1].append(f"Dropout {module.p:g}")
    return [
        {
            "out": lin.out_features,
            "in": lin.in_features,
            "ops": ops[i],
            "params": sum(p.numel() for p in lin.parameters()),
        }
        for i, lin in enumerate(linears)
    ]


def node_positions(count):
    """Y positions for the visible nodes, plus whether middle nodes are elided."""
    if count <= SHOW_MAX:
        spacing = 0.36 if count <= 3 else 0.3
        return [(i - (count - 1) / 2) * spacing for i in range(count)], False
    return [0.64, 0.48, 0.32, 0.16, -0.16, -0.32, -0.48, -0.64], True


def shorten(x1, y1, x2, y2, r1, r2):
    dx, dy = x2 - x1, y2 - y1
    dist = math.hypot(dx, dy)
    ux, uy = dx / dist, dy / dist
    return [(x1 + ux * r1, y1 + uy * r1), (x2 - ux * r2, y2 - uy * r2)]


def draw_diagram(model, activation, dropout, layer_norm, output):
    info = layer_info(model)
    hidden = info[:-1]
    out = info[-1]
    n_in = info[0]["in"]
    n_out = out["out"]
    total_params = sum(layer["params"] for layer in info)

    columns = [
        {
            "overline": "INPUT",
            "label": f"{n_in} {'feature' if n_in == 1 else 'features'}",
            "count": n_in,
            "color": INPUT_COLOR,
            "detail": "raw features",
            "op": None,
            "params": None,
        }
    ]
    for i, layer in enumerate(hidden):
        ops = layer["ops"]
        step = round(i * (len(HIDDEN_COLORS) - 1) / max(len(hidden) - 1, 1))
        columns.append(
            {
                "overline": f"HIDDEN {i + 1}",
                "label": f"{layer['out']} units",
                "count": layer["out"],
                "color": HIDDEN_COLORS[step],
                "detail": f"{layer['in']} × {layer['out']} + bias",
                "op": "Linear" + "".join(f" + {op}" for op in ops),
                "params": layer["params"],
            }
        )
    columns.append(
        {
            "overline": "OUTPUT",
            "label": f"{n_out} {'logit' if n_out == 1 else 'logits'}",
            "count": n_out,
            "color": OUTPUT_COLOR,
            "detail": f"{out['in']} × {out['out']} + bias",
            "op": "Linear",
            "params": out["params"],
        }
    )

    step = 1.6
    xs = [i * step for i in range(len(columns))]
    xlim = (xs[0] - 0.8, xs[-1] + 1.0)
    ylim = (-1.5, 1.6)
    fig_w = 3.1 + 1.85 * len(columns)
    fig_h = fig_w * (ylim[1] - ylim[0]) / (xlim[1] - xlim[0])

    plt.rcParams.update(
        {
            "font.family": ["Segoe UI", "DejaVu Sans", "sans-serif"],
            "mathtext.fontset": "dejavusans",
        }
    )
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=200)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    ax.set_position([0, 0, 1, 1])
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal")
    ax.axis("off")

    positions = []
    for col in columns:
        ys, elided = node_positions(col["count"])
        positions.append((ys, elided))

    # Sample the connections between visible nodes of adjacent columns.
    segments = []
    for i in range(len(columns) - 1):
        ys_a = positions[i][0]
        ys_b = positions[i + 1][0]
        r_a = R_IO if i == 0 else R_HIDDEN
        r_b = R_IO if i + 1 == len(columns) - 1 else R_HIDDEN
        for ya in ys_a:
            for yb in ys_b:
                segments.append(shorten(xs[i], ya, xs[i + 1], yb, r_a, r_b))
    ax.add_collection(
        LineCollection(segments, colors=EDGE, linewidths=0.45, alpha=0.55, zorder=1)
    )

    for i, (col, (ys, elided)) in enumerate(zip(columns, positions)):
        x = xs[i]
        is_io = i in (0, len(columns) - 1)
        ax.scatter(
            [x] * len(ys),
            ys,
            s=S_IO if is_io else S_HIDDEN,
            c=col["color"],
            edgecolors=SURFACE,
            linewidths=1.8,
            zorder=4,
        )
        if elided:
            ax.scatter(
                [x] * 3, [0.08, 0.0, -0.08],
                s=9, c=MUTED, edgecolors="none", zorder=5,
            )

        ax.text(
            x, 1.06, col["overline"], ha="center", va="bottom",
            fontsize=8.5, color=MUTED, fontweight="600",
        )
        ax.text(
            x, 0.82, col["label"], ha="center", va="bottom",
            fontsize=12, color=INK, fontweight="600",
        )
        if col["op"]:
            ax.text(
                x, -0.94, col["op"], ha="center", va="top",
                fontsize=8.5, color=INK_2,
            )
        ax.text(
            x, -1.10, col["detail"], ha="center", va="top",
            fontsize=8, color=MUTED,
        )
        if col["params"] is not None:
            ax.text(
                x, -1.24, f"{col['params']:,} params", ha="center", va="top",
                fontsize=8, color=MUTED,
            )

    # Input and output tick labels.
    input_ys = positions[0][0]
    for j, y in enumerate(input_ys):
        ax.text(
            xs[0] - 0.16, y, f"$x_{j + 1}$", ha="right", va="center",
            fontsize=9.5, color=INK_2,
        )
    output_ys = positions[-1][0]
    for j, y in enumerate(output_ys):
        ax.text(
            xs[-1] + 0.16, y, f"class {j}", ha="left", va="center",
            fontsize=9.5, color=INK_2,
        )

    arch = " → ".join([str(n_in)] + [str(layer["out"]) for layer in hidden] + [str(n_out)])
    extras = [f"{total_params:,} parameters", f"{activation.upper()} activation"]
    extras.append(f"dropout {dropout:g}" if dropout > 0 else "no dropout")
    if layer_norm:
        extras.append("LayerNorm")
    subtitle = f"{arch}  ·  " + "  ·  ".join(extras)

    ax.text(
        xlim[0] + 0.05, ylim[1] - 0.04, "MLP architecture",
        ha="left", va="top", fontsize=17, color=INK, fontweight="700",
    )
    ax.text(
        xlim[0] + 0.05, ylim[1] - 0.24, subtitle,
        ha="left", va="top", fontsize=9.5, color=INK_2,
    )

    if any(elided for _, elided in positions):
        elided_cols = [
            str(col["count"]) for col, (_, elided) in zip(columns, positions) if elided
        ]
        note = (
            f"Hidden columns show {SHOW_MAX} of {', '.join(sorted(set(elided_cols)))} units; "
            "connections are sampled. Output layer emits raw logits (CrossEntropyLoss applies softmax)."
        )
        ax.text(
            xlim[0] + 0.05, ylim[0] + 0.05, note,
            ha="left", va="bottom", fontsize=8, color=MUTED,
        )

    output = output.with_suffix(".png")
    fig.savefig(output, dpi=200, facecolor=SURFACE)
    fig.savefig(output.with_suffix(".svg"), facecolor=SURFACE)
    plt.close(fig)
    print(f"wrote {output} ({total_params:,} params)")
    return output


def main():
    parser = argparse.ArgumentParser(description="Draw the MLP architecture.")
    parser.add_argument("--hidden", type=int, nargs="+", default=[64, 64, 64])
    parser.add_argument("--activation", default="gelu", choices=["relu", "gelu", "silu", "tanh"])
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--layer-norm", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("mlp_architecture.png"))
    args = parser.parse_args()

    model = MLP(
        in_dim=2,
        hidden_dims=tuple(args.hidden),
        num_classes=3,
        activation=args.activation,
        dropout=args.dropout,
        use_layer_norm=args.layer_norm,
    )
    draw_diagram(model, args.activation, args.dropout, args.layer_norm, args.output)


if __name__ == "__main__":
    main()
