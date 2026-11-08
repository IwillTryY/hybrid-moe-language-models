"""Configurable MLP with training loop and toy spiral demo."""

import argparse
import math

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


class MLP(nn.Module):
    def __init__(
        self,
        in_dim,
        hidden_dims=(128, 128),
        num_classes=10,
        activation="gelu",
        dropout=0.1,
        use_layer_norm=False,
    ):
        super().__init__()
        acts = {
            "relu": nn.ReLU,
            "gelu": nn.GELU,
            "silu": nn.SiLU,
            "tanh": nn.Tanh,
        }
        if activation not in acts:
            raise ValueError(f"unknown activation {activation!r}, pick one of {list(acts)}")

        layers = []
        prev = in_dim
        for hidden in hidden_dims:
            layers.append(nn.Linear(prev, hidden))
            if use_layer_norm:
                layers.append(nn.LayerNorm(hidden))
            layers.append(acts[activation]())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev = hidden
        layers.append(nn.Linear(prev, num_classes))
        self.net = nn.Sequential(*layers)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=0.02)
            nn.init.zeros_(module.bias)

    def forward(self, x):
        if x.dim() > 2:
            x = torch.flatten(x, 1)  # accept image tensors directly
        return self.net(x)


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total, correct, loss_sum = 0, 0, 0.0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        logits = model(xb)
        loss_sum += criterion(logits, yb).item() * xb.size(0)
        correct += (logits.argmax(1) == yb).sum().item()
        total += xb.size(0)
    return loss_sum / total, correct / total


def train(model, train_ds, val_ds, epochs=30, batch_size=64, lr=1e-3,
          weight_decay=0.01, device="cpu", verbose=True):
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    model.to(device)

    for epoch in range(1, epochs + 1):
        model.train()
        total, correct, loss_sum = 0, 0, 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

            loss_sum += loss.item() * xb.size(0)
            correct += (logits.argmax(1) == yb).sum().item()
            total += xb.size(0)
        scheduler.step()

        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        if verbose:
            print(
                f"epoch {epoch:3d} | train loss {loss_sum / total:.4f} "
                f"acc {correct / total:.4f} | val loss {val_loss:.4f} acc {val_acc:.4f}"
            )
    return model


def make_spiral(n_per_class=500, num_classes=3, noise=0.15, seed=0):
    """Generated 2D spiral dataset: makes a toy nonlinear classification problem."""
    gen = torch.Generator().manual_seed(seed)
    xs, ys = [], []
    for c in range(num_classes):
        t = torch.rand(n_per_class, generator=gen)
        radius = 0.9 * t ** 0.8 + noise * 0.3 * torch.randn(n_per_class, generator=gen)
        angle = t * 3.0 + c * (2 * math.pi / num_classes)
        angle = angle + noise * torch.randn(n_per_class, generator=gen)
        xs.append(torch.stack([radius * torch.cos(angle), radius * torch.sin(angle)], dim=1))
        ys.append(torch.full((n_per_class,), c, dtype=torch.long))
    return torch.cat(xs), torch.cat(ys)


def main():
    parser = argparse.ArgumentParser(description="Train an MLP on a toy spiral dataset.")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--hidden", type=int, nargs="+", default=[64, 64, 64])
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--activation", default="gelu", choices=["relu", "gelu", "silu", "tanh"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    x, y = make_spiral(num_classes=3, seed=args.seed)
    perm = torch.randperm(x.size(0), generator=torch.Generator().manual_seed(args.seed))
    x, y = x[perm], y[perm]
    split = int(0.8 * x.size(0))
    train_ds = TensorDataset(x[:split], y[:split])
    val_ds = TensorDataset(x[split:], y[split:])

    model = MLP(
        in_dim=2,
        hidden_dims=tuple(args.hidden),
        num_classes=3,
        activation=args.activation,
        dropout=args.dropout,
    )
    print(f"params: {sum(p.numel() for p in model.parameters()):,} | device: {args.device}")
    model = train(
        model, train_ds, val_ds,
        epochs=args.epochs, batch_size=args.batch_size,
        lr=args.lr, device=args.device,
    )

    _, val_acc = evaluate(model, DataLoader(val_ds), nn.CrossEntropyLoss(), args.device)
    print(f"final val accuracy: {val_acc:.4f}")


if __name__ == "__main__":
    main()
