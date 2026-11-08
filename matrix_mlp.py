"""MLP variants whose hidden state is an 8x8 matrix instead of a flat vector.

Getting a matrix is not the interesting part; a 64-vector reshaped to 8x8 holds
the same numbers. What changes behavior is which matrix edges exist and whether
weights are shared across positions:

  LocalMatrixMLP  - each cell mixes with its 3x3 neighborhood, one shared kernel
  RowColMatrixMLP - each layer applies R @ H @ C^T, mixing all rows and columns

Both are compared against the dense flat MLP in matrix_experiment.py.
"""

import torch
import torch.nn as nn

GRID = 8


class LocalMatrixMLP(nn.Module):
    """Hidden state as GRID x GRID cells, mixed by shared 3x3 local kernels."""

    def __init__(self, n=GRID, blocks=3, num_classes=3):
        super().__init__()
        self.n = n
        self.proj = nn.Linear(2, n * n)
        self.blocks = nn.ModuleList(
            [nn.Conv2d(1, 1, 3, padding=1) for _ in range(blocks)]
        )
        self.act = nn.GELU()
        self.head = nn.Linear(n * n, num_classes)

    def forward(self, x):
        h = self.proj(x).view(-1, 1, self.n, self.n)
        for conv in self.blocks:
            h = h + self.act(conv(h))
        return self.head(h.flatten(1))


class RowColMatrixMLP(nn.Module):
    """Hidden state as a matrix; each block mixes rows and columns via R @ H @ C^T."""

    def __init__(self, n=GRID, blocks=3, num_classes=3):
        super().__init__()
        self.n = n
        self.proj = nn.Linear(2, n * n)
        self.rows = nn.ModuleList([nn.Linear(n, n, bias=False) for _ in range(blocks)])
        self.cols = nn.ModuleList([nn.Linear(n, n, bias=False) for _ in range(blocks)])
        self.act = nn.GELU()
        self.head = nn.Linear(n * n, num_classes)
        for row, col in zip(self.rows, self.cols):
            with torch.no_grad():
                row.weight.copy_(torch.eye(n) + 0.02 * torch.randn(n, n))
                col.weight.copy_(torch.eye(n) + 0.02 * torch.randn(n, n))

    def forward(self, x):
        h = self.proj(x).view(-1, self.n, self.n)
        for row, col in zip(self.rows, self.cols):
            h = h + self.act(row.weight @ h @ col.weight.T)
        return self.head(h.flatten(1))
