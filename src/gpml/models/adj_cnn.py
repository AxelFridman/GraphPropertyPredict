from __future__ import annotations
from typing import List
import torch
import torch.nn as nn
from gpml.registry import MODELS

@MODELS.register("adj_cnn")
class AdjacencyCNN(nn.Module):
    """Treat adjacency matrix as an image: [B,1,N,N]."""
    def __init__(self, channels: List[int] = [32, 64, 128], hidden_dim: int = 256, dropout: float = 0.2, out_dim: int = 1):
        super().__init__()
        layers = []
        in_ch = 1
        for ch in channels:
            layers += [
                nn.Conv2d(in_ch, ch, kernel_size=3, padding=1),
                nn.BatchNorm2d(ch),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2),
            ]
            in_ch = ch
        self.conv = nn.Sequential(*layers)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.LazyLinear(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, adj: torch.Tensor) -> torch.Tensor:
        x = adj.unsqueeze(1)
        x = self.conv(x)
        return self.head(x).squeeze(-1)
