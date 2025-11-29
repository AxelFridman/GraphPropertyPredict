from __future__ import annotations
import torch
import torch.nn as nn
from gpml.registry import MODELS

@MODELS.register("adj_mlp")
class AdjacencyMLP(nn.Module):
    """Flatten adjacency into a vector."""
    def __init__(self, hidden_dim: int = 512, depth: int = 3, dropout: float = 0.2, out_dim: int = 1):
        super().__init__()
        layers = [nn.Flatten(), nn.LazyLinear(hidden_dim), nn.ReLU(), nn.Dropout(dropout)]
        for _ in range(depth - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout)]
        layers += [nn.Linear(hidden_dim, out_dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, adj: torch.Tensor) -> torch.Tensor:
        return self.net(adj).squeeze(-1)
