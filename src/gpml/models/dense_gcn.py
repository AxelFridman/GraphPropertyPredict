from __future__ import annotations
import torch
import torch.nn as nn
from gpml.registry import MODELS

def normalize_adj(adj: torch.Tensor) -> torch.Tensor:
    b, n, _ = adj.shape
    eye = torch.eye(n, device=adj.device).unsqueeze(0).expand(b, -1, -1)
    a = adj + eye
    deg = a.sum(dim=-1)
    deg_inv_sqrt = torch.pow(deg.clamp(min=1.0), -0.5)
    d = torch.diag_embed(deg_inv_sqrt)
    return d @ a @ d

class DenseGCNLayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.lin = nn.Linear(in_dim, out_dim)

    def forward(self, x: torch.Tensor, a_norm: torch.Tensor) -> torch.Tensor:
        return self.lin(a_norm @ x)

@MODELS.register("dense_gcn")
class DenseGCN(nn.Module):
    def __init__(self, hidden_dim: int = 128, num_layers: int = 4, dropout: float = 0.1, out_dim: int = 1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.in_proj = nn.Linear(2, hidden_dim)
        self.layers = nn.ModuleList([DenseGCNLayer(hidden_dim, hidden_dim) for _ in range(num_layers)])
        self.norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(num_layers)])
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, adj: torch.Tensor) -> torch.Tensor:
        b, n, _ = adj.shape
        deg = adj.sum(dim=-1, keepdim=True)
        ones = torch.ones((b, n, 1), device=adj.device, dtype=adj.dtype)
        x = torch.cat([ones, deg], dim=-1)
        x = self.in_proj(x)
        a_norm = normalize_adj(adj)
        for layer, ln in zip(self.layers, self.norms):
            h = layer(x, a_norm)
            x = ln(torch.relu(h) + x)
            x = self.dropout(x)
        g = x.mean(dim=1)
        return self.head(g).squeeze(-1)
