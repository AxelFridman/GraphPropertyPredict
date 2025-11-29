from __future__ import annotations
from typing import Dict
import torch

@torch.no_grad()
def metrics_binary(logits: torch.Tensor, y: torch.Tensor) -> Dict[str, float]:
    probs = torch.sigmoid(logits)
    pred = (probs >= 0.5).float()
    acc = (pred == y).float().mean().item()
    tp = ((pred == 1) & (y == 1)).sum().item()
    fp = ((pred == 1) & (y == 0)).sum().item()
    fn = ((pred == 0) & (y == 1)).sum().item()
    prec = tp / (tp + fp + 1e-9)
    rec = tp / (tp + fn + 1e-9)
    f1 = 2 * prec * rec / (prec + rec + 1e-9)
    return {"accuracy": acc, "f1": float(f1)}

@torch.no_grad()
def metrics_regression(pred: torch.Tensor, y: torch.Tensor) -> Dict[str, float]:
    err = pred - y
    mae = err.abs().mean().item()
    rmse = (err.pow(2).mean().sqrt()).item()
    return {"mae": mae, "rmse": rmse}
