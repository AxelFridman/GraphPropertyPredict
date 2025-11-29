from __future__ import annotations
import torch.nn as nn

def make_loss(mode: str):
    if mode == "binary":
        return nn.BCEWithLogitsLoss()
    if mode == "regression":
        return nn.MSELoss()
    if mode == "multiclass":
        return nn.CrossEntropyLoss()
    raise ValueError(f"Unknown mode: {mode}")
