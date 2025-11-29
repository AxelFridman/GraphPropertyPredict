from __future__ import annotations
from typing import Any, Dict, List
import torch

def collate_batch(batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
    adj = torch.stack([b["adj"] for b in batch], dim=0)
    y = torch.tensor([float(b["y"]) for b in batch], dtype=torch.float32)
    return {"adj": adj, "y": y}
