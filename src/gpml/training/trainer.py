from __future__ import annotations
import os, json
from dataclasses import dataclass
from typing import Dict
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from gpml.training.losses import make_loss
from gpml.training.metrics import metrics_binary, metrics_regression

@dataclass
class TrainerConfig:
    epochs: int
    grad_clip: float
    log_every: int
    out_dir: str
    metric_for_best: str

def _is_better(metric_name: str, new: float, best: float) -> bool:
    lower_better = any(k in metric_name for k in ["mae", "rmse", "loss"])
    return new < best if lower_better else new > best

class Trainer:
    def __init__(self, model: torch.nn.Module, mode: str, device: torch.device, cfg: TrainerConfig):
        self.model = model.to(device)
        self.mode = mode
        self.device = device
        self.cfg = cfg
        self.loss_fn = make_loss(mode)
        os.makedirs(cfg.out_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir=cfg.out_dir)

    def _step_metrics(self, logits: torch.Tensor, y: torch.Tensor) -> Dict[str, float]:
        if self.mode == "binary":
            return metrics_binary(logits, y)
        if self.mode == "regression":
            return metrics_regression(logits, y)
        raise NotImplementedError("multiclass not wired in this starter.")

    def run_epoch(self, loader, optim=None, split="train") -> Dict[str, float]:
        train = optim is not None
        self.model.train(train)
        total_loss = 0.0
        n = 0
        agg: Dict[str, float] = {}

        for batch in tqdm(loader, desc=split, leave=False):
            adj = batch["adj"].to(self.device)
            y = batch["y"].to(self.device).float()

            logits = self.model(adj)
            loss = self.loss_fn(logits, y)

            if train:
                optim.zero_grad(set_to_none=True)
                loss.backward()
                if self.cfg.grad_clip and self.cfg.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip)
                optim.step()

            bs = adj.shape[0]
            total_loss += loss.item() * bs
            n += bs

            m = self._step_metrics(logits.detach(), y.detach())
            for k, v in m.items():
                agg[k] = agg.get(k, 0.0) + v * bs

        out = {"loss": total_loss / max(1, n)}
        for k, v in agg.items():
            out[k] = v / max(1, n)
        return out

    def fit(self, train_loader, val_loader, optim) -> str:
        best_ckpt = os.path.join(self.cfg.out_dir, "best.pt")
        last_ckpt = os.path.join(self.cfg.out_dir, "last.pt")

        best_val = float("inf") if any(k in self.cfg.metric_for_best for k in ["mae", "rmse", "loss"]) else -float("inf")
        history = []

        for epoch in range(1, self.cfg.epochs + 1):
            tr = self.run_epoch(train_loader, optim=optim, split="train")
            va = self.run_epoch(val_loader, optim=None, split="val")

            for k, v in tr.items():
                self.writer.add_scalar(f"train/{k}", v, epoch)
            for k, v in va.items():
                self.writer.add_scalar(f"val/{k}", v, epoch)

            metric_key = self.cfg.metric_for_best.split("/", 1)[1]
            current = va.get(metric_key)
            if current is None:
                raise KeyError(f"metric_for_best='{self.cfg.metric_for_best}' not found in val metrics: {list(va.keys())}")

            if _is_better(metric_key, current, best_val):
                best_val = current
                torch.save({"model": self.model.state_dict(), "epoch": epoch, "val": va}, best_ckpt)

            torch.save({"model": self.model.state_dict(), "epoch": epoch, "val": va}, last_ckpt)

            history.append({"epoch": epoch, "train": tr, "val": va})
            with open(os.path.join(self.cfg.out_dir, "history.json"), "w", encoding="utf-8") as f:
                json.dump(history, f, indent=2)

        return best_ckpt
