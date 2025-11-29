import argparse
import torch
from torch.utils.data import DataLoader, random_split

from gpml.utils.config import load_config
from gpml.utils.seed import seed_all
from gpml.utils.device import pick_device
from gpml.registry import MODELS
from gpml.data.dataset import DatasetConfig, build_or_load_dataset
from gpml.data.collate import collate_batch
from gpml.training.metrics import metrics_binary, metrics_regression

import gpml.graphs, gpml.tasks, gpml.models

@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", required=True)
    args = ap.parse_args()

    cfg = load_config(args.config)
    seed_all(cfg["seed"])
    device = pick_device(cfg.get("device", "auto"))

    dcfg = DatasetConfig(**cfg["dataset"])
    task = cfg["task"]
    ds = build_or_load_dataset(cfg=dcfg, task_name=task["name"], task_params=task.get("params"), seed=cfg["seed"])

    split = cfg["split"]
    n = len(ds)
    n_train = int(split["train"] * n)
    n_val = int(split["val"] * n)
    n_test = n - n_train - n_val
    _, _, test_ds = random_split(
        ds, [n_train, n_val, n_test],
        generator=torch.Generator().manual_seed(cfg["seed"])
    )

    dlcfg = cfg["dataloader"]
    test_loader = DataLoader(test_ds, batch_size=dlcfg["batch_size"], shuffle=False, num_workers=dlcfg["num_workers"], collate_fn=collate_batch)

    mode = task["mode"]
    mcfg = cfg["model"]
    model_cls = MODELS.get(mcfg["name"])
    model = model_cls(out_dim=1, **(mcfg.get("params") or {})).to(device)

    blob = torch.load(args.ckpt, map_location="cpu")
    model.load_state_dict(blob["model"])
    model.eval()

    all_logits, all_y = [], []
    for batch in test_loader:
        adj = batch["adj"].to(device)
        y = batch["y"].to(device).float()
        logits = model(adj)
        all_logits.append(logits.detach().cpu())
        all_y.append(y.detach().cpu())

    logits = torch.cat(all_logits)
    y = torch.cat(all_y)

    m = metrics_binary(logits, y) if mode == "binary" else metrics_regression(logits, y)

    print("TEST METRICS")
    for k, v in m.items():
        print(f"  {k}: {v:.4f}")

if __name__ == "__main__":
    main()
