import argparse
import torch
from torch.utils.data import DataLoader, random_split

from gpml.utils.config import load_config
from gpml.utils.seed import seed_all
from gpml.utils.device import pick_device
from gpml.registry import MODELS
from gpml.data.dataset import DatasetConfig, build_or_load_dataset
from gpml.data.collate import collate_batch
from gpml.training.trainer import Trainer, TrainerConfig

import gpml.graphs
import gpml.tasks
import gpml.models

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()

    cfg = load_config(args.config)
    seed_all(cfg["seed"])
    device = pick_device(cfg.get("device", "auto"))

    dcfg = DatasetConfig(**cfg["dataset"])
    task = cfg["task"]
    ds = build_or_load_dataset(
        cfg=dcfg,
        task_name=task["name"],
        task_params=task.get("params"),
        seed=cfg["seed"],
    )

    split = cfg["split"]
    n = len(ds)
    n_train = int(split["train"] * n)
    n_val = int(split["val"] * n)
    n_test = n - n_train - n_val
    train_ds, val_ds, _ = random_split(
        ds, [n_train, n_val, n_test],
        generator=torch.Generator().manual_seed(cfg["seed"])
    )

    dlcfg = cfg["dataloader"]
    train_loader = DataLoader(
        train_ds, batch_size=dlcfg["batch_size"], shuffle=True,
        num_workers=dlcfg["num_workers"], collate_fn=collate_batch
    )
    val_loader = DataLoader(
        val_ds, batch_size=dlcfg["batch_size"], shuffle=False,
        num_workers=dlcfg["num_workers"], collate_fn=collate_batch
    )

    mode = task["mode"]
    mcfg = cfg["model"]
    model_cls = MODELS.get(mcfg["name"])
    model = model_cls(out_dim=1, **(mcfg.get("params") or {}))

    ocfg = cfg["optim"]
    name = ocfg["name"].lower()
    if name == "adamw":
        optim = torch.optim.AdamW(model.parameters(), lr=ocfg["lr"], weight_decay=ocfg.get("weight_decay", 0.0))
    elif name == "adam":
        optim = torch.optim.Adam(model.parameters(), lr=ocfg["lr"], weight_decay=ocfg.get("weight_decay", 0.0))
    else:
        raise ValueError(f"Unknown optimizer: {ocfg['name']}")

    tcfg = TrainerConfig(**cfg["trainer"])
    trainer = Trainer(model=model, mode=mode, device=device, cfg=tcfg)
    ckpt = trainer.fit(train_loader, val_loader, optim)
    print(f"Training done. Best checkpoint: {ckpt}")

if __name__ == "__main__":
    main()
