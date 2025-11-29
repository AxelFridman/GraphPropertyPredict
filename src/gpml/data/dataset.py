from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List
import os
import numpy as np
import torch
import networkx as nx
from tqdm import tqdm

from gpml.registry import TASKS, TRANSFORMS
from gpml.graphs.sampler import GraphSampler, GraphSourceSpec
from gpml.data.transforms import Compose

@dataclass
class DatasetConfig:
    cache_path: str
    num_graphs: int
    n_nodes: int
    balance_binary: bool
    sources: List[Dict[str, Any]]
    transforms: List[Dict[str, Any]]

class GraphPropertyDataset(torch.utils.data.Dataset):
    def __init__(self, adj: torch.Tensor, y: torch.Tensor, meta: Dict[str, Any] | None = None):
        self.adj = adj
        self.y = y
        self.meta = meta or {}

    def __len__(self) -> int:
        return self.adj.shape[0]

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return {"adj": self.adj[idx], "y": self.y[idx]}

def _instantiate_transforms(specs: List[Dict[str, Any]]) -> Compose:
    ts = []
    for s in specs:
        name = s["name"]
        params = s.get("params", {}) or {}
        cls = TRANSFORMS.get(name)
        ts.append(cls(**params))
    return Compose(ts)

def build_or_load_dataset(
    *,
    cfg: DatasetConfig,
    task_name: str,
    task_params: Dict[str, Any] | None,
    seed: int,
) -> GraphPropertyDataset:
    os.makedirs(os.path.dirname(cfg.cache_path), exist_ok=True)
    if os.path.exists(cfg.cache_path):
        blob = torch.load(cfg.cache_path, map_location="cpu")
        return GraphPropertyDataset(blob["adj"], blob["y"], meta=blob.get("meta", {}))

    task_cls = TASKS.get(task_name)
    task = task_cls(**(task_params or {}))

    sources = [GraphSourceSpec(**s) for s in cfg.sources]
    sampler = GraphSampler(sources, seed=seed)

    tfm = _instantiate_transforms(cfg.transforms)

    adj_list = []
    y_list = []

    want_balance = bool(cfg.balance_binary) and getattr(task, "mode", None) == "binary"
    target_each = cfg.num_graphs // 2 if want_balance else None
    count_pos = 0
    count_neg = 0

    pbar = tqdm(total=cfg.num_graphs, desc=f"Building dataset ({task_name})")
    i = 0
    while len(adj_list) < cfg.num_graphs:
        g = sampler.sample(cfg.n_nodes, seed=seed + i)
        i += 1

        adj = nx.to_numpy_array(g, dtype=np.uint8)
        np.fill_diagonal(adj, 0)

        y = task.label(g)

        if want_balance:
            if y == 1 and count_pos >= target_each:
                continue
            if y == 0 and count_neg >= target_each:
                continue

        sample = {"adj": adj, "y": y, "n": cfg.n_nodes}
        sample = tfm(sample)
        adj_t = sample["adj"]
        y_val = sample["y"]

        if isinstance(adj_t, np.ndarray):
            adj_t = torch.from_numpy(adj_t).float()

        adj_list.append(adj_t.unsqueeze(0))
        y_list.append(float(y_val))

        if want_balance:
            if y == 1:
                count_pos += 1
            else:
                count_neg += 1

        pbar.update(1)

    pbar.close()

    adj = torch.cat(adj_list, dim=0).contiguous()
    y = torch.tensor(y_list, dtype=torch.float32)

    blob = {"adj": adj, "y": y, "meta": {"task": task_name, "n_nodes": cfg.n_nodes}}
    torch.save(blob, cfg.cache_path)
    return GraphPropertyDataset(adj, y, meta=blob["meta"])
