from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any, List
import numpy as np
import torch
from gpml.registry import TRANSFORMS

Sample = Dict[str, Any]  # keys: adj, n, y, meta...

class Compose:
    def __init__(self, transforms: List):
        self.transforms = transforms

    def __call__(self, sample: Sample) -> Sample:
        for t in self.transforms:
            sample = t(sample)
        return sample

@TRANSFORMS.register("random_node_permutation")
@dataclass
class RandomNodePermutation:
    p: float = 1.0
    rng_seed: int = 0

    def __post_init__(self):
        self.rng = np.random.default_rng(self.rng_seed)

    def __call__(self, sample: Sample) -> Sample:
        if self.rng.random() > self.p:
            return sample
        adj = sample["adj"]
        if isinstance(adj, torch.Tensor):
            adj = adj.detach().cpu().numpy()
        n = adj.shape[0]
        perm = self.rng.permutation(n)
        sample["adj"] = adj[perm][:, perm]
        return sample

@TRANSFORMS.register("degree_sort_canonicalize")
@dataclass
class DegreeSortCanonicalize:
    descending: bool = True

    def __call__(self, sample: Sample) -> Sample:
        adj = sample["adj"]
        if isinstance(adj, torch.Tensor):
            adj = adj.detach().cpu().numpy()
        deg = adj.sum(axis=1)
        order = np.argsort(deg)
        if self.descending:
            order = order[::-1]
        sample["adj"] = adj[order][:, order]
        return sample

@TRANSFORMS.register("pad_to_size")
@dataclass
class PadToSize:
    size: int

    def __call__(self, sample: Sample) -> Sample:
        adj = sample["adj"]
        if isinstance(adj, torch.Tensor):
            adj = adj.detach().cpu().numpy()
        n = adj.shape[0]
        if n == self.size:
            return sample
        if n > self.size:
            sample["adj"] = adj[: self.size, : self.size]
            return sample
        out = np.zeros((self.size, self.size), dtype=adj.dtype)
        out[:n, :n] = adj
        sample["adj"] = out
        return sample

@TRANSFORMS.register("to_float_tensor")
@dataclass
class ToFloatTensor:
    def __call__(self, sample: Sample) -> Sample:
        adj = sample["adj"]
        if not isinstance(adj, torch.Tensor):
            adj = torch.from_numpy(np.asarray(adj))
        sample["adj"] = adj.float()
        return sample
