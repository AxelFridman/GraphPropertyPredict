from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List
import random
import networkx as nx
from gpml.registry import GRAPH_FAMILIES

@dataclass
class GraphSourceSpec:
    family: str
    params: Dict[str, Any]
    weight: float

class GraphSampler:
    def __init__(self, sources: List[GraphSourceSpec], seed: int = 0):
        self.sources = sources
        self.rng = random.Random(seed)
        weights = [s.weight for s in sources]
        s = sum(weights) if sum(weights) > 0 else 1.0
        self.weights = [w / s for w in weights]

    def sample(self, n: int, seed: int | None = None) -> nx.Graph:
        spec = self.rng.choices(self.sources, weights=self.weights, k=1)[0]
        fn = GRAPH_FAMILIES.get(spec.family)
        kwargs = dict(spec.params)
        kwargs["seed"] = seed
        return fn(n=n, **kwargs)
