from __future__ import annotations
import networkx as nx
from gpml.registry import TASKS

@TASKS.register("max_clique")
class MaxCliqueTask:
    """
    Target: clique number ω(G).
    Exact computation is NP-hard; keep graphs small for dataset generation if you want exact labels.
    """
    name = "max_clique"
    mode = "regression"
    num_classes = None

    def __init__(self, **_: object):
        pass

    def label(self, g: nx.Graph) -> float:
        try:
            from networkx.algorithms.clique import graph_clique_number
            return float(graph_clique_number(g))
        except Exception:
            cliques = list(nx.find_cliques(g))
            return float(max((len(c) for c in cliques), default=1))
