from __future__ import annotations
import networkx as nx
from gpml.registry import TASKS

@TASKS.register("bipartite")
class BipartiteTask:
    name = "bipartite"
    mode = "binary"
    num_classes = None

    def __init__(self, **_: object):
        pass

    def label(self, g: nx.Graph) -> int:
        return 1 if nx.is_bipartite(g) else 0
