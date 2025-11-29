from __future__ import annotations
from typing import Any
import networkx as nx
from gpml.registry import GRAPH_FAMILIES

@GRAPH_FAMILIES.register("erdos_renyi")
def erdos_renyi(n: int, p: float = 0.1, seed: int | None = None) -> nx.Graph:
    return nx.gnp_random_graph(n=n, p=p, seed=seed, directed=False)

@GRAPH_FAMILIES.register("watts_strogatz")
def watts_strogatz(n: int, k: int = 4, p: float = 0.2, seed: int | None = None) -> nx.Graph:
    k = min(k if k % 2 == 0 else k + 1, n - (n % 2 == 1))
    k = max(2, min(k, n - 1))
    return nx.watts_strogatz_graph(n=n, k=k, p=p, seed=seed)

@GRAPH_FAMILIES.register("barabasi_albert")
def barabasi_albert(n: int, m: int = 2, seed: int | None = None) -> nx.Graph:
    m = max(1, min(m, n - 1))
    return nx.barabasi_albert_graph(n=n, m=m, seed=seed)

@GRAPH_FAMILIES.register("random_regular")
def random_regular(n: int, d: int = 3, seed: int | None = None) -> nx.Graph:
    d = max(0, min(d, n - 1))
    if (n * d) % 2 == 1:
        d = max(0, d - 1)
    return nx.random_regular_graph(d=d, n=n, seed=seed)

@GRAPH_FAMILIES.register("cycle")
def cycle(n: int, **_: Any) -> nx.Graph:
    return nx.cycle_graph(n)

@GRAPH_FAMILIES.register("complete")
def complete(n: int, **_: Any) -> nx.Graph:
    return nx.complete_graph(n)

@GRAPH_FAMILIES.register("random_bipartite")
def random_bipartite(n: int, p: float = 0.2, seed: int | None = None) -> nx.Graph:
    n1 = n // 2
    n2 = n - n1
    return nx.bipartite.random_graph(n1, n2, p=p, seed=seed)
