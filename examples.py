# ordering_coloring_dataset.py

from __future__ import annotations

import json
import random
from dataclasses import dataclass, asdict
from itertools import permutations, combinations
from typing import Dict, Iterable, List, Tuple, Optional

import networkx as nx


# ----------------------------
# Your original predicate
# ----------------------------

def colorings(iterable, colors: int):
    """(Brute force) Yield all colorings of iterable with colors 1..colors as dict."""
    iterable = list(iterable)
    if len(iterable) == 0:
        yield {}
    else:
        first, *rest = iterable
        for smaller in colorings(rest, colors):
            for color in range(1, colors + 1):
                yield {first: color} | smaller


def is_forbidden_pattern(graph: nx.Graph, coloring: Dict[int, int], u: int, v: int, w: int) -> bool:
    # exactly your condition:
    return graph.has_edge(u, w) and (coloring[u] == coloring[v] or coloring[v] == coloring[w])


def is_solution(graph: nx.Graph, ordering: List[int], coloring: Dict[int, int]) -> bool:
    return all(
        not is_forbidden_pattern(graph, coloring, u, v, w)
        for u, v, w in combinations(ordering, 3)
    )


# ----------------------------
# Key trick: build constraint graph H(ordering)
# For each original edge (a,b), every vertex v between them must differ from a and b.
# So we add edges (v,a) and (v,b) in the derived graph H.
# Then your "solution" is exactly: "proper coloring of H".
# ----------------------------

def constraint_graph_for_ordering(G: nx.Graph, ordering: List[int]) -> nx.Graph:
    pos = {v: i for i, v in enumerate(ordering)}
    H = nx.Graph()
    H.add_nodes_from(G.nodes())

    for a, b in G.edges():
        ia, ib = pos[a], pos[b]
        if ia == ib:
            continue
        if ia > ib:
            a, b = b, a
            ia, ib = ib, ia

        # vertices strictly between a and b in the ordering
        for idx in range(ia + 1, ib):
            v = ordering[idx]
            H.add_edge(v, a)
            H.add_edge(v, b)

    return H


# ----------------------------
# Exact chromatic number (and witness coloring) via DSATUR branch & bound
# Works well for small graphs (n ~ 5-10ish).
# ----------------------------

def greedy_upper_bound_coloring(G: nx.Graph) -> Dict[int, int]:
    """Simple greedy coloring (gives an upper bound)."""
    nodes = sorted(G.nodes(), key=lambda v: G.degree(v), reverse=True)
    coloring: Dict[int, int] = {}
    for v in nodes:
        used = {coloring[u] for u in G.neighbors(v) if u in coloring}
        c = 1
        while c in used:
            c += 1
        coloring[v] = c
    return coloring


def exact_chromatic_number_dsatur(G: nx.Graph) -> Tuple[int, Dict[int, int]]:
    """
    Returns (k, coloring) where coloring is a proper coloring using exactly k colors,
    and k is the chromatic number of G (exact).
    """
    nodes = list(G.nodes())
    if not nodes:
        return 0, {}
    if G.number_of_edges() == 0:
        return 1, {v: 1 for v in nodes}

    adj = {v: set(G.neighbors(v)) for v in nodes}

    # initial best from greedy
    best_coloring = greedy_upper_bound_coloring(G)
    best_k = max(best_coloring.values())

    colors: Dict[int, int] = {}
    neighbor_colors = {v: set() for v in nodes}  # sat sets
    uncolored = set(nodes)

    def pick_vertex() -> int:
        # max saturation degree, tie-break by degree
        return max(uncolored, key=lambda v: (len(neighbor_colors[v]), len(adj[v])))

    def backtrack(current_max: int) -> None:
        nonlocal best_k, best_coloring

        if not uncolored:
            if current_max < best_k:
                best_k = current_max
                best_coloring = colors.copy()
            return

        # cannot improve if already >= best
        if current_max >= best_k:
            return

        v = pick_vertex()
        uncolored.remove(v)

        forbidden = {colors[u] for u in adj[v] if u in colors}

        # try existing colors first
        for c in range(1, current_max + 1):
            if c in forbidden:
                continue

            colors[v] = c
            changed = []
            for u in adj[v]:
                if u in uncolored and c not in neighbor_colors[u]:
                    neighbor_colors[u].add(c)
                    changed.append(u)

            backtrack(current_max)

            for u in changed:
                neighbor_colors[u].remove(c)
            del colors[v]

        # try introducing a new color
        new_c = current_max + 1
        if new_c < best_k:
            colors[v] = new_c
            changed = []
            for u in adj[v]:
                if u in uncolored and new_c not in neighbor_colors[u]:
                    neighbor_colors[u].add(new_c)
                    changed.append(u)

            backtrack(new_c)

            for u in changed:
                neighbor_colors[u].remove(new_c)
            del colors[v]

        uncolored.add(v)

    backtrack(0)

    # ensure full coloring
    if set(best_coloring.keys()) != set(nodes):
        # fallback shouldn't happen, but keep it safe:
        best_coloring = greedy_upper_bound_coloring(G)
        best_k = max(best_coloring.values())

    return best_k, best_coloring


# ----------------------------
# Dataset records
# ----------------------------

@dataclass
class GraphRecord:
    graph_id: int
    n: int
    m: int
    edges: List[Tuple[int, int]]

    # globally best over all orderings
    best_k: int
    best_ordering: List[int]
    best_coloring: Dict[int, int]

    # optional light stats
    num_orderings: int
    num_optimal_orderings: int


def analyze_graph_bruteforce_all_orderings(G: nx.Graph, graph_id: int) -> GraphRecord:
    nodes = list(G.nodes())
    best_k = len(nodes) + 1
    best_ordering: Optional[List[int]] = None
    best_coloring: Optional[Dict[int, int]] = None

    num_opt = 0
    total = 0

    for ordering in permutations(nodes):
        total += 1
        ordering = list(ordering)
        H = constraint_graph_for_ordering(G, ordering)
        k, coloring = exact_chromatic_number_dsatur(H)

        if k < best_k:
            best_k = k
            best_ordering = ordering
            best_coloring = coloring
            num_opt = 1
        elif k == best_k:
            num_opt += 1

    assert best_ordering is not None and best_coloring is not None

    # sanity: verify the witness solves your original constraint on G
    assert is_solution(G, best_ordering, best_coloring), "Internal error: witness doesn't satisfy constraint."

    edges = [tuple(sorted(e)) for e in G.edges()]
    edges.sort()
    return GraphRecord(
        graph_id=graph_id,
        n=G.number_of_nodes(),
        m=G.number_of_edges(),
        edges=edges,
        best_k=best_k,
        best_ordering=best_ordering,
        best_coloring=best_coloring,
        num_orderings=total,
        num_optimal_orderings=num_opt,
    )


def generate_random_graphs_dataset_jsonl(
    path: str,
    num_graphs: int = 200,
    n: int = 7,
    p: float = 0.35,
    seed: int = 0,
    require_connected: bool = False,
) -> None:
    rng = random.Random(seed)

    with open(path, "w", encoding="utf-8") as f:
        for gid in range(num_graphs):
            # sample until matches criteria
            while True:
                # fixed nodes 0..n-1, so orderings are consistent
                G = nx.gnp_random_graph(n, p, seed=rng.randint(0, 10**9))
                G = nx.convert_node_labels_to_integers(G, first_label=0)
                if not require_connected or (n <= 1 or nx.is_connected(G)):
                    break

            rec = analyze_graph_bruteforce_all_orderings(G, graph_id=gid)
            # JSON needs string keys for dicts sometimes; we keep ints but json will convert keys to strings.
            f.write(json.dumps(asdict(rec)) + "\n")


def load_dataset_jsonl(path: str) -> List[GraphRecord]:
    out: List[GraphRecord] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            # restore tuple edges + int-key coloring
            edges = [tuple(e) for e in d["edges"]]
            coloring = {int(k): int(v) for k, v in d["best_coloring"].items()}
            out.append(
                GraphRecord(
                    graph_id=int(d["graph_id"]),
                    n=int(d["n"]),
                    m=int(d["m"]),
                    edges=edges,
                    best_k=int(d["best_k"]),
                    best_ordering=[int(x) for x in d["best_ordering"]],
                    best_coloring=coloring,
                    num_orderings=int(d["num_orderings"]),
                    num_optimal_orderings=int(d["num_optimal_orderings"]),
                )
            )
    return out


def record_to_graph(rec: GraphRecord) -> nx.Graph:
    G = nx.Graph()
    G.add_nodes_from(range(rec.n))
    G.add_edges_from(rec.edges)
    return G


# ----------------------------
# Show 10 random graphs + how the optimum ordering realizes the minimum
# ----------------------------

def show_10_random_examples(dataset: List[GraphRecord], seed: int = 0) -> None:
    import matplotlib.pyplot as plt

    rng = random.Random(seed)
    samples = rng.sample(dataset, k=min(10, len(dataset)))

    for rec in samples:
        G = record_to_graph(rec)
        ordering = rec.best_ordering
        coloring = rec.best_coloring

        # build constraint graph and compute which edges were "added" by ordering
        H = constraint_graph_for_ordering(G, ordering)
        orig = {tuple(sorted(e)) for e in G.edges()}
        extra = [e for e in H.edges() if tuple(sorted(e)) not in orig]

        pos = nx.spring_layout(G, seed=rec.graph_id)

        order_idx = {v: i for i, v in enumerate(ordering)}
        labels = {v: f"{v}\n@{order_idx[v]}" for v in G.nodes()}

        # draw G with node colors
        plt.figure(figsize=(5.5, 5.5))
        nx.draw_networkx_nodes(
            G, pos,
            node_size=800,
            node_color=[coloring[v] for v in G.nodes()],
            cmap=plt.cm.tab20,
        )
        nx.draw_networkx_labels(G, pos, labels=labels, font_size=10)
        nx.draw_networkx_edges(G, pos, width=2)

        # overlay "extra" constraint edges in dashed style
        if extra:
            nx.draw_networkx_edges(G, pos, edgelist=extra, style="dashed", width=2, alpha=0.6)

        plt.title(
            f"graph_id={rec.graph_id} | n={rec.n} m={rec.m}\n"
            f"GLOBAL MIN colors={rec.best_k} via ordering={ordering}\n"
            f"(labels show node and its position @i in ordering)\n"
            f"dashed edges = constraints induced by ordering"
        )
        plt.axis("off")
        plt.show()

        # also print a readable witness
        print(f"graph_id={rec.graph_id}")
        print(f"  best_k = {rec.best_k}")
        print(f"  best_ordering = {rec.best_ordering}")
        print(f"  best_coloring = {rec.best_coloring}")
        print(f"  Solution check = {is_solution(G, rec.best_ordering, rec.best_coloring)}")
        print(f"  optimal orderings count = {rec.num_optimal_orderings} / {rec.num_orderings}")
        print("-" * 80)


# ----------------------------
# Example usage
# ----------------------------

if __name__ == "__main__":
    out_path = "ordering_coloring_dataset.jsonl"

    # Keep n small if you're brute-forcing all orderings.
    generate_random_graphs_dataset_jsonl(
        path=out_path,
        num_graphs=50,
        n=7,
        p=0.35,
        seed=123,
        require_connected=False,
    )

    dataset = load_dataset_jsonl(out_path)
    show_10_random_examples(dataset, seed=999)
