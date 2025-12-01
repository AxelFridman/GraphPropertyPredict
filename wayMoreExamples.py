# build_ordering_coloring_dataset_with_csv_no_images_varied_graphs.py

from __future__ import annotations

import csv
import json
import os
import random
import time
from dataclasses import dataclass, asdict
from itertools import permutations
from typing import Dict, List, Tuple, Optional

import networkx as nx

try:
    from tqdm.auto import tqdm
except ImportError:
    tqdm = None


# ----------------------------
# Constraint graph for an ordering
# ("solution" <-> proper coloring of derived graph H)
# ----------------------------

def constraint_graph_for_ordering(G: nx.Graph, ordering: List[int]) -> nx.Graph:
    pos = {v: i for i, v in enumerate(ordering)}
    H = nx.Graph()
    H.add_nodes_from(G.nodes())

    for a, b in G.edges():
        ia, ib = pos[a], pos[b]
        if ia > ib:
            a, b = b, a
            ia, ib = ib, ia

        # any vertex v strictly between endpoints must differ from both endpoints
        for idx in range(ia + 1, ib):
            v = ordering[idx]
            H.add_edge(v, a)
            H.add_edge(v, b)

    return H


def is_solution(G: nx.Graph, ordering: List[int], coloring: Dict[int, int]) -> bool:
    H = constraint_graph_for_ordering(G, ordering)
    for u, v in H.edges():
        if coloring[u] == coloring[v]:
            return False
    return True


# ----------------------------
# Exact chromatic number (and witness coloring): DSATUR B&B
# ----------------------------

def greedy_upper_bound_coloring(G: nx.Graph) -> Dict[int, int]:
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
    nodes = list(G.nodes())
    if not nodes:
        return 0, {}
    if G.number_of_edges() == 0:
        return 1, {v: 1 for v in nodes}

    adj = {v: set(G.neighbors(v)) for v in nodes}

    best_coloring = greedy_upper_bound_coloring(G)
    best_k = max(best_coloring.values())

    colors: Dict[int, int] = {}
    neighbor_colors = {v: set() for v in nodes}
    uncolored = set(nodes)

    def pick_vertex() -> int:
        return max(uncolored, key=lambda v: (len(neighbor_colors[v]), len(adj[v])))

    def backtrack(current_max: int) -> None:
        nonlocal best_k, best_coloring

        if not uncolored:
            if current_max < best_k:
                best_k = current_max
                best_coloring = colors.copy()
            return

        if current_max >= best_k:
            return

        v = pick_vertex()
        uncolored.remove(v)

        forbidden = {colors[u] for u in adj[v] if u in colors}

        # try existing colors
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

        # try introducing new color
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

    if set(best_coloring.keys()) != set(nodes):
        best_coloring = greedy_upper_bound_coloring(G)
        best_k = max(best_coloring.values())

    return best_k, best_coloring


# ----------------------------
# Exact Hamiltonian path (undirected): DP over subsets (bitmask)
# ----------------------------

def has_hamiltonian_path_exact(G: nx.Graph) -> bool:
    n = G.number_of_nodes()
    if n <= 1:
        return True

    # Fast impossibility checks
    if nx.number_connected_components(G) != 1:
        return False

    deg = dict(G.degree())
    if any(deg[v] == 0 for v in G.nodes()):
        return False

    # In an undirected graph, >2 degree-1 vertices => impossible for Hamiltonian path
    if sum(1 for v in G.nodes() if deg[v] == 1) > 2:
        return False

    # Nodes are assumed 0..n-1 (we enforce during generation)
    # Bitmask adjacency
    adjmask = [0] * n
    for u, v in G.edges():
        adjmask[u] |= 1 << v
        adjmask[v] |= 1 << u

    full = (1 << n) - 1
    dp = [0] * (1 << n)  # dp[mask] = bitset of possible endpoints for a Hamiltonian path visiting mask

    for i in range(n):
        dp[1 << i] = 1 << i

    for mask in range(1 << n):
        ends = dp[mask]
        if ends == 0:
            continue
        if mask == full:
            return True

        remaining = (~mask) & full
        e = ends
        while e:
            vbit = e & -e
            e -= vbit
            v = (vbit.bit_length() - 1)
            nbrs = adjmask[v] & remaining
            nb = nbrs
            while nb:
                wbit = nb & -nb
                nb -= wbit
                dp[mask | wbit] |= wbit

    return dp[full] != 0


# ----------------------------
# Dataset record (JSONL-friendly)
# ----------------------------

@dataclass
class GraphRecord:
    graph_id: int
    n: int
    m: int
    edges: List[Tuple[int, int]]

    best_k: int
    best_ordering: List[int]
    best_coloring: Dict[int, int]

    num_orderings: int
    num_optimal_orderings: int

    has_hamiltonian_path: bool


def record_to_graph(rec: GraphRecord) -> nx.Graph:
    G = nx.Graph()
    G.add_nodes_from(range(rec.n))
    G.add_edges_from(rec.edges)
    return G


# ----------------------------
# Ordering search: exhaustive for small n, sampled for larger n
# ----------------------------

def _bfs_ordering(G: nx.Graph, start: int) -> List[int]:
    seen = set()
    order: List[int] = []
    for comp in nx.connected_components(G):
        if start not in comp:
            continue
        q = [start]
        seen.add(start)
        while q:
            v = q.pop(0)
            order.append(v)
            for u in G.neighbors(v):
                if u not in seen:
                    seen.add(u)
                    q.append(u)
        break

    # add remaining components (disconnected case)
    for comp in nx.connected_components(G):
        if any(v in seen for v in comp):
            continue
        root = next(iter(comp))
        q = [root]
        seen.add(root)
        while q:
            v = q.pop(0)
            order.append(v)
            for u in G.neighbors(v):
                if u not in seen:
                    seen.add(u)
                    q.append(u)
    return order


def _dfs_ordering(G: nx.Graph, start: int) -> List[int]:
    seen = set()
    order: List[int] = []

    def dfs(v: int) -> None:
        seen.add(v)
        order.append(v)
        for u in G.neighbors(v):
            if u not in seen:
                dfs(u)

    # first component containing start
    for comp in nx.connected_components(G):
        if start in comp:
            dfs(start)
            break

    # then the rest (disconnected)
    for comp in nx.connected_components(G):
        if any(v in seen for v in comp):
            continue
        root = next(iter(comp))
        dfs(root)

    return order


def candidate_orderings(G: nx.Graph, rng: random.Random, max_samples: int) -> List[List[int]]:
    nodes = list(G.nodes())
    n = len(nodes)
    if n == 0:
        return [[]]

    seen = set()
    out: List[List[int]] = []

    def push(order: List[int]) -> None:
        t = tuple(order)
        if t not in seen:
            seen.add(t)
            out.append(order)

    # deterministic-ish heuristics
    push(sorted(nodes, key=lambda v: G.degree(v), reverse=True))  # high degree first
    push(sorted(nodes, key=lambda v: G.degree(v)))                # low degree first

    # core-based heuristics (gives good variety on “mesh-like” graphs)
    try:
        core = nx.core_number(G) if G.number_of_nodes() > 0 else {}
        push(sorted(nodes, key=lambda v: (core.get(v, 0), G.degree(v)), reverse=True))
        push(sorted(nodes, key=lambda v: (core.get(v, 0), G.degree(v))))
    except nx.NetworkXError:
        pass

    # component-aware: big components first, then by degree
    comps = sorted(nx.connected_components(G), key=len, reverse=True)
    comp_order: List[int] = []
    for comp in comps:
        comp_order.extend(sorted(comp, key=lambda v: G.degree(v), reverse=True))
    push(comp_order)

    # BFS/DFS from different starts
    starts = []
    starts.append(max(nodes, key=lambda v: G.degree(v)))
    starts.append(min(nodes, key=lambda v: G.degree(v)))
    starts.append(rng.choice(nodes))
    for s in starts:
        push(_bfs_ordering(G, s))
        push(_dfs_ordering(G, s))

    # random permutations (true variety)
    base = nodes[:]
    while len(out) < max_samples:
        rng.shuffle(base)
        push(base[:])

    return out[:max_samples]


def analyze_graph_orderings(
    G: nx.Graph,
    graph_id: int,
    *,
    exhaustive_n_max: int = 8,
    sampled_orderings: int = 512,
    ordering_seed: int = 0,
) -> GraphRecord:
    nodes = list(G.nodes())
    n = len(nodes)

    best_k = n + 1
    best_ordering: Optional[List[int]] = None
    best_coloring: Optional[Dict[int, int]] = None
    total = 0
    num_opt = 0

    if n <= exhaustive_n_max:
        iterable = permutations(nodes)
        for ordering in iterable:
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
    else:
        rng = random.Random(ordering_seed)
        orders = candidate_orderings(G, rng, max_samples=sampled_orderings)
        for ordering in orders:
            total += 1
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
    assert is_solution(G, best_ordering, best_coloring)

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
        has_hamiltonian_path=has_hamiltonian_path_exact(G),
    )


# ----------------------------
# Extra attributes for CSV
# ----------------------------

def compute_graph_attributes(G: nx.Graph) -> Dict[str, object]:
    degrees = dict(G.degree())
    deg_vals = list(degrees.values())

    max_degree = max(deg_vals) if deg_vals else 0
    avg_degree = (sum(deg_vals) / len(deg_vals)) if deg_vals else 0.0

    num_deg_ge_2 = sum(1 for d in deg_vals if d >= 2)
    num_deg_ge_3 = sum(1 for d in deg_vals if d >= 3)
    num_deg_ge_5 = sum(1 for d in deg_vals if d >= 5)

    num_isolated = sum(1 for d in deg_vals if d == 0)
    num_components = nx.number_connected_components(G) if G.number_of_nodes() > 0 else 0
    component_sizes = sorted((len(c) for c in nx.connected_components(G)), reverse=True)

    # exact clique number via maximal cliques enumeration (OK for small n)
    clique_number = max((len(c) for c in nx.find_cliques(G)), default=0)

    return {
        "max_degree": max_degree,
        "avg_degree": avg_degree,
        "clique_number": clique_number,
        "num_deg_ge_2": num_deg_ge_2,
        "num_deg_ge_3": num_deg_ge_3,
        "num_deg_ge_5": num_deg_ge_5,
        "num_isolated": num_isolated,
        "num_components": num_components,
        "component_sizes": component_sizes,  # store as JSON in CSV
        "density": nx.density(G) if G.number_of_nodes() > 1 else 0.0,
    }


# ----------------------------
# Varied graph generation (tons of variety + bridge-compositions)
# ----------------------------

def _ensure_int_labels(G: nx.Graph) -> nx.Graph:
    return nx.convert_node_labels_to_integers(G, first_label=0)


def _rand_even_k(rng: random.Random, n: int, k_min: int = 2) -> int:
    if n <= 2:
        return 0
    k = rng.randint(k_min, max(k_min, n - 1))
    if k % 2 == 1:
        k = max(k_min, k - 1)
    return min(k, n - 1)


def _combine_with_bridges(
    G1: nx.Graph,
    G2: nx.Graph,
    rng: random.Random,
    *,
    num_bridges: int = 1,
    bridge_path_len: int = 0,
) -> nx.Graph:
    A = _ensure_int_labels(G1)
    B = _ensure_int_labels(G2)
    G = nx.disjoint_union(A, B)

    nodes_A = list(range(A.number_of_nodes()))
    offset = A.number_of_nodes()
    nodes_B = list(range(offset, offset + B.number_of_nodes()))

    if not nodes_A or not nodes_B:
        return _ensure_int_labels(G)

    for _ in range(num_bridges):
        u = rng.choice(nodes_A)
        v = rng.choice(nodes_B)

        if bridge_path_len <= 0:
            G.add_edge(u, v)
        else:
            last = u
            for _k in range(bridge_path_len):
                new_node = G.number_of_nodes()
                G.add_node(new_node)
                G.add_edge(last, new_node)
                last = new_node
            G.add_edge(last, v)

    return _ensure_int_labels(G)


def _force_connected(G: nx.Graph, rng: random.Random) -> nx.Graph:
    if G.number_of_nodes() <= 1:
        return _ensure_int_labels(G)
    if nx.is_connected(G):
        return _ensure_int_labels(G)

    comps = [list(c) for c in nx.connected_components(G)]
    # connect components with random bridge edges (keeps "bridge-y" nature)
    for i in range(len(comps) - 1):
        u = rng.choice(comps[i])
        v = rng.choice(comps[i + 1])
        G.add_edge(u, v)
    return _ensure_int_labels(G)


def _force_disconnected(G: nx.Graph, rng: random.Random) -> nx.Graph:
    n = G.number_of_nodes()
    if n <= 1:
        return _ensure_int_labels(G)
    if nx.number_connected_components(G) >= 2:
        return _ensure_int_labels(G)

    # cut by partition: remove all crossing edges between S and V\S
    nodes = list(G.nodes())
    k = rng.randint(1, n - 1)
    rng.shuffle(nodes)
    S = set(nodes[:k])
    cut_edges = [(u, v) for (u, v) in G.edges() if (u in S) ^ (v in S)]
    G.remove_edges_from(cut_edges)

    # if still connected (rare), nuke a random spanning edge
    if nx.number_connected_components(G) == 1 and G.number_of_edges() > 0:
        u, v = rng.choice(list(G.edges()))
        G.remove_edge(u, v)

    return _ensure_int_labels(G)


def generate_varied_graph(
    gid: int,
    *,
    seed: int,
    n_min: int,
    n_max: int,
    connectivity: str = "mixed",   # "mixed" | "all_connected" | "all_disconnected"
    _depth: int = 0,
    max_depth: int = 3,
) -> nx.Graph:
    base_seed = (seed * 1_000_003 + gid) % (2**32)
    rng = random.Random(base_seed)

    # heavy-tail-ish size: muchos chicos, algunos grandes
    if rng.random() < 0.72:
        n = rng.randint(n_min, max(n_min, (n_min + n_max) // 2))
    else:
        n = rng.randint(max(n_min, (n_min + n_max) // 2), n_max)
    n = max(1, n)

    # base cases para n pequeño (evita rangos inválidos en recetas)
    if n == 1:
        G = nx.empty_graph(1)
        G = _ensure_int_labels(nx.Graph(G))
        if connectivity == "all_disconnected":
            return G
        if connectivity == "all_connected":
            return G
        return G

    recipes = [
        ("empty", 0.06),
        ("complete", 0.06),
        ("path", 0.06),
        ("cycle", 0.04),
        ("star", 0.04),
        ("tree", 0.06),
        ("erdos_sparse", 0.08),
        ("erdos_dense", 0.10),
        ("gnm", 0.06),
        ("ba", 0.08),
        ("ws", 0.06),
        ("sbm", 0.08),
        ("regular", 0.04),
        ("geometric", 0.04),
        ("barbell", 0.03),
        ("lollipop", 0.03),
        ("disjoint_union", 0.08),
        ("bridge_combo", 0.10),
    ]

    # evita recursión “en cadena” (bridge_combo -> disjoint_union -> bridge_combo -> ...)
    if _depth >= max_depth:
        recipes = [(t, w) for (t, w) in recipes if t not in ("disjoint_union", "bridge_combo")]

    tags, weights = zip(*recipes)
    tag = rng.choices(tags, weights=weights, k=1)[0]

    def rand_seed() -> int:
        return rng.randrange(0, 2**32)

    if tag == "empty":
        G = nx.empty_graph(n)

    elif tag == "complete":
        G = nx.complete_graph(n)

    elif tag == "path":
        G = nx.path_graph(n)

    elif tag == "cycle":
        if n < 3:
            G = nx.path_graph(n)
        else:
            G = nx.cycle_graph(n)

    elif tag == "star":
        G = nx.star_graph(n - 1)  # total n nodes

    elif tag == "tree":
        G = nx.random_tree(n, seed=rand_seed())

    elif tag == "erdos_sparse":
        p = rng.uniform(0.02, min(0.18, 1.0))
        G = nx.gnp_random_graph(n, p, seed=rand_seed())

    elif tag == "erdos_dense":
        p = rng.uniform(0.55, 0.95)
        G = nx.gnp_random_graph(n, p, seed=rand_seed())

    elif tag == "gnm":
        max_m = n * (n - 1) // 2
        if rng.random() < 0.5:
            m = rng.randint(0, max(1, max_m // 5))
        else:
            m = rng.randint(max(0, max_m // 3), max_m)
        G = nx.gnm_random_graph(n, m, seed=rand_seed())

    elif tag == "ba":
        if n <= 2:
            G = nx.path_graph(n)
        else:
            m = rng.randint(1, min(5, n - 1))
            G = nx.barabasi_albert_graph(n, m, seed=rand_seed())

    elif tag == "ws":
        if n <= 3:
            G = nx.path_graph(n)
        else:
            k = _rand_even_k(rng, n, k_min=2)
            beta = rng.uniform(0.0, 0.85)
            G = nx.watts_strogatz_graph(n, k, beta, seed=rand_seed())

    elif tag == "sbm":
        if n <= 3:
            G = nx.path_graph(n)
        else:
            blocks = rng.randint(2, min(5, n))
            sizes = [1] * blocks
            for _ in range(n - blocks):
                sizes[rng.randrange(blocks)] += 1
            p_in = rng.uniform(0.35, 0.95)
            p_out = rng.uniform(0.0, min(0.25, p_in * 0.6))
            probs = [[p_out] * blocks for _ in range(blocks)]
            for i in range(blocks):
                probs[i][i] = p_in
            G = nx.stochastic_block_model(sizes, probs, seed=rand_seed())

    elif tag == "regular":
        if n <= 3:
            G = nx.path_graph(n)
        else:
            d = rng.randint(1, min(n - 1, 5))
            if (n * d) % 2 == 1:
                d = max(1, d - 1)
            if d == 0:
                G = nx.empty_graph(n)
            else:
                try:
                    G = nx.random_regular_graph(d, n, seed=rand_seed())
                except nx.NetworkXError:
                    G = nx.gnp_random_graph(n, rng.uniform(0.1, 0.6), seed=rand_seed())

    elif tag == "geometric":
        if n <= 2:
            G = nx.path_graph(n)
        else:
            radius = rng.uniform(0.15, 0.55)
            G = nx.random_geometric_graph(n, radius, seed=rand_seed())
            G = nx.Graph(G)

    elif tag == "barbell":
        if n < 6:
            G = nx.path_graph(n)
        else:
            c = rng.randint(2, max(2, (n - 2) // 2))
            p = max(0, n - 2 * c)
            G = nx.barbell_graph(c, p)

    elif tag == "lollipop":
        if n < 5:
            G = nx.path_graph(n)
        else:
            c = rng.randint(2, n - 2)
            p = max(1, n - c)
            G = nx.lollipop_graph(c, p)

    elif tag == "disjoint_union":
        # FIX: si n<2 no se puede pedir randint(2, ...)
        if n < 2:
            G = nx.empty_graph(n)
        else:
            kcomps = rng.randint(2, min(5, n))
            sizes = [1] * kcomps
            for _ in range(n - kcomps):
                sizes[rng.randrange(kcomps)] += 1

            comps: List[nx.Graph] = []
            for s in sizes:
                # genera EXACTAMENTE s nodos para que el total sea n
                comps.append(
                    generate_varied_graph(
                        rand_seed(),
                        seed=seed ^ 0x9E3779B9,
                        n_min=s,
                        n_max=s,
                        connectivity="mixed",
                        _depth=_depth + 1,
                        max_depth=max_depth,
                    )
                )
            G = nx.disjoint_union_all([_ensure_int_labels(C) for C in comps])

    elif tag == "bridge_combo":
        if n <= 2:
            G = nx.path_graph(n)
        else:
            path_len = rng.choice([0, 0, 1, 2, 3])
            bridges = rng.choice([1, 1, 1, 2, 3])

            # reservar nodos para el camino puente (si lo hay)
            n_rem = max(2, n - path_len)
            n1 = rng.randint(1, n_rem - 1)
            n2 = max(1, n_rem - n1)

            G1 = generate_varied_graph(
                rand_seed(),
                seed=seed ^ 0xA5A5A5A5,
                n_min=n1, n_max=n1,
                connectivity="mixed",
                _depth=_depth + 1,
                max_depth=max_depth,
            )
            G2 = generate_varied_graph(
                rand_seed(),
                seed=seed ^ 0x5A5A5A5A,
                n_min=n2, n_max=n2,
                connectivity="mixed",
                _depth=_depth + 1,
                max_depth=max_depth,
            )
            G = _combine_with_bridges(G1, G2, rng, num_bridges=bridges, bridge_path_len=path_len)

    else:
        G = nx.gnp_random_graph(n, 0.3, seed=rand_seed())

    G = _ensure_int_labels(nx.Graph(G))

    # connectivity target
    if connectivity == "all_connected":
        G = _force_connected(G, rng)
    elif connectivity == "all_disconnected":
        G = _force_disconnected(G, rng)
    else:
        goal = rng.choices(["any", "connected", "disconnected"], weights=[0.45, 0.30, 0.25], k=1)[0]
        if goal == "connected":
            G = _force_connected(G, rng)
        elif goal == "disconnected":
            G = _force_disconnected(G, rng)

    return _ensure_int_labels(G)


# ----------------------------
# Progressive/resumable writing helpers
# ----------------------------

def count_jsonl_rows(path: str) -> int:
    if not os.path.exists(path):
        return 0
    with open(path, "r", encoding="utf-8") as f:
        return sum(1 for _ in f)


# ----------------------------
# Build everything: JSONL + CSV (PROGRESSIVE + RESUMABLE)
# ----------------------------

def build_dataset(
    out_dir: str = "out_dataset",
    num_graphs: int = 10_000,
    seed: int = 123,
    n_min: int = 4,
    n_max: int = 12,
    connectivity: str = "mixed",  # "mixed"|"all_connected"|"all_disconnected"
    exhaustive_n_max: int = 8,
    sampled_orderings: int = 512,
    resume: bool = True,
    flush_every: int = 25,
) -> Tuple[str, str]:
    """
    Creates (progressively):
      - out_dataset/ordering_dataset.jsonl   (append-only)
      - out_dataset/ordering_dataset.csv     (append-only)

    Notes:
      - Best ordering search is exhaustive only if n <= exhaustive_n_max.
        For larger graphs it samples `sampled_orderings` candidate orderings.
      - Hamiltonian path is exact DP (bitmask); keep n_max reasonable for speed.
    """
    os.makedirs(out_dir, exist_ok=True)

    jsonl_path = os.path.join(out_dir, "ordering_dataset.jsonl")
    csv_path = os.path.join(out_dir, "ordering_dataset.csv")

    start_gid = count_jsonl_rows(jsonl_path) if resume else 0
    if start_gid >= num_graphs:
        print(f"Already complete: {start_gid}/{num_graphs}")
        return jsonl_path, csv_path

    fieldnames = [
        "graph_id", "n", "m", "edges",
        "best_k", "best_ordering", "best_coloring",
        "num_orderings", "num_optimal_orderings",
        "has_hamiltonian_path",
        "max_degree", "avg_degree", "clique_number",
        "num_deg_ge_2", "num_deg_ge_3", "num_deg_ge_5",
        "num_isolated", "num_components", "component_sizes",
        "density",
    ]

    csv_exists = os.path.exists(csv_path) and os.path.getsize(csv_path) > 0

    with open(jsonl_path, "a", encoding="utf-8") as jf, open(csv_path, "a", newline="", encoding="utf-8") as cf:
        writer = csv.DictWriter(cf, fieldnames=fieldnames)

        if not csv_exists:
            writer.writeheader()
            cf.flush()

        gids = range(start_gid, num_graphs)
        iterator = tqdm(gids, total=num_graphs, initial=start_gid, dynamic_ncols=True) if tqdm else gids

        t0 = time.perf_counter()
        done = 0

        for gid in iterator:
            t_graph_start = time.perf_counter()
            base_seed = (seed * 1_000_003 + gid) % (2**32)

            G = generate_varied_graph(
                gid,
                seed=seed,
                n_min=n_min,
                n_max=n_max,
                connectivity=connectivity,
            )
            G = _ensure_int_labels(G)

            rec = analyze_graph_orderings(
                G,
                graph_id=gid,
                exhaustive_n_max=exhaustive_n_max,
                sampled_orderings=sampled_orderings,
                ordering_seed=(base_seed ^ 0xC0FFEE),
            )
            attrs = compute_graph_attributes(G)

            # JSONL row
            jf.write(json.dumps(asdict(rec)) + "\n")

            # CSV row
            row = {
                "graph_id": rec.graph_id,
                "n": rec.n,
                "m": rec.m,
                "edges": json.dumps(rec.edges),

                "best_k": rec.best_k,
                "best_ordering": json.dumps(rec.best_ordering),
                "best_coloring": json.dumps(rec.best_coloring),

                "num_orderings": rec.num_orderings,
                "num_optimal_orderings": rec.num_optimal_orderings,

                "has_hamiltonian_path": int(bool(rec.has_hamiltonian_path)),

                "max_degree": attrs["max_degree"],
                "avg_degree": attrs["avg_degree"],
                "clique_number": attrs["clique_number"],
                "num_deg_ge_2": attrs["num_deg_ge_2"],
                "num_deg_ge_3": attrs["num_deg_ge_3"],
                "num_deg_ge_5": attrs["num_deg_ge_5"],
                "num_isolated": attrs["num_isolated"],
                "num_components": attrs["num_components"],
                "component_sizes": json.dumps(attrs["component_sizes"]),
                "density": attrs["density"],
            }
            writer.writerow(row)

            done += 1
            if done % flush_every == 0:
                jf.flush()
                cf.flush()

            dt = time.perf_counter() - t_graph_start
            if tqdm:
                iterator.set_postfix({"n": rec.n, "k": rec.best_k, "sec/graph": f"{dt:.2f}"})
            else:
                elapsed = time.perf_counter() - t0
                avg = elapsed / done
                remaining = (num_graphs - (gid + 1)) * avg
                print(
                    f"[{gid+1}/{num_graphs}] n={rec.n} k={rec.best_k} sec/graph={dt:.2f} "
                    f"avg={avg:.2f}s est_left={remaining:.0f}s",
                    flush=True,
                )

    return jsonl_path, csv_path


if __name__ == "__main__":
    # pip install tqdm  (opcional)
    jsonl_path, csv_path = build_dataset(
        out_dir="out_dataset2",
        num_graphs=10_000,
        seed=123,

        # variedad real (tamaños mezclados)
        n_min=2,
        n_max=12,

        # "mixed" | "all_connected" | "all_disconnected"
        connectivity="mixed",

        # exacto factorial sólo hasta acá; después samplea
        exhaustive_n_max=8,
        sampled_orderings=512,

        resume=True,
        flush_every=25,
    )
    print("Wrote (append/resume):", jsonl_path)
    print("Wrote (append/resume):", csv_path)
