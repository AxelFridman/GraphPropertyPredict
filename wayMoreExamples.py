# build_ordering_coloring_dataset_with_csv_no_images_varied_graphs_parallel.py

from __future__ import annotations

import csv
import json
import os
import random
import time
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
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

    if nx.number_connected_components(G) != 1:
        return False

    deg = dict(G.degree())
    if any(deg[v] == 0 for v in G.nodes()):
        return False

    if sum(1 for v in G.nodes() if deg[v] == 1) > 2:
        return False

    adjmask = [0] * n
    for u, v in G.edges():
        adjmask[u] |= 1 << v
        adjmask[v] |= 1 << u

    full = (1 << n) - 1
    dp = [0] * (1 << n)

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

    for comp in nx.connected_components(G):
        if start in comp:
            dfs(start)
            break

    for comp in nx.connected_components(G):
        if any(v in seen for v in comp):
            continue
        root = next(iter(comp))
        dfs(root)

    return order


def candidate_orderings(G: nx.Graph, rng: random.Random, max_samples: int) -> List[List[int]]:
    nodes = list(G.nodes())
    if not nodes:
        return [[]]

    seen = set()
    out: List[List[int]] = []

    def push(order: List[int]) -> None:
        t = tuple(order)
        if t not in seen:
            seen.add(t)
            out.append(order)

    push(sorted(nodes, key=lambda v: G.degree(v), reverse=True))
    push(sorted(nodes, key=lambda v: G.degree(v)))

    try:
        core = nx.core_number(G) if G.number_of_nodes() > 0 else {}
        push(sorted(nodes, key=lambda v: (core.get(v, 0), G.degree(v)), reverse=True))
        push(sorted(nodes, key=lambda v: (core.get(v, 0), G.degree(v))))
    except nx.NetworkXError:
        pass

    comps = sorted(nx.connected_components(G), key=len, reverse=True)
    comp_order: List[int] = []
    for comp in comps:
        comp_order.extend(sorted(comp, key=lambda v: G.degree(v), reverse=True))
    push(comp_order)

    starts = [
        max(nodes, key=lambda v: G.degree(v)),
        min(nodes, key=lambda v: G.degree(v)),
        rng.choice(nodes),
    ]
    for s in starts:
        push(_bfs_ordering(G, s))
        push(_dfs_ordering(G, s))

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
# Extra attributes for CSV (+ max_clique witness)
# ----------------------------

def maximum_clique_witness(G: nx.Graph) -> List[int]:
    """
    Exact: enumerates maximal cliques and keeps the largest.
    Deterministic tie-break: lexicographically smallest among max-size cliques.
    """
    best: List[int] = []
    for c in nx.find_cliques(G):
        c_sorted = sorted(c)
        if len(c_sorted) > len(best) or (len(c_sorted) == len(best) and c_sorted < best):
            best = c_sorted
    return best


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

    max_clique = maximum_clique_witness(G)
    clique_number = len(max_clique)

    return {
        "max_degree": max_degree,
        "avg_degree": avg_degree,
        "clique_number": clique_number,
        "max_clique": max_clique,  # NEW
        "num_deg_ge_2": num_deg_ge_2,
        "num_deg_ge_3": num_deg_ge_3,
        "num_deg_ge_5": num_deg_ge_5,
        "num_isolated": num_isolated,
        "num_components": num_components,
        "component_sizes": component_sizes,
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


def _random_tree_compat(n: int, seed: int) -> nx.Graph:
    """
    NetworkX 3.4+: nx.random_labeled_tree
    Older: attempt import from generators
    """
    if hasattr(nx, "random_labeled_tree"):
        return nx.random_labeled_tree(n, seed=seed)
    try:
        from networkx.generators.trees import random_labeled_tree
        return random_labeled_tree(n, seed=seed)
    except Exception as e:
        raise RuntimeError("No random labeled tree generator found in your NetworkX.") from e


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

    nodes = list(G.nodes())
    k = rng.randint(1, n - 1)
    rng.shuffle(nodes)
    S = set(nodes[:k])
    cut_edges = [(u, v) for (u, v) in G.edges() if (u in S) ^ (v in S)]
    G.remove_edges_from(cut_edges)

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
    connectivity: str = "mixed",
    _depth: int = 0,
    max_depth: int = 3,
) -> nx.Graph:
    base_seed = (seed * 1_000_003 + gid) % (2**32)
    rng = random.Random(base_seed)

    if rng.random() < 0.72:
        n = rng.randint(n_min, max(n_min, (n_min + n_max) // 2))
    else:
        n = rng.randint(max(n_min, (n_min + n_max) // 2), n_max)
    n = max(1, n)

    if n == 1:
        return _ensure_int_labels(nx.empty_graph(1))

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
        G = nx.path_graph(n) if n < 3 else nx.cycle_graph(n)

    elif tag == "star":
        G = nx.star_graph(n - 1)

    elif tag == "tree":
        # FIX for NetworkX 3.4+: use random_labeled_tree
        G = _random_tree_compat(n, seed=rand_seed())

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
        if n < 2:
            G = nx.empty_graph(n)
        else:
            kcomps = rng.randint(2, min(5, n))
            sizes = [1] * kcomps
            for _ in range(n - kcomps):
                sizes[rng.randrange(kcomps)] += 1

            comps: List[nx.Graph] = []
            for s in sizes:
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
# Parallel worker (must be top-level for Windows pickling)
# ----------------------------

def _compute_one_graph(args: Tuple[int, int, int, int, str, int, int]) -> Tuple[dict, dict]:
    gid, seed, n_min, n_max, connectivity, exhaustive_n_max, sampled_orderings = args
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
    return asdict(rec), attrs


# ----------------------------
# Build everything: JSONL + CSV (PARALLEL + RESUMABLE)
# ----------------------------

def build_dataset_parallel(
    out_dir: str = "out_dataset",
    num_graphs: int = 10_000,
    seed: int = 123,
    n_min: int = 4,
    n_max: int = 12,
    connectivity: str = "mixed",
    exhaustive_n_max: int = 8,
    sampled_orderings: int = 512,
    resume: bool = True,
    flush_every: int = 50,
    workers: Optional[int] = None,
    chunksize: int = 25,
) -> Tuple[str, str]:
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
        "max_degree", "avg_degree", "clique_number", "max_clique",  # NEW
        "num_deg_ge_2", "num_deg_ge_3", "num_deg_ge_5",
        "num_isolated", "num_components", "component_sizes",
        "density",
    ]

    csv_exists = os.path.exists(csv_path) and os.path.getsize(csv_path) > 0
    if workers is None:
        workers = max(1, (os.cpu_count() or 2) - 1)

    gids = list(range(start_gid, num_graphs))
    task_args = [(gid, seed, n_min, n_max, connectivity, exhaustive_n_max, sampled_orderings) for gid in gids]

    with open(jsonl_path, "a", encoding="utf-8") as jf, open(csv_path, "a", newline="", encoding="utf-8") as cf:
        writer = csv.DictWriter(cf, fieldnames=fieldnames)
        if not csv_exists:
            writer.writeheader()
            cf.flush()

        pbar = tqdm(total=num_graphs, initial=start_gid, dynamic_ncols=True) if tqdm else None
        t0 = time.perf_counter()
        done = 0

        with ProcessPoolExecutor(max_workers=workers) as ex:
            for payload, attrs in ex.map(_compute_one_graph, task_args, chunksize=chunksize):
                jf.write(json.dumps(payload) + "\n")

                row = {
                    "graph_id": payload["graph_id"],
                    "n": payload["n"],
                    "m": payload["m"],
                    "edges": json.dumps(payload["edges"]),

                    "best_k": payload["best_k"],
                    "best_ordering": json.dumps(payload["best_ordering"]),
                    "best_coloring": json.dumps(payload["best_coloring"]),

                    "num_orderings": payload["num_orderings"],
                    "num_optimal_orderings": payload["num_optimal_orderings"],

                    "has_hamiltonian_path": int(bool(payload["has_hamiltonian_path"])),

                    "max_degree": attrs["max_degree"],
                    "avg_degree": attrs["avg_degree"],
                    "clique_number": attrs["clique_number"],
                    "max_clique": json.dumps(attrs["max_clique"]),  # NEW

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

                if pbar:
                    pbar.update(1)
                    pbar.set_postfix({"n": payload["n"], "k": payload["best_k"]})
                elif done % 50 == 0:
                    elapsed = time.perf_counter() - t0
                    print(f"done={start_gid+done}/{num_graphs} avg_sec_graph={elapsed/done:.3f}", flush=True)

        if pbar:
            pbar.close()

    return jsonl_path, csv_path


if __name__ == "__main__":
    mp.freeze_support()

    jsonl_path, csv_path = build_dataset_parallel(
        out_dir="out_datasetJust8",
        num_graphs=10_000,
        
        seed=786,

        n_min=8,
        n_max=8,

        connectivity="mixed",

        exhaustive_n_max=8,   # WARNING: factorial if n<=exhaustive_n_max (10! per graph if n=10)
        sampled_orderings=512,

        resume=True,
        flush_every=25,
        workers=None,          # auto: cpu_count()-1
        chunksize=25,
    )

    print("Wrote (append/resume):", jsonl_path)
    print("Wrote (append/resume):", csv_path)
