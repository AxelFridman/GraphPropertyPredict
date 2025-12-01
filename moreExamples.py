# build_ordering_coloring_dataset_with_csv_and_images.py

from __future__ import annotations

import csv
import json
import os
import time
from dataclasses import dataclass, asdict
from itertools import permutations
from typing import Dict, List, Tuple, Optional

import matplotlib.pyplot as plt
import networkx as nx

try:
    from tqdm.auto import tqdm
except ImportError:
    tqdm = None


# ----------------------------
# Constraint graph for an ordering
# (Your "solution" <-> proper coloring of this derived graph H)
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
    # quick check using derived constraint graph:
    H = constraint_graph_for_ordering(G, ordering)
    for u, v in H.edges():
        if coloring[u] == coloring[v]:
            return False
    return True


# ----------------------------
# Exact chromatic number (and witness coloring): DSATUR branch & bound
# Good for small n (like 5-10ish).
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

    # safety fallback
    if set(best_coloring.keys()) != set(nodes):
        best_coloring = greedy_upper_bound_coloring(G)
        best_k = max(best_coloring.values())

    return best_k, best_coloring


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


def analyze_graph_bruteforce_all_orderings(G: nx.Graph, graph_id: int) -> GraphRecord:
    nodes = list(G.nodes())

    best_k = len(nodes) + 1
    best_ordering: Optional[List[int]] = None
    best_coloring: Optional[Dict[int, int]] = None

    total = 0
    num_opt = 0

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
    )


def record_to_graph(rec: GraphRecord) -> nx.Graph:
    G = nx.Graph()
    G.add_nodes_from(range(rec.n))
    G.add_edges_from(rec.edges)
    return G


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

    # exact for small graphs; for large graphs, this can get expensive
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
# Image saving
# ----------------------------

def folder_for_k(best_k: int) -> str:
    return {
        1: "one_color",
        2: "two_color",
        3: "three_color",
        5: "five_color",
    }.get(best_k, "other_color")


def save_graph_image(
    rec: GraphRecord,
    out_root: str,
    draw_dashed_constraints: bool = True,
) -> str:
    G = record_to_graph(rec)
    ordering = rec.best_ordering
    coloring = rec.best_coloring

    # folder by k
    folder = folder_for_k(rec.best_k)
    out_dir = os.path.join(out_root, folder)
    os.makedirs(out_dir, exist_ok=True)

    path = os.path.join(out_dir, f"graph_{rec.graph_id}.png")

    # layout + labels
    pos = nx.spring_layout(G, seed=rec.graph_id)
    order_idx = {v: i for i, v in enumerate(ordering)}
    labels = {v: f"{v}\n@{order_idx[v]}" for v in G.nodes()}

    plt.figure(figsize=(6.2, 6.2))

    nx.draw_networkx_nodes(
        G, pos,
        node_size=850,
        node_color=[coloring[v] for v in G.nodes()],
        cmap=plt.cm.tab20,
    )
    nx.draw_networkx_edges(G, pos, width=2)
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=10)

    if draw_dashed_constraints:
        H = constraint_graph_for_ordering(G, ordering)
        orig = {tuple(sorted(e)) for e in G.edges()}
        extra = [e for e in H.edges() if tuple(sorted(e)) not in orig]
        if extra:
            nx.draw_networkx_edges(G, pos, edgelist=extra, style="dashed", width=2, alpha=0.55)

    plt.title(f"id={rec.graph_id} | ordering-min colors={rec.best_k}\nordering={ordering}")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()

    return path


# ----------------------------
# Progressive/resumable writing helpers
# ----------------------------

def count_jsonl_rows(path: str) -> int:
    if not os.path.exists(path):
        return 0
    with open(path, "r", encoding="utf-8") as f:
        return sum(1 for _ in f)


# ----------------------------
# Build everything: JSONL + CSV + images (PROGRESSIVE + RESUMABLE)
# ----------------------------

def build_dataset(
    out_dir: str = "out_dataset",
    num_graphs: int = 50,
    n: int = 7,
    p: float = 0.35,
    seed: int = 123,
    require_connected: bool = False,
    resume: bool = True,
    flush_every: int = 1,
) -> Tuple[str, str]:
    """
    Creates (progressively):
      - out_dataset/ordering_dataset.jsonl   (append-only)
      - out_dataset/ordering_dataset.csv     (append-only)
      - out_dataset/{one_color,two_color,three_color,five_color,other_color}/graph_*.png

    Resume behavior:
      - If resume=True, it counts lines in JSONL and continues from that graph_id.
      - Graph generation is deterministic per graph_id so resumed runs produce identical remaining graphs.

    Progress:
      - Uses tqdm (with ETA) if installed: pip install tqdm
      - Otherwise prints lines with an estimated remaining time.
    """
    os.makedirs(out_dir, exist_ok=True)

    # Ensure all folders exist up front
    for folder in ["one_color", "two_color", "three_color", "five_color", "other_color"]:
        os.makedirs(os.path.join(out_dir, folder), exist_ok=True)

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
        "image_path",
        "max_degree", "avg_degree", "clique_number",
        "num_deg_ge_2", "num_deg_ge_3", "num_deg_ge_5",
        "num_isolated", "num_components", "component_sizes",
        "density",
    ]

    csv_exists = os.path.exists(csv_path) and os.path.getsize(csv_path) > 0

    # Append progressively, flush so you can inspect partial outputs while running.
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

            # Deterministic graph per gid (important for resume!)
            base_seed = (seed * 1_000_003 + gid) % (2**32)
            while True:
                G = nx.gnp_random_graph(n, p, seed=base_seed)
                G = nx.convert_node_labels_to_integers(G, first_label=0)
                if not require_connected or (n <= 1 or nx.is_connected(G)):
                    break
                base_seed = (base_seed + 1) % (2**32)

            rec = analyze_graph_bruteforce_all_orderings(G, graph_id=gid)

            attrs = compute_graph_attributes(G)
            img_path = save_graph_image(rec, out_root=out_dir, draw_dashed_constraints=True)

            # JSONL row now
            jf.write(json.dumps(asdict(rec)) + "\n")

            # CSV row now
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
                "image_path": img_path,

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
                iterator.set_postfix({"k": rec.best_k, "sec/graph": f"{dt:.2f}"})
            else:
                elapsed = time.perf_counter() - t0
                avg = elapsed / done
                remaining = (num_graphs - (gid + 1)) * avg
                print(
                    f"[{gid+1}/{num_graphs}] k={rec.best_k} sec/graph={dt:.2f} "
                    f"avg={avg:.2f}s est_left={remaining:.0f}s",
                    flush=True,
                )

    return jsonl_path, csv_path


if __name__ == "__main__":
    # For ETA/progress bar: pip install tqdm
    # For large runs, being resumable is essential.
    jsonl_path, csv_path = build_dataset(
        out_dir="out_dataset",
        num_graphs=1280,
        n=8,
        p=0.35,
        seed=123,
        require_connected=False,
        resume=True,
        flush_every=1,
    )
    print("Wrote (append/resume):", jsonl_path)
    print("Wrote (append/resume):", csv_path)
    print("Images in: out_dataset/one_color, two_color, three_color, five_color (and other_color if needed)")
