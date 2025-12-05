# graph_2tnn.py
#
# Core data structures and algorithms to check whether a graph
# admits a k-track nearest-neighbor layout for k in {1,2}.
#
# Every function has a short complexity comment in the docstring.

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Dict, Set
import itertools


# -----------------------------------------------------------
# Basic graph representation
# -----------------------------------------------------------

@dataclass
class Graph:
    n: int                          # number of vertices (0..n-1)
    edges: List[Tuple[int, int]]    # list of undirected edges (u < v)
    adj: List[Set[int]]             # adjacency sets for fast lookup


def make_graph(n: int, edges: List[Tuple[int, int]]) -> Graph:
    """
    Build an undirected simple graph structure from n and edge list.
    Complexity: O(n + m), where m = len(edges).
    """
    adj: List[Set[int]] = [set() for _ in range(n)]
    norm_edges: List[Tuple[int, int]] = []
    for (u, v) in edges:
        if u == v:  # ignore self-loops if present
            continue
        a, b = (u, v) if u < v else (v, u)
        if b not in adj[a]:
            adj[a].add(b)
            adj[b].add(a)
            norm_edges.append((a, b))
    return Graph(n=n, edges=norm_edges, adj=adj)


def num_edges(g: Graph) -> int:
    """
    Return |E|.
    Complexity: O(1), using stored edge list.
    """
    return len(g.edges)


def degrees(g: Graph) -> List[int]:
    """
    Return list of degrees for all vertices.
    Complexity: O(n).
    """
    return [len(g.adj[v]) for v in range(g.n)]


def max_degree(g: Graph) -> int:
    """
    Return maximum degree Δ(G).
    Complexity: O(n).
    """
    return max(degrees(g)) if g.n > 0 else 0


def connected_components(g: Graph) -> List[List[int]]:
    """
    Return list of connected components, each as a list of vertices.
    BFS-based.
    Complexity: O(n + m).
    """
    seen = [False] * g.n
    comps: List[List[int]] = []
    for s in range(g.n):
        if not seen[s]:
            q = [s]
            seen[s] = True
            comp = [s]
            for u in q:
                for w in g.adj[u]:
                    if not seen[w]:
                        seen[w] = True
                        q.append(w)
                        comp.append(w)
            comps.append(comp)
    return comps


def induced_subgraph(g: Graph, verts: List[int]) -> Graph:
    """
    Return the induced subgraph on 'verts', relabeled to 0..k-1.
    Complexity: O(k + m_sub), where k = |verts| and m_sub is edges inside verts.
    """
    mapping = {v: i for i, v in enumerate(verts)}
    sub_edges: List[Tuple[int, int]] = []
    for v in verts:
        for w in g.adj[v]:
            if v < w and w in mapping:
                sub_edges.append((mapping[v], mapping[w]))
    return make_graph(len(verts), sub_edges)


def has_k4(g: Graph) -> bool:
    """
    Naively detect a K4 by enumerating all 4-vertex subsets.
    Complexity: O(n^4) in worst case, OK for small graphs.
    """
    n = g.n
    if n < 4:
        return False
    adj = g.adj
    for a, b, c, d in itertools.combinations(range(n), 4):
        if (b in adj[a] and c in adj[a] and d in adj[a] and
            c in adj[b] and d in adj[b] and
            d in adj[c]):
            return True
    return False


# -----------------------------------------------------------
# Nearest-neighbor host graph for a given layout
# -----------------------------------------------------------

def legal_host_edges_for_layout(
    n: int,
    edges: List[Tuple[int, int]],
    order: List[int],
    tracks: List[int],
    k: int,
) -> Set[Tuple[int, int]]:
    """
    Compute the set of legal host edges under a k-track layout:
      - order: a permutation of vertices, position 0..n-1
      - tracks[v] in {1, ..., k}
    Legal edges connect pred/succ along each track and pred/succ on every
    other track (nearest cross-track neighbors).
    Returns a set of edges (u, v) with u < v.
    Complexity: O(n k + m_host), where m_host is number of legal host edges.
    """
    # position[v] = index in order
    pos = [-1] * n
    for i, v in enumerate(order):
        pos[v] = i

    # vertices per track, sorted by position
    track_lists: Dict[int, List[int]] = {t: [] for t in range(1, k + 1)}
    for v in range(n):
        track_lists[tracks[v]].append(v)
    for t in range(1, k + 1):
        track_lists[t].sort(key=lambda v: pos[v])

    legal: Set[Tuple[int, int]] = set()

    def add_edge(u: int, v: int) -> None:
        a, b = (u, v) if u < v else (v, u)
        legal.add((a, b))

    # Same-track pred/succ
    for t in range(1, k + 1):
        lst = track_lists[t]
        for i, v in enumerate(lst):
            if i > 0:
                add_edge(lst[i - 1], v)       # pred_t(v) -- v
            if i < len(lst) - 1:
                add_edge(v, lst[i + 1])       # v -- succ_t(v)

    # Cross-track pred/succ
    if k >= 2:
        for t in range(1, k + 1):
            lst = track_lists[t]
            for v in lst:
                pv = pos[v]
                for s in range(1, k + 1):
                    if s == t:
                        continue
                    other = track_lists[s]
                    # binary search in other for nearest neighbors in position
                    lo, hi = 0, len(other)
                    while lo < hi:
                        mid = (lo + hi) // 2
                        if pos[other[mid]] < pv:
                            lo = mid + 1
                        else:
                            hi = mid
                    # left neighbor: lo-1; right neighbor: lo
                    if lo - 1 >= 0:
                        add_edge(v, other[lo - 1])  # pred_s(v)
                    if lo < len(other):
                        add_edge(v, other[lo])      # succ_s(v)

    return legal


def is_layout_nearest_neighbor(
    n: int,
    edges: List[Tuple[int, int]],
    order: List[int],
    tracks: List[int],
    k: int,
) -> bool:
    """
    Check if (order, tracks) is a k-track nearest-neighbor layout for all edges.
    Complexity: O(n k + m), where m = |E|.
    """
    legal = legal_host_edges_for_layout(n, edges, order, tracks, k)
    for (u, v) in edges:
        a, b = (u, v) if u < v else (v, u)
        if (a, b) not in legal:
            return False
    return True


# -----------------------------------------------------------
# Exhaustive search recognition for k in {1, 2}
# -----------------------------------------------------------

def tn_leq_k_by_search(g: Graph, k: int, n_limit_for_search: int = 9) -> bool:
    """
    Decide if nearest-neighbor track number tn(G) <= k (k in {1,2})
    by exhaustive search over:
      - all permutations of vertices (global orders)
      - all track assignments (k^n possibilities) with simple symmetry breaking.
    Complexity (worst-case): O(n! * k^n * (n + m)), exponential in n.
    Intended only for small graphs; n_limit_for_search bounds the size.
    """
    n = g.n
    m = num_edges(g)

    # Quick necessary condition for k = 1: Δ(G) <= 2
    if k == 1 and max_degree(g) > 2:
        return False

    if n == 0 or n == 1:
        return True

    if n > n_limit_for_search:
        # Avoid blow-up for large graphs: treat as unknown/False.
        return False

    verts = list(range(n))
    edge_list = g.edges

    # Enumerate all vertex orders
    for order in itertools.permutations(verts):
        if k == 1:
            tracks = [1] * n
            if is_layout_nearest_neighbor(n, edge_list, list(order), tracks, k=1):
                return True
        else:
            # k == 2: brute-force all 2^n assignments via bitmask
            # Symmetry-breaking: enforce that the first vertex in the order
            # is on track 1 to cut the search roughly in half.
            for mask in range(1 << n):
                tracks = [1 + ((mask >> v) & 1) for v in range(n)]
                if tracks[order[0]] != 1:
                    continue
                if is_layout_nearest_neighbor(n, edge_list, list(order), tracks, k=2):
                    return True

    return False


def tn_leq_2_by_components(g: Graph, n_limit_for_search: int = 9) -> bool:
    """
    Decide if tn(G) <= 2 using connected components:
       tn(G) = max_i tn(G_i).
    For each component, apply fast filters and then exhaustive search for
    k = 1 and k = 2.
    Complexity: sum over components of the exponential search on each,
    plus linear-time filters. Exponential in component size.
    """
    if g.n == 0:
        return True

    comps = connected_components(g)
    for comp in comps:
        H = induced_subgraph(g, comp)
        nH = H.n
        mH = num_edges(H)

        # Small components are always representable with <= 2 tracks
        if nH <= 3:
            continue

        # Edge bound for k=2: m <= 2n - 3
        if mH > 2 * nH - 3:
            return False

        # Degree bound: Δ(G) <= 4 for k=2
        if max_degree(H) > 4:
            return False

        # K4 forces k >= 3
        if has_k4(H):
            return False

        # Try k=1, then k=2
        if tn_leq_k_by_search(H, k=1, n_limit_for_search=n_limit_for_search):
            continue
        if tn_leq_k_by_search(H, k=2, n_limit_for_search=n_limit_for_search):
            continue

        # No layout with k <= 2 found for this component
        return False

    # All components accept
    return True
