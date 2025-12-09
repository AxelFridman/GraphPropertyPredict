
# Core data structures and algorithms to check whether a graph
# admits a k-track nearest-neighbor layout for k in {1,2}.
#
# Improved version:
#   - adds fast recognition + explicit (verified) constructions for known
#     YES-classes (paths/linear forests, cycles, path+one-extra-edge,
#     and bounded-degree caterpillars),
#   - uses a fast K4 detector once Δ<=4 is known,
#   - preserves the overall structure: component cuts -> cheap YES -> search.
#

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Dict, Set, Optional
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
        if u == v:
            continue
        a, b = (u, v) if u < v else (v, u)
        if b not in adj[a]:
            adj[a].add(b)
            adj[b].add(a)
            norm_edges.append((a, b))
    return Graph(n=n, edges=norm_edges, adj=adj)


def num_edges(g: Graph) -> int:
    """Return |E|.  Complexity: O(1)."""
    return len(g.edges)


def degrees(g: Graph) -> List[int]:
    """Return list of degrees for all vertices.  Complexity: O(n)."""
    return [len(g.adj[v]) for v in range(g.n)]


def max_degree(g: Graph) -> int:
    """Return maximum degree Δ(G).  Complexity: O(n)."""
    return max(degrees(g)) if g.n > 0 else 0


def connected_components(g: Graph) -> List[List[int]]:
    """
    Return list of connected components, each as a list of vertices.
    BFS-based.  Complexity: O(n + m).
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
    Detect a K4. If Δ<=4 (as in our k<=2 pipeline), this runs in O(n).
    Falls back to O(n^4) only if Δ is large (still OK for small graphs).

    Complexity:
      - If Δ(G) <= 4: O(n) (each vertex checks <=4 choose 3 neighbor-triples)
      - Worst case: O(n^4)
    """
    n = g.n
    if n < 4:
        return False

    degs = degrees(g)
    if max(degs) <= 4:
        adj = g.adj
        for v in range(n):
            N = list(adj[v])
            if len(N) < 3:
                continue
            for a, b, c in itertools.combinations(N, 3):
                if (b in adj[a]) and (c in adj[a]) and (c in adj[b]):
                    return True
        return False

    # fallback
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

    Complexity: O(n k + m_host).
    """
    pos = [-1] * n
    for i, v in enumerate(order):
        pos[v] = i

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
                add_edge(lst[i - 1], v)
            if i < len(lst) - 1:
                add_edge(v, lst[i + 1])

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
                    # lower_bound by position
                    lo, hi = 0, len(other)
                    while lo < hi:
                        mid = (lo + hi) // 2
                        if pos[other[mid]] < pv:
                            lo = mid + 1
                        else:
                            hi = mid
                    if lo - 1 >= 0:
                        add_edge(v, other[lo - 1])
                    if lo < len(other):
                        add_edge(v, other[lo])

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
    Complexity: O(n k + m).
    """
    legal = legal_host_edges_for_layout(n, edges, order, tracks, k)
    for (u, v) in edges:
        a, b = (u, v) if u < v else (v, u)
        if (a, b) not in legal:
            return False
    return True


# -----------------------------------------------------------
# Known YES-classes + explicit constructions (verified)
# -----------------------------------------------------------

def _path_order_if_path(g: Graph) -> Optional[List[int]]:
    """
    If g is a connected path (or single vertex), return its vertex order along the path.
    Otherwise return None.

    Complexity: O(n + m).
    """
    n = g.n
    if n == 0:
        return []
    m = num_edges(g)
    deg = degrees(g)

    if n == 1:
        return [0]
    if m != n - 1:
        return None
    if max(deg) > 2:
        return None

    ends = [v for v in range(n) if deg[v] == 1]
    if len(ends) != 2:
        return None

    start = ends[0]
    order: List[int] = []
    prev = -1
    cur = start
    while True:
        order.append(cur)
        nxts = [w for w in g.adj[cur] if w != prev]
        if not nxts:
            break
        prev, cur = cur, nxts[0]

    if len(order) != n:
        return None
    return order


def _cycle_order_if_cycle(g: Graph) -> Optional[List[int]]:
    """
    If g is a connected cycle C_n (n>=3), return the cyclic order list [v0,v1,...,v_{n-1}]
    in global order. Else None.

    Complexity: O(n + m).
    """
    n = g.n
    if n < 3:
        return None
    if num_edges(g) != n:
        return None
    deg = degrees(g)
    if any(d != 2 for d in deg):
        return None

    start = 0
    order = [start]
    prev = -1
    cur = start
    while True:
        neigh = list(g.adj[cur])
        nxt = neigh[0] if neigh[0] != prev else neigh[1]
        if nxt == start:
            break
        order.append(nxt)
        prev, cur = cur, nxt
        if len(order) > n:
            return None

    if len(order) != n:
        return None
    return order


def _strip_leaves_core(g: Graph) -> Set[int]:
    """
    Return the set of vertices remaining after repeatedly stripping degree<=1 vertices
    (the 2-core). For a connected unicyclic graph, this is exactly its unique cycle.

    Complexity: O(n + m).
    """
    n = g.n
    deg = [len(g.adj[v]) for v in range(n)]
    alive = [True] * n
    q = [v for v in range(n) if deg[v] <= 1]
    for v in q:
        alive[v] = False
    head = 0
    while head < len(q):
        v = q[head]
        head += 1
        for w in g.adj[v]:
            if alive[w]:
                deg[w] -= 1
                if deg[w] <= 1:
                    alive[w] = False
                    q.append(w)
    return {v for v in range(n) if alive[v]}


def _try_layout_cycle(g: Graph) -> bool:
    """
    Construct the standard 2-track layout for a cycle as in the writeup:
      order: v0 < v1 < ... < v_{n-1}
      track1: {v0, v_{n-1}}, track2: others

    Complexity: O(n + m).
    """
    cyc = _cycle_order_if_cycle(g)
    if cyc is None:
        return False
    n = g.n
    order = cyc[:]  # already a permutation
    tracks = [2] * n
    tracks[order[0]] = 1
    tracks[order[-1]] = 1
    return is_layout_nearest_neighbor(n, g.edges, order, tracks, k=2)


def _try_layout_path_plus_one_edge(g: Graph) -> bool:
    """
    Try to recognize "path + one extra edge" and build the canonical 2-track layout:
      - Find the unique chord e* such that removing it yields a path.
      - Order vertices along that path.
      - Put chord endpoints on track 2, all others on track 1.

    Complexity: O(n + m). (We only test cycle-edges as chord candidates.)
    """
    n = g.n
    if n <= 2:
        return True
    if num_edges(g) != n:  # connected unicyclic prerequisite for this class
        return False

    deg0 = degrees(g)
    if max(deg0) > 4:
        return False

    core = _strip_leaves_core(g)
    if len(core) < 3:
        return False  # should not happen for simple connected unicyclic with n>=3

    core_edges = [(u, v) for (u, v) in g.edges if u in core and v in core]
    if not core_edges:
        return False

    # Candidate chord = a cycle-edge whose removal makes max degree <= 2 (i.e., a path).
    chord: Optional[Tuple[int, int]] = None
    for (a, b) in core_edges:
        deg = deg0[:]
        deg[a] -= 1
        deg[b] -= 1
        if max(deg) <= 2:
            # Removing this core-edge keeps the graph connected (unicyclic property),
            # and max degree <= 2 implies the resulting tree is a path.
            chord = (a, b)
            break

    if chord is None:
        return False

    # Build adjacency of the path tree = all edges except chord
    adj_path: List[Set[int]] = [set(neis) for neis in g.adj]
    a, b = chord
    if b in adj_path[a]:
        adj_path[a].remove(b)
        adj_path[b].remove(a)

    deg_path = [len(adj_path[v]) for v in range(n)]
    ends = [v for v in range(n) if deg_path[v] == 1]
    if n > 1 and len(ends) != 2:
        return False

    # Extract the path order by walking from an endpoint
    start = ends[0] if n > 1 else 0
    order: List[int] = []
    prev = -1
    cur = start
    while True:
        order.append(cur)
        nxts = [w for w in adj_path[cur] if w != prev]
        if not nxts:
            break
        prev, cur = cur, nxts[0]

    if len(order) != n:
        return False

    # Canonical track assignment
    tracks = [1] * n
    tracks[a] = 2
    tracks[b] = 2

    return is_layout_nearest_neighbor(n, g.edges, order, tracks, k=2)


def _is_tree(g: Graph) -> bool:
    """Connected component g is a tree iff m = n-1. Complexity: O(1) with stored m."""
    return g.n <= 1 or num_edges(g) == g.n - 1


def _is_caterpillar_tree(g: Graph) -> bool:
    """
    Caterpillar test (connected tree): removing all original leaves yields a path (or empty/single).
    Equivalent: induced subgraph on vertices with degree >= 2 is a path (or size <= 1).

    Complexity: O(n + m).
    """
    if not _is_tree(g):
        return False
    n = g.n
    if n <= 2:
        return True

    deg = degrees(g)
    core = [v for v in range(n) if deg[v] >= 2]
    if len(core) <= 1:
        return True

    core_set = set(core)
    core_adj: Dict[int, List[int]] = {v: [w for w in g.adj[v] if w in core_set] for v in core}
    # core must be connected and have max degree <=2 and |E_core| = |V_core|-1
    max_core_deg = max(len(core_adj[v]) for v in core)
    if max_core_deg > 2:
        return False

    e_core2 = sum(len(core_adj[v]) for v in core)  # 2*|E_core|
    if e_core2 != 2 * (len(core) - 1):
        return False

    # connected check on core
    start = core[0]
    seen = {start}
    stack = [start]
    while stack:
        v = stack.pop()
        for w in core_adj[v]:
            if w not in seen:
                seen.add(w)
                stack.append(w)
    return len(seen) == len(core)


def _try_layout_caterpillar(g: Graph) -> bool:
    """
    Construct and verify a 2-track layout for a caterpillar with Δ<=4:
      - Put spine (core) on track 1 in path order.
      - Put leaves on track 2, placed near their spine neighbor (some before, some after).

    Complexity: O(n + m).
    """
    if not _is_caterpillar_tree(g):
        return False
    if max_degree(g) > 4:
        return False

    n = g.n
    if n <= 1:
        return True

    deg = degrees(g)

    # Identify spine/core
    spine = [v for v in range(n) if deg[v] >= 2]
    spine_set = set(spine)

    tracks = [2] * n
    if len(spine) == 0:
        # connected implies n<=2 (already handled), but be safe
        tracks = [1] * n
        order = list(range(n))
        return is_layout_nearest_neighbor(n, g.edges, order, tracks, k=1)

    for v in spine:
        tracks[v] = 1

    # If spine size 1: it's a star-like caterpillar, easy: put center first then all leaves.
    if len(spine) == 1:
        center = spine[0]
        leaves = [v for v in range(n) if v != center]
        order = [center] + leaves
        return is_layout_nearest_neighbor(n, g.edges, order, tracks, k=2)

    # Build the spine path order
    spine_graph = make_graph(
        len(spine),
        [(spine.index(u), spine.index(v)) for (u, v) in g.edges if u in spine_set and v in spine_set],
    )
    spine_order_idx = _path_order_if_path(spine_graph)
    if spine_order_idx is None:
        return False
    spine_order = [spine[i] for i in spine_order_idx]

    # Build global order by interleaving leaves around spine vertices
    used = set()
    order: List[int] = []
    for s in spine_order:
        # Leaves are degree-1 neighbors of s
        leaves = [u for u in g.adj[s] if deg[u] == 1]
        leaves.sort()
        if len(leaves) >= 2:
            # put one leaf immediately before s, one immediately after
            order.append(leaves[0]); used.add(leaves[0])
            order.append(s); used.add(s)
            order.append(leaves[1]); used.add(leaves[1])
            for extra in leaves[2:]:
                # shouldn't happen under Δ<=4, but keep deterministic
                order.append(extra); used.add(extra)
        elif len(leaves) == 1:
            order.append(s); used.add(s)
            order.append(leaves[0]); used.add(leaves[0])
        else:
            order.append(s); used.add(s)

    # Append any leftover vertices (defensive; should be none)
    for v in range(n):
        if v not in used:
            order.append(v)

    if len(order) != n or len(set(order)) != n:
        return False
    return is_layout_nearest_neighbor(n, g.edges, order, tracks, k=2)


def _try_fast_yes_component(g: Graph) -> bool:
    """
    Try fast YES checks/constructions for known classes (for tn<=2).
    Returns True if accepted, False otherwise.

    Complexity: O(n + m).
    """
    # k=1: connected linear forest means "path (or isolate)"
    path_order = _path_order_if_path(g)
    if path_order is not None:
        # 1-track layout: order along the path, all on track 1
        tracks = [1] * g.n
        if is_layout_nearest_neighbor(g.n, g.edges, path_order, tracks, k=1):
            return True

    # cycle
    if _try_layout_cycle(g):
        return True

    # path + one extra edge (unicyclic with special chord)
    if _try_layout_path_plus_one_edge(g):
        return True

    # caterpillar (tree) with Δ<=4
    if _try_layout_caterpillar(g):
        return True

    return False


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

    # Quick necessary condition for k = 1: Δ(G) <= 2
    if k == 1 and max_degree(g) > 2:
        return False

    if n <= 1:
        return True

    # Fast known-class acceptance before brute force (helps even for small n)
    if k == 1:
        # If it's a path, accept immediately (path_order + verify)
        path_order = _path_order_if_path(g)
        if path_order is not None:
            tracks = [1] * n
            return is_layout_nearest_neighbor(n, g.edges, path_order, tracks, k=1)

    if k == 2 and _try_fast_yes_component(g):
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

    For each component:
      - apply fast filters (cuts),
      - apply cheap YES recognizers + verified constructions for known families,
      - finally fall back to exhaustive search (size-bounded).

    Complexity: linear-time filters + (optional) exponential search on small components.
    """
    if g.n == 0:
        return True

    comps = connected_components(g)
    for comp in comps:
        H = induced_subgraph(g, comp)
        nH = H.n
        mH = num_edges(H)

        # Cut 0: tiny graphs are always <=2
        if nH <= 3:
            continue

        # Cut 1: (your existing density bound for k=2)
        if mH > 2 * nH - 3:
            return False

        # Cut 2: Δ(G) <= 4 for k=2 (your existing filter)
        if max_degree(H) > 4:
            return False

        # Cut 3: K4 forces k >= 3
        if has_k4(H):
            return False

        # Cheap YES: known classes + verified constructions (works for large n too)
        if _try_fast_yes_component(H):
            continue

        # Fallback: try k=1, then k=2 by bounded exhaustive search
        if tn_leq_k_by_search(H, k=1, n_limit_for_search=n_limit_for_search):
            continue
        if tn_leq_k_by_search(H, k=2, n_limit_for_search=n_limit_for_search):
            continue

        return False

    return True
