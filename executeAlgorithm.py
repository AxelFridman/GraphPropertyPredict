# run_2tnn_batch.py
#
# Batch driver:
#   - reads graphs from a CSV
#   - runs tn(G) <= 2 recognition
#   - times each call
#   - writes results.csv with timing info
#
# Required input columns:
#   graph_id, n, m, edges
#
# 'edges' must be a string representation of a list of pairs, e.g.:
#   "[[0, 1], [1, 2]]"
#

from __future__ import annotations
import ast
import csv
import time
from typing import List, Tuple

import pandas as pd

from finalAlgorithm import Graph, make_graph, tn_leq_2_by_components


# -----------------------------------------------------------
# CSV / parsing utilities
# -----------------------------------------------------------

def parse_edges_field(s: str) -> List[Tuple[int, int]]:
    """
    Parse an 'edges' field like '[[0, 1], [1, 2]]' into a list of (u, v) ints.
    Complexity: O(m), where m is the number of edges.
    """
    try:
        data = ast.literal_eval(s)
        edges: List[Tuple[int, int]] = [(int(u), int(v)) for (u, v) in data]
        return edges
    except Exception as e:
        raise ValueError(f"Could not parse edges field: {s!r}. Error: {e}")


def load_graphs_csv(path: str) -> List[Tuple[str, Graph, str, int, int]]:
    """
    Load graphs from CSV.
    Expects columns: graph_id, n, m, edges.
    Returns: list of (graph_id, Graph, raw_edges_str, n, m).
    Complexity: O(sum(n_i + m_i)) over all rows.
    """
    rows = []
    with open(path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        required = {'graph_id', 'n', 'm', 'edges'}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"CSV must contain columns: {required}")
        for row in reader:
            gid = str(row['graph_id'])
            n = int(row['n'])
            m = int(row['m'])
            raw_edges = row['edges']
            edges = parse_edges_field(raw_edges)
            g = make_graph(n, edges)
            # We trust 'm' column; if you want, you can assert len(g.edges) == m
            rows.append((gid, g, raw_edges, n, m))
    return rows


def process_graphs(
    input_csv: str,
    output_csv: str,
    n_limit_for_search: int = 9,
) -> pd.DataFrame:
    """
    For each graph in input_csv:
      - run tn_leq_2_by_components
      - measure elapsed time in milliseconds
      - store row: graph_id, graph (edges string), n, m, time_ms
    Write all rows to output_csv and return the DataFrame.
    Complexity: sum over graphs of recognition cost (exponential in size)
                plus linear I/O.
    """
    rows = load_graphs_csv(input_csv)
    out_rows = []
    for (gid, g, raw_edges, n, m) in rows:
        t0 = time.perf_counter()
        _ = tn_leq_2_by_components(g, n_limit_for_search=n_limit_for_search)
        t1 = time.perf_counter()
        ms = (t1 - t0) * 1000.0

        out_rows.append({
            "graph_id": gid,
            "graph": raw_edges,
            "n": n,
            "m": m,
            "time_ms": round(ms, 3),
        })

    df = pd.DataFrame(out_rows, columns=["graph_id", "graph", "n", "m", "time_ms"])
    df.to_csv(output_csv, index=False)
    return df


# -----------------------------------------------------------
# CLI entry point
# -----------------------------------------------------------

if __name__ == "__main__":
    # Adjust these paths as needed
    input_csv = r"C:\Users\af46294\Downloads\GraphPropertyPredict-main\GraphPropertyPredict-main\ordering_dataset.csv"    # your multi-graph CSV
    output_csv = "results.csv"  # results with timings

    df = process_graphs(input_csv, output_csv, n_limit_for_search=9)
    print("Wrote results to", output_csv)
    print(df)
