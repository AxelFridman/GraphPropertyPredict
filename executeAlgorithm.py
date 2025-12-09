from __future__ import annotations
import ast
import csv
import time
import logging  ### NEW
from typing import List, Tuple

import pandas as pd

from finalAlgorithm import Graph, make_graph, tn_leq_2_by_components

# -----------------------------------------------------------
# Logging config
# -----------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,  # change to DEBUG if you want even more detail
    format="%(asctime)s [%(levelname)s] %(message)s"
)

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

    NEW BEHAVIOR:
      - Logs timing for each row.
      - Writes each row to `output_csv` immediately (progress is saved
        continuously), instead of only at the end.
    """
    rows = load_graphs_csv(input_csv)
    total = len(rows)
    logging.info("Loaded %d graphs from %s", total, input_csv)

    fieldnames = ["graph_id", "graph", "n", "m", "time_ms", "result"]
    out_rows = []

    # Open output file once, write header, then write & flush every row
    with open(output_csv, "w", newline="", encoding="utf-8") as f_out:
        writer = csv.DictWriter(f_out, fieldnames=fieldnames)
        writer.writeheader()

        for idx, (gid, g, raw_edges, n, m) in enumerate(rows, start=1):
            t0 = time.perf_counter()
            res = tn_leq_2_by_components(g, n_limit_for_search=n_limit_for_search)
            t1 = time.perf_counter()
            ms = (t1 - t0) * 1000.0
            ms_rounded = round(ms, 3)

            row_dict = {
                "graph_id": gid,
                "graph": raw_edges,
                "n": n,
                "m": m,
                "time_ms": ms_rounded,
                "result": str(int(res)),
            }

            # Save row in memory (for the final DataFrame)
            out_rows.append(row_dict)

            # Immediately append to CSV and flush so progress is persisted
            writer.writerow(row_dict)
            f_out.flush()

            # Log progress for this row
            logging.info(
                "Processed row %d/%d: graph_id=%s, n=%d, m=%d, result=%s, time=%.3f ms",
                idx, total, gid, n, m, int(res), ms_rounded
            )


    df = pd.DataFrame(out_rows, columns=fieldnames)
    logging.info("Finished processing all graphs. Results written to %s", output_csv)
    return df

def make_complete_binary_tree(depth: int) -> Tuple[int, List[Tuple[int, int]]]:
    """
    Build a complete binary tree of given depth.
    Depth 1 => 1 node
    Depth 2 => 3 nodes
    Depth 3 => 7 nodes
    etc.
    Returns: (n, edges)
    """
    n = 2**depth - 1  # number of nodes
    edges = []
    for i in range(n):
        left = 2*i + 1
        right = 2*i + 2
        if left < n:
            edges.append((i, left))
        if right < n:
            edges.append((i, right))
    return n, edges

if __name__ == "__main__":
    # Adjust these paths as needed
    depths = [2, 3, 4,5]
    rows = []

    for d in depths:
        n, edges = make_complete_binary_tree(d)
        m = len(edges)
        g = make_graph(n, edges)

        # run your algorithm
        t0 = time.perf_counter()
        res = tn_leq_2_by_components(g, n_limit_for_search=9)
        t1 = time.perf_counter()
        ms = round((t1 - t0) * 1000.0, 3)

        rows.append({
            "graph_id": f"binary_depth_{d}",
            "graph": str(edges),
            "n": n,
            "m": m,
            "time_ms": ms,
            "result": int(res),
        })

        print(f"Depth {d} → n={n}, m={m}, result={res}, time={ms} ms")

    # convert to DataFrame
    df = pd.DataFrame(rows)
    print("\nGenerated test dataframe:")
    print(df)


    # input_csv = r"C:\Users\fridm\Desktop\GraphPropertyPredict\combinedData.csv"    # your multi-graph CSV
    # output_csv = "results.csv"  # results with timings

    # df = process_graphs(input_csv, output_csv, n_limit_for_search=9)
    # print("Wrote results to", output_csv)
    # print(df)
