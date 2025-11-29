# graph-property-ml

A modular research repo for learning **graph properties** from **adjacency matrices**.

You provide a (possibly slow) graph algorithm that returns either:

- **Binary label**: `0/1` (e.g., *is bipartite?*)
- **Numeric target**: real/int (e.g., *max clique size*, *chromatic number*, *diameter*, etc.)

The repo generates graphs, labels them with your task function, trains a neural net on adjacency matrices, evaluates, and runs inference on new graphs.

---

## What this repo is (and isn’t)

### ✅ Good for

- Building a **fast surrogate** for a slow verifier/solver `f(G)` on the kinds of graphs you care about.
- Benchmarking which models generalize across **graph families**.
- Experimenting quickly: swap **task**, **graph generators**, **transforms**, **model**, **loss/metrics** via YAML.

### ⚠️ Not magic

- Learned surrogates can fail out-of-distribution (OOD). If you train on ER graphs and deploy on planar graphs, expect surprises.
- Some labels are NP-hard (clique / coloring). Exact labels require small graphs or expensive preprocessing.

---

## Key caveat: adjacency matrices are node-order dependent

Two isomorphic graphs can have different adjacency matrices due to node relabeling. If you train directly on `A`, your model can learn ordering artifacts instead of the graph property.

Use at least one of:

1) **Permutation augmentation**: randomly permute nodes during training (**included**: `random_node_permutation`)
2) **Canonicalization**: reorder nodes deterministically (**included**: `degree_sort_canonicalize`)
3) **Permutation-friendly models**: message passing + pooling (**included**: `dense_gcn`)

---

## Repository structure

```
configs/                 YAML experiment configs
scripts/                 CLI entrypoints (build dataset, train, eval, predict)
src/gpml/
  graphs/                graph family generators (ER, WS, BA, etc.)
  tasks/                 label functions (bipartite, max_clique, templates)
  data/                  dataset caching + transforms + collation
  models/                interchangeable models (CNN/MLP/GCN on adjacency)
  training/              trainer, losses, metrics
```

---

## Installation

### Option A: editable install (recommended)

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

### Option B: no install (quick hacking)

```bash
export PYTHONPATH=src
pip install -r requirements.txt
```

---

## Quickstart 1: Binary task (bipartite)

### 1) Build the dataset cache

```bash
python scripts/build_dataset.py --config configs/bipartite_adjcnn.yaml
```

This creates something like:

- `data/cache/bipartite_n32_50k.pt`

### 2) Train

```bash
python scripts/train.py --config configs/bipartite_adjcnn.yaml
```

Outputs go to:

- `runs/bipartite_adjcnn/`
  - `best.pt`, `last.pt`, `history.json`, TensorBoard logs

### 3) Evaluate best checkpoint on test split

```bash
python scripts/evaluate.py --config configs/bipartite_adjcnn.yaml --ckpt runs/bipartite_adjcnn/best.pt
```

### 4) Predict on a new adjacency matrix

Create an `NxN` `.npy` file:

```python
import numpy as np
A = np.zeros((32, 32), dtype=np.float32)
np.save("adj.npy", A)
```

Then run:

```bash
python scripts/predict.py --ckpt runs/bipartite_adjcnn/best.pt --model adj_cnn --adj adj.npy
```

---

## Quickstart 2: Numeric task (max clique size)

> Max clique is NP-hard. For exact labels, keep `n_nodes` small (the default config uses 18).

```bash
python scripts/build_dataset.py --config configs/maxclique_densegcn.yaml
python scripts/train.py        --config configs/maxclique_densegcn.yaml
python scripts/evaluate.py     --config configs/maxclique_densegcn.yaml --ckpt runs/maxclique_densegcn/best.pt
```

---

## TensorBoard

```bash
tensorboard --logdir runs
```

Typical signals:

- `train/loss`, `val/loss`
- binary: `val/accuracy`, `val/f1`
- regression: `val/mae`, `val/rmse`

---

## Config system (how to customize everything)

Configs live in `configs/*.yaml`. Key sections:

- `task`: which label function to use (binary/regression)
- `dataset`: how to generate graphs + where to cache
- `model`: which architecture to use
- `optim`: optimizer settings
- `trainer`: epochs, logging, checkpointing

Example (snippets):

```yaml
task:
  name: bipartite
  mode: binary

dataset:
  num_graphs: 50000
  n_nodes: 32
  sources:
    - family: erdos_renyi
      params: {p: 0.10}
      weight: 0.5
```

---

## Add your own task (plug in your slow function)

Create a new file: `src/gpml/tasks/my_task.py`

```python
import networkx as nx
from gpml.registry import TASKS

@TASKS.register("my_property")
class MyPropertyTask:
    name = "my_property"
    mode = "binary"   # or "regression"
    num_classes = None

    def __init__(self, **params):
        self.params = params

    def label(self, g: nx.Graph):
        # Call your slow verifier/solver here.
        # Return 0/1 for binary or float/int for regression.
        return 1
```

Make sure it gets imported so it registers:

- easiest: add to `src/gpml/tasks/__init__.py`, e.g. `from .my_task import MyPropertyTask`
- or import it in your script before building datasets/training

Then update your config:

```yaml
task:
  name: my_property
  mode: binary
```

---

## Add / modify graph sources (simulation)

Graph generator functions live in: `src/gpml/graphs/families.py`.

Add a new family:

```python
import networkx as nx
from gpml.registry import GRAPH_FAMILIES

@GRAPH_FAMILIES.register("my_family")
def my_family(n: int, seed: int | None = None, **params) -> nx.Graph:
    # return a networkx.Graph
    ...
```

Use it in the config:

```yaml
dataset:
  sources:
    - family: my_family
      params: {foo: 1.0}
      weight: 1.0
```

### “Download known families of graphs”

This starter repo focuses on simulation/generation. A clean way to add downloads is to:

- create `src/gpml/graphs/downloaders/` (or `src/gpml/data/external/`)
- implement code that returns a list of `networkx.Graph` objects
- keep the rest of the pipeline unchanged (graphs → adjacency → transforms → labels)

---

## Transforms (augmentation + canonicalization)

Configured under `dataset.transforms`.

Available transforms (starter):

- `random_node_permutation` (recommended)
- `degree_sort_canonicalize` (deterministic baseline)
- `pad_to_size` (forces fixed N)
- `to_float_tensor`

Example:

```yaml
transforms:
  - name: random_node_permutation
    params: {p: 1.0}
  - name: pad_to_size
    params: {size: 32}
  - name: to_float_tensor
    params: {}
```

---

## Models (swap without touching training code)

Implemented in `src/gpml/models/` and selected by:

```yaml
model:
  name: adj_cnn     # adj_mlp | dense_gcn
```

Notes:

- `adj_cnn`: fast baseline; treats adjacency as an image; not permutation-invariant → use augmentation.
- `adj_mlp`: simplest baseline (flatten); usually weaker than CNN/GCN.
- `dense_gcn`: message passing + pooling; typically more robust.

---

## Losses & metrics

Defined in:

- `src/gpml/training/losses.py`
- `src/gpml/training/metrics.py`

Defaults:

- **binary**: `BCEWithLogitsLoss`, metrics = accuracy, F1
- **regression**: `MSELoss`, metrics = MAE, RMSE

---

## Practical tips for real generalization

1) **Hold out a graph family**
   Train on ER+WS, test only on BA (or vice versa). This is the fastest sanity check for OOD robustness.
2) **Control density / size**
   For each family, vary parameters (`p`, `k`, `m`) to avoid learning “density” proxies.
3) **Balance binary datasets**
   Some properties are rare at certain densities. Use `balance_binary: true` (included for bipartite config).
4) **NP-hard targets**
   For clique/coloring:

- keep `n_nodes` small if you need exact labels,
- or label with heuristics and treat it as an approximate target.
