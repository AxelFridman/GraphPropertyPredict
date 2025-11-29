# graph-property-ml

Train neural nets on adjacency matrices to approximate expensive graph-property algorithms.

## Install
```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

## Quickstart (binary: bipartite)
1) Build a dataset cache:
```bash
python scripts/build_dataset.py --config configs/bipartite_adjcnn.yaml
```

2) Train:
```bash
python scripts/train.py --config configs/bipartite_adjcnn.yaml
```

3) Evaluate best checkpoint:
```bash
python scripts/evaluate.py --config configs/bipartite_adjcnn.yaml --ckpt runs/bipartite_adjcnn/best.pt
```

4) Predict on a new adjacency matrix stored as .npy:
```bash
python scripts/predict.py --ckpt runs/bipartite_adjcnn/best.pt --model adj_cnn --adj path/to/adj.npy
```

## Notes / design choices
- Adjacency matrices are **node-order dependent**. Use:
  - `RandomNodePermutation` augmentation, and/or
  - `DegreeSortCanonicalize`, and/or
  - a permutation-friendlier model (the provided `DenseGCN` is closer).
- For NP-hard labels (max clique, coloring, etc.), generate *small* graphs if you need exact labels.
- Swap tasks by changing `task.name` in config or by adding a new file in `src/gpml/tasks/` and registering it.

## TensorBoard
```bash
tensorboard --logdir runs
```
