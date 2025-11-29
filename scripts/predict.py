import argparse
import numpy as np
import torch

from gpml.registry import MODELS
from gpml.utils.device import pick_device

import gpml.models  # register models

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--adj", required=True, help="Path to adjacency .npy (NxN)")
    ap.add_argument("--model", required=True, help="Model name: adj_cnn | adj_mlp | dense_gcn")
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    device = pick_device(args.device)

    adj = np.load(args.adj)
    adj = torch.from_numpy(adj).float().unsqueeze(0)  # [1,N,N]

    blob = torch.load(args.ckpt, map_location="cpu")
    model_cls = MODELS.get(args.model)
    model = model_cls(out_dim=1).to(device)
    model.load_state_dict(blob["model"])
    model.eval()

    with torch.no_grad():
        logit = model(adj.to(device)).item()

    print(f"logit/pred: {logit:.6f}")
    print(f"sigmoid(prob=1): {1/(1+np.exp(-logit)):.6f}")

if __name__ == "__main__":
    main()
