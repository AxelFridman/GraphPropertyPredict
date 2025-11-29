import argparse
from gpml.utils.config import load_config
from gpml.utils.seed import seed_all
from gpml.data.dataset import DatasetConfig, build_or_load_dataset
import gpml.graphs  # registers families
import gpml.tasks   # registers tasks

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()

    cfg = load_config(args.config)
    seed_all(cfg["seed"])

    dcfg = DatasetConfig(**cfg["dataset"])
    task = cfg["task"]
    ds = build_or_load_dataset(
      cfg=dcfg,
      task_name=task["name"],
      task_params=task.get("params"),
      seed=cfg["seed"],
    )
    print(f"Dataset ready: {len(ds)} graphs. Cache: {dcfg.cache_path}")

if __name__ == "__main__":
    main()
