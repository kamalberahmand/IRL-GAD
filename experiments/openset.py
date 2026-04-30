"""Reproduce the open-set evaluation from Table 5.

Protocol
--------
1. Train on a benchmark with *structural* anomalies only.
2. Reuse the trained reward / scorer (no fine-tuning) on the same
   benchmark with the held-out anomaly type:
     - attribute   (features replaced)
     - contextual  (community-swapped features)
     - hybrid      (structural + attribute)
3. Report AUC-ROC for each anomaly type and the average drop vs.
   in-distribution performance.

Usage
-----
    python -m experiments.openset --config configs/yelpchi.yaml
"""
from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path

import torch
import yaml

from evaluate import EvalConfig, evaluate
from main import _build_train_cfg
from train import train
from utils.data_utils import DatasetSpec, load_dataset
from utils.metrics import compute_metrics
from models.irl_gad import IRLGAD, IRLGADConfig


HELD_OUT_TYPES = ["attribute", "contextual", "hybrid"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--out",    default="./experiments/openset_results.json")
    args = p.parse_args()

    with open(args.config) as f:
        yaml_cfg = yaml.safe_load(f)

    # 1. train on structural anomalies
    yaml_train = deepcopy(yaml_cfg)
    yaml_train["dataset"]["anomaly_type"] = "structural"
    cfg = _build_train_cfg(yaml_train, {})
    cfg.run_name = f"{cfg.dataset.name}_openset_train_struct_seed{cfg.seed}"
    train_out = train(cfg)
    ckpt = train_out["ckpt"]
    in_dist_auc = train_out["metrics"]["auc_roc"]
    print(f"in-distribution AUC-ROC = {in_dist_auc*100:.2f}%")

    # 2. evaluate on each held-out anomaly type using the same checkpoint
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    blob = torch.load(ckpt, map_location=device)
    model_cfg = IRLGADConfig(**blob["model_cfg"])
    model = IRLGAD(model_cfg).to(device)
    state = {k: v.to(device) for k, v in blob["state_dict"].items()}
    model.load_state_dict(state, strict=False)
    model.eval()

    results: dict = {"in_distribution_auc_roc": in_dist_auc, "held_out": {}}
    for atype in HELD_OUT_TYPES:
        spec = DatasetSpec(
            name=cfg.dataset.name,
            root=cfg.dataset.root,
            anomaly_ratio=cfg.dataset.anomaly_ratio,
            anomaly_type=atype,
            # use a different seed so the injected anomalies don't overlap
            # with the training mask
            seed=cfg.dataset.seed + 1000,
        )
        data = load_dataset(spec).to(device)
        with torch.no_grad():
            scored = model.score(data.x, data.edge_index)
            metrics = compute_metrics(scored["score"].cpu(), data.y_anom.cpu())
        results["held_out"][atype] = metrics.as_dict()
        print(f"  [{atype}] {metrics}")

    # 3. compute the average drop
    drops = [
        in_dist_auc - results["held_out"][a]["auc_roc"]
        for a in HELD_OUT_TYPES
    ]
    results["mean_drop_pp"] = float(sum(drops) / len(drops) * 100)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"open-set results -> {args.out}")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
