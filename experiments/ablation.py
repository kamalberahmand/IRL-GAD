"""Reproduce the component ablation from Table 6.

Each variant retrains from scratch on YelpChi (or whichever dataset is
selected) and reports the final AUC-ROC. The full table from the paper
covers seven variants:

    full              : all reward heads + IRL scoring
    no_R_tmp          : drop temporal head
    no_R_sem          : drop semantic head
    no_R_str          : drop structural head
    GAT + recon       : same backbone, replace IRL score with reconstruction
    GAT + contrast    : same backbone, replace IRL score with contrastive
    single_reward     : fuse all heads into one (no decomposition)

Usage
-----
    python -m experiments.ablation --config configs/yelpchi.yaml
"""
from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path

import yaml

from main import _build_train_cfg
from train import train


VARIANTS = [
    ("full",            {}),
    ("no_R_tmp",        {"model.use_tmp": False}),
    ("no_R_sem",        {"model.use_sem": False}),
    ("no_R_str",        {"model.use_str": False}),
    ("recon_score",     {"model.score_mode": "reconstruction"}),
    ("contrast_score",  {"model.score_mode": "contrastive"}),
    ("single_reward",   {"model.use_sem": False, "model.use_tmp": False}),
]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--out",    default="./experiments/ablation_results.json")
    args = p.parse_args()

    with open(args.config) as f:
        yaml_cfg = yaml.safe_load(f)

    results = {}
    for name, overrides in VARIANTS:
        print(f"\n========== ablation variant: {name} ==========")
        cfg = _build_train_cfg(deepcopy(yaml_cfg), deepcopy(overrides))
        cfg.run_name = f"{cfg.dataset.name}_ablate_{name}_seed{cfg.seed}"
        out = train(cfg)
        results[name] = out["metrics"]

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nablation results -> {args.out}")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
