"""Measure wall-clock time and GPU memory per training epoch.

Reproduces the columns of Table 7. We run a small, fixed number of
training epochs and report:

    time_per_epoch_sec : averaged over the last (epochs - warmup) epochs
    gpu_mb             : peak allocated CUDA memory (MB) during the loop

Usage
-----
    python -m experiments.scalability --config configs/ogbn_arxiv.yaml \\
        --epochs 10 --warmup 2
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import yaml

from main import _build_train_cfg
from models.irl_gad import IRLGAD
from utils.data_utils import load_dataset, split_normal_indices
from utils.seed import set_seed


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--out",    default="./experiments/scalability_results.json")
    args = p.parse_args()

    with open(args.config) as f:
        yaml_cfg = yaml.safe_load(f)

    cfg = _build_train_cfg(yaml_cfg, {})
    set_seed(cfg.seed)
    device = torch.device(cfg.device)

    data = load_dataset(cfg.dataset).to(device)
    train_idx, _ = split_normal_indices(data, val_frac=0.10, seed=cfg.seed)
    train_idx = train_idx.to(device)

    cfg.model.in_dim = data.x.size(1)
    model = IRLGAD(cfg.model).to(device)
    optim = torch.optim.Adam(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
    )

    # GPU memory tracking
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    times = []
    for ep in range(args.epochs):
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.time()
        model.train()
        out = model.compute_loss(data.x, data.edge_index, node_mask=train_idx)
        out["loss"].backward()
        optim.step()
        optim.zero_grad()
        if device.type == "cuda":
            torch.cuda.synchronize()
        dt = time.time() - t0
        if ep >= args.warmup:
            times.append(dt)
        print(f"  ep {ep:2d}  {dt:.3f}s")

    avg = sum(times) / max(len(times), 1)
    peak_mb = (
        torch.cuda.max_memory_allocated(device) / (1024 * 1024)
        if device.type == "cuda" else 0.0
    )

    results = {
        "dataset":             cfg.dataset.name,
        "epochs_measured":     len(times),
        "warmup":              args.warmup,
        "time_per_epoch_sec":  round(avg, 3),
        "peak_gpu_mb":         round(peak_mb, 1),
        "device":              str(device),
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
