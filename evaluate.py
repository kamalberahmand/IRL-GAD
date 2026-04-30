"""Standalone evaluation: load a checkpoint and report metrics."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch

from models.irl_gad import IRLGAD, IRLGADConfig
from utils.data_utils import DatasetSpec, load_dataset
from utils.metrics import compute_metrics
from utils.seed import set_seed


@dataclass
class EvalConfig:
    dataset: DatasetSpec
    ckpt: str                              # path to a `best.pt`
    seed: int = 0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    score_mode: Optional[str] = None       # override saved cfg's score_mode if given


def evaluate(cfg: EvalConfig) -> dict:
    set_seed(cfg.seed)
    device = torch.device(cfg.device)

    data = load_dataset(cfg.dataset).to(device)

    blob = torch.load(cfg.ckpt, map_location=device)
    model_cfg = IRLGADConfig(**blob["model_cfg"])
    if cfg.score_mode is not None:
        model_cfg.score_mode = cfg.score_mode

    model = IRLGAD(model_cfg).to(device)
    state = {k: v.to(device) for k, v in blob["state_dict"].items()}
    model.load_state_dict(state, strict=False)
    model.eval()

    with torch.no_grad():
        scored = model.score(data.x, data.edge_index)
        scores = scored["score"].cpu()
        metrics = compute_metrics(scores, data.y_anom.cpu())

    print(f"[{cfg.dataset.name} | mode={model_cfg.score_mode}] {metrics}")
    return {
        "dataset":     cfg.dataset.name,
        "score_mode":  model_cfg.score_mode,
        "ckpt":        cfg.ckpt,
        "metrics":     metrics.as_dict(),
        "saved_epoch": int(blob.get("epoch", -1)),
    }
