"""Training entry point for IRL-GAD.

Usage (programmatic):
    from train import train
    metrics = train(cfg)

Or via main.py / scripts.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import torch
from torch_geometric.data import Data

from models.irl_gad import IRLGAD, IRLGADConfig
from utils.data_utils import DatasetSpec, load_dataset, split_normal_indices
from utils.metrics import compute_metrics
from utils.seed import set_seed


# ---------------------------------------------------------------------------
# top-level train config
# ---------------------------------------------------------------------------
@dataclass
class TrainConfig:
    dataset: DatasetSpec
    model: IRLGADConfig
    epochs: int = 200
    lr: float = 1e-3
    weight_decay: float = 1e-4
    early_stop_patience: int = 50
    log_every: int = 10
    seed: int = 0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir: str = "./experiments/runs"
    run_name: Optional[str] = None
    val_frac: float = 0.10


# ---------------------------------------------------------------------------
# core training loop
# ---------------------------------------------------------------------------
def train(cfg: TrainConfig) -> dict:
    set_seed(cfg.seed)
    device = torch.device(cfg.device)

    # --- data ----------------------------------------------------------
    data: Data = load_dataset(cfg.dataset)
    data = data.to(device)
    train_idx, val_idx = split_normal_indices(
        data, val_frac=cfg.val_frac, seed=cfg.seed
    )
    train_idx = train_idx.to(device)
    val_idx   = val_idx.to(device)

    # set in_dim from data
    cfg.model.in_dim = data.x.size(1)

    # --- model ---------------------------------------------------------
    model = IRLGAD(cfg.model).to(device)
    optim = torch.optim.Adam(
        model.parameters(),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
        betas=(0.9, 0.999),
    )

    # --- bookkeeping ---------------------------------------------------
    out_dir = Path(cfg.out_dir) / (cfg.run_name or _default_run_name(cfg))
    out_dir.mkdir(parents=True, exist_ok=True)

    best = {"val_auc_roc": -1.0, "epoch": -1, "state": None}
    no_improve = 0
    start = time.time()

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        out = model.compute_loss(
            data.x, data.edge_index, node_mask=train_idx
        )
        loss = out["loss"]
        optim.zero_grad()
        loss.backward()
        optim.step()

        # --- validation ------------------------------------------------
        model.eval()
        with torch.no_grad():
            scored = model.score(data.x, data.edge_index)
            scores_all = scored["score"].cpu()
            y_anom = data.y_anom.cpu()
            metrics = compute_metrics(scores_all, y_anom)

        if metrics.auc_roc > best["val_auc_roc"]:
            best = {
                "val_auc_roc": metrics.auc_roc,
                "epoch": epoch,
                "state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
            }
            no_improve = 0
        else:
            no_improve += 1

        if epoch % cfg.log_every == 0 or epoch == 1:
            print(
                f"[ep {epoch:4d}] loss={loss.item():.4f}  "
                f"val_auc={metrics.auc_roc*100:.2f}%  "
                f"best={best['val_auc_roc']*100:.2f}%  "
                f"λ1={out['lambda1'].item():.3f}  "
                f"λ2={out['lambda2'].item():.3f}"
            )

        if no_improve >= cfg.early_stop_patience:
            print(f"[ep {epoch}] early stopping (patience={cfg.early_stop_patience})")
            break

    elapsed = time.time() - start
    print(f"finished in {elapsed:.1f}s ({(epoch)/elapsed:.2f} ep/s)")

    # --- save best checkpoint -----------------------------------------
    ckpt_path = out_dir / "best.pt"
    torch.save(
        {
            "state_dict": best["state"],
            "model_cfg": cfg.model.__dict__,
            "epoch": best["epoch"],
            "val_auc_roc": best["val_auc_roc"],
        },
        ckpt_path,
    )
    print(f"checkpoint -> {ckpt_path}")

    # --- final metrics on full graph ----------------------------------
    model.load_state_dict({k: v.to(device) for k, v in best["state"].items()})
    model.eval()
    with torch.no_grad():
        scored = model.score(data.x, data.edge_index)
        scores_all = scored["score"].cpu()
        final = compute_metrics(scores_all, data.y_anom.cpu())
    print(f"final: {final}")
    return {
        "best_epoch": best["epoch"],
        "metrics": final.as_dict(),
        "elapsed_sec": elapsed,
        "ckpt": str(ckpt_path),
    }


def _default_run_name(cfg: TrainConfig) -> str:
    return f"{cfg.dataset.name}_seed{cfg.seed}_K{cfg.model.num_layers}"
