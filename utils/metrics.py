"""Anomaly-detection metrics used in the paper."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)


@dataclass
class AnomalyMetrics:
    auc_roc: float
    auc_pr: float
    tpr_at_5_fpr: float

    def as_dict(self) -> dict[str, float]:
        return {
            "auc_roc": self.auc_roc,
            "auc_pr": self.auc_pr,
            "tpr@5fpr": self.tpr_at_5_fpr,
        }

    def __str__(self) -> str:
        return (
            f"AUC-ROC: {self.auc_roc*100:.2f}% | "
            f"AUC-PR: {self.auc_pr*100:.2f}% | "
            f"TPR@5%FPR: {self.tpr_at_5_fpr*100:.2f}%"
        )


def compute_metrics(scores: torch.Tensor | np.ndarray,
                    y_true: torch.Tensor | np.ndarray,
                    target_fpr: float = 0.05) -> AnomalyMetrics:
    """Compute the three headline metrics.

    `scores` higher = more anomalous. `y_true` is a binary mask
    (1 = anomalous).
    """
    if isinstance(scores, torch.Tensor):
        scores = scores.detach().cpu().numpy()
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    scores = np.asarray(scores).astype(np.float64)
    y_true = np.asarray(y_true).astype(np.int64)

    if y_true.sum() == 0 or y_true.sum() == y_true.size:
        # degenerate: all one class
        return AnomalyMetrics(0.5, float(y_true.mean()), 0.0)

    auc_roc = float(roc_auc_score(y_true, scores))
    auc_pr  = float(average_precision_score(y_true, scores))
    fpr, tpr, _ = roc_curve(y_true, scores)
    # interpolate TPR at the requested FPR
    tpr_at = float(np.interp(target_fpr, fpr, tpr))
    return AnomalyMetrics(auc_roc, auc_pr, tpr_at)


def best_threshold_at_fpr(scores: torch.Tensor | np.ndarray,
                          y_true: torch.Tensor | np.ndarray,
                          target_fpr: float = 0.05) -> float:
    """Threshold giving FPR closest to `target_fpr` (without exceeding it)."""
    if isinstance(scores, torch.Tensor):
        scores = scores.detach().cpu().numpy()
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    fpr, _, thr = roc_curve(y_true, scores)
    # roc_curve returns thresholds in decreasing order; pick largest thr
    # whose fpr is <= target.
    valid = fpr <= target_fpr
    if not valid.any():
        return float(thr[0])
    idx = int(np.where(valid)[0][-1])
    return float(thr[idx])
