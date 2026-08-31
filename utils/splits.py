"""Normal-only split construction (contract section 4).

Replaces `utils.data_utils.split_normal_indices`, which read `y_anom` to
define "benign", produced only train/val, and had no calibration stage.

Protocol, per seed in {0,1,2,3,4}, over *valid* nodes only:

    15%  train        (known-benign)
     5%  validation   (benign-only, model selection)
     5%  calibration  (benign-only, disjoint, threshold only)
    rest test         (remaining benign + all anomalies)

The four masks are disjoint by construction. Anomaly labels are read
exactly once, here, to define the known-benign pool -- which is what
`OC` access means -- and the resulting `test` mask is the only place
anomalies appear. Nothing downstream may consult `y_anom` until the
checkpoint, hyperparameters, and threshold are frozen; `assert_frozen`
below is the guard.

`valid_mask` exists for Amazon, where users without a fraud/benign label
stay in the graph for message passing but are excluded from every mask.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, asdict
from typing import Optional

import numpy as np
import torch

TRAIN_FRAC = 0.15
VAL_FRAC = 0.05
CALIB_FRAC = 0.05


@dataclass
class SplitManifest:
    """Immutable record of one (dataset, seed) split."""
    dataset: str
    seed: int
    num_nodes: int
    num_valid: int
    num_benign: int
    num_anomalies: int
    n_train: int
    n_val: int
    n_calib: int
    n_test: int
    split_hash: str

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, sort_keys=True)


@dataclass
class Splits:
    train: torch.Tensor      # bool (N,)
    val: torch.Tensor
    calib: torch.Tensor
    test: torch.Tensor
    manifest: SplitManifest

    def assert_disjoint(self) -> None:
        stack = torch.stack([self.train, self.val, self.calib, self.test])
        overlap = stack.long().sum(dim=0)
        if int(overlap.max()) > 1:
            bad = int((overlap > 1).sum())
            raise AssertionError(
                f"splits overlap on {bad} node(s); masks must be disjoint"
            )


def _hash_masks(*masks: torch.Tensor) -> str:
    h = hashlib.sha256()
    for m in masks:
        idx = torch.nonzero(m, as_tuple=False).flatten().cpu().numpy()
        h.update(idx.astype(np.int64).tobytes())
        h.update(b"|")
    return h.hexdigest()


def make_splits(
    y_anom: torch.Tensor,
    dataset: str,
    seed: int,
    valid_mask: Optional[torch.Tensor] = None,
) -> Splits:
    """Build the 15/5/5/rest normal-only split for one seed.

    `y_anom` is consulted here and only here. `valid_mask` marks nodes
    eligible for any mask (Amazon: labelled users only).
    """
    n = int(y_anom.numel())
    if valid_mask is None:
        valid_mask = torch.ones(n, dtype=torch.bool)
    valid_mask = valid_mask.to(torch.bool)

    benign = (y_anom == 0) & valid_mask
    anomalous = (y_anom == 1) & valid_mask

    benign_idx = torch.nonzero(benign, as_tuple=False).flatten()
    n_benign = int(benign_idx.numel())
    if n_benign < 20:
        raise ValueError(
            f"{dataset}: only {n_benign} benign nodes; cannot form a 15/5/5 split"
        )

    # Permutation depends on the seed only, so a fixed seed reproduces the
    # split exactly (test 15). numpy Generator is used rather than torch's
    # global RNG so that model-side seeding cannot perturb the split.
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n_benign)
    shuffled = benign_idx[torch.from_numpy(perm)]

    n_train = int(round(TRAIN_FRAC * n_benign))
    n_val = int(round(VAL_FRAC * n_benign))
    n_calib = int(round(CALIB_FRAC * n_benign))

    train_idx = shuffled[:n_train]
    val_idx = shuffled[n_train:n_train + n_val]
    calib_idx = shuffled[n_train + n_val:n_train + n_val + n_calib]
    held_benign = shuffled[n_train + n_val + n_calib:]

    def _mask(idx: torch.Tensor) -> torch.Tensor:
        m = torch.zeros(n, dtype=torch.bool)
        m[idx] = True
        return m

    train = _mask(train_idx)
    val = _mask(val_idx)
    calib = _mask(calib_idx)
    test = _mask(held_benign) | anomalous

    split_hash = _hash_masks(train, val, calib, test)
    manifest = SplitManifest(
        dataset=dataset,
        seed=seed,
        num_nodes=n,
        num_valid=int(valid_mask.sum()),
        num_benign=n_benign,
        num_anomalies=int(anomalous.sum()),
        n_train=int(train.sum()),
        n_val=int(val.sum()),
        n_calib=int(calib.sum()),
        n_test=int(test.sum()),
        split_hash=split_hash,
    )

    splits = Splits(train=train, val=val, calib=calib, test=test, manifest=manifest)
    splits.assert_disjoint()
    return splits


# ---------------------------------------------------------------------------
# calibration threshold
# ---------------------------------------------------------------------------
def calibrate_threshold(scores: torch.Tensor, calib_mask: torch.Tensor) -> float:
    """Deployment threshold = 95th percentile of benign-calibration scores.

    Contract section 4. This is *not* the TPR@5%FPR operating point, which
    is a separate test-set statistic computed in utils.metrics.
    """
    if int(calib_mask.sum()) == 0:
        raise ValueError("calibration mask is empty")
    calib_scores = scores[calib_mask].detach().cpu().numpy()
    return float(np.percentile(calib_scores, 95.0))


# ---------------------------------------------------------------------------
# leakage guard
# ---------------------------------------------------------------------------
class LabelAccessError(RuntimeError):
    """Raised when anomaly labels are touched before the model is frozen."""


class FrozenLabels:
    """Wraps `y_anom` so it cannot be read until explicitly unlocked.

    Training, validation, model selection, and calibration all run with the
    labels locked. `unlock()` is called only by the evaluation path, after
    the checkpoint and threshold are fixed. Tests 8 and 9 rely on this.
    """

    def __init__(self, y_anom: torch.Tensor, stage: str = "training"):
        self._y = y_anom
        self._unlocked = False
        self._stage = stage

    def unlock(self, reason: str) -> torch.Tensor:
        self._unlocked = True
        self._reason = reason
        return self._y

    @property
    def value(self) -> torch.Tensor:
        if not self._unlocked:
            raise LabelAccessError(
                f"anomaly labels read during '{self._stage}' before freezing. "
                "Model selection and calibration must use benign nodes only "
                "(contract section 4)."
            )
        return self._y
