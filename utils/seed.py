"""Deterministic seeding utilities for reproducibility."""
from __future__ import annotations

import os
import random
from typing import Optional

import numpy as np
import torch


def set_seed(seed: int = 0, deterministic_cudnn: bool = True) -> None:
    """Seed all relevant RNGs.

    Parameters
    ----------
    seed : int
        Master seed shared across Python `random`, NumPy, and PyTorch.
    deterministic_cudnn : bool
        If True, force cuDNN into deterministic-but-slower mode. Set this
        to True for paper-quality runs and False for development speed.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic_cudnn:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True


def seeded_generator(seed: int) -> torch.Generator:
    """Return a torch.Generator seeded for DataLoader shuffling, etc."""
    g = torch.Generator()
    g.manual_seed(seed)
    return g


def worker_init_fn(worker_id: int, base_seed: Optional[int] = None) -> None:
    """For passing to torch DataLoader's `worker_init_fn`."""
    s = (base_seed or 0) + worker_id
    np.random.seed(s)
    random.seed(s)
