"""Dataset loading for the six benchmarks used in the paper.

Datasets:
  Cora, Citeseer       : torch_geometric.datasets.Planetoid (auto-download)
  Amazon, YelpChi      : DGL fraud datasets (auto-download via dgl)
  JODIE                : Wikipedia/Reddit user-item interaction graph
                         (auto-download via the JODIE/CTDNE shim)
  ogbn-arxiv           : Open Graph Benchmark (auto-download via ogb)

Each loader returns a torch_geometric.data.Data object with
`x`, `edge_index`, and a binary `y_anom` (1 = anomalous). For the
synthetic homophilic graphs (Cora, Citeseer) we inject anomalies via
utils.anomaly_injection. For the fraud datasets (Amazon, YelpChi) we
use the labels provided by the dataset.

If a dataset library is missing at import time we raise an informative
`ImportError` rather than silently failing.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
from torch_geometric.data import Data

from utils.anomaly_injection import AnomalyConfig, inject_anomalies


@dataclass
class DatasetSpec:
    name: str               # cora | citeseer | amazon | yelpchi | jodie | ogbn_arxiv
    root: str = "./data"
    anomaly_ratio: float = 0.05
    anomaly_type: str = "structural"   # only used for synthetic injection
    seed: int = 0


# ---------------------------------------------------------------------------
# individual loaders
# ---------------------------------------------------------------------------
def _load_planetoid(spec: DatasetSpec) -> Data:
    """Cora / Citeseer with structurally injected anomalies."""
    from torch_geometric.datasets import Planetoid

    name_map = {"cora": "Cora", "citeseer": "Citeseer"}
    if spec.name not in name_map:
        raise ValueError(f"Unsupported planetoid dataset: {spec.name}")
    ds = Planetoid(root=os.path.join(spec.root, "planetoid"), name=name_map[spec.name])
    data = ds[0]
    # Inject anomalies according to the requested type.
    cfg = AnomalyConfig(
        anomaly_ratio=spec.anomaly_ratio,
        type=spec.anomaly_type,
        seed=spec.seed,
    )
    return inject_anomalies(data, cfg)


def _load_dgl_fraud(spec: DatasetSpec) -> Data:
    """Amazon / YelpChi via DGL's FraudYelpDataset / FraudAmazonDataset."""
    try:
        import dgl  # noqa: F401
        from dgl.data import FraudYelpDataset, FraudAmazonDataset
    except ImportError as e:
        raise ImportError(
            "Loading Amazon / YelpChi requires `dgl`. Install via:\n"
            "    pip install dgl\n"
        ) from e

    cls = FraudYelpDataset if spec.name == "yelpchi" else FraudAmazonDataset
    g = cls(raw_dir=os.path.join(spec.root, spec.name))[0]
    # Use homogeneous edge view (collapse multi-relational edges)
    src_list, dst_list = [], []
    for etype in g.canonical_etypes:
        s, d = g.edges(etype=etype)
        src_list.append(s)
        dst_list.append(d)
    src = torch.cat(src_list)
    dst = torch.cat(dst_list)
    edge_index = torch.stack([src, dst], dim=0).long()

    # Pick the dominant node type, which holds features and labels.
    main_ntype = g.ntypes[0]
    x = g.nodes[main_ntype].data["feature"].float()
    y = g.nodes[main_ntype].data["label"].long()
    data = Data(x=x, edge_index=edge_index)
    data.y_anom = (y != 0).long()
    data.anomaly_type = "organic"
    return data


def _load_jodie(spec: DatasetSpec) -> Data:
    """JODIE-style temporal interaction graph (Wikipedia by default).

    We treat the union of edges over all timestamps as the graph and use
    the per-edge label to mark "anomalous" interactions; node-level
    anomaly labels are derived as nodes incident to any anomalous edge.
    """
    try:
        from torch_geometric.datasets import JODIEDataset
    except ImportError as e:
        raise ImportError(
            "Loading JODIE requires torch_geometric.datasets.JODIEDataset.\n"
        ) from e

    ds = JODIEDataset(root=os.path.join(spec.root, "jodie"), name="wikipedia")
    raw = ds[0]
    n_users = int(raw.src.max().item()) + 1
    n_items = int(raw.dst.max().item()) + 1
    n = n_users + n_items

    # Treat user/item ids in a single namespace; offset items by n_users.
    src = raw.src.long()
    dst = raw.dst.long() + n_users
    edge_index = torch.stack([src, dst], dim=0)

    # Use msg as features; pad nodes without messages with zeros.
    feat_dim = raw.msg.size(1)
    x = torch.zeros(n, feat_dim)
    # Aggregate by mean of incident messages
    counts = torch.zeros(n)
    for i in range(src.size(0)):
        x[src[i]] += raw.msg[i]
        x[dst[i]] += raw.msg[i]
        counts[src[i]] += 1
        counts[dst[i]] += 1
    x = x / counts.clamp(min=1).unsqueeze(1)

    # Derive node-level anomaly labels from edge-level labels
    y_anom = torch.zeros(n, dtype=torch.long)
    if hasattr(raw, "y"):
        anomalous_edges = raw.y.long().nonzero(as_tuple=False).flatten()
        for ei in anomalous_edges.tolist():
            y_anom[src[ei]] = 1
            y_anom[dst[ei]] = 1

    data = Data(x=x, edge_index=edge_index)
    data.y_anom = y_anom
    data.anomaly_type = "temporal"
    return data


def _load_ogbn_arxiv(spec: DatasetSpec) -> Data:
    """ogbn-arxiv with structurally injected anomalies (large-scale test)."""
    try:
        from ogb.nodeproppred import PygNodePropPredDataset
    except ImportError as e:
        raise ImportError(
            "Loading ogbn-arxiv requires `ogb`. Install via:\n"
            "    pip install ogb\n"
        ) from e

    ds = PygNodePropPredDataset(name="ogbn-arxiv", root=os.path.join(spec.root, "ogb"))
    data = ds[0]
    cfg = AnomalyConfig(
        anomaly_ratio=spec.anomaly_ratio,
        type=spec.anomaly_type,
        seed=spec.seed,
    )
    return inject_anomalies(data, cfg)


# ---------------------------------------------------------------------------
# dispatch
# ---------------------------------------------------------------------------
DISPATCH = {
    "cora":         _load_planetoid,
    "citeseer":     _load_planetoid,
    "amazon":       _load_dgl_fraud,
    "yelpchi":      _load_dgl_fraud,
    "jodie":        _load_jodie,
    "ogbn_arxiv":   _load_ogbn_arxiv,
    "ogbn-arxiv":   _load_ogbn_arxiv,
}


def load_dataset(spec: DatasetSpec) -> Data:
    """Return a Data object with `x`, `edge_index`, `y_anom`."""
    key = spec.name.lower()
    if key not in DISPATCH:
        raise ValueError(
            f"Unknown dataset {spec.name!r}. Choices: {sorted(DISPATCH)}"
        )
    data = DISPATCH[key](spec)

    # sanity: ensure binary anomaly mask exists
    if not hasattr(data, "y_anom"):
        raise RuntimeError(
            f"Loader for {spec.name!r} did not produce `y_anom` mask."
        )
    return data


# ---------------------------------------------------------------------------
# train/val split helper for normality-only training
# ---------------------------------------------------------------------------
def split_normal_indices(
    data: Data, val_frac: float = 0.10, seed: int = 0
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (train_idx, val_idx) over **normal** nodes only.

    IRL-GAD trains on normal trajectories only (the "expert" set);
    anomalies are held out for test-time scoring.
    """
    g = torch.Generator().manual_seed(seed)
    normal_idx = (data.y_anom == 0).nonzero(as_tuple=False).flatten()
    perm = normal_idx[torch.randperm(normal_idx.numel(), generator=g)]
    n_val = int(round(val_frac * perm.numel()))
    val_idx = perm[:n_val]
    train_idx = perm[n_val:]
    return train_idx, val_idx
