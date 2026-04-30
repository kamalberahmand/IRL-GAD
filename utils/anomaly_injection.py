"""Anomaly injection for synthetic benchmarks (Cora, Citeseer).

Implements four anomaly types matching the protocol used in DOMINANT
[Ding et al. 2019] and CoLA [Liu et al. 2021]:

  (i)   structural  : random nodes placed into small dense cliques
  (ii)  attribute   : feature replaced by the most-distant node's feature
  (iii) contextual  : feature swapped with a node from a different community
  (iv)  hybrid      : (i) and (ii) combined on the same nodes

The function returns the modified Data object together with a binary
label vector (1 = anomalous).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
from torch_geometric.data import Data


@dataclass
class AnomalyConfig:
    anomaly_ratio: float = 0.05         # fraction of total nodes to make anomalous
    clique_size: int = 15               # for structural anomalies
    attribute_k: int = 50               # candidate pool size for attribute swap
    seed: int = 0
    type: str = "structural"            # structural | attribute | contextual | hybrid


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _louvain_communities(edge_index: torch.Tensor, num_nodes: int) -> np.ndarray:
    """Best-effort community detection. Falls back to random partition.

    We avoid a hard dependency on `python-louvain`; if it is unavailable
    we partition nodes randomly into sqrt(N) groups, which is sufficient
    for "draw a feature from a different community" semantics.
    """
    try:
        import networkx as nx
        import community as community_louvain  # python-louvain
        G = nx.Graph()
        G.add_nodes_from(range(num_nodes))
        ei = edge_index.cpu().numpy()
        G.add_edges_from(zip(ei[0].tolist(), ei[1].tolist()))
        partition = community_louvain.best_partition(G, random_state=0)
        return np.array([partition[i] for i in range(num_nodes)], dtype=np.int64)
    except Exception:
        rng = np.random.default_rng(0)
        n_groups = max(2, int(np.sqrt(num_nodes)))
        return rng.integers(0, n_groups, size=num_nodes)


def _add_clique(edge_index: torch.Tensor, nodes: np.ndarray) -> torch.Tensor:
    """Add a fully-connected clique among `nodes` to `edge_index`."""
    n = len(nodes)
    if n < 2:
        return edge_index
    src, dst = np.meshgrid(nodes, nodes, indexing="xy")
    mask = src != dst
    extra = np.stack([src[mask], dst[mask]], axis=0)
    extra_t = torch.from_numpy(extra).to(edge_index.device).long()
    return torch.cat([edge_index, extra_t], dim=1)


# ---------------------------------------------------------------------------
# main API
# ---------------------------------------------------------------------------
def inject_anomalies(data: Data, cfg: AnomalyConfig) -> Data:
    """Return a copy of `data` with synthetic anomalies and a binary `y_anom`.

    The original `data.y` (class labels) is preserved untouched; the new
    anomaly mask is stored in `data.y_anom`.
    """
    rng = np.random.default_rng(cfg.seed)
    n = data.num_nodes
    n_anom = max(1, int(round(cfg.anomaly_ratio * n)))
    y_anom = torch.zeros(n, dtype=torch.long)

    new_x = data.x.clone()
    new_edge_index = data.edge_index.clone()

    chosen = rng.choice(n, size=n_anom, replace=False)
    chosen = np.sort(chosen)

    if cfg.type in ("structural", "hybrid"):
        # group `chosen` into cliques of size `clique_size`
        for start in range(0, len(chosen), cfg.clique_size):
            group = chosen[start : start + cfg.clique_size]
            if len(group) >= 2:
                new_edge_index = _add_clique(new_edge_index, group)

    if cfg.type in ("attribute", "hybrid"):
        # for each anomalous node, replace its features with the most
        # dissimilar node's features within a candidate pool
        pool = rng.choice(n, size=min(cfg.attribute_k, n), replace=False)
        pool_feats = data.x[pool]
        for i in chosen:
            v = data.x[i].unsqueeze(0)
            # cosine distance ~ 1 - cos_sim; pick the largest
            num = (v * pool_feats).sum(dim=1)
            denom = v.norm(dim=1) * pool_feats.norm(dim=1) + 1e-12
            cos_sim = num / denom
            most_distant = pool[int(torch.argmin(cos_sim).item())]
            new_x[i] = data.x[most_distant]

    if cfg.type == "contextual":
        comms = _louvain_communities(data.edge_index, n)
        for i in chosen:
            other_comm_nodes = np.where(comms != comms[i])[0]
            if len(other_comm_nodes) == 0:
                continue
            j = rng.choice(other_comm_nodes)
            new_x[i] = data.x[j]

    y_anom[chosen] = 1

    out = data.clone()
    out.x = new_x
    out.edge_index = new_edge_index
    out.y_anom = y_anom
    out.anomaly_type = cfg.type
    return out
