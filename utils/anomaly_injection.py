"""Reproducible 5% node-level anomaly construction (contract section 3).

The four operators share one target set and one benign-mask set per seed;
only the corruption differs. That is what makes the Q2 drop statistic a
paired difference. `select_targets` is therefore separated from
`apply_operator`: the caller draws targets once and applies each operator
to the same IDs.

Changes against the legacy implementation (audit F10):

  - attribute: 50 donor candidates are now sampled *per target*, and the
    donor is the maximum-cosine-distance candidate. Legacy sampled one
    shared pool of 50 for all targets.
  - contextual: donor is the maximum-cosine-distance node from a
    *different* community. Legacy picked uniformly at random.
  - Louvain: seeded from the run seed, computed on the clean graph. Legacy
    hardcoded random_state=0 and silently fell back to a random partition
    on any exception.
  - targets are drawn from an eligible pool that excludes train,
    validation, and calibration nodes.
  - target IDs, donor IDs, the Louvain seed, and graph hashes are saved.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional

import numpy as np
import torch
from torch_geometric.data import Data

OPERATORS = ("structural", "attribute", "contextual", "hybrid")
GROUP_SIZE = 15          # structural: densely connect groups of at most 15
N_DONORS = 50            # attribute: candidate donors sampled per target
ANOMALY_RATIO = 0.05


class LouvainUnavailable(RuntimeError):
    """Raised when contextual injection cannot compute a Louvain partition."""


@dataclass
class InjectionManifest:
    dataset: str
    seed: int
    operator: str
    anomaly_ratio: float
    target_ids: List[int]
    donor_ids: Dict[str, int] = field(default_factory=dict)
    louvain_seed: Optional[int] = None
    num_communities: Optional[int] = None
    clean_graph_hash: str = ""
    corrupted_graph_hash: str = ""
    edges_added: int = 0
    features_replaced: int = 0

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, sort_keys=True)


# ---------------------------------------------------------------------------
# hashing
# ---------------------------------------------------------------------------
def graph_hash(x: torch.Tensor, edge_index: torch.Tensor) -> str:
    h = hashlib.sha256()
    ei = edge_index.cpu().numpy().astype(np.int64)
    order = np.lexsort((ei[1], ei[0]))
    h.update(np.ascontiguousarray(ei[:, order]).tobytes())
    h.update(np.ascontiguousarray(x.cpu().numpy(), dtype=np.float32).tobytes())
    return h.hexdigest()


# ---------------------------------------------------------------------------
# target selection (shared across all four operators)
# ---------------------------------------------------------------------------
def select_targets(
    num_nodes: int,
    seed: int,
    excluded: Optional[torch.Tensor] = None,
    anomaly_ratio: float = ANOMALY_RATIO,
) -> np.ndarray:
    """Draw the anomaly target IDs for one seed.

    `excluded` is the union of the train, validation, and calibration
    masks. Contract section 3: those nodes must not enter the target pool.
    The draw depends only on (num_nodes, seed, excluded), never on the
    operator, so all four corrupted graphs share targets (test 10).
    """
    eligible = np.ones(num_nodes, dtype=bool)
    if excluded is not None:
        eligible[excluded.cpu().numpy().astype(bool)] = False

    pool = np.nonzero(eligible)[0]
    n_anom = max(1, int(round(anomaly_ratio * num_nodes)))
    if pool.size < n_anom:
        raise ValueError(
            f"eligible pool has {pool.size} nodes but {n_anom} targets are needed; "
            "reduce anomaly_ratio or the benign split fractions"
        )

    rng = np.random.default_rng(seed)
    targets = rng.choice(pool, size=n_anom, replace=False)
    return np.sort(targets)


# ---------------------------------------------------------------------------
# operators
# ---------------------------------------------------------------------------
def _densely_connect(edge_index: torch.Tensor, group: np.ndarray) -> torch.Tensor:
    if len(group) < 2:
        return edge_index
    src, dst = np.meshgrid(group, group, indexing="xy")
    keep = src != dst
    extra = torch.from_numpy(np.stack([src[keep], dst[keep]], axis=0)).long()
    return torch.cat([edge_index, extra.to(edge_index.device)], dim=1)


def _structural(edge_index: torch.Tensor, targets: np.ndarray):
    before = edge_index.size(1)
    out = edge_index
    for start in range(0, len(targets), GROUP_SIZE):
        out = _densely_connect(out, targets[start:start + GROUP_SIZE])
    return out, out.size(1) - before


def _max_cosine_distance_donor(
    x: torch.Tensor, target: int, candidates: np.ndarray
) -> int:
    """Return the candidate with maximum cosine distance from `target`."""
    v = x[target].unsqueeze(0)
    cand = x[candidates]
    num = (v * cand).sum(dim=1)
    denom = v.norm(dim=1) * cand.norm(dim=1) + 1e-12
    cos_sim = num / denom
    return int(candidates[int(torch.argmin(cos_sim))])


def _attribute(x: torch.Tensor, targets: np.ndarray, seed: int):
    """Replace each target's features with its most-distant of 50 donors."""
    n = x.size(0)
    new_x = x.clone()
    donors: Dict[str, int] = {}
    target_set = set(int(t) for t in targets)
    rng = np.random.default_rng(seed + 1)

    non_target = np.array([i for i in range(n) if i not in target_set])
    for t in targets:
        k = min(N_DONORS, non_target.size)
        candidates = rng.choice(non_target, size=k, replace=False)
        donor = _max_cosine_distance_donor(x, int(t), candidates)
        new_x[int(t)] = x[donor]
        donors[str(int(t))] = donor
    return new_x, donors


def _louvain(edge_index: torch.Tensor, num_nodes: int, seed: int) -> np.ndarray:
    """Seeded Louvain partition of the clean graph.

    Raises rather than falling back to a random partition: a random
    partition silently changes what "different community" means and made
    the legacy contextual operator unreproducible (audit F10).
    """
    try:
        import networkx as nx
        import community as community_louvain
    except ImportError as e:
        raise LouvainUnavailable(
            "contextual injection requires networkx and python-louvain:\n"
            "    pip install networkx python-louvain"
        ) from e

    G = nx.Graph()
    G.add_nodes_from(range(num_nodes))
    ei = edge_index.cpu().numpy()
    G.add_edges_from(zip(ei[0].tolist(), ei[1].tolist()))
    partition = community_louvain.best_partition(G, random_state=seed)
    return np.array([partition[i] for i in range(num_nodes)], dtype=np.int64)


def _contextual(
    x: torch.Tensor,
    edge_index: torch.Tensor,
    targets: np.ndarray,
    seed: int,
):
    """Max-cosine-distance donor drawn from a different Louvain community."""
    n = x.size(0)
    comms = _louvain(edge_index, n, seed)
    new_x = x.clone()
    donors: Dict[str, int] = {}

    for t in targets:
        t = int(t)
        other = np.nonzero(comms != comms[t])[0]
        if other.size == 0:
            continue
        donor = _max_cosine_distance_donor(x, t, other)
        new_x[t] = x[donor]
        donors[str(t)] = donor
    return new_x, donors, int(comms.max()) + 1


# ---------------------------------------------------------------------------
# public API
# ---------------------------------------------------------------------------
def apply_operator(
    data: Data,
    operator: str,
    targets: np.ndarray,
    seed: int,
    dataset: str = "",
):
    """Apply one corruption operator to a fixed target set.

    `hybrid` applies structural first, then attribute, to the same targets
    (contract section 3).
    """
    if operator not in OPERATORS:
        raise ValueError(f"unknown operator {operator!r}; choices: {OPERATORS}")

    clean_hash = graph_hash(data.x, data.edge_index)
    n = data.num_nodes
    new_x = data.x.clone()
    new_edge_index = data.edge_index.clone()
    donors: Dict[str, int] = {}
    louvain_seed = None
    n_comms = None
    edges_added = 0
    feats_replaced = 0

    if operator in ("structural", "hybrid"):
        new_edge_index, edges_added = _structural(new_edge_index, targets)

    if operator in ("attribute", "hybrid"):
        new_x, donors = _attribute(data.x, targets, seed)
        feats_replaced = len(donors)

    if operator == "contextual":
        new_x, donors, n_comms = _contextual(data.x, data.edge_index, targets, seed)
        louvain_seed = seed
        feats_replaced = len(donors)

    y_anom = torch.zeros(n, dtype=torch.long)
    y_anom[torch.from_numpy(targets).long()] = 1

    out = data.clone()
    out.x = new_x
    out.edge_index = new_edge_index
    out.y_anom = y_anom
    out.anomaly_type = operator

    manifest = InjectionManifest(
        dataset=dataset,
        seed=seed,
        operator=operator,
        anomaly_ratio=ANOMALY_RATIO,
        target_ids=[int(t) for t in targets],
        donor_ids=donors,
        louvain_seed=louvain_seed,
        num_communities=n_comms,
        clean_graph_hash=clean_hash,
        corrupted_graph_hash=graph_hash(new_x, new_edge_index),
        edges_added=edges_added,
        features_replaced=feats_replaced,
    )
    return out, manifest


def build_matched_graphs(
    data: Data,
    seed: int,
    excluded: Optional[torch.Tensor] = None,
    dataset: str = "",
):
    """Build all four corrupted graphs from one shared target set (Q2)."""
    targets = select_targets(data.num_nodes, seed, excluded=excluded)
    return {
        op: apply_operator(data, op, targets, seed, dataset=dataset)
        for op in OPERATORS
    }


# ---------------------------------------------------------------------------
# DEPRECATED legacy shim
# ---------------------------------------------------------------------------
# Kept so existing callers import cleanly. Excluded from every reported
# configuration: it draws targets from all nodes (no benign-mask exclusion)
# and cannot produce the paired Q2 design. Use `build_matched_graphs`.
@dataclass
class AnomalyConfig:
    anomaly_ratio: float = ANOMALY_RATIO
    clique_size: int = GROUP_SIZE
    attribute_k: int = N_DONORS
    seed: int = 0
    type: str = "structural"


def inject_anomalies(data: Data, cfg: "AnomalyConfig") -> Data:
    """DEPRECATED. Use select_targets + apply_operator, or build_matched_graphs."""
    import warnings

    warnings.warn(
        "inject_anomalies() is deprecated and excluded from reported "
        "configurations: it does not exclude benign train/val/calibration "
        "nodes from the target pool. Use build_matched_graphs().",
        DeprecationWarning,
        stacklevel=2,
    )
    targets = select_targets(
        data.num_nodes, cfg.seed, excluded=None, anomaly_ratio=cfg.anomaly_ratio
    )
    graph, _ = apply_operator(data, cfg.type, targets, cfg.seed)
    return graph
