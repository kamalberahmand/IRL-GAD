"""Contract section 7 tests.

Numbering follows the contract. Tests 6, 7, 12, 13, 14 are marked where
they depend on modules not yet migrated; see CODE_PAPER_ALIGNMENT.md.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch
from torch_geometric.data import Data

from models.policy import (
    assert_valid_policies,
    observed_log_policy,
    reference_log_policy,
    score_from_policies,
    segment_log_softmax,
)
from utils.anomaly_injection import (
    OPERATORS,
    apply_operator,
    build_matched_graphs,
    graph_hash,
    select_targets,
)
from utils.splits import (
    FrozenLabels,
    LabelAccessError,
    calibrate_threshold,
    make_splits,
)

N_NODES = 240
K_HOPS = 2


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def toy_graph():
    g = torch.Generator().manual_seed(0)
    n = N_NODES
    # every node gets at least one outgoing edge
    src = torch.arange(n).repeat_interleave(4)
    dst = torch.randint(0, n, (src.numel(),), generator=g)
    edge_index = torch.stack([src, dst])
    x = torch.randn(n, 16, generator=g)
    return Data(x=x, edge_index=edge_index)


@pytest.fixture
def policies(toy_graph):
    ei = toy_graph.edge_index
    index = ei[0]
    n = toy_graph.num_nodes
    g = torch.Generator().manual_seed(1)
    e = ei.size(1)
    raw = torch.randn(e, generator=g)
    alpha = torch.exp(segment_log_softmax(raw, index, n))
    rewards = [torch.randn(e, generator=g) for _ in range(K_HOPS)]
    log_p = observed_log_policy([alpha] * K_HOPS, index, n)
    log_q = reference_log_policy(rewards, index, n, beta=0.1)
    return log_p, log_q, index, n


@pytest.fixture
def y_anom():
    y = torch.zeros(N_NODES, dtype=torch.long)
    y[:12] = 1
    return y


# ---------------------------------------------------------------------------
# 1. identical supports
# ---------------------------------------------------------------------------
def test_01_policies_share_support(policies):
    log_p, log_q, index, n = policies
    for k in range(K_HOPS):
        assert log_p[k].shape == log_q[k].shape
        assert log_p[k].numel() == index.numel()
    # no entry is masked out of one policy but present in the other
    for k in range(K_HOPS):
        assert torch.isfinite(log_p[k]).all()
        assert torch.isfinite(log_q[k]).all()


# ---------------------------------------------------------------------------
# 2. normalisation per node and hop
# ---------------------------------------------------------------------------
def test_02_policies_normalise(policies):
    log_p, log_q, index, n = policies
    for pol in (log_p, log_q):
        for k in range(K_HOPS):
            mass = torch.zeros(n).index_add_(0, index, torch.exp(pol[k]))
            has = torch.zeros(n, dtype=torch.bool)
            has[index] = True
            assert torch.allclose(mass[has], torch.ones(int(has.sum())), atol=1e-5)
    # the assertion helper agrees
    assert_valid_policies(log_p, log_q, index, n)


def test_02b_assertions_catch_unnormalised(policies):
    log_p, log_q, index, n = policies
    broken = [log_p[0] + 3.0] + log_p[1:]
    with pytest.raises(AssertionError, match="off-normal"):
        assert_valid_policies(broken, log_q, index, n)


def test_02c_assertions_catch_misaligned_support(policies):
    log_p, log_q, index, n = policies
    truncated = [log_q[0][:-5]] + log_q[1:]
    with pytest.raises(AssertionError, match="support mismatch"):
        assert_valid_policies(log_p, truncated, index, n)


# ---------------------------------------------------------------------------
# 3. KL finite and non-negative
# ---------------------------------------------------------------------------
def test_03_kl_finite_nonnegative(policies):
    log_p, log_q, index, n = policies
    score, d = score_from_policies(log_p, log_q, index, n)
    for d_k in d:
        assert torch.isfinite(d_k).all()
        assert float(d_k.min()) >= 0.0
    assert torch.isfinite(score).all()


def test_03b_kl_zero_when_policies_match(policies):
    log_p, _, index, n = policies
    score, d = score_from_policies(log_p, log_p, index, n)
    assert float(score.abs().max()) < 1e-6


# ---------------------------------------------------------------------------
# 4. total score == sum of per-hop KL terms
# ---------------------------------------------------------------------------
def test_04_score_equals_sum_of_per_hop(policies):
    log_p, log_q, index, n = policies
    score, d = score_from_policies(log_p, log_q, index, n)
    assert len(d) == K_HOPS
    recomputed = d[0] + d[1]
    assert torch.allclose(score, recomputed, atol=1e-10, rtol=0)


# ---------------------------------------------------------------------------
# 5. no neighbour action is sampled
# ---------------------------------------------------------------------------
def test_05_no_sampling_in_score_path():
    import inspect
    import models.policy as policy_mod

    src = inspect.getsource(policy_mod)
    for banned in ("multinomial", "torch.randint", "Categorical", ".sample("):
        assert banned not in src, f"sampling primitive {banned!r} present in score path"


def test_05b_score_is_deterministic(policies):
    log_p, log_q, index, n = policies
    s1, _ = score_from_policies(log_p, log_q, index, n)
    s2, _ = score_from_policies(log_p, log_q, index, n)
    assert torch.equal(s1, s2)


# ---------------------------------------------------------------------------
# 8. anomaly labels cannot enter training / selection / calibration
# ---------------------------------------------------------------------------
def test_08_labels_locked_until_frozen(y_anom):
    frozen = FrozenLabels(y_anom, stage="model selection")
    with pytest.raises(LabelAccessError, match="benign nodes only"):
        _ = frozen.value
    unlocked = frozen.unlock(reason="checkpoint and threshold frozen")
    assert torch.equal(unlocked, y_anom)
    assert torch.equal(frozen.value, y_anom)


def test_08b_calibration_uses_benign_only(y_anom):
    sp = make_splits(y_anom, dataset="toy", seed=0)
    # no anomaly appears in train, val, or calibration
    for mask in (sp.train, sp.val, sp.calib):
        assert int((y_anom[mask] == 1).sum()) == 0
    # every anomaly is in test
    assert int(sp.test[y_anom == 1].sum()) == int((y_anom == 1).sum())


def test_08c_threshold_is_95th_percentile(y_anom):
    sp = make_splits(y_anom, dataset="toy", seed=0)
    scores = torch.arange(N_NODES, dtype=torch.float)
    thr = calibrate_threshold(scores, sp.calib)
    calib_vals = scores[sp.calib].numpy()
    assert abs(thr - float(np.percentile(calib_vals, 95.0))) < 1e-9
    # threshold is not the TPR@5%FPR operating point
    assert thr <= float(scores.max())


# ---------------------------------------------------------------------------
# 9. masks are disjoint, and fractions match the contract
# ---------------------------------------------------------------------------
def test_09_masks_disjoint(y_anom):
    sp = make_splits(y_anom, dataset="toy", seed=0)
    sp.assert_disjoint()
    total = sp.train | sp.val | sp.calib | sp.test
    assert int(total.sum()) == N_NODES


def test_09b_split_fractions(y_anom):
    sp = make_splits(y_anom, dataset="toy", seed=0)
    n_benign = int((y_anom == 0).sum())
    assert sp.manifest.n_train == round(0.15 * n_benign)
    assert sp.manifest.n_val == round(0.05 * n_benign)
    assert sp.manifest.n_calib == round(0.05 * n_benign)


def test_09c_invalid_nodes_excluded(y_anom):
    valid = torch.ones(N_NODES, dtype=torch.bool)
    valid[200:] = False           # unlabelled users (Amazon case)
    sp = make_splits(y_anom, dataset="amazon", seed=0, valid_mask=valid)
    for mask in (sp.train, sp.val, sp.calib, sp.test):
        assert int(mask[~valid].sum()) == 0


# ---------------------------------------------------------------------------
# 10. Q2 target IDs and masks identical across constructions
# ---------------------------------------------------------------------------
def test_10_targets_identical_across_operators(toy_graph):
    built = build_matched_graphs(toy_graph, seed=0, dataset="toy")
    manifests = [built[op][1] for op in OPERATORS]
    reference = manifests[0].target_ids
    for m in manifests[1:]:
        assert m.target_ids == reference, f"{m.operator} drew different targets"
    for op in OPERATORS:
        g = built[op][0]
        assert torch.equal(
            torch.nonzero(g.y_anom).flatten(),
            torch.tensor(reference, dtype=torch.long),
        )


def test_10b_targets_exclude_benign_masks(toy_graph, y_anom):
    sp = make_splits(y_anom, dataset="toy", seed=0)
    excluded = sp.train | sp.val | sp.calib
    targets = select_targets(toy_graph.num_nodes, seed=0, excluded=excluded)
    assert int(excluded[torch.from_numpy(targets).long()].sum()) == 0


def test_10c_only_the_operator_differs(toy_graph):
    built = build_matched_graphs(toy_graph, seed=0, dataset="toy")
    # structural changes edges but not features
    assert torch.equal(built["structural"][0].x, toy_graph.x)
    assert built["structural"][0].edge_index.size(1) > toy_graph.edge_index.size(1)
    # attribute changes features but not edges
    assert torch.equal(built["attribute"][0].edge_index, toy_graph.edge_index)
    assert not torch.equal(built["attribute"][0].x, toy_graph.x)
    # hybrid does both
    assert built["hybrid"][0].edge_index.size(1) > toy_graph.edge_index.size(1)
    assert not torch.equal(built["hybrid"][0].x, toy_graph.x)


def test_10d_hybrid_is_structural_then_attribute(toy_graph):
    built = build_matched_graphs(toy_graph, seed=0, dataset="toy")
    assert torch.equal(
        built["hybrid"][0].edge_index, built["structural"][0].edge_index
    )
    assert torch.equal(built["hybrid"][0].x, built["attribute"][0].x)


def test_10e_attribute_donors_are_per_target(toy_graph):
    _, manifest = apply_operator(
        toy_graph, "attribute", select_targets(toy_graph.num_nodes, 0), 0
    )
    donors = list(manifest.donor_ids.values())
    assert len(donors) == len(manifest.target_ids)
    # legacy bug: one shared pool made donors collapse onto a few nodes
    assert len(set(donors)) > 1


# ---------------------------------------------------------------------------
# 15. fixed seeds reproduce split and corruption hashes
# ---------------------------------------------------------------------------
def test_15_split_hash_reproducible(y_anom):
    a = make_splits(y_anom, dataset="toy", seed=3)
    b = make_splits(y_anom, dataset="toy", seed=3)
    assert a.manifest.split_hash == b.manifest.split_hash
    c = make_splits(y_anom, dataset="toy", seed=4)
    assert c.manifest.split_hash != a.manifest.split_hash


def test_15b_corruption_hash_reproducible(toy_graph):
    a = build_matched_graphs(toy_graph, seed=1, dataset="toy")
    b = build_matched_graphs(toy_graph, seed=1, dataset="toy")
    for op in OPERATORS:
        assert a[op][1].corrupted_graph_hash == b[op][1].corrupted_graph_hash

    c = build_matched_graphs(toy_graph, seed=2, dataset="toy")
    assert c["structural"][1].target_ids != a["structural"][1].target_ids


def test_15c_graph_hash_detects_change(toy_graph):
    h0 = graph_hash(toy_graph.x, toy_graph.edge_index)
    x2 = toy_graph.x.clone()
    x2[0, 0] += 1.0
    assert graph_hash(x2, toy_graph.edge_index) != h0
