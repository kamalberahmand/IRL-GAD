"""Full-attention policies and the exact per-hop KL score (contract 2.1-2.2).

Replaces `models.soft_value_iteration`, which built the reference policy
through a Bellman loop parameterised by `T` (svi_iterations) and `gamma`.
Contract 2.4 removes both from the active protocol, so the reference
policy here is a plain temperature softmax of the combined reward over the
same neighbour support as the observed policy:

    p_k^v = pi_v^(k)                    (the GAT attention distribution)
    q_k^v = softmax_{u in N(v)} R_theta(v, u) / beta

    d_k(v) = KL( p_k^v || q_k^v )
    S(v)   = sum_{k=1..K} d_k(v)

Both distributions are normalised over the *same* index axis, the same
edge set, and the same masking. `score_from_policies` returns the per-hop
terms so that `S == sum_k d_k` is directly testable (test 4).

Two conflicts with the legacy code are resolved here and recorded in
CODE_PAPER_ALIGNMENT.md:

  - Direction. The legacy encoder softmaxed attention over incoming edges
    per destination, then `attention_to_log_policy` re-normalised it per
    source, so `p` was neither the GAT distribution nor aligned with `q`.
    Both are now normalised over the same `index` axis, supplied by the
    caller, and the encoder is configured to match.
  - Masking. Dropout was applied to attention after softmax and before
    export, so masked neighbours survived as ~1e-12 probability actions.
    Attention used as a policy is now exported pre-dropout.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import torch

NORM_TOL = 1e-4


@dataclass
class PolicyConfig:
    beta: float = 0.10
    check: bool = True          # run the contract 2.1 assertions
    tol: float = NORM_TOL


# ---------------------------------------------------------------------------
# segment ops
# ---------------------------------------------------------------------------
def _segment_max(values: torch.Tensor, index: torch.Tensor, n: int) -> torch.Tensor:
    out = values.new_full((n,), float("-inf"))
    out = out.index_reduce_(0, index, values, reduce="amax", include_self=True)
    return out


def _segment_logsumexp(values: torch.Tensor, index: torch.Tensor, n: int) -> torch.Tensor:
    """Numerically stable per-segment logsumexp.

    Segments with no members return -inf, which propagates to an empty
    support rather than a silent -1e30 shift (legacy bug, audit F9).
    """
    mx = _segment_max(values, index, n)
    safe_mx = torch.where(torch.isfinite(mx), mx, torch.zeros_like(mx))
    shifted = torch.exp(values - safe_mx[index])
    denom = torch.zeros(n, device=values.device, dtype=values.dtype)
    denom.index_add_(0, index, shifted)
    return safe_mx + torch.log(denom.clamp(min=torch.finfo(values.dtype).tiny))


def segment_log_softmax(
    values: torch.Tensor, index: torch.Tensor, n: int
) -> torch.Tensor:
    """Log-softmax of `values` within each segment given by `index`."""
    return values - _segment_logsumexp(values, index, n)[index]


# ---------------------------------------------------------------------------
# policies
# ---------------------------------------------------------------------------
def observed_log_policy(
    alpha_per_hop: List[torch.Tensor],
    index: torch.Tensor,
    n: int,
) -> List[torch.Tensor]:
    """Observed policy p_k = pi_v^(k) in log space.

    `alpha_per_hop[k]` holds the GAT attention already normalised over
    `index` by the encoder. It is re-expressed in log space and
    renormalised only to absorb floating-point drift; the distribution is
    not redirected.
    """
    out = []
    for alpha in alpha_per_hop:
        log_a = torch.log(alpha.clamp_min(torch.finfo(alpha.dtype).tiny))
        out.append(segment_log_softmax(log_a, index, n))
    return out


def reference_log_policy(
    rewards_per_hop: List[torch.Tensor],
    index: torch.Tensor,
    n: int,
    beta: float,
) -> List[torch.Tensor]:
    """Reward-induced reference policy q_k over the same support as p_k."""
    if beta <= 0:
        raise ValueError(f"beta must be positive, got {beta}")
    return [segment_log_softmax(r / beta, index, n) for r in rewards_per_hop]


# ---------------------------------------------------------------------------
# assertions (contract 2.1)
# ---------------------------------------------------------------------------
def assert_valid_policies(
    log_p_per_hop: List[torch.Tensor],
    log_q_per_hop: List[torch.Tensor],
    index: torch.Tensor,
    n: int,
    tol: float = NORM_TOL,
) -> None:
    """Finite, non-negative, aligned, and summing to one per node and hop."""
    if len(log_p_per_hop) != len(log_q_per_hop):
        raise AssertionError(
            f"hop count mismatch: p has {len(log_p_per_hop)}, q has {len(log_q_per_hop)}"
        )

    # Nodes with at least one outgoing edge; empty supports are skipped
    # rather than asserted to sum to one.
    has_support = torch.zeros(n, dtype=torch.bool, device=index.device)
    has_support[index] = True

    for k, (log_p, log_q) in enumerate(zip(log_p_per_hop, log_q_per_hop)):
        if log_p.shape != log_q.shape:
            raise AssertionError(
                f"hop {k}: support mismatch, p{tuple(log_p.shape)} vs q{tuple(log_q.shape)}"
            )
        if log_p.numel() != index.numel():
            raise AssertionError(
                f"hop {k}: policy has {log_p.numel()} entries but index has {index.numel()}"
            )
        if not torch.isfinite(log_p).all():
            raise AssertionError(f"hop {k}: observed policy has non-finite entries")
        if not torch.isfinite(log_q).all():
            raise AssertionError(f"hop {k}: reference policy has non-finite entries")

        for name, log_pi in (("observed", log_p), ("reference", log_q)):
            prob = torch.exp(log_pi)
            if (prob < -tol).any():
                raise AssertionError(f"hop {k}: {name} policy has negative mass")
            mass = torch.zeros(n, device=log_pi.device, dtype=log_pi.dtype)
            mass.index_add_(0, index, prob)
            err = (mass[has_support] - 1.0).abs().max()
            if float(err) > tol:
                raise AssertionError(
                    f"hop {k}: {name} policy off-normal by {float(err):.2e} (tol {tol:.0e})"
                )


# ---------------------------------------------------------------------------
# score
# ---------------------------------------------------------------------------
def score_from_policies(
    log_p_per_hop: List[torch.Tensor],
    log_q_per_hop: List[torch.Tensor],
    index: torch.Tensor,
    n: int,
    check: bool = True,
    tol: float = NORM_TOL,
) -> Tuple[torch.Tensor, List[torch.Tensor]]:
    """Return (S, [d_1..d_K]) with S exactly the sum of the per-hop terms.

    The total is accumulated by summing the stored `d_k` tensors, not by
    a separate reduction, so `S == sum_k d_k` holds to floating point by
    construction (contract 2.2, test 4).
    """
    if check:
        assert_valid_policies(log_p_per_hop, log_q_per_hop, index, n, tol)

    d_per_hop: List[torch.Tensor] = []
    for log_p, log_q in zip(log_p_per_hop, log_q_per_hop):
        p = torch.exp(log_p)
        contrib = p * (log_p - log_q)
        d_k = torch.zeros(n, device=log_p.device, dtype=log_p.dtype)
        d_k.index_add_(0, index, contrib)
        # KL is non-negative in exact arithmetic; clamp only cancellation noise.
        neg = float(d_k.min())
        if neg < -tol:
            raise AssertionError(f"per-hop KL is negative ({neg:.2e}); supports differ")
        d_per_hop.append(d_k.clamp_min(0.0))

    score = torch.stack(d_per_hop, dim=0).sum(dim=0)
    return score, d_per_hop
