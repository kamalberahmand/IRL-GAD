"""Soft value iteration for MaxEnt IRL on the Node-MDP.

The Node-MDP for a node v has:
  state    s_t  : augmented neighborhood representation at hop t
  action   a    : choice of neighbor u in N(v)
  reward   R_θ(s_t, u)  : per-edge scalar from RewardNetwork
  transition T  : deterministic GAT update
  horizon  K    : number of GNN layers

For policy deviation scoring we need the *soft Bellman optimal* policy
at every (state, hop) pair:

    π*_soft(u | s_t)  =  exp(Q*(s_t, u) / β) / Σ_{u'} exp(Q*(s_t, u') / β)

with the soft Bellman equations

    Q*(s_t, u)  =  R_θ(s_t, u)  +  γ V*(s_{t+1})
    V*(s_t)     =  β · log Σ_u exp(Q*(s_t, u) / β)

Because in the Node-MDP, transitioning means *moving one hop forward
through the GNN*, the next state value V*(s_{t+1}) is a property of
the destination node at the next hop — not a separate continuous-state
backup. We therefore run the soft Bellman backup *along edges, hop by
hop, in reverse time order*, which is what is implemented below.

This makes a full soft VI pass cost O(K · |E|), matching the cost of a
GNN forward pass (the paper's complexity claim).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import torch

from torch_geometric.utils import softmax as edge_softmax


@dataclass
class SVIConfig:
    beta: float = 0.10        # MaxEnt temperature
    gamma: float = 1.0        # hop discount
    num_iterations: int = 5   # number of inner-loop fixpoint sweeps


def soft_value_iteration(
    rewards_per_hop: List[torch.Tensor],   # list length K, each (E,)
    edge_index: torch.Tensor,              # (2, E)
    num_nodes: int,
    cfg: SVIConfig,
) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    """Compute (Q*_per_hop, log_pi_star_per_hop).

    Each Q*_per_hop[t] has shape (E,). The soft optimal policy at hop t
    is recovered by edge-softmax over destinations:

        log π*_soft(u | s_t) = Q*_per_hop[t] / β  -  V*_per_hop[t][src]

    where V*_per_hop[t] is the per-source-node log-sum-exp.

    We compute iteratively for `num_iterations` sweeps:
        Q*[t]   = R[t] + γ V*[t+1]
        V*[t]   = β logsumexp_u (Q*[t,u]/β) per source node

    Initialisation: V*[K] = 0 everywhere (terminal value).
    """
    K = len(rewards_per_hop)
    src = edge_index[0]

    # initialise V*_t for t = K..0 with zeros (terminal value)
    V_per_hop = [
        torch.zeros(num_nodes, device=edge_index.device)
        for _ in range(K + 1)
    ]
    Q_per_hop: List[torch.Tensor] = [None] * K  # type: ignore[assignment]

    for _it in range(cfg.num_iterations):
        # backward pass: t = K-1, K-2, ..., 0
        for t in reversed(range(K)):
            # Q*[t,e] = R[t,e] + γ V*[t+1, dst(e)]   (next-state value at destination)
            dst = edge_index[1]
            Q_t = rewards_per_hop[t] + cfg.gamma * V_per_hop[t + 1][dst]
            Q_per_hop[t] = Q_t

            # V*[t, v] = β · log Σ_{u in N(v)} exp(Q*[t, (v,u)] / β)
            scaled = Q_t / cfg.beta
            # for numerical stability, subtract per-source max before exp
            src_max = _scatter_max_per_source(scaled, src, num_nodes)
            shifted = scaled - src_max[src]
            exp_shifted = torch.exp(shifted)
            sum_exp = torch.zeros(num_nodes, device=Q_t.device)
            sum_exp.index_add_(0, src, exp_shifted)
            log_sum_exp = src_max + torch.log(sum_exp.clamp(min=1e-20))
            V_per_hop[t] = cfg.beta * log_sum_exp

    # build log π*_soft per hop, normalised over outgoing-from-src edges
    log_pi_per_hop: List[torch.Tensor] = []
    for t in range(K):
        Q_t = Q_per_hop[t]
        # log π*(e) = Q*[t,e]/β - V*[t, src(e)]/β
        log_pi = Q_t / cfg.beta - V_per_hop[t][src] / cfg.beta
        log_pi_per_hop.append(log_pi)

    return Q_per_hop, log_pi_per_hop


def _scatter_max_per_source(
    values: torch.Tensor,
    src: torch.Tensor,
    num_nodes: int,
) -> torch.Tensor:
    """Return per-source max of `values` for use as a logsumexp shift."""
    # initialise to a very negative number; index_reduce with amax requires
    # PyTorch >= 1.13 — fall back to a manual implementation otherwise.
    out = values.new_full((num_nodes,), -1e30)
    if hasattr(out, "index_reduce_"):
        out = out.index_reduce_(0, src, values, reduce="amax", include_self=True)
    else:
        # slow but always correct fallback
        for i in range(values.numel()):
            s = int(src[i].item())
            v = values[i]
            if v > out[s]:
                out[s] = v
    return out


def kl_observed_vs_optimal(
    log_pi_obs_per_hop: List[torch.Tensor],   # observed policy α^{(t)} per edge (in log-space if available)
    log_pi_star_per_hop: List[torch.Tensor],  # optimal policy in log-space per edge
    edge_index: torch.Tensor,
    num_nodes: int,
) -> torch.Tensor:
    """Per-source-node KL(π_v || π*_soft), summed over hops.

    Both `log_pi_obs_per_hop` and `log_pi_star_per_hop` must contain
    log-probabilities in the same edge ordering used to build them.
    Returns a tensor of shape (num_nodes,).
    """
    src = edge_index[0]
    K = len(log_pi_obs_per_hop)
    score = torch.zeros(num_nodes, device=edge_index.device)
    for t in range(K):
        # KL contribution per edge: π_obs * (log π_obs - log π*_soft)
        log_p = log_pi_obs_per_hop[t]
        log_q = log_pi_star_per_hop[t]
        p = torch.exp(log_p)
        contrib = p * (log_p - log_q)
        score.index_add_(0, src, contrib)
    return score


def attention_to_log_policy(
    alpha_per_hop: List[torch.Tensor],   # list of (E,) attention values from GAT
    edge_index: torch.Tensor,
    num_nodes: int,
) -> List[torch.Tensor]:
    """Convert raw attention to per-source log-probabilities.

    The GAT layer normalises attention with softmax over *incoming*
    edges (per destination). For policy semantics we need the
    distribution to be over *outgoing* edges (per source) — the
    aggregation policy of node v over its neighbors. We therefore
    re-normalise per-source.
    """
    src = edge_index[0]
    log_pi_per_hop: List[torch.Tensor] = []
    for alpha in alpha_per_hop:
        # ensure positivity & convert to a per-source distribution
        alpha = alpha.clamp(min=1e-12)
        # log alpha minus per-source logsumexp (= per-source softmax in log space)
        log_a = torch.log(alpha)
        src_max = _scatter_max_per_source(log_a, src, num_nodes)
        shifted = log_a - src_max[src]
        exp_shift = torch.exp(shifted)
        denom = torch.zeros(num_nodes, device=alpha.device)
        denom.index_add_(0, src, exp_shift)
        log_norm = src_max + torch.log(denom.clamp(min=1e-20))
        log_pi_per_hop.append(log_a - log_norm[src])
    return log_pi_per_hop
