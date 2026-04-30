"""Decomposed reward network used by MaxEnt-GIRL.

The reward is decomposed into three interpretable axes:

    R_θ(s, a) = R^str_{θ_1}(s, a)
              + λ_1 · R^sem_{θ_2}(s, a)
              + λ_2 · R^tmp_{θ_3}(s, a)

Each head is a small MLP on top of a shared GNN backbone φ_θ.
For computational efficiency we operate on per-edge inputs:
each (s, a) pair corresponds to an edge (v, u) at hop t, with state
features taken as the source representation h_v^{(t)} and the action
features taken as the destination representation h_u^{(t)} together
with a small set of structural / semantic / temporal cues.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class RewardConfig:
    state_dim: int          # dimensionality of h_v^{(t)}
    hidden_dim: int = 256
    structural_extra: int = 4   # dim of hand-crafted structural features
    use_temporal: bool = True


class _MLPHead(nn.Module):
    def __init__(self, in_dim: int, hidden: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class RewardNetwork(nn.Module):
    """R_θ(s, a) = R^str + λ_1 R^sem + λ_2 R^tmp.

    The mixing coefficients λ_1, λ_2 are *learned* via a softplus
    parameterization so they remain non-negative.
    """

    def __init__(self, cfg: RewardConfig):
        super().__init__()
        self.cfg = cfg
        # Each head sees a different feature subset:
        #   structural  : [h_v, h_u, structural_features]
        #   semantic    : [h_v, h_u, |h_v - h_u|, h_v * h_u]
        #   temporal    : [h_v, h_u, time_features]
        self.head_str = _MLPHead(2 * cfg.state_dim + cfg.structural_extra, cfg.hidden_dim)
        self.head_sem = _MLPHead(4 * cfg.state_dim, cfg.hidden_dim)
        self.head_tmp = (
            _MLPHead(2 * cfg.state_dim + 2, cfg.hidden_dim)
            if cfg.use_temporal else None
        )

        # Learnable non-negative mixing coefficients (initialised at log(1)).
        self.raw_lambda1 = nn.Parameter(torch.zeros(()))
        self.raw_lambda2 = nn.Parameter(torch.zeros(()))

    @property
    def lambda1(self) -> torch.Tensor:
        return F.softplus(self.raw_lambda1)

    @property
    def lambda2(self) -> torch.Tensor:
        return F.softplus(self.raw_lambda2)

    @staticmethod
    def _structural_features(
        edge_index: torch.Tensor,
        num_nodes: int,
        edge_subset: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Cheap structural cues per edge: src_deg, dst_deg, deg_ratio, common_neighbors_proxy.

        We use degrees as a proxy for richer features so the cost stays
        O(|E|).
        """
        deg = torch.zeros(num_nodes, device=edge_index.device)
        deg.index_add_(0, edge_index[1], torch.ones(edge_index.size(1), device=edge_index.device))
        src_deg = deg[edge_index[0]]
        dst_deg = deg[edge_index[1]]
        ratio   = src_deg / (dst_deg + 1.0)
        # `common_neighbors_proxy` ~ log(1 + min(src_deg, dst_deg))
        proxy   = torch.log1p(torch.minimum(src_deg, dst_deg))
        feats = torch.stack([
            torch.log1p(src_deg),
            torch.log1p(dst_deg),
            ratio.clamp(0, 10),
            proxy,
        ], dim=1)
        if edge_subset is not None:
            feats = feats[edge_subset]
        return feats

    def forward(
        self,
        h_src: torch.Tensor,        # (E, d) — h_v^{(t)} for each edge's source
        h_dst: torch.Tensor,        # (E, d) — h_u^{(t)} for each edge's destination
        struct_feats: torch.Tensor, # (E, structural_extra)
        time_feats: torch.Tensor | None = None,  # (E, 2) or None
    ) -> dict[str, torch.Tensor]:
        """Return per-edge rewards as well as per-component values."""
        x_str = torch.cat([h_src, h_dst, struct_feats], dim=-1)
        r_str = self.head_str(x_str)

        diff = (h_src - h_dst).abs()
        prod = h_src * h_dst
        x_sem = torch.cat([h_src, h_dst, diff, prod], dim=-1)
        r_sem = self.head_sem(x_sem)

        if self.head_tmp is not None and time_feats is not None:
            x_tmp = torch.cat([h_src, h_dst, time_feats], dim=-1)
            r_tmp = self.head_tmp(x_tmp)
        else:
            r_tmp = torch.zeros_like(r_str)

        total = r_str + self.lambda1 * r_sem + self.lambda2 * r_tmp
        return {
            "total":  total,
            "str":    r_str,
            "sem":    r_sem,
            "tmp":    r_tmp,
            "lambda1": self.lambda1.detach(),
            "lambda2": self.lambda2.detach(),
        }
