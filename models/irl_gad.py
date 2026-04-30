"""IRL-GAD: Graph Anomaly Detection via Inverse Reinforcement Learning.

This module ties together:

  M1 — TrajectoryGAT  (gat_encoder.TrajectoryGAT)
  M2 — RewardNetwork  (reward_network.RewardNetwork)
  M3 — SoftValueIteration / KL scoring  (soft_value_iteration.*)

The MaxEnt-IRL likelihood we maximise on normal trajectories is

    log P(τ | R_θ)  ∝  Σ_t  log π*_soft(a_t | s_t)

which under the Bellman-consistent soft policy is

    log π*_soft(a_t | s_t)  =  Q*(s_t, a_t)/β  -  V*(s_t)/β

The empirical "actions" along a normal trajectory are sampled
proportional to GAT attention α_v^{(t)}; we take the expected
per-trajectory log-likelihood under that distribution, which gives:

    L_IRL(θ)  =  - E_v∈V+ [ Σ_t Σ_u  α_v^{(t)}(u) · log π*_soft(u | s_t) ]
                 +  λ ‖θ‖²

This is what `compute_loss` returns (without the regularisation term;
that is added by the optimizer via weight_decay).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import torch
import torch.nn as nn

from models.gat_encoder import GATConfig, TrajectoryGAT
from models.reward_network import RewardConfig, RewardNetwork
from models.soft_value_iteration import (
    SVIConfig,
    attention_to_log_policy,
    kl_observed_vs_optimal,
    soft_value_iteration,
)


# ---------------------------------------------------------------------------
# config
# ---------------------------------------------------------------------------
@dataclass
class IRLGADConfig:
    in_dim: int                              # set by data
    hidden_dim: int = 256
    num_heads: int = 8
    num_layers: int = 2                      # K in the paper
    dropout: float = 0.1
    reward_hidden: int = 256
    use_temporal_reward: bool = True
    structural_extra: int = 4
    beta: float = 0.10                       # MaxEnt temperature
    gamma: float = 1.0                       # hop discount
    svi_iterations: int = 5
    # Ablation switches: set any to False to zero-out that head.
    use_str: bool = True
    use_sem: bool = True
    use_tmp: bool = True
    # Score override for the "GAT + recon score" / "GAT + contrastive score"
    # ablations. Default is "irl"; switching forces a non-IRL score
    # while keeping the GAT backbone fixed.
    score_mode: str = "irl"   # irl | reconstruction | contrastive


# ---------------------------------------------------------------------------
# model
# ---------------------------------------------------------------------------
class IRLGAD(nn.Module):
    def __init__(self, cfg: IRLGADConfig):
        super().__init__()
        self.cfg = cfg

        self.encoder = TrajectoryGAT(
            GATConfig(
                in_dim=cfg.in_dim,
                hidden_dim=cfg.hidden_dim,
                num_layers=cfg.num_layers,
                num_heads=cfg.num_heads,
                dropout=cfg.dropout,
            )
        )
        self.reward = RewardNetwork(
            RewardConfig(
                state_dim=cfg.hidden_dim,
                hidden_dim=cfg.reward_hidden,
                structural_extra=cfg.structural_extra,
                use_temporal=cfg.use_temporal_reward,
            )
        )
        self.svi = SVIConfig(
            beta=cfg.beta,
            gamma=cfg.gamma,
            num_iterations=cfg.svi_iterations,
        )

        # heads used only by ablation score modes
        self.feature_decoder = nn.Linear(cfg.hidden_dim, cfg.in_dim)
        self.proj_head = nn.Sequential(
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
            nn.ReLU(),
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
        )

    # ------------------------------------------------------------------
    # forward / training
    # ------------------------------------------------------------------
    def encode(
        self, x: torch.Tensor, edge_index: torch.Tensor
    ) -> dict[str, list[torch.Tensor] | torch.Tensor]:
        """Run the GAT encoder and return trajectory components."""
        hidden_per_hop, attn_per_hop = self.encoder(x, edge_index)
        return {
            "hidden_per_hop": hidden_per_hop,
            "attn_per_hop":   attn_per_hop,
            "h_final":        hidden_per_hop[-1],
        }

    def compute_per_hop_rewards(
        self,
        hidden_per_hop: List[torch.Tensor],
        edge_index: torch.Tensor,
        num_nodes: int,
        time_feats: Optional[torch.Tensor] = None,
    ) -> List[torch.Tensor]:
        """Return list length K of per-edge total rewards."""
        # Pre-compute structural features once (they only depend on graph).
        struct_feats = self.reward._structural_features(edge_index, num_nodes)

        rewards_per_hop: List[torch.Tensor] = []
        for t in range(self.cfg.num_layers):
            # state at hop t = hidden_per_hop[t] (input of layer t+1)
            h = hidden_per_hop[t]
            # need to align hidden dim for hop 0 (raw features have in_dim ≠ hidden_dim).
            # We project the raw feature trajectory into the encoder's
            # hidden space using the first layer's linear.
            if h.size(-1) != self.cfg.hidden_dim:
                # use the first GAT layer's projection (heads concatenated)
                first_layer = self.encoder.layers[0]
                with torch.no_grad():
                    h_proj = first_layer.lin(h).view(
                        h.size(0),
                        first_layer.heads * first_layer.out_dim_per_head,
                    )
                # detach? we want gradients to flow into the encoder
                h_proj = first_layer.lin(h).view(
                    h.size(0),
                    first_layer.heads * first_layer.out_dim_per_head,
                )
                h = h_proj

            h_src = h[edge_index[0]]
            h_dst = h[edge_index[1]]
            r_dict = self.reward(h_src, h_dst, struct_feats, time_feats)

            # ablation switches
            r = (
                (r_dict["str"] if self.cfg.use_str else 0.0)
                + (self.reward.lambda1 * r_dict["sem"] if self.cfg.use_sem else 0.0)
                + (self.reward.lambda2 * r_dict["tmp"] if self.cfg.use_tmp else 0.0)
            )
            if isinstance(r, float):
                r = torch.zeros_like(r_dict["str"])
            rewards_per_hop.append(r)
        return rewards_per_hop

    def compute_loss(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        node_mask: Optional[torch.Tensor] = None,
        time_feats: Optional[torch.Tensor] = None,
    ) -> dict[str, torch.Tensor]:
        """MaxEnt-IRL loss on the (normal) `node_mask` subset.

        Computes:
          (1) trajectory via GAT (hidden + attention per hop)
          (2) per-edge rewards
          (3) soft Q*, log π*_soft via soft value iteration
          (4) negative expected log-likelihood under empirical π_v (= attention)
        """
        n = x.size(0)
        out = self.encode(x, edge_index)
        hidden = out["hidden_per_hop"]
        attn   = out["attn_per_hop"]

        rewards = self.compute_per_hop_rewards(hidden, edge_index, n, time_feats)
        Q_per_hop, log_pi_star = soft_value_iteration(
            rewards, edge_index, n, self.svi
        )

        # Empirical policy from attention (per-source distribution).
        log_pi_obs = attention_to_log_policy(attn, edge_index, n)

        # NLL = - Σ_t Σ_e π_obs(e) · log π*_soft(e), aggregated per source node, then averaged
        # over the requested mask of normal nodes.
        src = edge_index[0]
        per_src_nll = torch.zeros(n, device=x.device)
        for t in range(len(attn)):
            p = torch.exp(log_pi_obs[t])
            contrib = -p * log_pi_star[t]
            per_src_nll.index_add_(0, src, contrib)

        if node_mask is None:
            loss = per_src_nll.mean()
        else:
            loss = per_src_nll[node_mask].mean()

        # Also report the un-masked anomaly score (for monitoring) —
        # this is just the KL up to a constant.
        score = kl_observed_vs_optimal(log_pi_obs, log_pi_star, edge_index, n)

        return {
            "loss":   loss,
            "score":  score.detach(),
            "lambda1": self.reward.lambda1.detach(),
            "lambda2": self.reward.lambda2.detach(),
        }

    # ------------------------------------------------------------------
    # inference / scoring
    # ------------------------------------------------------------------
    @torch.no_grad()
    def score(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        time_feats: Optional[torch.Tensor] = None,
    ) -> dict[str, torch.Tensor]:
        """Return anomaly scores per node according to `score_mode`."""
        n = x.size(0)
        out = self.encode(x, edge_index)
        hidden = out["hidden_per_hop"]
        attn   = out["attn_per_hop"]
        h_final = out["h_final"]

        if self.cfg.score_mode == "irl":
            rewards = self.compute_per_hop_rewards(hidden, edge_index, n, time_feats)
            _, log_pi_star = soft_value_iteration(rewards, edge_index, n, self.svi)
            log_pi_obs = attention_to_log_policy(attn, edge_index, n)
            score = kl_observed_vs_optimal(log_pi_obs, log_pi_star, edge_index, n)
            # also expose a "reward feature" suitable for t-SNE
            reward_feat = torch.stack(
                [r.new_zeros(n).index_add_(0, edge_index[0], r)
                 for r in rewards],
                dim=1,
            )
            return {"score": score, "h_final": h_final, "reward_feat": reward_feat}

        if self.cfg.score_mode == "reconstruction":
            x_hat = self.feature_decoder(h_final)
            recon_err = ((x_hat - x) ** 2).mean(dim=-1)
            return {"score": recon_err, "h_final": h_final, "reward_feat": h_final}

        if self.cfg.score_mode == "contrastive":
            # Simple instance-discrimination style score: distance to the
            # mean projection. Higher = more anomalous.
            z = self.proj_head(h_final)
            z = torch.nn.functional.normalize(z, dim=-1)
            mu = z.mean(dim=0, keepdim=True)
            score = 1.0 - (z * mu).sum(dim=-1)
            return {"score": score, "h_final": h_final, "reward_feat": h_final}

        raise ValueError(f"Unknown score_mode: {self.cfg.score_mode}")
