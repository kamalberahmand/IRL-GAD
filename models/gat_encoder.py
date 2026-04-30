"""GAT encoder that exposes per-hop attention coefficients.

Why a custom GAT layer rather than `torch_geometric.nn.GATConv`?
The IRL framework treats α_v^{(t)} as the *aggregation policy* of node
v at hop t and needs the per-edge attention values made available at
every hop alongside the hidden representations h_v^{(t)}. Stock GATConv
does expose attention via `return_attention_weights=True`, but the API
returns it on a sparse (edge, head) layout that is awkward to align
with policy-deviation scoring. We implement a thin custom layer that
explicitly returns both the hidden state and an `(E,)` attention
tensor in the same edge ordering as `edge_index`.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import softmax as edge_softmax


@dataclass
class GATConfig:
    in_dim: int
    hidden_dim: int = 256
    num_layers: int = 2
    num_heads: int = 8
    dropout: float = 0.1
    negative_slope: float = 0.2


class _PolicyGATLayer(nn.Module):
    """Single multi-head GAT layer that returns averaged-head attention.

    For policy deviation scoring we need *one* distribution per
    (node, hop) pair, so the multi-head attention is averaged across
    heads after the per-head softmax. The hidden representation is
    concatenated across heads as in the original GAT.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        heads: int,
        dropout: float,
        negative_slope: float,
        concat: bool = True,
    ):
        super().__init__()
        self.heads = heads
        self.out_dim_per_head = out_dim
        self.concat = concat
        self.dropout = dropout
        self.negative_slope = negative_slope

        self.lin = nn.Linear(in_dim, heads * out_dim, bias=False)
        # attention: a^T [Wh_i || Wh_j] split into source/target halves
        self.att_src = nn.Parameter(torch.empty(1, heads, out_dim))
        self.att_dst = nn.Parameter(torch.empty(1, heads, out_dim))
        self.bias = nn.Parameter(torch.zeros(heads * out_dim if concat else out_dim))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.lin.weight)
        nn.init.xavier_uniform_(self.att_src)
        nn.init.xavier_uniform_(self.att_dst)
        nn.init.zeros_(self.bias)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (h_out, alpha_per_edge).

        h_out : (N, heads * out_dim) if concat else (N, out_dim)
        alpha_per_edge : (E,)  — head-averaged attention weights aligned with edge_index
        """
        n = x.size(0)
        h = self.lin(x).view(n, self.heads, self.out_dim_per_head)

        src, dst = edge_index[0], edge_index[1]
        alpha_src = (h * self.att_src).sum(dim=-1)  # (N, heads)
        alpha_dst = (h * self.att_dst).sum(dim=-1)  # (N, heads)
        alpha = alpha_src[src] + alpha_dst[dst]     # (E, heads)
        alpha = F.leaky_relu(alpha, self.negative_slope)
        # softmax over incoming edges of each destination node
        alpha = edge_softmax(alpha, dst, num_nodes=n)
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        # propagate
        msg = h[src] * alpha.unsqueeze(-1)          # (E, heads, out_dim)
        out = torch.zeros(n, self.heads, self.out_dim_per_head,
                          device=x.device, dtype=x.dtype)
        out.index_add_(0, dst, msg)

        if self.concat:
            out = out.reshape(n, self.heads * self.out_dim_per_head)
        else:
            out = out.mean(dim=1)
        out = out + self.bias

        # alpha for policy = head-averaged attention (one number per edge)
        alpha_pooled = alpha.mean(dim=-1)
        return out, alpha_pooled


class TrajectoryGAT(nn.Module):
    """K-layer GAT that returns the full state-policy trajectory.

    forward returns:
      hidden_per_hop : list of length K+1 of (N, d) tensors
                       hidden_per_hop[0] = input features
                       hidden_per_hop[t] = h_v^{(t)}
      attn_per_hop   : list of length K of (E,) tensors aligned with
                       `edge_index` — these are the policies π_v^{(t)}.
    """

    def __init__(self, cfg: GATConfig):
        super().__init__()
        self.cfg = cfg
        layers = []
        in_dim = cfg.in_dim
        per_head = cfg.hidden_dim // cfg.num_heads
        for layer_i in range(cfg.num_layers):
            out_dim = per_head
            concat = True
            layers.append(
                _PolicyGATLayer(
                    in_dim=in_dim,
                    out_dim=out_dim,
                    heads=cfg.num_heads,
                    dropout=cfg.dropout,
                    negative_slope=cfg.negative_slope,
                    concat=concat,
                )
            )
            in_dim = cfg.hidden_dim
        self.layers = nn.ModuleList(layers)
        self.out_dim = cfg.hidden_dim

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        hidden_per_hop = [x]
        attn_per_hop = []
        h = x
        for layer in self.layers:
            h, alpha = layer(h, edge_index)
            h = F.elu(h)
            hidden_per_hop.append(h)
            attn_per_hop.append(alpha)
        return hidden_per_hop, attn_per_hop
