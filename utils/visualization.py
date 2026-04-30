"""Visualization utilities (mirrors Section 6.6 of the paper)."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import torch


def reward_landscape_tsne(
    embeddings: torch.Tensor | np.ndarray,
    reward_features: torch.Tensor | np.ndarray,
    y_anom: torch.Tensor | np.ndarray,
    out_path: str = "figure_reward_landscape.pdf",
    seed: int = 0,
    sample_per_class: int = 250,
    structural_mask: Optional[torch.Tensor | np.ndarray] = None,
) -> None:
    """Render the two-panel t-SNE figure used in the paper.

    Parameters
    ----------
    embeddings : (N, d) GAT-final embeddings.
    reward_features : (N, d') per-hop concatenated reward vector.
    y_anom : (N,) binary anomaly mask.
    structural_mask : optional (N,) binary mask distinguishing
        structural anomalies from camouflaged ones, used purely for
        coloring. If None, all anomalies are drawn as "camouflaged".
    """
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from sklearn.manifold import TSNE

    def _to_numpy(t):
        return t.detach().cpu().numpy() if isinstance(t, torch.Tensor) else np.asarray(t)

    h = _to_numpy(embeddings)
    r = _to_numpy(reward_features)
    y = _to_numpy(y_anom).astype(int)
    s = _to_numpy(structural_mask).astype(int) if structural_mask is not None else None

    rng = np.random.default_rng(seed)
    n = h.shape[0]
    normal_idx = np.where(y == 0)[0]
    anom_idx   = np.where(y == 1)[0]
    if s is None:
        camo_idx, struct_idx = anom_idx, np.array([], dtype=int)
    else:
        camo_idx   = anom_idx[s[anom_idx] == 0]
        struct_idx = anom_idx[s[anom_idx] == 1]

    # subsample for visualization
    def _samp(idx, k):
        if len(idx) <= k:
            return idx
        return rng.choice(idx, size=k, replace=False)
    n_idx = _samp(normal_idx, sample_per_class)
    c_idx = _samp(camo_idx,   max(40, sample_per_class // 6))
    s_idx = _samp(struct_idx, max(20, sample_per_class // 10))

    keep = np.concatenate([n_idx, c_idx, s_idx])
    labels = np.concatenate([
        np.zeros(len(n_idx), dtype=int),
        np.ones(len(c_idx), dtype=int),
        np.full(len(s_idx), 2, dtype=int),
    ])

    Z_emb = TSNE(n_components=2, perplexity=30,
                 init="pca", random_state=seed).fit_transform(h[keep])
    Z_rew = TSNE(n_components=2, perplexity=30,
                 init="pca", random_state=seed).fit_transform(r[keep])

    COL = ["#9E9E9E", "#FF8C00", "#E63946"]
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(7, 3.2),
                                    gridspec_kw={"wspace": 0.18})
    for cls, name in enumerate(["Normal", "Camouflaged", "Structural"]):
        m = labels == cls
        axL.scatter(Z_emb[m, 0], Z_emb[m, 1], c=COL[cls], s=18,
                    alpha=0.78, edgecolors="white", linewidths=0.4)
        axR.scatter(Z_rew[m, 0], Z_rew[m, 1], c=COL[cls], s=18,
                    alpha=0.78, edgecolors="white", linewidths=0.4)

    axL.set_title(r"Embedding space $\mathbf{h}_v^{(K)}$",
                  fontsize=11, style="italic", pad=6)
    axR.set_title(r"Reward space $\mathcal{S}(v)$",
                  fontsize=11, style="italic", pad=6)
    for ax in (axL, axR):
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_color("#BFBFBF"); sp.set_linewidth(0.8)
    axL.set_xlabel("t-SNE 1", fontsize=9)
    axL.set_ylabel("t-SNE 2", fontsize=9)
    axR.set_xlabel("policy deviation (t-SNE 1)", fontsize=9)
    axR.set_ylabel("t-SNE 2", fontsize=9)

    handles = [Line2D([0],[0], marker="o", color="none",
                      markerfacecolor=COL[i], markeredgecolor="white",
                      markeredgewidth=0.4, markersize=8,
                      label=name)
               for i, name in enumerate(["Normal","Camouflaged anomaly","Structural anomaly"])]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False,
               fontsize=9, bbox_to_anchor=(0.5, -0.04),
               handletextpad=0.4, columnspacing=2.0)

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
