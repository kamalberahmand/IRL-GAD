# IRL-GAD: Graph Anomaly Detection via Inverse Reinforcement Learning as Normality Modeling

> **Anonymous code release accompanying the IEEE TNNLS submission of the same title.**
> All author identifying information has been removed for double-blind review.

This repository contains a clean, modular PyTorch / PyTorch-Geometric
implementation of IRL-GAD, a graph anomaly detection method that
recasts the problem as inverse reinforcement learning: a latent reward
function is inferred from the aggregation trajectories of normal
nodes, and anomalies are scored by the KL divergence between the
observed aggregation policy and the soft Bellman-optimal policy
under that reward.

---

## ⚠️ Reproducibility note

The code in this repository is implemented to be **mathematically
faithful** to the manuscript: the Node-MDP construction, the
MaxEnt-IRL likelihood, the soft value iteration, the per-hop KL
score, and the three-component reward decomposition match
Equations (1)–(12) of the paper.

However, the **specific numerical values reported in the paper's
tables (Tables 3–7 and the sensitivity grid) reflect the authors'
ongoing experimental measurements**; the values committed to a
given camera-ready version may differ from those produced by an
arbitrary single run of this code, depending on hardware, dataset
versions, and random seeds. We recommend that any third-party
benchmarking treat the manuscript numbers as the *target* and
recompute on local hardware with the seeds documented below
(`{0, 1, 2, 3, 4}`).

If a number you produce here disagrees materially with the paper,
please verify (i) the dataset version, (ii) the hyperparameters in
`configs/<dataset>.yaml`, and (iii) that the unsupervised training
protocol (no anomaly labels at training time) is being applied. Do
**not** assume a discrepancy implies a code bug without checking
these three.

---

## Repository layout

```
IRL-GAD/
├── README.md                       (this file)
├── requirements.txt
├── LICENSE
├── .gitignore
├── main.py                         (CLI dispatch)
├── train.py                        (training loop)
├── evaluate.py                     (standalone eval from checkpoint)
├── configs/
│   ├── default.yaml
│   ├── cora.yaml
│   ├── citeseer.yaml
│   ├── amazon.yaml
│   ├── yelpchi.yaml
│   ├── jodie.yaml
│   └── ogbn_arxiv.yaml
├── models/
│   ├── gat_encoder.py              (M1 — TrajectoryGAT)
│   ├── reward_network.py           (M2 — three-head reward, R^str + λ_1 R^sem + λ_2 R^tmp)
│   ├── soft_value_iteration.py     (M3a — soft Bellman backup)
│   └── irl_gad.py                  (top-level model wiring all three)
├── utils/
│   ├── data_utils.py               (six benchmark loaders)
│   ├── anomaly_injection.py        (structural / attribute / contextual / hybrid)
│   ├── metrics.py                  (AUC-ROC, AUC-PR, TPR@5%FPR)
│   ├── visualization.py            (t-SNE reward landscape, Fig. 2)
│   └── seed.py                     (deterministic seeding)
├── scripts/
│   ├── run_all_benchmarks.sh
│   ├── ablation.sh
│   └── openset.sh
└── experiments/
    ├── ablation.py                 (Table 6)
    ├── openset.py                  (Table 5)
    └── scalability.py              (Table 7)
```

---

## Installation

```bash
# 1. Create a fresh environment (Python 3.10+ recommended)
python -m venv .venv
source .venv/bin/activate

# 2. Install PyTorch first (consult the official wheel for your CUDA)
#    https://pytorch.org/get-started/locally/
pip install "torch>=2.0,<2.5"

# 3. Install the rest
pip install -r requirements.txt
```

GPU is recommended for `yelpchi`, `amazon`, `jodie`, and
`ogbn_arxiv`; the homophilic benchmarks (Cora, Citeseer) train on
CPU in a few minutes.

---

## Datasets

All six datasets download automatically into `./data/` on first use:

| Dataset      | Source                                             | Notes                                              |
| ------------ | -------------------------------------------------- | -------------------------------------------------- |
| Cora         | `torch_geometric.datasets.Planetoid`               | structural anomalies injected (5%)                 |
| Citeseer     | `torch_geometric.datasets.Planetoid`               | structural anomalies injected (5%)                 |
| Amazon       | `dgl.data.FraudAmazonDataset`                      | organic anomaly labels                             |
| YelpChi      | `dgl.data.FraudYelpDataset`                        | organic anomaly labels                             |
| JODIE        | `torch_geometric.datasets.JODIEDataset`            | Wikipedia by default; node-level anomaly labels    |
| ogbn-arxiv   | `ogb.nodeproppred.PygNodePropPredDataset`          | structural anomalies injected (5%)                 |

The first dataset access on a fresh machine will trigger a one-time
download of a few hundred MB.

---

## Quick start

### Train on Cora

```bash
python main.py train --config configs/cora.yaml
```

This writes a checkpoint to
`experiments/runs/cora_seed0_K2/best.pt` and prints validation /
final metrics.

### Run all six benchmarks

```bash
bash scripts/run_all_benchmarks.sh
```

### Component ablation (Table 6)

```bash
bash scripts/ablation.sh                 # YelpChi by default
bash scripts/ablation.sh configs/cora.yaml
```

Output: `experiments/ablation_results.json` with AUC-ROC for the
seven variants:

```
{ "full":            { "auc_roc": ... },
  "no_R_tmp":        { "auc_roc": ... },
  "no_R_sem":        { "auc_roc": ... },
  "no_R_str":        { "auc_roc": ... },
  "recon_score":     { "auc_roc": ... },     # GAT + reconstruction
  "contrast_score":  { "auc_roc": ... },     # GAT + contrastive
  "single_reward":   { "auc_roc": ... } }
```

### Open-set generalization (Table 5)

```bash
bash scripts/openset.sh                 # YelpChi
```

Trains on structural anomalies only, then evaluates the same
checkpoint on attribute, contextual, and hybrid anomalies.

### Scalability benchmark (Table 7)

```bash
python -m experiments.scalability \
    --config configs/ogbn_arxiv.yaml \
    --epochs 10 --warmup 2
```

Reports `time_per_epoch_sec` (averaged over `epochs - warmup`
iterations) and `peak_gpu_mb`.

### Reward-landscape visualization (Figure 2)

```bash
python main.py visualize \
    --config configs/yelpchi.yaml \
    --ckpt   experiments/runs/yelpchi_seed0_K2/best.pt \
    --out    figure_reward_landscape.pdf
```

---

## Configuration overrides

`main.py train` supports inline overrides:

```bash
python main.py train --config configs/yelpchi.yaml \
    --set training.lr=5e-4 model.beta=0.05 model.num_layers=3
```

The override syntax is `<section>.<field>=<value>` where `<section>`
is one of `dataset`, `model`, `training`.

---

## Reproducibility

* All RNGs are seeded via `utils.seed.set_seed(seed)`.
  cuDNN is forced into deterministic mode by default.
* The standard seed list used in the paper is `{0, 1, 2, 3, 4}`.
  Re-run with `--set training.seed=K dataset.seed=K`.
* Datasets, hyperparameters, and code revision should all be
  reported alongside any reproduction attempt.
* Hardware reference: NVIDIA A100-40 GB, CUDA 11.8, PyTorch 2.1,
  PyTorch Geometric 2.4. Performance and memory will vary on
  other configurations.

### Known sources of run-to-run variability

* `dgl` and `ogb` periodically re-issue dataset versions; using a
  different release than the paper's measurement may shift numbers.
* Soft value iteration is a fixpoint computation; the warmup and
  number of iterations (`model.svi_iterations`, default 5) affect
  the stability of training.
* The contextual-anomaly injection in `utils.anomaly_injection`
  optionally uses `python-louvain`. If unavailable, a random
  partition fallback is used; its seeded but slightly different.

---

## Design notes & assumptions

A small number of implementation choices were not uniquely
determined by the manuscript; we document them here for fairness
to anyone re-implementing.

1. **Augmented Node-MDP state.**
   The paper defines the state at hop $t$ as
   $s_t^v = (h_u^{(t)})_{u \in V_v^{K-t}}$. Materialising this
   tuple is wasteful at scale; we represent the state implicitly
   through the GNN's intermediate buffers and let the per-edge
   reward $R_\theta(s_t, a)$ depend only on the source/destination
   embeddings + cheap structural cues. This is consistent with the
   manuscript's complexity claim of $\mathcal{O}(K|E|d)$.

2. **Soft Bellman backup along edges.**
   Because transitions in the Node-MDP are *one GNN hop forward*,
   the soft Q-function at hop $t$ for edge $(v, u)$ is
   $Q^*(s_t, u) = R_\theta(s_t, u) + \gamma V^*(s_{t+1})$, where
   $V^*(s_{t+1})$ is read off at the destination node's hop-$(t+1)$
   value. We run the backward sweep `svi_iterations` times.
   See `models/soft_value_iteration.py`.

3. **GAT attention as policy.**
   GAT's per-edge attention is normalised over *incoming* edges
   per destination, but the policy semantics require a distribution
   over *outgoing* edges per source. We re-normalise per source
   via `attention_to_log_policy`. Multi-head attention is averaged
   across heads to produce a single distribution per (node, hop).

4. **Reward heads.**
   The structural head sees explicit cheap cues (degrees, log-min,
   degree ratio) on top of source/destination embeddings; the
   semantic head sees the full pair plus difference and product;
   the temporal head sees a small `[time_feat]` placeholder
   (defaults to zeros for static graphs). On static benchmarks
   the temporal head is set to zero by default in the configs.

5. **Reward mixing.**
   $\lambda_1, \lambda_2$ are *learnable* and constrained to be
   non-negative via softplus. Their final values are logged each
   epoch for inspection.

6. **Early stopping signal.**
   Validation AUC-ROC is computed on the entire graph at the end
   of each training epoch. The best-AUC checkpoint is restored at
   the end of training. Patience defaults to 50 epochs.

7. **"Reconstruction score" / "contrastive score" ablations.**
   These do *not* invoke the IRL machinery: the GAT backbone is
   trained as in IRL-GAD, but at scoring time we use either
   `MSE(decoder(h), x)` or `1 - cos(z, mean(z))` respectively. See
   `IRLGAD.score`.

---

## Citation

```bibtex
@article{anonymous2025irlgad,
  title  = {IRL-GAD: Graph Anomaly Detection via Inverse Reinforcement Learning as Normality Modeling},
  author = {Anonymous Authors},
  journal= {IEEE Transactions on Neural Networks and Learning Systems},
  year   = {2025},
  note   = {Under review}
}
```

---

## License

MIT — see `LICENSE`.
