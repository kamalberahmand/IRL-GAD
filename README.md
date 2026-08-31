# IRL-GAD: Graph Anomaly Detection via Inverse Reinforcement Learning as Normality Modeling

> **Anonymous code release accompanying the TMLR submission of the same title.**
> All author identifying information has been removed for double-blind review.

This repository contains a modular PyTorch / PyTorch-Geometric
implementation of IRL-GAD, a graph anomaly detection method that
recasts the problem as inverse reinforcement learning: a latent reward
function is inferred from the aggregation trajectories of known-benign
nodes, and anomalies are scored by the KL divergence between the
observed aggregation policy and the reward-induced reference policy.

---

## ⚠️ Status of this release

This repository is **mid-migration**. A protocol audit
(`AUDIT_BEFORE_CHANGES.md`) identified several places where the code did
not implement the evaluation protocol the manuscript describes. The
protocol core has been rewritten and tested; the experiment runners have
not yet been migrated onto it.

**Concretely, the numbers in the manuscript tables were produced by the
pre-audit code path and should not be treated as reproducible from this
repository in its current state.** The specific reasons are recorded per
finding in `AUDIT_BEFORE_CHANGES.md`; the two that matter most are that
checkpoint selection read test-set anomaly labels (finding F1), and that
two of the six benchmark columns had no valid label source in code
(findings F3, F4).

`CODE_PAPER_ALIGNMENT.md` maps every requirement to its status. Migrated
and tested components are marked `DONE`; the rest are marked `PENDING`
and the corresponding audit finding still applies.

We would rather ship this honestly than present numbers we cannot
currently regenerate.

---

## What has been migrated

| Component | File | Tested |
| --- | --- | --- |
| Normal-only splits (15/5/5/rest), calibration, manifests, hashes | `utils/splits.py` | yes |
| Label-access guard (`FrozenLabels`) | `utils/splits.py` | yes |
| Full-attention policies, exact per-hop KL, contract assertions | `models/policy.py` | yes |
| Four anomaly operators with shared paired targets | `utils/anomaly_injection.py` | yes |
| Contract test suite (23 tests) | `tests/test_contract.py` | yes |

Run the suite with:

```bash
pip install -r requirements.txt
python -m pytest tests/ -q
```

## What has not been migrated

* `train.py` still selects checkpoints on a full-graph objective (F1).
* `models/soft_value_iteration.py` is retained but **deprecated**; the
  active formulation in `models/policy.py` uses neither `T` nor `gamma`.
* No Reddit loader exists (F4). No ogbn-arxiv binary label artifact
  loader exists; the current path injects synthetic anomalies, which the
  protocol forbids (F3).
* `experiments/openset.py` does not implement the paired design (F11);
  `experiments/ablation.py` covers 7 of the 10 required controls (F12).
* No sensitivity runner, no per-node diagnostics export, no result
  manifests or resumability (F13–F16).

---

## Repository layout

```
IRL-GAD/
├── README.md                       (this file)
├── AUDIT_BEFORE_CHANGES.md         (protocol audit, file-and-line evidence)
├── CODE_PAPER_ALIGNMENT.md         (requirement → code status)
├── requirements.txt
├── LICENSE
├── main.py                         (CLI dispatch)
├── train.py                        (training loop — PENDING migration)
├── evaluate.py                     (standalone eval from checkpoint)
├── configs/
│   ├── default.yaml
│   ├── cora.yaml  citeseer.yaml  amazon.yaml  yelpchi.yaml
│   ├── jodie.yaml                  (DEPRECATED — retired from the active suite)
│   └── ogbn_arxiv.yaml
├── models/
│   ├── gat_encoder.py              (M1 — TrajectoryGAT)
│   ├── reward_network.py           (M2 — decomposed reward heads)
│   ├── policy.py                   (M3 — full-attention policies + per-hop KL)
│   ├── soft_value_iteration.py     (DEPRECATED — excluded from reported configs)
│   └── irl_gad.py                  (top-level wiring — PENDING migration)
├── utils/
│   ├── splits.py                   (normal-only protocol, calibration, hashes)
│   ├── data_utils.py               (benchmark loaders)
│   ├── anomaly_injection.py        (structural / attribute / contextual / hybrid)
│   ├── metrics.py                  (AUC-ROC, AUC-PR, TPR@5%FPR)
│   ├── visualization.py            (t-SNE reward landscape)
│   └── seed.py                     (deterministic seeding)
├── tests/
│   └── test_contract.py            (protocol test suite)
├── scripts/
└── experiments/
```

---

## Installation

```bash
python -m venv .venv
source .venv/bin/activate

# Install PyTorch first (consult the official wheel for your CUDA)
# https://pytorch.org/get-started/locally/
pip install "torch>=2.0,<2.5"

pip install -r requirements.txt
```

`python-louvain` is **required**, not optional: contextual injection now
fails loudly if it is unavailable rather than silently substituting a
random partition.

GPU is recommended for `yelpchi`, `amazon`, and `ogbn_arxiv`; Cora and
Citeseer train on CPU in a few minutes.

---

## Datasets

| Dataset | Source | Status |
| --- | --- | --- |
| Cora | `Planetoid` | 5% node-level injection, four matched operators |
| Citeseer | `Planetoid` | 5% node-level injection, four matched operators |
| Amazon | `dgl.data.FraudAmazonDataset` | native fraud labels; unlabelled users excluded from all masks |
| YelpChi | `dgl.data.FraudYelpDataset` | native spam labels; no injection |
| Reddit | BOND (366 native outliers) | **loader not implemented** — see F4 |
| ogbn-arxiv | `ogb.nodeproppred` | **requires a binary anomaly artifact**; the 40 subject classes must never be reinterpreted as anomaly labels, and synthetic injection is not permitted — see F3 |

JODIE / Wikipedia is retired from the active suite. The loader is kept
for reuse but is excluded from all reported configurations.

---

## Evaluation protocol

For each seed in `{0, 1, 2, 3, 4}`, over valid nodes only:

| Split | Fraction | Use |
| --- | --- | --- |
| train | 15% | known-benign training |
| validation | 5% | benign-only model selection |
| calibration | 5% | benign-only threshold, disjoint from validation |
| test | remainder + all anomalies | evaluation only |

The four masks are disjoint by construction. Anomaly labels are read
once, to define the known-benign pool, and are otherwise locked behind
`FrozenLabels` until the checkpoint, hyperparameters, and threshold are
frozen. The deployment threshold is the 95th percentile of
benign-calibration scores; TPR at 5% test FPR is a **separate** statistic
and is not the calibrated operating point.

Split manifests and mask hashes are saved for every dataset and seed.

---

## Anomaly construction (Cora, Citeseer)

All four operators share **one** target set and **one** set of benign
masks per seed; only the corruption differs. This is what makes the
cross-type drop a paired difference.

| Operator | Construction |
| --- | --- |
| `structural` | partition targets into groups of at most 15, densely connect each group |
| `attribute` | sample 50 non-target donors **per target**, replace features with the maximum-cosine-distance candidate |
| `contextual` | seeded Louvain partition of the clean graph, maximum-cosine-distance donor from a different community |
| `hybrid` | structural first, then attribute, on the same targets |

Targets are drawn from a pool that excludes all train, validation, and
calibration nodes. Target IDs, donor IDs, the Louvain seed, and clean and
corrupted graph hashes are saved to an `InjectionManifest`.

---

## Quick start

```bash
# train on Cora
python main.py train --config configs/cora.yaml

# inline overrides: <section>.<field>=<value>, section in {dataset, model, training}
python main.py train --config configs/yelpchi.yaml \
    --set training.lr=5e-4 model.beta=0.05 model.num_layers=3
```

The runners under `experiments/` execute but do not yet implement the
protocol above; see `CODE_PAPER_ALIGNMENT.md` before using their output.

---

## Reproducibility

* RNGs are seeded via `utils.seed.set_seed(seed)`; cuDNN is forced into
  deterministic mode.
* Splits are drawn from a seed-only `numpy` generator so that model-side
  seeding cannot perturb them. A fixed seed reproduces the split hash and
  the corruption hashes exactly (tests 15, 15b).
* Hardware reference: NVIDIA A100-40 GB, CUDA 11.8, PyTorch 2.1,
  PyTorch Geometric 2.4.

### Known sources of run-to-run variability

* `dgl` and `ogb` periodically re-issue dataset versions.
* Multi-head attention aggregation is still under review (see below), and
  the choice affects scores.

---

## Design notes & assumptions

Choices not uniquely determined by the manuscript, documented for anyone
re-implementing.

1. **Augmented Node-MDP state.**
   The paper defines the state at hop $t$ as
   $s_t^v = (h_u^{(t)})_{u \in V_v^{K-t}}$. Materialising this tuple is
   wasteful at scale; the state is represented implicitly through the
   GNN's intermediate buffers, and the per-edge reward
   $R_\theta(s_t, a)$ depends only on source/destination embeddings plus
   cheap structural cues. This is consistent with the $O(K|E|d)$
   complexity claim.

2. **No dynamic-programming loop.**
   The deployed method operates directly on full attention
   distributions. The reference policy is a temperature softmax of the
   combined reward over the neighbour support:
   $q_k^v = \mathrm{softmax}_{u \in N(v)} R_\theta(v,u)/\beta$.
   `models/soft_value_iteration.py` is retained only as deprecated legacy
   and is excluded from every reported configuration. Independently of
   that decision, the legacy loop was not a fixpoint: the terminal value
   was initialised to zero and never updated, so iterations beyond the
   first recomputed identical values.

3. **No sampled actions.**
   No neighbour action is sampled anywhere in the score path. The
   observed policy is the complete attention distribution
   $p_k^v = \pi_v^{(k)}$, and the loss takes the full expectation under
   it. Test 5 asserts that no sampling primitive appears in the score
   path.

4. **Aligned supports.**
   $p_k^v$ and $q_k^v$ are normalised over the same index axis, the same
   edge set, and the same masking. Masked neighbours are excluded from the
   support rather than represented as near-zero-probability actions.
   Attention used as a policy is exported before dropout.

5. **Exact score.**
   $d_k(v) = D_{\mathrm{KL}}(\pi_v^{(k)} \,\|\, \pi_{\theta,v}^{*(k)})$ is
   stored per hop and $S(v) = \sum_k d_k(v)$ is accumulated by summing
   those stored terms, so the identity holds to floating point by
   construction (test 4, residual 0). Reward-head outputs are
   diagnostics and do **not** sum to the score.

6. **Checkpoint selection.**
   Selection uses the benign-validation objective under a fixed epoch
   budget; ties select the earliest checkpoint. Anomaly labels are not
   read during training, validation, model selection, or calibration.
   *(Implemented in `utils/splits.py`; wiring into `train.py` is
   PENDING — see F1.)*

7. **Open question: multi-head aggregation.**
   The observed policy averages 8 attention heads; the reference policy
   has no head dimension. Whether to average $q$ identically, sum
   per-head KL terms, or make the reward head-aware is unresolved, and
   the choice changes reported scores. Documented rather than silently
   fixed.

---

## Citation

```bibtex
@article{anonymous2026irlgad,
  title   = {IRL-GAD: Graph Anomaly Detection via Inverse Reinforcement Learning
             as Normality Modeling},
  author  = {Anonymous Authors},
  journal = {Transactions on Machine Learning Research},
  issn    = {2835-8856},
  year    = {2026},
  note    = {Under review}
}
```

---

## License

MIT — see `LICENSE`.
