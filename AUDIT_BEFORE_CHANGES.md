# AUDIT_BEFORE_CHANGES.md

## 0. Provenance and access blocker

**The specified repository could not be accessed.**

| Endpoint | Result |
|---|---|
| `https://anonymous.4open.science/r/IRL-GAD-4245/` (fetch tool) | `ROBOTS_DISALLOWED` — site disallows automated access |
| `https://anonymous.4open.science/r/IRL-GAD-4245/` (container curl) | `HTTP/2 403`, header `x-deny-reason: host_not_allowed` |
| `https://anonymous.4open.science/api/repo/IRL-GAD-4245/files/` | `403` |
| `https://anonymous.4open.science/r/IRL-GAD-4245/README.md` | `403` |

`anonymous.4open.science` is not on the sandbox egress allowlist, and it additionally
disallows automated fetching. No amount of retrying will reach it from here.

**Source actually audited:** `github.com/kamalberahmand/IRL-GAD`, commit `92ad74a`
("Add full repository"), cloned over an allowlisted domain. It is a plausible mirror —
it contains `models/soft_value_iteration.py`, `configs/jodie.yaml`, and
`utils/anomaly_injection.py`, i.e. exactly the artifacts §1.3 of the contract asks
to search for. A second repo, `kamalberahmand/IRL-GAD2`, is empty (0 commits).

**This audit is provisional until the mirror is confirmed to match IRL-GAD-4245.**
Line numbers below are from commit `92ad74a` and will not be reliable if the anonymous
copy has diverged.

Repository scale: 31 files, 2036 Python LOC, 424 KB. No `tests/` directory exists.

---

## 1. Repository map

```
main.py                          CLI: train | evaluate | ablate | visualize
train.py                         training loop, checkpoint selection
evaluate.py                      checkpoint -> metrics
models/gat_encoder.py            _PolicyGATLayer, TrajectoryGAT (attention extraction)
models/reward_network.py         RewardNetwork: head_str / head_sem / head_tmp, lambda1/lambda2
models/soft_value_iteration.py   soft_value_iteration, kl_observed_vs_optimal, attention_to_log_policy
models/irl_gad.py                IRLGAD: compute_loss, score, compute_per_hop_rewards
utils/data_utils.py              loaders + split_normal_indices
utils/anomaly_injection.py       inject_anomalies (4 types)
utils/metrics.py                 compute_metrics
utils/seed.py, utils/visualization.py
experiments/openset.py           Q2 analogue
experiments/ablation.py          Q3 analogue
experiments/scalability.py       Q5 analogue
configs/{default,cora,citeseer,amazon,yelpchi,jodie,ogbn_arxiv}.yaml
scripts/{run_all_benchmarks,ablation,openset}.sh
```

No `reddit.yaml`. No sensitivity runner. No Q4 (reward-space diagnostics) runner.
No result-manifest writer. No resumability. No tests.

---

## 2. Findings against the contract

Severity: **S1** blocks any published number; **S2** violates the contract but is
locally fixable; **S3** cleanup.

### F1 (S1) — Anomaly labels drive checkpoint selection and early stopping

`train.py:91-106`

```python
model.eval()
with torch.no_grad():
    scored = model.score(data.x, data.edge_index)
    scores_all = scored["score"].cpu()
    y_anom = data.y_anom.cpu()                    # <-- test-set anomaly labels
    metrics = compute_metrics(scores_all, y_anom)

if metrics.auc_roc > best["val_auc_roc"]:         # <-- selection objective
```

The selection objective is AUC-ROC computed on **every node in the graph**, including
all anomalies. Early stopping (`train.py:117`) keys on the same quantity. This violates
contract §4 ("validation/model selection uses benign nodes only"; "test anomaly labels
are inaccessible until the model, checkpoint, hyperparameters, and threshold are
frozen") and defeats the `OC` access claim in the manuscript. Every current number is
selected on the test labels.

Also note `best["val_auc_roc"]` is a misnomer: `val_idx` is computed at `train.py:54`
and then **never used**. There is no validation-restricted evaluation anywhere.

### F2 (S1) — Splits are constructed from anomaly labels, and the required 15/5/5 split does not exist

`utils/data_utils.py:200-214`

```python
normal_idx = (data.y_anom == 0).nonzero(as_tuple=False).flatten()
...
n_val = int(round(val_frac * perm.numel()))
val_idx  = perm[:n_val]
train_idx = perm[n_val:]
```

Three violations of §4:
- "known-benign" is defined by reading `y_anom`, i.e. by the ground truth the protocol
  says must be inaccessible. For Cora/Citeseer this is circular (labels are injected);
  for Amazon/YelpChi/Reddit it is leakage.
- There is **no calibration split**. `val_frac=0.10` yields 90% train / 10% val, not
  the required 15% train / 5% val / 5% calibration / remainder test.
- Consequently there is no 95th-percentile benign-calibration threshold, no deployment
  threshold, and no TPR@5%FPR statistic anywhere in the repository.

### F3 (S1) — ogbn-arxiv receives synthetic injection

`utils/data_utils.py:146-163`

```python
ds = PygNodePropPredDataset(name="ogbn-arxiv", ...)
data = ds[0]
cfg = AnomalyConfig(anomaly_ratio=..., type=spec.anomaly_type, seed=spec.seed)
return inject_anomalies(data, cfg)
```

Contract §3 forbids this outright and requires loading a specific processed binary
label artifact (~5.0% prevalence), failing loudly if absent. The current code silently
manufactures labels instead.

### F4 (S1) — Reddit is not implemented

`DISPATCH` (`utils/data_utils.py:169-177`) has keys
`cora, citeseer, amazon, yelpchi, jodie, ogbn_arxiv, ogbn-arxiv`. There is **no
`reddit` key and no BOND loader**. The only Reddit-adjacent code is `_load_jodie`
(`utils/data_utils.py:95-143`), which is hardcoded to `name="wikipedia"` and derives
node labels by marking every endpoint of an anomalous edge.

The manuscript's Reddit column (71.1 / 40.9, described as "newly completed executions
under the locked Reddit protocol") has **no corresponding code path in this repository**.

### F5 (S1) — The observed policy is not the GAT attention distribution

`models/gat_encoder.py:91` normalises attention over **incoming** edges per destination:

```python
alpha = edge_softmax(alpha, dst, num_nodes=n)
```

`models/soft_value_iteration.py:153-179` then **re-normalises the same values per
source**, and the docstring is explicit that it is changing the semantics:

```python
"""The GAT layer normalises attention with softmax over *incoming*
edges (per destination). For policy semantics we need the
distribution to be over *outgoing* edges (per source) ... We therefore
re-normalise per-source."""
```

So `p_k^v` is a per-source renormalisation of a per-destination softmax — it is not
`π_v^(k)`, the quantity the paper defines and the quantity the reference policy `q_k^v`
is built on (`soft_value_iteration.py:101` normalises per `src`). Contract §2.1 requires
edge direction to be identical for `p` and `q` and requires `p` to be the actual
attention distribution. Either the encoder must softmax over the source axis, or the
paper's definition must change. **This is a formulation-level conflict and must be
resolved by you, not silently by me.**

Secondary: `gat_encoder.py:107` collapses eight heads by `alpha.mean(dim=-1)` before
the renormalisation, so multi-head aggregation for `p` has no counterpart in `q`.

### F6 (S1) — Dropout is applied to attention, so the "observed policy" is stochastic during training

`models/gat_encoder.py:92`

```python
alpha = F.dropout(alpha, p=self.dropout, training=self.training)
```

Attention is dropped *after* softmax and *before* being exported as the policy. During
training, `p_k^v` is therefore a randomly-zeroed, unnormalised vector. Zeroed entries
survive into `attention_to_log_policy` via `alpha.clamp(min=1e-12)`
(`soft_value_iteration.py:170`), which converts a masked-out neighbour into an active
action with probability ~`1e-12/Z` instead of removing it from the support. Contract
§2.1: "Masked neighbours must be excluded from the support, not represented as active
zero-probability actions."

### F7 (S1) — Soft value iteration, `T`, and `gamma` are live in the default protocol

`models/soft_value_iteration.py:46-104` implements the Bellman loop.
`configs/default.yaml:20-21` sets `gamma: 1.0`, `svi_iterations: 5`.
`models/irl_gad.py:96-100` wires both into every forward pass, and
`irl_gad.py:193, 244` call it in both `compute_loss` and `score`.

Contract §2.4 requires `T` and `gamma` to be removed from or disabled in the active
default protocol unless repository evidence proves they are genuinely executed by the
verified formulation, and forbids reporting sensitivity to them.

Note the loop as written is also **not a fixpoint iteration**: `V_per_hop[K]` is
initialised to zeros and never updated (only indices `0..K-1` are written at line 94),
so with `K=2` the backward sweep converges after one pass and iterations 2–5 recompute
identical values. `svi_iterations=5` is dead compute.

### F8 (S2) — Per-hop KL terms `d_k` are never exposed; `score == Σ d_k` is untestable

`models/soft_value_iteration.py:128-150` accumulates into a single tensor via
`score.index_add_(0, src, contrib)` inside the hop loop and returns only the total.
Contract §2.2 requires `d_k(v)` to be stored and exposed, and §7.4 requires a test that
the total equals their sum. Neither is currently possible. `score()`
(`irl_gad.py:253`) returns `{"score", "h_final", "reward_feat"}` — no `d_k`.

### F9 (S2) — No normalisation, finiteness, or alignment assertions

No assertions exist anywhere that `p` and `q` are finite, non-negative, share support,
or sum to one. Contract §2.1 requires them. `soft_value_iteration.py:115` seeds the
per-source max with `-1e30`; for a node with no outgoing edges this propagates a
`-1e30` shift into `log_sum_exp` at line 93 unchecked.

### F10 (S2) — Anomaly injection does not match the four specified operators

`utils/anomaly_injection.py:73-127`.

| Contract §3 requirement | Current code | Line |
|---|---|---|
| `attribute`: sample **50 donors per target**, take max cosine distance | one shared pool of 50 sampled **once** for all targets | 100 |
| `contextual`: max-cosine-distance donor from a different community | **uniform random** pick from another community | 117 |
| `contextual`: seeded Louvain on the clean graph | `random_state=0` hardcoded, ignores `cfg.seed`; silent `except Exception` fallback to a random partition | 50-55 |
| targets excluded from train/val/calibration pools | targets drawn from **all** `n` nodes with no mask awareness | 87 |
| save target IDs, masks, Louvain seed, donor IDs, graph hashes | nothing is persisted | — |
| `hybrid`: structural first, then attribute | correct (both `if` blocks fire in order) | 90, 97 |
| `structural`: groups of ≤15, densely connected | correct | 92-95 |

One thing does work: because `chosen` is drawn from `rng` seeded only by `cfg.seed`
(line 79, 87) before any type-specific branch, **target IDs are identical across the
four types for a fixed seed**. That property is accidental but load-bearing for Q2, and
it is worth preserving explicitly.

### F11 (S1) — Q2 runner breaks the paired design it is supposed to implement

`experiments/openset.py:70-78`

```python
spec = DatasetSpec(
    ...
    anomaly_type=atype,
    # use a different seed so the injected anomalies don't overlap
    # with the training mask
    seed=cfg.dataset.seed + 1000,
)
```

The `+1000` offset gives each held-out type **different anomaly targets on a different
graph**, destroying the exact property F10 accidentally provides. Contract §5-Q2
requires identical target identities and benign masks across all four constructions,
varying only the corruption operator.

Further Q2 violations:
- `in_dist_auc` (line 55) comes from `train_out["metrics"]`, which is the
  label-selected quantity from F1.
- The structural condition is evaluated on a *different* graph object than the three
  held-out conditions, so `D_{m,s}^M` is not a paired difference.
- Only AUC-ROC is recorded (line 88); AUC-PR is dropped.
- `mean_drop_pp` (line 91) aggregates across types but there is no per-seed loop at
  all — one seed per invocation, so the required "aggregate paired seed-level values
  only after this calculation" cannot happen.
- No checkpoint-hash verification.

### F12 (S2) — Ablation set does not match Q3

`experiments/ablation.py:32-40` defines 7 variants. Contract §5-Q3 requires 10.

Missing: `empirical benign-policy KL`, `behaviour-cloning likelihood`,
`one-class attention-sequence scorer`, `fixed encoder + reward reference`,
`randomized attention`.

Present but not in the contract: `no_R_tmp`, `single_reward`.

`no_R_tmp` and the `lambda2` machinery must be disabled on static graphs per §5-Q3,
yet `configs/default.yaml:17` ships `use_temporal_reward: true` and `default.yaml:24`
`use_tmp: true` for Cora. `RewardNetwork.lambda2` (`reward_network.py:80`) is a live
learnable parameter on every dataset, and `irl_gad.py:222` reports it.

In practice `r_tmp` is zeroed at `reward_network.py:131` whenever `time_feats is None`
(which is always, since no caller ever passes it), so `lambda2` receives gradient only
through the zero tensor — it is a dangling parameter, not a functioning head.

Also: every variant calls `train()` (line 57), so every ablation number inherits F1.

### F13 (S2) — Scalability instrumentation is materially incomplete

`experiments/scalability.py`. Present: CUDA sync around the timed region (61-70),
`reset_peak_memory_stats` (57), `max_memory_allocated` (78), warmup (72).

Missing against §5-Q5: separate inference latency; repeated measurements with mean
**and standard deviation** (only `avg` at line 76); hardware identification; software
versions; batch size; epoch definition; whether data loading is included (it is
excluded, but this is not recorded); separately-recorded data-loading time.

`reset_peak_memory_stats` is called once before the loop, so `peak_gpu_mb` is a
whole-run maximum rather than a per-epoch statistic.

### F14 (S2) — No sensitivity runner exists

Contract §5-Q5 requires sweeps over `K∈{1..5}`, `β∈{0.01,0.05,0.10,0.50,1.00,5.00}`,
`λ1∈{0.0,0.1,0.5,1.0,2.0,5.0}`, plus the joint `β×λ1` grid at `K=2`, on
Amazon/YelpChi/Reddit across 5 seeds. No such script exists.

Note `λ1` is currently a **learned** parameter (`reward_network.py:72,76`), not a
configurable one, so a `λ1` sweep is not expressible without a config path to fix it.

### F15 (S2) — No Q4 runner, and the diagnostic claim is unsupported

No per-node export exists. `score()` returns `reward_feat` (`irl_gad.py:248-252`) as a
per-hop *summed reward*, which is neither the structural nor semantic diagnostic mean
required by §5-Q4, and `utils/visualization.py` is a t-SNE helper, not the audited
class-conditional export.

### F16 (S2) — No result integrity layer

`train.py:126-134` saves `state_dict`, `model_cfg`, `epoch`, `val_auc_roc`. §6 requires
dataset version/hash, split hash, code revision, checkpoint hash, selection objective,
raw node-level scores and labels, status, and failure reason. None are recorded. No
resumability (§6) — re-running repeats completed work.

### F17 (S2) — No tests

There is no `tests/` directory, no `pytest.ini`/`conftest.py`, and no CI config. All
15 tests required by §7 are absent.

### F18 (S3) — Dead code in the reward path

`models/irl_gad.py:142-155`:

```python
if h.size(-1) != self.cfg.hidden_dim:
    first_layer = self.encoder.layers[0]
    with torch.no_grad():                 # <-- computed
        h_proj = first_layer.lin(h).view(...)
    # detach? we want gradients to flow into the encoder
    h_proj = first_layer.lin(h).view(...)  # <-- recomputed, overwrites
    h = h_proj
```

The `no_grad` projection is computed and immediately discarded. Doubles this projection's
cost on hop 0 for every forward pass. The `# detach?` comment indicates the author was
undecided; the live behaviour (gradients flow) matches the contract's joint-optimisation
requirement, so only the dead block should go.

`from torch_geometric.utils import softmax as edge_softmax` at
`soft_value_iteration.py:36` is imported and never used.

`_scatter_max_per_source` (`soft_value_iteration.py:107-125`) has a Python-loop fallback
over `values.numel()`; on ogbn-arxiv (~1.2M edges) this would be catastrophic if
`index_reduce_` were ever unavailable. It should hard-fail instead.

### F19 (S3) — Wikipedia/JODIE still active

`configs/jodie.yaml` exists and `DISPATCH["jodie"]` is live. §3 says retire from the
active six-dataset configuration without destructively deleting the loader.

---

## 3. Contract §1.3 targeted search results

| Search target | Found? | Evidence |
|---|---|---|
| Sampled neighbour actions | **No** | No sampling in the score path. `irl_gad.py:21` docstring says "sampled proportional to GAT attention" but the implementation takes the full expectation (`irl_gad.py:204-207`). Docstring is stale; **code already satisfies §2.1's no-sampling rule.** |
| `multinomial` / categorical sampling | **No** | `grep` over all `.py`: zero hits. |
| Soft value iteration / Bellman loops | **Yes** | `models/soft_value_iteration.py:46-104`; called `irl_gad.py:193, 244`. |
| Parameter `T` | **Yes**, as `svi_iterations` | `soft_value_iteration.py:43`; `default.yaml:21`. Dead beyond iteration 1 (see F7). |
| Parameter `gamma` | **Yes** | `soft_value_iteration.py:42`; `irl_gad.py:59`; `default.yaml:20`. |
| Parameter `lambda2` | **Yes** | `reward_network.py:73,80`; `irl_gad.py:165,222`. Dangling (see F12). |
| Synthetic injection on Reddit | **N/A** | Reddit is not implemented (F4). |
| Synthetic injection on ogbn-arxiv | **Yes** | `data_utils.py:158-163` (F3). |
| Anomaly labels in training | No | `compute_loss` is label-free. |
| Anomaly labels in validation / early stopping / selection | **Yes** | `train.py:95-106, 117` (F1). |
| Anomaly labels in split construction | **Yes** | `data_utils.py:209` (F2). |
| Anomaly labels in calibration | **N/A** | No calibration stage exists (F2). |

---

## 4. Formulation conflicts requiring your decision

These cannot be resolved by "smallest safe diff" because either choice changes reported
numbers. Per the contract's instruction to document conflicts rather than silently
change the science:

1. **F5, attention direction.** `p` must equal `π_v^(k)`. Does `π_v^(k)` mean the GAT's
   own per-destination softmax (fix: build `q` on `dst`, delete the renormalisation), or
   an aggregation-over-neighbours distribution (fix: change the encoder to softmax on
   `src`)? These give different models.
2. **F5, multi-head.** `p` averages 8 heads; `q` has one head. Options: average `q` the
   same way, compute per-head KL and sum, or make the reward head-aware.
3. **F7, SVI removal.** Removing the Bellman loop makes `q` a plain softmax of the
   combined reward over the neighbour support — closer to §2.1's
   `q = π*_θ,v` and §2.4's "operates directly on full attention distributions". Confirm
   this is the intended target formulation, since it changes every number.
4. **F12/F14, `λ1`.** Learned (current) vs. fixed hyperparameter (required for the §5-Q5
   sweep). I propose making it configurable with `learned` as a legacy option.

---

## 5. Bottom line

The audited code cannot currently produce a defensible number for any of Q1–Q5:

- **Q1**: F1 + F2 (selection on test labels) invalidate all six columns; F3 and F4 mean
  two of the six datasets have no valid label source at all.
- **Q2**: F11 breaks the paired design; the drop statistic is not a paired difference.
- **Q3**: 5 of 10 controls are absent; all inherit F1.
- **Q4**: no runner, no per-hop terms (F8), no export (F15).
- **Q5**: no sensitivity runner (F14); scalability instrumentation incomplete (F13).

Repository-wide, the `OC` access label claimed in Table 3 is not supported by the code
as written.
