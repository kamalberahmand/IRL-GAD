# CODE_PAPER_ALIGNMENT.md

Status of each contract requirement in this patch. `DONE` = implemented and
tested here. `PENDING` = not yet migrated; the audit finding still stands.

## Section 2 — implementation contract

| Req | Status | Where |
|---|---|---|
| 2.1 no sampled neighbour action | DONE | `models/policy.py` (whole module); test 5 greps the score path for sampling primitives |
| 2.1 `p_k^v` = full GAT attention | DONE | `policy.observed_log_policy` — no redirection; caller supplies the `index` axis |
| 2.1 `q_k^v` on identical support | DONE | `policy.reference_log_policy` — same `index`, same edge set |
| 2.1 masked neighbours excluded, not zero-probability | DONE | `policy.segment_log_softmax`; empty segments return `-inf`, not `-1e30` |
| 2.1 renormalised per source and hop | DONE | `policy.segment_log_softmax` |
| 2.1 stable log-softmax / KL | DONE | `policy._segment_logsumexp` (max-shift, `finfo.tiny` floor) |
| 2.1 assertions: finite, non-negative, aligned, sum to one | DONE | `policy.assert_valid_policies`; tests 2b, 2c prove it fires |
| 2.2 expose `d_k(v)` and `S(v)` | DONE | `policy.score_from_policies` returns `(S, [d_1..d_K])` |
| 2.2 `S` exactly the sum of stored `d_k` | DONE | accumulated by summing the stored tensors; test 4 asserts residual 0 |
| 2.3 one Adam over encoder + reward | PENDING | `train.py:65` already does this; needs the new score path wired in |
| 2.3 no encoder freezing in full model | PENDING | no freezing exists today; `fixed_encoder_reward_reference` control not yet added |
| 2.4 remove `T` / `gamma` from active protocol | PARTIAL | `models/policy.py` has neither. `models/soft_value_iteration.py` still exists and `irl_gad.py` still calls it |

### Formulation decisions taken

The contract states it supersedes inconsistent README text, so where the
README and the contract disagree the contract wins. Three choices were
forced; all change reported numbers.

1. **SVI removed.** Contract 2.4 says the method "operates directly on full
   attention distributions". `q` is now a temperature softmax of the combined
   reward over the neighbour support. The README design note "Soft Bellman
   backup along edges" is superseded. Independently, the legacy loop was not a
   fixpoint: `V[K]` was initialised to zero and never updated, so iterations
   2–5 recomputed identical values (audit F7).
2. **Direction unified.** `p` and `q` are normalised over the same `index`
   axis, passed in by the caller. The legacy split — encoder softmax over
   `dst`, then renormalisation over `src` — is gone. **The encoder must now be
   configured to emit attention over the same axis; this edit to
   `gat_encoder.py` is PENDING and is the one place a wrong choice silently
   changes the model.**
3. **Attention exported pre-dropout.** Legacy applied dropout after softmax
   and before export, making `p` a randomly-zeroed unnormalised vector during
   training (audit F6). PENDING in `gat_encoder.py`.

Still needs your decision: **multi-head aggregation.** `p` averages 8 heads;
`q` has no head dimension. Options are (a) average `q` identically, (b) per-head
KL summed, (c) head-aware reward. Not implemented pending your call.

## Section 3 — datasets

| Req | Status | Where |
|---|---|---|
| structural: groups ≤15, densely connected | DONE | `anomaly_injection._structural` |
| attribute: 50 donors **per target**, max cosine distance | DONE | `anomaly_injection._attribute`; test 10e |
| contextual: seeded Louvain, max-distance donor from another community | DONE | `anomaly_injection._contextual`, `._louvain` |
| Louvain fail-loud instead of random fallback | DONE | `LouvainUnavailable` |
| hybrid: structural then attribute, same targets | DONE | `apply_operator`; test 10d |
| targets exclude train/val/calibration | DONE | `select_targets(excluded=...)`; test 10b |
| same targets + masks across all four types | DONE | `build_matched_graphs`; test 10 |
| save targets, masks, Louvain seed, donors, graph hashes | DONE | `InjectionManifest` |
| Amazon: unlabelled users in graph, out of masks | DONE | `make_splits(valid_mask=...)`; test 9c |
| YelpChi: no synthetic injection | DONE | untouched loader |
| Reddit: BOND graph, 366 outliers, no injection | PENDING | no loader exists (audit F4) |
| ogbn-arxiv: load binary artifact, fail if absent | PENDING | `data_utils.py:158` still injects (audit F3) |
| retire JODIE non-destructively | PENDING | still in `DISPATCH` |

## Section 4 — split and evaluation protocol

| Req | Status | Where |
|---|---|---|
| 15 / 5 / 5 / rest per seed | DONE | `utils/splits.make_splits`; test 9b |
| training on known-benign only | DONE | `Splits.train` |
| selection on benign validation only | DONE | `Splits.val`; enforcement via `FrozenLabels` |
| calibration disjoint benign-only | DONE | `Splits.calib`; tests 8b, 9 |
| test labels inaccessible until frozen | DONE | `FrozenLabels`; test 8 |
| threshold = 95th percentile of benign calibration | DONE | `calibrate_threshold`; test 8c |
| TPR@5%FPR kept separate from threshold | DONE | documented in `splits.py`; `utils/metrics.best_threshold_at_fpr` untouched |
| immutable split manifests and hashes | DONE | `SplitManifest`, `_hash_masks` |
| checkpoint selection on benign-val objective, earliest tie | PENDING | `train.py:98` still selects on full-graph AUC against `y_anom` (audit F1) |

## Sections 5–7 — runners, integrity, tests

| Req | Status |
|---|---|
| Q1 / Q2 / Q3 / Q4 / Q5 runners | PENDING — `experiments/openset.py` still uses `seed+1000` (audit F11); ablation has 7 of 10 variants (F12); no Q4 or sensitivity runner (F14, F15) |
| result manifests, resumability | PENDING (F16) — smoke test emits a sample manifest shape only |
| tests 1,2,3,4,5,8,9,10,15 | DONE — `tests/test_contract.py`, 23 passing |
| tests 6,7 (optimizer/gradients) | PENDING — need the new score path wired into `irl_gad.py` |
| tests 11,12,13,14 | PENDING — depend on the pending Reddit/arxiv loaders and Q2 runner |
