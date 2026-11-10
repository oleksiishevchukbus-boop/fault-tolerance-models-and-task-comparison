# Metrics reference

Every quantity produced by these experiments: what it measures, how it is
computed, and where it is used. Each entry names the function that computes it,
so the code is the authority if this file ever drifts from it.

Nothing here is estimated or approximated unless the entry says so.

## Contents

1. [Notation](#1-notation)
2. [The fault model](#2-the-fault-model)
3. [Clean performance](#3-clean-performance)
4. [Per-sample damage: classification](#4-per-sample-damage-classification)
5. [Per-sample damage: forecasting](#5-per-sample-damage-forecasting)
6. [Per-head metrics](#6-per-head-metrics)
7. [Per-layer metrics: fragility](#7-per-layer-metrics-fragility)
8. [Per-model summary metrics](#8-per-model-summary-metrics)
9. [Per-trial fault-severity descriptors](#9-per-trial-fault-severity-descriptors)
10. [Aggregations](#10-aggregations)
11. [Dependence between severity and degradation](#11-dependence-between-severity-and-degradation)
12. [Fault-tolerance regions](#12-fault-tolerance-regions)
13. [Attention statistics](#13-attention-statistics)
14. [Measures computed only in the reports](#14-measures-computed-only-in-the-reports)
15. [Protocol constants](#15-protocol-constants)
16. [Which column plays which role](#16-which-column-plays-which-role)
17. [Caveats](#17-caveats)

---

## 1. Notation

| Symbol | Meaning |
|---|---|
| `L` | number of encoder layers (4 in `base_transformer`, 6 in `tiny_vit`) |
| `H` | heads per layer (8 / 6) |
| `M = L x H` | total attention heads (32 / 36) |
| `(l, h)` | a head: layer `l`, head index `h`, written `Ll_Hh` |
| `F` | the set of heads crashed in one trial |
| `N` | samples in the fixed fault-evaluation subset (`FT_N_FAULT`, default 1000) |
| `K` | random multi-head trials per model (`FT_K_TRIALS`, default 10000) |
| `y_i` | ground truth for sample `i` |
| `tau` | an acceptable-degradation threshold |
| `eps` | `1e-12`, guards the divisions in `single_head.py` |

Two families of task appear, and they are **not measured in the same unit**:
classification damage is a loss of probability (dimensionless, in `[-1, 1]`),
forecasting damage is an increase in absolute error on a min-max normalised
series, or in MAPE on the original megawatt scale. See [Caveats](#17-caveats).

---

## 2. The fault model

**Definition (attention-head crash).** A head is crashed when its output tensor
is replaced with zeros *after* the attention weights multiply V and *before*
head concatenation and the output projection `W_o`.

Implemented as a per-head binary buffer `head_mask` in
`src/faulttol/model.py::MultiHeadAttention`. In `forward`:

```
head_output = attn @ V                      # (B, H, L, d_k)
head_output = head_output * head_mask.view(1, -1, 1, 1)
```

so `head_mask[h] = 0` removes head `h`'s entire contribution while leaving every
trained parameter untouched.

- Faults are injected **at inference only**; the model is trained clean and
  frozen, so a measurement is the resilience of a fixed model, not a learned
  compensation.
- A configuration is active for exactly one evaluation pass, after which
  `reset_head_masks` restores every mask to 1.
- Injection helpers: `src/faulttol/injection.py::disable_single_head`,
  `disable_multiple_heads`, `reset_head_masks`.

---

## 3. Clean performance

Measured before any fault, on the same fixed subset the fault trials use.

### Classification — `metrics.py::classification_clean_metrics`

| Metric | Computation |
|---|---|
| `accuracy` | `accuracy_score(y, pred)` |
| `precision`, `recall`, `f1` | scikit-learn, `average="macro"` (`"binary"` for 2 classes), `zero_division=0` |
| `roc_auc` | `roc_auc_score`; multiclass uses `multi_class="ovr", average="macro"`. Returns `NaN` if scikit-learn raises `ValueError` (e.g. a class missing from the subset) |

Macro averaging weights every class equally regardless of support, which suits
the balanced subsets used here.

### Forecasting — `metrics.py::regression_clean_metrics`

| Metric | Computation | Scale |
|---|---|---|
| `mse` | `mean((y - yhat)^2)` | normalised |
| `rmse` | `sqrt(mse)` | normalised |
| `mae` | `mean(|y - yhat|)` | normalised |
| `mape` | `100 * mean(|(y_o - yhat_o) / y_o|)`, where `_o` is de-normalised | **megawatts** |

`mape` appears only when a `denorm` callable is supplied; the notebooks always
supply one, so it is the only clean metric on the original physical scale.

---

## 4. Per-sample damage: classification

`metrics.py::classification_damage(clean, fault)`. It first asserts that clean
and faulted outputs cover the same samples in the same order, then works from

```
p_true_i  = probability the model assigns to sample i's TRUE class
D_i       = p_true_i(clean) - p_true_i(fault)
positive  = max(D_i, 0)     negative = min(D_i, 0)     absolute = |D_i|
```

`D_i > 0` means the crash made the model less certain of the right answer;
`D_i < 0` means it accidentally helped, which happens for 7-16% of samples in
practice.

| Column | Formula | What it is for |
|---|---|---|
| `D_mean` | `mean(D_i)` | net damage; help cancels harm. The **degradation variable** for the severity analysis |
| `D_pos_mean` | `mean(max(D_i, 0))` | **harm only**, help counted as 0. The **criticality signal** and the safe-region variable |
| `D_neg_mean` | `mean(min(D_i, 0))` | how much the crash helped. `D_mean = D_pos_mean + D_neg_mean` holds to float precision in the computation (residual ~1e-18); after the CSV round-trip the residual is ~1e-9 |
| `D_abs_mean` | `mean(|D_i|)` | total disturbance regardless of direction |
| `D_q90`, `D_q95`, `D_q99` | quantiles of `D_i` | tail of the signed damage |
| `D_pos_q90/95/99` | quantiles of `max(D_i, 0)` | tail of the harmful damage |
| `D_abs_q95` | 95th percentile of `|D_i|` | tail disturbance |
| `harmful_rate_001/005/010` | `mean(D_i > 0.01 / 0.05 / 0.10)` | fraction of samples harmed beyond a visible amount |
| `helpful_rate_001/005/010` | `mean(D_i < -0.01 / -0.05 / -0.10)` | fraction accidentally improved |
| `f1_clean`, `f1_fault`, `delta_f1` | macro F1 before/after, and the drop | task-level effect |
| `acc_clean`, `acc_fault`, `delta_accuracy` | accuracy before/after, and the drop | task-level effect |

**Why the one-sided measure is the criticality signal.** Because crashes help
some samples, a signed mean lets help cancel harm and makes a genuinely
disruptive head look safe. In the AG News run, head `L1_H4` has
`D_pos_mean = 0.0112` (91% of the worst head) but `D_mean = 0.0039` (under half
of it), because it improves 9.2% of samples. Ranking on `D_mean` would badly
understate it.

---

## 5. Per-sample damage: forecasting

`metrics.py::regression_damage(clean, fault, denorm)`, from

```
abs_clean_i    = |y_i - yhat_i(clean)|
abs_fault_i    = |y_i - yhat_i(fault)|
error_change_i = abs_fault_i - abs_clean_i      # > 0 means degradation
harmful_i      = max(error_change_i, 0)
```

| Column | Formula | What it is for |
|---|---|---|
| `mae_clean`, `mae_fault` | `mean(abs_clean)`, `mean(abs_fault)` | before/after |
| `delta_mae` | `mae_fault - mae_clean` | the **degradation variable** |
| `mse_clean`, `mse_fault`, `delta_mse` | squared-error equivalents | penalises large misses harder |
| `mean_error_change` | `mean(error_change_i)` | net change; improvements cancel |
| `mean_harmful_error_change` | `mean(max(error_change_i, 0))` | **criticality signal**, one-sided like `D_pos_mean` |
| `q95_harmful_error_change`, `q99_...` | quantiles of `harmful_i` | tail of the harm |
| `harmful_rate` | `mean(error_change_i > 0)` | fraction of samples made worse at all (note: no threshold, unlike the classification rates) |
| `mape_clean`, `mape_fault`, `delta_mape` | MAPE before/after and the rise, in **percentage points of megawatts** | the **safe-region variable**; the only damage figure on the physical scale |

---

## 6. Per-head metrics

`single_head.py::run_single_head_analysis` crashes each of the `M` heads on its
own, with every other head alive, so each clean-versus-faulted comparison
differs in exactly one head. That is `M` evaluations. For each head it stores
every damage column above, plus:

| Metric | Formula | Meaning |
|---|---|---|
| `criticality` | `C = max(criticality_col, 0)` | how much damage this head's failure causes on its own. The `max(.,0)` is a safety clip; both criticality columns are already non-negative, so it is a no-op in practice |
| `redundancy` | `R = 1 - C / (max_j C_j + eps)` | 0 for the single worst head, approaching 1 for a head whose loss costs nothing. **Relative to this model's worst head**, so it is not comparable across models |
| `severity_weight` | `W = C / (mean_j C_j + eps)` | criticality normalised so the average head is `1.0`. A head at `W = 2` is twice as damaging as the average |

**Critical head set** — `single_head.py::critical_head_set`: the top
`max(1, round(CRITICAL_TOP_FRACTION * M))` heads by criticality, so 3 heads of
32 and 4 of 36 at the default 10%. This set defines `N_crit_fail` and the
critical-head axis of the fault-tolerance regions.

---

## 7. Per-layer metrics: fragility

`single_head.py::build_layer_summary`, grouping the single-head table by layer.

| Metric | Formula |
|---|---|
| `mean_layer_criticality` | mean `C` over the layer's heads |
| `max_head_criticality` | largest `C` in the layer |
| `min_head_criticality` | smallest `C` in the layer |
| `mean_redundancy` | mean `R` over the layer's heads |
| `n_heads` | heads in the layer |
| `most_critical_head` | name of the layer's worst head |

**"Fragile layer" is reported two ways**, in
`single_head.py::build_model_summary`, because they can disagree:

| Metric | Definition | Reads as |
|---|---|---|
| `most_fragile_layer_by_mean` | `argmax_l mean_layer_criticality` | the layer that matters *as a whole* |
| `most_fragile_layer_by_max` | `argmax_l max_head_criticality` | the layer holding the single worst head |

Each has a `_value` companion holding the quantity that was maximised. When the
two disagree, one layer contains a critical outlier head among otherwise
redundant ones while another is uniformly moderately important. AG News shows
this: `by_mean` is layer 1, `by_max` is layer 0.

---

## 8. Per-model summary metrics

`single_head.py::build_model_summary`, written into each `summary.json`.

| Metric | Formula | Meaning |
|---|---|---|
| `most_critical_head` | head with max `C` | the model's single point of failure |
| `max_criticality` | `max C` | worst single-head cost, in the criticality column's own units |
| `mean_criticality`, `median_criticality`, `min_criticality` | over heads | the distribution |
| `mean_redundancy` | mean `R` | how expendable the typical head is |
| `n_heads_above_half_max` | `count(C > 0.5 * max C)` | how many heads are in the same league as the worst. Low means concentrated |
| `most_fragile_layer_by_mean/_by_max` (+`_value`) | see above | where criticality sits |
| `clean_*` | the clean metrics, prefixed | context for reading damages |

Written alongside in the notebooks' `summary.json`:

| Field | Formula | Meaning |
|---|---|---|
| `median_over_max_criticality` | `median C / (max C + 1e-12)` | **concentration**. 1 = every head matters equally; near 0 = one head dominates |
| `n_heads_zero_effect` | `count(C <= 1e-6)` | heads whose solo failure is unmeasurable |
| `worst_single_head` | head name plus its criticality, degradation and safe-degradation | the worst case in all three columns |
| `criticality_profile` | all `C` values, descending | the raw curve behind the profile figure |

---

## 9. Per-trial fault-severity descriptors

`multihead.py::run_random_multihead` runs `K` trials. Each draws
`N_fail ~ Uniform{1..M}`, samples that many heads without replacement, crashes
them simultaneously, and evaluates once.

| Column | Formula | Meaning |
|---|---|---|
| `N_fail` | `|F|` | how many heads failed |
| `fail_fraction` | `N_fail / M` | the same, as a share of the model |
| `L_fault` | number of layers containing at least one crashed head | how spread out the damage is |
| `N_crit_fail` | `|F ∩ critical set|` | how many *critical* heads failed |
| `W_fail` | `sum_{h in F} W_h` | total severity. **Grows with `N_fail` almost by construction** |
| `W_fail_mean` | `mean_{h in F} W_h` | average severity of the failed heads. **Independent of how many failed** |
| `n_fail_layer_0..L-1` | crashed heads per layer | the layer profile of the fault |
| `failed_heads` | `"L0_H3;L2_H5"` | the exact configuration, for reproducing a trial |
| `run_id` | trial index | joins back to the seed sequence |

plus every damage column from section 4 or 5.

`W_fail` versus `W_fail_mean` is the central distinction in the analysis. The
sum is nearly a restatement of the count (measured Spearman with `N_fail`:
0.999 AG News, 0.996 CIFAR-10, 0.943-0.973 PJM), so only the mean can answer
"did it matter *which* heads failed?".

---

## 10. Aggregations

| Aggregation | Definition | Where |
|---|---|---|
| **Fault-fraction bins** | `fail_fraction` cut at 0-10%, 10-25%, 25-50%, 50-75%, 75-100% | `multihead.py::fault_fraction_bins` |
| **Criticality quartiles** | `qcut(rank(W_fail_mean), 4)` into `Q1..Q4`; Q1 = failed heads were least critical, Q4 = most. Ranking before cutting keeps the four bins exactly equal when values tie | `multihead.py::criticality_quartiles` |
| **Degradation grid** | mean degradation pivoted over fault-fraction bin (rows) x criticality quartile (columns) | `multihead.py::degradation_grid` |
| **Fault-count bands** | mean degradation for `N_fail` in 1, 2-3, 4-8, 9-16, 17-24, 25+ | `scripts/make_experiment_report.py` |

---

## 11. Dependence between severity and degradation

For each descriptor in `N_fail`, `L_fault`, `N_crit_fail`, `W_fail`,
`W_fail_mean`, the notebooks compute **Pearson**, **Spearman** and **Kendall**
correlations against the degradation column. The two classification notebooks
also record `spearman_p`, the p-value of the Spearman figure; the PJM notebook
does not, so that column is absent from its `severity_correlations.csv`.

Also reported explicitly: **`Spearman(W_fail, N_fail)`** — the confound. A high
correlation between `W_fail` and degradation is not evidence that criticality
carries information, because most of `W_fail` *is* the fault count.

The copula-based dependence modelling of the source study is deliberately not
reproduced. `src/faulttol/copula.py` remains in the tree, unused, along with the
`COPULA_FAMILIES` and `CVM_BOOTSTRAP_REPS` settings.

---

## 12. Fault-tolerance regions

`saferegion.py`. A **boundary** `(n, c)` admits every trial with
`N_fail <= n` **and** `N_crit_fail <= c`. For each candidate boundary, over the
degradation variable `Y`:

| Metric | Formula | Meaning |
|---|---|---|
| `n_configs` | trials inside the boundary | sample size |
| `coverage_percent` | `100 * n_configs / K` | share of the sampled fault space admitted |
| `p_safe` | `mean(Y <= tau)` inside the boundary | empirical probability of staying acceptable |
| `is_safe` | `p_safe >= SAFE_PROB` (0.95) | does this boundary certify? |
| `required_tau` | `quantile(Y, 0.95)` inside the boundary | the threshold you would have to accept for it to certify |

Boundaries with fewer than `MIN_CONFIGS_PER_BOUNDARY` (100) trials are skipped
entirely, so a boundary is never certified off a handful of samples.

`safe_region_grid.csv` carries those columns plus `tau` (the threshold in the
degradation variable's own units) and `threshold_name` (its label, e.g.
`tau_0.010` or `25%_of_clean_mape`), which is appended whenever the notebook
supplies threshold labels.

`required_tau.csv` answers the inverse question — "given this boundary, what
would I have to tolerate?" — and adds two distribution columns:

| Column | Formula |
|---|---|
| `required_tau` | `quantile(Y, 0.95)` inside the boundary |
| `median_y` | `median(Y)` inside the boundary, the typical case |
| `max_y` | `max(Y)` inside the boundary, the worst observed case |

**`widest_safe_boundary`** picks, among the `is_safe` boundaries, the one with
the largest `coverage_percent`, and reports:

| Field | Meaning |
|---|---|
| `safe_boundary` | e.g. `"N_fail<=10, N_crit_fail<=3"`, or absent if none certifies |
| `n_fail_max`, `n_crit_fail_max` | the two caps numerically |
| `max_p_safe` | the best `p_safe` reached (of *any* boundary when none certifies) |
| `coverage_percent` | share of the fault space it admits |
| `smallest_required_tau` | the most forgiving threshold any boundary needs |
| `n_boundaries_evaluated` | candidates that cleared the 100-trial floor |

**Thresholds differ by task family**, which matters when comparing:

- classification: `CLS_TAUS = (0.010, 0.025, 0.050)`, **absolute** `D_pos_mean`
- forecasting: `REG_ALPHAS = (0.10, 0.25, 0.50)` times *that region's own clean
  MAPE*, so **relative** and considerably stricter at a clean MAPE near 1%

A cap on critical failures only constrains anything when `n_crit_fail_max` is
**strictly below** the number of critical heads. `N_crit_fail<=3` in a model
with exactly 3 critical heads is not a constraint, and the report generators
test for this rather than assuming it.

---

## 13. Attention statistics

`injection.py::collect_attention_stats_per_head`, on the clean model with all
masks restored. For attention map `A` of shape `(B, H, T, T)`:

```
cumulative_{b,h,j} = sum_i A_{b,h,i,j}     # total attention key j receives
```

| Column | Formula |
|---|---|
| `mean_cumulative_attention` | mean of `cumulative` over batch and keys, averaged over batches |
| `max_cumulative_attention` | running max of `cumulative` over batch and keys |

This is the same **cumulative attention contribution** that *Dropout and the
Outliers* uses to define attention outliers. That study additionally counts
values above `tau * mean` and tracks a max-to-median ratio; adding a median and
a threshold here would make the two directly comparable.

Caching of attention maps is off by default (`store_maps`) and switched on only
for this collection, which keeps the 10000-trial loop cheap. Numerical results
are unaffected.

---

## 14. Measures computed only in the reports

These are not in any CSV; `scripts/make_experiment_report.py`
derives them.

| Measure | Formula | Purpose |
|---|---|---|
| **Concentration** | `median C / max C` | one number for "few critical heads or many equal ones". Dimensionless, so comparable across task families |
| **Head-span** | `min C / max C` | how far apart the weakest and worst heads are |
| **Count effect** | band(`25+`) / band(`1 head`) | how much more damage many failures do than one. Dimensionless |
| **Identity effect** | `Q4 / Q1` mean degradation among trials with `N_fail` in 4-16 | does *which* head fails matter, with the count roughly held fixed |
| **Dominance** | count effect / identity effect | which of the two drives the risk |
| **Cap binding** | `n_crit_fail_max < |critical set|` | whether a safe boundary really restricts critical failures |

The verdict text in the reports is derived from these numbers rather than
fixed: concentration is bucketed at `< 0.35` (concentrated), `0.35-0.60`
(mixed), `>= 0.60` (graded), and each hypothesis is reported as supported or
not according to what the run measured.

---

## 15. Protocol constants

`src/faulttol/config.py`, all overridable by environment variable.

| Constant | Default | Env var | Role |
|---|---|---|---|
| `K_TRIALS` | 10000 | `FT_K_TRIALS` | random multi-head trials per model |
| `N_FAULT` | 1000 | `FT_N_FAULT` | samples in the fixed evaluation subset |
| `CRITICAL_TOP_FRACTION` | 0.10 | - | share of heads called critical |
| `SAFE_PROB` | 0.95 | - | safety level for a boundary |
| `MIN_CONFIGS_PER_BOUNDARY` | 100 | - | minimum trials before a boundary is judged |
| `CLS_TAUS` | 0.010, 0.025, 0.050 | - | absolute classification thresholds |
| `REG_ALPHAS` | 0.10, 0.25, 0.50 | - | forecasting thresholds, as fractions of clean MAPE |
| `DATA_SEED`, `SPLIT_SEED`, `TRAIN_SEED` | 42 | - | data, split and training seeds |
| `FAULT_SPLIT_SEED` | 2026 | - | choice of the fixed evaluation subset |
| `MULTIHEAD_FAULT_SEED` | 1024 | - | the multi-head trial sequence |

Reducing `N_FAULT` rather than `K_TRIALS` is the better economy: `K` drives the
coverage counts of the regions and the sample size of the severity analysis,
while `N_FAULT` only adds estimation noise to each individual trial.

---

## 16. Which column plays which role

Each notebook picks three columns. They are not the same, and the choice is
what the whole analysis hangs on.

| Notebook | Criticality (`CRIT_COL`) | Degradation (`DEG_COL`) | Safe region (`SAFE_COL`) | Thresholds |
|---|---|---|---|---|
| `ag_news` | `D_pos_mean` | `D_mean` | `D_pos_mean` | `CLS_TAUS`, absolute |
| `cifar10` | `D_pos_mean` | `D_mean` | `D_pos_mean` | `CLS_TAUS`, absolute |
| `pjm_regions` | `mean_harmful_error_change` | `delta_mae` | `delta_mape` | `REG_ALPHAS` x clean MAPE |

The `tiny_vit` family uses the same three per dataset, since its notebooks are
derived from these.

---

## 17. Caveats

**Units do not cross task families.** A criticality of 0.058 for CIFAR-10 and
0.041 for a PJM region are different measurements and must not be ranked
against each other. Only the dimensionless measures of
[section 14](#14-measures-computed-only-in-the-reports) compare across
modalities. AG News and CIFAR-10 do share a unit, so their absolute levels are
mutually comparable.

**Redundancy is model-relative.** `R = 1 - C/max(C)` is normalised by the
model's own worst head, so a high mean redundancy means "most heads matter much
less than our worst one", not "this model is robust in absolute terms".

**The identity effect is only approximately count-controlled.** Restricting to
`N_fail` in 4-16 does not fix the count exactly, and `mean(N_fail)` per
quartile comes out U-shaped, because an extreme `W_fail_mean` is likelier when
few heads fail. Computing the ratio stratified by exact fault count instead
gives 1.43 (AG News, versus 1.41 pooled), 1.45 (CIFAR-10, versus 1.61) and
2.15-2.32 (PJM, versus 2.45-2.76). The conclusions hold, but the pooled figure
overstates the effect by 11-19% outside AG News, and on the stratified numbers
text and image are indistinguishable rather than ordered. The reports currently
carry the pooled figure.

**An absent safe boundary arrives as `NaN`, not `None`.** The summary passes
through a pandas DataFrame, and pandas 3.0 stores a mixed `None`/`str` column
in its string dtype, mapping `None` to `NaN`. `NaN` is truthy, so a plain
`if boundary:` test reports a safe region that does not exist. Both report
generators use an explicit presence check; anything new reading
`safe_regions` must do the same.

**`harmful_rate` is not the same shape in both families.** Classification has
three thresholded rates (`> 0.01`, `0.05`, `0.10`); forecasting has a single
unthresholded `harmful_rate` (`error_change > 0`), which is therefore much
larger and not comparable to them.

**Four constants in `metrics.py` are dead.** `CLS_CRITICALITY_COL`,
`CLS_DEGRADATION_COL`, `REG_CRITICALITY_COL` and `REG_DEGRADATION_COL` are
defined but referenced nowhere; the notebooks set their own `CRIT_COL`,
`DEG_COL` and `SAFE_COL`. Note that `CLS_DEGRADATION_COL = "D_pos_mean"`
disagrees with the `DEG_COL = "D_mean"` the classification notebooks actually
use, so the constants should not be treated as documentation of current
behaviour.

**Criticality measured on an under-trained model is not clean.** A head that
has not finished specialising is neither critical nor redundant in a stable
way, so criticality figures should be read next to the clean performance in the
same table.
