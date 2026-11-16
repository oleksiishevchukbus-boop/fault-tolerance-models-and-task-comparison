# W_o weight-fault experiments

The main study simulates a head failure as a **crash**: the head's output is
multiplied by 0 before the heads are concatenated and projected by `W_o`
(`MultiHeadAttention.forward`, `src/faulttol/model.py`). These experiments
simulate head faults by **changing the weights of `W_o`** instead.

| | |
|---|---|
| Notebooks | `notebooks/{base_transformer,tiny_gpt2,tiny_vit}/{ag_news,cifar10}_wo_faults.ipynb` (`python scripts/make_wo_notebooks.py --family <family>`) |
| Library | `src/faulttol/wo_faults.py`, figures in `src/faulttol/plots.py` |
| Settings | `WO_*` in `src/faulttol/config.py` |
| Results | `results/<family>/<dataset>/wo_faults/` (`summary.json`, `tables/`, `figures/`) |
| Models | the trained checkpoints of the crash study, `results/<family>/<dataset>/models/best.pt` |
| Reports | `reports/<family>/<dataset>_wo_faults_report_EN.docx`, built by `python scripts/make_wo_report.py` |

The notebooks only load a checkpoint and never train. They write to
`wo_faults/` and never touch the crash study's results.

## Where a head lives in W_o

`W_o` is a `d_model x d_model` linear layer. Its input is the concatenation of
the heads' outputs, so

```
W_o(concat(head_0, ..., head_H)) = sum_h  W_o[:, h*d_k:(h+1)*d_k] @ head_h  +  b_o
```

Head *h* is only ever multiplied by its own **`d_k` columns** of `W_o.weight`
(256 x 32 = 8,192 weights in `base_transformer`). Those columns are the fault
target. The bias `b_o` is shared by all heads and is never changed.

Zeroing a head's columns removes exactly the term the crash mask removes, so
**the crash is one particular W_o fault**. Experiment E0 checks that the two give
bit-identical outputs.

**A consequence of post-norm.** Every layer of `base_transformer` ends in a
LayerNorm over the `d_model` features, and LayerNorm subtracts the mean over
the features. A change that adds **the same number to every output feature** is
therefore removed again. Setting one whole column of the head's `W_o` block to
a constant `v` does exactly that: it adds `v * head_h[c]` to all features. So:

- a whole column, or the whole block, set to **any** constant gives the same
  result as setting it to 0, i.e. the crash. This holds until the values are so
  large (about 1e6) that float32 rounding breaks the cancellation;
- stuck-at faults with fraction 1.0 equal the crash for every stuck value.

Single weights and rows (the `d_k` weights that write one output feature) are
not cancelled, and their damage does depend on the value.

## Fault models

All faults are applied in place by the `wo_fault` context manager. It copies the
touched columns, applies the faults, and restores the copy on exit. Every
experiment block ends by asserting that a SHA-1 of the whole state dict equals
the checkpoint's.

| Kind (`spec["kind"]`) | What changes |
|---|---|
| `zero` | the head's whole block becomes 0 (= the crash) |
| `bitflip_ber`, `p` | each of the block's float32 bits flips independently with probability `p` |
| `stuck_at`, `value`/`value_rel`, `frac` | a random fraction `frac` of the block's weights is set to the value |
| `set_value`, `idx`, `value`/`value_rel` | the given weights are set to the value |
| `flip_bit`, `idx`, `bit` | bit `bit` of the given weights is flipped (0 = mantissa LSB, 23-30 exponent, 30 = exponent MSB, 31 = sign) |
| `scale`, `idx`, `factor` | the given weights are multiplied by `factor` (`-1` negates) |

`value_rel` is a multiple of **m = max|W_o| of the head's layer**, so "stuck at
+m" means stuck at the largest weight magnitude that layer has learned. In the
two checkpoints m is 0.156-0.177 (CIFAR-10) and 0.130-0.141 (AG News).

Each trial records, alongside the damage metrics of the crash study:

| Column | Meaning |
|---|---|
| `n_changed` | weights whose bit pattern changed |
| `n_bitflips`, `n_exp_msb_flips` | bits flipped, and how many of them were the exponent MSB |
| `delta_norm` | `‖ΔW‖_F` over the faulted heads' blocks (inf if a weight became non-finite) |
| `rel_change` | `‖ΔW‖_F / ‖W‖_F` |
| `nonfinite_weights` | weights that became inf or NaN |
| `nonfinite_rate` | share of evaluation samples whose output was not finite |

### Non-finite outputs

Flipping the exponent MSB of a weight smaller than 2 multiplies it by 2^128. One
such weight overflows the forward pass, and the class probabilities become NaN.
`sanitize_outputs` counts every sample with non-finite probabilities as a
**failed prediction**: uniform probabilities (so `p_true = 1/C`) and a wrong
predicted label. The damage metrics stay finite and comparable, and
`nonfinite_rate` (per trial) and `catastrophic_rate` (share of trials with
`nonfinite_rate > 0`) keep these outcomes visible.

With the clean model's mean `p_true` near 0.6, a fully failed model shows
`D_pos_mean` of about 0.5-0.6. That is the ceiling of every damage curve.

## Experiments

### Crash reference

The single-head crash analysis is re-run on the loaded checkpoint (32
evaluations). If the evaluation subset has the shipped size, the result is
checked against the shipped `tables/single_head_damage.csv`. It provides the
reference criticality, the severity weights `W`, the critical-head set (top
10%), and the deep-dive heads: **the top 3 by crash criticality plus the least
critical head**.

### E0: sanity check

| Check | Must hold |
|---|---|
| `zero_vs_mask_single` | zero W_o columns == crash mask, for each of the 32 heads |
| `zero_vs_mask_multi` | the same, for the first 50 multi-head sets of the crash study |
| `ber0_vs_clean` | a bit flip at BER 0 == clean model |
| `double_flip_vs_clean` | the same bit of 16 weights flipped twice == clean model |

All comparisons are exact (`np.array_equal` on the class probabilities). The
notebook stops if any fails. Table: `sanity_checks.csv`.

### E1: single-head W_o faults, every head

Each head in turn gets one fault while all other heads stay intact:

- **bit flips** at BER 1e-6, 1e-5, 1e-4, 1e-3: about 0.26, 2.6, 26 and 262
  flipped bits per head;
- **stuck-at** 0, +m or -m, for a random 1%, 10%, 50% or 100% of the head's
  weights.

Stochastic levels are repeated `WO_REPEATS` (10) times. The **W_o criticality**
of a head is its mean `D_pos_mean` over repeats, the same formula as the crash
criticality.

| Table / figure | Content |
|---|---|
| `wo_single_head.csv` | every trial |
| `wo_single_head_criticality.csv` | per head x fault level: criticality, q95, max, catastrophic rate |
| `wo_vs_crash_single_head.csv` | per fault level: mean criticality, ratio to the crash, Spearman vs the crash ranking, top-3 overlap |
| `wo_single_head_levels.png` | criticality vs level, one panel per fault model |
| `crash_vs_wo_criticality.png` | crash criticality vs W_o criticality per head |

### E2: random multi-head W_o faults

The head sets are drawn with the crash study's seed (`MULTIHEAD_FAULT_SEED`), so
trial *k* faults exactly the heads that crash trial *k* crashed; only the fault
differs. `WO_K_TRIALS` (2000) trials for each of two settings:

- `bitflip_ber1e-4`: every failed head gets bit flips at BER 1e-4;
- `stuck_+m_frac0.10`: 10% of every failed head's weights are stuck at +m.

The analysis is the crash study's: degradation grid by fault fraction x
criticality quartile, rank correlations of severity vs degradation (plus
`rel_change`, `n_bitflips`, `n_exp_msb_flips`), and the 95% safe regions. When
the evaluation subset has the shipped size, `share_trials_worse_than_crash`
compares each trial with the crash trial on the same heads.

| Table / figure | Content |
|---|---|
| `wo_multihead_<setting>.csv` | every trial (same columns as the crash `random_multihead.csv`, plus the W_o columns) |
| `wo_degradation_grid_<setting>.csv/.png`, `wo_severity_correlations_<setting>.csv`, `wo_safe_region_*_<setting>.csv` | the crash study's analysis on W_o faults |
| `wo_multihead_curves.png` | mean degradation vs `N_fail`: crash and both W_o settings |

### Deep-dive on the most critical heads

Every trial changes the W_o columns of **one** deep-dive head.

**D1: the value a weight is changed to.** Values 0, ±m/2, ±m, ±10m, ±1e2,
±1e4, ±1e6 are written to

- **single**: 50 random single weights, one per trial;
- **row**: 8 random rows, the `d_k` weights that write one output feature;
- **column**: 8 random columns, the `d_model` weights that read one head
  dimension;
- **block**: the whole block.

The same targets are reused for every value, so values are compared on paired
targets. Tables `deep_value_sweep.csv` and `deep_value_sweep_summary.csv`,
figure `deep_value_sweep.png`.

**D2: the original magnitude of the changed weight.** The head's weights are
ranked by |w| into deciles (`d01` smallest ... `d10` largest) plus the top 1%
and top 0.1%. From each bin, 30 random single weights are zeroed, negated,
multiplied by 10, or get their exponent MSB flipped, one weight per trial. Each
whole bin is also zeroed at once. Tables `deep_magnitude.csv`,
`deep_magnitude_summary.csv` and `deep_magnitude_correlation.csv` (Spearman of
|w| vs damage), figure `deep_magnitude.png`.

**D3: the flipped bit.** For each bit 0-31, the same 30 random weights get that
bit flipped, one per trial. Tables `deep_bit_position.csv` and
`deep_bit_position_summary.csv`, figure `deep_bit_position.png`.

**D4: the size of the change.** All deep-dive trials pooled as damage vs
`‖ΔW‖_F`, with binned medians per head role. If the critical heads' curve lies
above the least critical head's at the same `‖ΔW‖_F`, where the change lands
matters, not only how large it is. Table `deep_damage_vs_delta.csv`, figure
`deep_damage_vs_delta.png`.

## Running

```bash
python scripts/make_wo_notebooks.py          # regenerate (clears outputs)
python -m nbconvert --to notebook --execute --inplace \
  --ExecutePreprocessor.timeout=-1 --ExecutePreprocessor.kernel_name=ft-python \
  notebooks/base_transformer/cifar10_wo_faults.ipynb
```

The generator copies the import, global-variable, data-split and model cells
verbatim from the original notebook, so the evaluation subset is the crash
study's. It stops if a copied cell can't be found exactly once.

| Variable | Meaning | Default |
|---|---|---|
| `FT_N_FAULT` | evaluation subset size; comparisons with the shipped crash results run only at the shipped size | 1000 |
| `FT_WO_REPEATS` | repeats per stochastic E1 level | 10 |
| `FT_WO_K_TRIALS` | multi-head trials per E2 setting | 2000 |
| `FT_WO_SWEEP_SINGLE`, `FT_WO_SWEEP_ROWS`, `FT_WO_SWEEP_COLUMNS` | D1 targets per value | 50, 8, 8 |
| `FT_WO_SAMPLES` | weights per D2 bin and per D3 bit | 30 |

A smoke run: `FT_N_FAULT=250 FT_WO_REPEATS=1 FT_WO_K_TRIALS=120 FT_WO_SAMPLES=2
FT_WO_SWEEP_SINGLE=3 FT_WO_SWEEP_ROWS=3 FT_WO_SWEEP_COLUMNS=2`, about 3 minutes
per notebook.

**Cost.** About 22,000 evaluations per model. On an RTX 4070 Laptop at the
defaults one evaluation took 0.27 s, so each notebook ran for about 1 h 50 min.

## Results

Both notebooks ran at the defaults above (`N_FAULT` = 1000, 10 repeats, 2000
multi-head trials, 30 samples). Every model checksum matched the checkpoint
after every block. "Damage" is `D_pos_mean`. Its ceiling (every sample failed)
is 0.605 for CIFAR-10 and 0.653 for AG News.

| | CIFAR-10 | AG News |
|---|---|---|
| Clean accuracy | 0.811 | 0.907 |
| Deep-dive heads: top 3 by crash criticality | L1_H7 (0.067), L2_H7 (0.033), L2_H3 (0.032) | L0_H4 (0.012), L1_H4 (0.011), L1_H2 (0.011) |
| Deep-dive heads: least critical | L0_H5 (0.010) | L0_H2 (0.005) |
| Mean crash criticality over the 32 heads | 0.024 | 0.0083 |

### E0: the crash is reproduced exactly

All four checks passed with a maximum difference of 0.0 on both models: zeroed
W_o columns vs the crash mask for 32 single heads and 50 multi-head sets, BER 0,
and the double flip.

### E1: single-head faults

**Stuck-at.** With fraction 1.0, every stuck value gives the crash criticality
of every head (ratio to the crash 1.000, Spearman 1.0), as the LayerNorm argument
predicts. Partial faults grow with the fraction:

| Fraction stuck | CIFAR-10: stuck 0 | CIFAR-10: stuck ±m | AG News: stuck 0 | AG News: stuck ±m |
|---|---|---|---|---|
| 1% | 0.03 | 0.12 | 0.02 | 0.07 |
| 10% | 0.12 | 0.39 | 0.11 | 0.24 |
| 50% | 0.47 | 0.88 | 0.47 | 0.63 |
| 100% | 1.00 | 1.00 | 1.00 | 1.00 |

*Mean criticality over heads, as a share of the crash's.*

Zeroing half of a head's weights does about half of the crash's damage.
Sticking them at ±m does more, because it moves each weight further than
zeroing does. Spearman against the crash ranking rises with the fraction for
stuck-at-0 (CIFAR-10: 0.35, 0.73, 0.98; AG News: 0.18, 0.57, 0.92), but stays
around 0.2-0.5 for ±m. A partial stuck-at-±m fault does not hit the
crash-critical heads hardest.

**Bit flips are decided by one bit.** In all 1,280 bit-flip trials per model,
a trial produced non-finite outputs **if and only if** at least one flip landed
on the exponent MSB (bit 30). This held in 520 of 520 trials with such a flip,
and in 0 of 760 trials without one. Every weight of `W_o` has |w| < 2, so that
flip multiplies it by 2^128 and the forward pass overflows.

| BER | Flips per head | Trials catastrophic | Mean damage of the other trials (CIFAR-10 / AG News) |
|---|---|---|---|
| 1e-6 | 0.26 | 0.6% | 4e-6 / 1e-6 |
| 1e-5 | 2.6 | 4.7% | 9e-5 / 4e-5 |
| 1e-4 | 26 | 57% | 7e-4 / 2e-4 |
| 1e-3 | 262 | 100% | - |

The catastrophic shares are the same for both models, because the same seed
flips the same bit positions. The mean bit-flip criticality is therefore about
P(bit 30 is hit) x ceiling. At BER 1e-5 it already exceeds the mean crash
criticality: 1.2x on CIFAR-10 and 3.7x on AG News. At 1e-4 it is 14x and 45x.
The bit-flip ranking of heads is unrelated to the crash ranking (Spearman
between -0.14 and 0.19, no overlap in the top 3). Which head suffers depends on
where the bit-30 flips land, not on the head.

### E2: random multi-head faults

On the same head sets as the crash study's trials:

| | CIFAR-10 crash | CIFAR-10 BER 1e-4 | CIFAR-10 stuck +m 10% | AG News crash | AG News BER 1e-4 | AG News stuck +m 10% |
|---|---|---|---|---|---|---|
| Mean `D_mean` | 0.307 | 0.585 | 0.043 | 0.149 | 0.623 | 0.0045 |
| Trials worse than the crash on the same heads | - | 94.5% | 0.4% | - | 97.5% | 0.55% |
| Catastrophic trials | 0 | 97.5% | 0 | 0 | 97.5% | 0 |
| Spearman(`N_fail`, damage) | | 0.26 | 0.98 | | 0.26 | 0.84 |

**Safe regions (τ, widest boundary, coverage):**

| τ | CIFAR-10 crash | CIFAR-10 stuck +m 10% | AG News crash | AG News stuck +m 10% |
|---|---|---|---|---|
| 0.01 | none | none | N_fail ≤ 1, N_crit_fail ≤ 0 (2.6%) | N_fail ≤ 13, N_crit_fail ≤ 2 (37.7%) |
| 0.025 | none | N_fail ≤ 3, N_crit_fail ≤ 2 (9.4%) | N_fail ≤ 4, N_crit_fail ≤ 3 (12.3%) | all (100%) |
| 0.05 | N_fail ≤ 2, N_crit_fail ≤ 0 (5.2%) | N_fail ≤ 12, N_crit_fail ≤ 3 (35.9%) | N_fail ≤ 10, N_crit_fail ≤ 3 (30.4%) | all (100%) |
| Required τ (Q0.95) | 0.030 | 0.017 | 0.0098 | 0.0041 |

BER 1e-4 has no safe region at any τ, and its required τ is the ceiling. With
26 flips per failed head, almost every trial hits bit 30 at least once. Only 51
of the 2000 trials avoided that, and their mean damage was below 1e-4. The
number of failed heads explains little (Spearman 0.26), since even one failed
head is usually enough.

### D1: the value a weight is changed to

Mean damage of a single weight set to the value, top-1 head / least critical
head:

| Value | CIFAR-10 | AG News |
|---|---|---|
| 0 | 1.0e-4 / 3.3e-5 | 1.9e-5 / 1.8e-5 |
| ±m | 5.5e-4 / 2.1e-4 | 8.4e-5 / 7.8e-5 |
| ±10m | 5.8e-3 / 2.2e-3 | 8.8e-4 / 8.0e-4 |
| ±1e2 | 0.40 / 0.27 | 0.30 / 0.29 |
| ±1e4, ±1e6 | ceiling | ceiling |

- **The sign does not matter.** +v and -v agree within noise everywhere.
- **Up to 10m, damage grows linearly with the value.** Even a single weight at
  10m (about 1.5) costs under 1% of probability.
- **Between 10m and 1e2 a single weight becomes dangerous.** At 1e2 one weight
  out of 8,192 costs 0.2-0.4 of the true-class probability, more than the crash
  of the most critical head (0.067 / 0.012) by a factor of 4-25. At 1e4 the
  model is destroyed.
- **Rows behave like single weights, but stronger.** A row of `d_k` weights
  set to 1e2 reaches 0.52-0.62, close to the ceiling. At 10m a row is about 5x
  a single weight.
- **Columns and the whole block are value-invariant up to 1e4.** Every value
  gives exactly the damage of setting them to 0, as the LayerNorm argument
  predicts. At ±1e6 float32 rounding breaks the cancellation and the damage
  rises a little, up to 2.2x the crash for the block of L0_H5 (CIFAR-10).

### D2: the original magnitude of the changed weight

Larger weights matter more. Spearman(|w|, damage) is 0.89-0.97 for zeroing,
negating and x10, on every head of both models. For the top-1 head, the damage
from zeroing one weight rises 60x on CIFAR-10 and 38x on AG News from the
smallest decile to the top 0.1%. Negating doubles the damage of zeroing, since
the change is twice as large. x10 multiplies it by about 9.

Zeroing whole deciles: the 10% largest weights (`d10`) carry **about 40% of the
crash damage** (CIFAR-10 top-1: 0.026 of 0.067; AG News top-1: 0.0050 of
0.012). The 10% smallest carry 0.4-0.7%. Flipping the exponent MSB is
catastrophic in every bin. Its damage does not depend on |w| at all.

### D3: the flipped bit

| Field | Bits | Mean damage (CIFAR-10 / AG News) | Catastrophic |
|---|---|---|---|
| Mantissa | 0-22 | 2e-6 / 5e-7 | never |
| Sign | 31 | 1.2e-4 / 3.4e-5 | never |
| Exponent | 23-29 | up to 1.7e-3 / 3.2e-4 | never |
| Exponent MSB | 30 | ceiling | **always** (120 / 120 per model) |

- **Mantissa damage doubles with every bit**, because each bit up doubles the
  size of the change. Bit 22, the largest mantissa bit, still costs at most
  4e-5.
- **Bits 25 and 26 are the most harmful exponent bits after bit 30.** For these
  small weights the flip moves the exponent by 4 or 8, i.e. it scales the weight
  by 2^±4 or 2^±8.
- **Bits 27-29 give exactly the same damage.** All three are 1 in every sampled weight,
  and clearing any of them divides the weight by 2^16 or more, which amounts to
  zeroing it.
- **The sign bit (31) is a negation.** It costs about twice the damage of
  zeroing the weight.

### D4: does only the size of the change matter?

Mostly. Over all deep-dive trials, Spearman(`‖ΔW‖_F`, damage) is 0.98-0.99 for
every head of both models. The one break is the value-invariant column/block
writes at 1e6. Their `‖ΔW‖_F` reaches 1e7, yet their damage stays near the
crash level, because the LayerNorm removes the change.

**Where the change lands matters on CIFAR-10, not on AG News.** For changes
of `‖ΔW‖_F` between about 1e-4 and 100, the median damage on CIFAR-10's top-1
head is about 2x that of the least critical head (6.8e-5 vs 3.0e-5, and 9.0e-4
vs 4.2e-4). On AG News the four heads lie on one curve: the least critical head
L0_H2 is as sensitive to single-weight changes as the top-1 head. This matches
the crash criticalities, which differ by 6.6x on CIFAR-10 but only 2.4x on
AG News. Large changes reach the ceiling on every head.

### Summary

1. **The crash is one W_o fault.** Zeroing a head's W_o columns reproduces it
   exactly. Setting a whole column or the whole block to any constant gives the
   same result, because post-norm LayerNorm removes the constant.
2. **Under bit flips, one bit decides everything.** A flip of the exponent MSB
   of any `W_o` weight destroys the model (100% of trials). Every other bit,
   alone, costs under 0.2% of probability. The damage of a bit-flip fault is
   therefore the probability that bit 30 is hit, about 1 - (1 - BER)^8192 per
   head. From BER 1e-5 up, bit flips are worse than crashes on average, and at
   1e-4 the crash study's safe regions do not exist.
3. **Stuck-at faults are milder than the crash.** They were worse than the crash on the
   same heads in under 1% of trials. 10% of every failed head's weights stuck at +m
   widens the safe regions about 7x on CIFAR-10 (τ = 0.05: 35.9% vs 5.2%
   coverage).
4. **For a single changed value, the new value matters far more than which
   weight changes.** The damage grows linearly up to about 10m. Somewhere
   between 1.5 and 1e2, one weight out of 8,192 becomes more harmful than
   crashing the whole most critical head. Among weights changed in the same
   way, larger original weights do more damage (Spearman ≈ 0.95), and the 10%
   largest carry about 40% of the crash damage.
5. **Crash criticality transfers only partly.** It predicts the ranking of
   heads under faults that remove a large part of the head, such as stuck-at-0
   at 50% (Spearman 0.92-0.98). It does not predict the ranking under bit
   flips or stuck-at-±m. Under small, equal-sized changes it separates the
   heads on CIFAR-10 but not on AG News.

### tiny_gpt2, tiny_vit and comparison across models

The same notebooks were generated for `tiny_gpt2`
(`make_wo_notebooks.py --family tiny_gpt2`). This model has 6 x 6 = 36 heads,
d_model 384, d_k 64 (24,576 W_o weights per head), LayerNorm before each
sub-layer, and no biases. Both runs passed every E0 check exactly and every
checksum. The same holds for `tiny_vit` (6 x 6 = 36 heads, pre-norm), run
afterwards with `--family tiny_vit`. Per-model reports: `reports/<family>/<dataset>_wo_faults_report_EN.docx`.

| | bt CIFAR-10 | bt AG News | gpt2 CIFAR-10 | gpt2 AG News | vit CIFAR-10 | vit AG News |
|---|---|---|---|---|---|---|
| Clean accuracy | 0.811 | 0.907 | 0.758 | 0.908 | 0.711 | 0.898 |
| Single-head bit flips, x crash at BER 1e-6 / 1e-5 / 1e-4 | 0.16 / 1.2 / 14 | 0.49 / 3.7 / 45 | 0.32 / 3.2 / 13 | 1.8 / 18 / 76 | 0.33 / 3.3 / 14 | 2.0 / 20 / 85 |
| Stuck-at-0 on 50%: x crash (Spearman vs crash) | 0.47 (0.98) | 0.47 (0.92) | 0.39 (1.00) | 0.43 (0.99) | 0.42 (1.00) | 0.43 (1.00) |
| Stuck-at +m on 10%: Spearman vs crash | 0.20 | 0.24 | 0.80 | 0.98 | 0.80 | 0.98 |
| Multi-head BER 1e-4: trials worse than crash | 94.5% | 97.5% | 89.3% | 96.7% | 94.5% | 96.8% |
| Multi-head stuck +m 10%: trials worse than crash | 0.4% | 0.55% | 0% | 0.75% | 0.05% | 2.9% |
| Safe region at τ = 0.05: crash / stuck +m 10% | 5.2% / 35.9% | 30.4% / 100% | 2.7% / 39.2% | 20.9% / 100% | 2.6% / 40.8% | 25.0% / 100% |

*bt = `base_transformer`, gpt2 = `tiny_gpt2`, vit = `tiny_vit`. Safe regions are given as the
coverage of fault configurations.*

What carries over to the second and third architectures:

- **Every finding about the crash and the LayerNorm holds.** Zeroed W_o equals
  the crash exactly. A whole column or block set to any constant up to 1e4
  equals the crash, even though the LayerNorm now sits before each sub-layer:
  every LayerNorm that reads the residual stream removes the constant.
- **Bit 30 still decides everything.** In all 786 single-head trials with a
  bit-30 flip, per model, the model was destroyed; in none of the others.
  Because a `tiny_gpt2` head has 3x more W_o weights, bit 30 is hit 3x more
  often at the same BER, so bit flips overtake the crash at a lower BER on AG
  News.
- **Value, magnitude and bit position behave the same.** Damage is linear up
  to 10m, sign-symmetric and large at 1e2. Spearman(|w|, damage) is 0.89-0.97.
  Bit 30 is always fatal, and bits 27-29 act like zeroing.

What differs:

- **A bit-30 flip does not always give NaN.** In about 22% of destroyed
  `tiny_gpt2` and `tiny_vit` trials the outputs stayed finite, but the predictions collapsed
  (damage at the ceiling). `catastrophic_rate` counts only non-finite outputs,
  so for this model it under-counts. The reports also count a trial as
  "destroyed" when its damage reaches 90% of the ceiling.
- **Crash criticality transfers better in the 36-head models.** Stuck-at ±m ranks the heads like the
  crash (Spearman 0.80 and 0.98, vs about 0.2 in `base_transformer`). At the
  same `‖ΔW‖_F`, the most critical head takes 10-14x the damage of the least
  critical one, against 1-2x in `base_transformer`. In `tiny_gpt2` the crash
  criticalities are also far more uneven (top/least 32x on CIFAR-10, 13x on
  AG News). `tiny_vit` matches `tiny_gpt2` here: the same Spearman values
  (0.80 / 0.98), top/least crash ratios of 50x (CIFAR-10) and 28x (AG News),
  and 12-21x more damage on the top head than on the least critical one for
  the same single-weight change.
- **CIFAR-10's top head is critical enough that one weight cannot outdo it.**
  L0_H5's crash costs 0.38. One of its weights set to 1e2 costs 0.37, close to
  but below that, whereas in every other model one weight at 1e2 exceeds the
  crash of the most critical head by 3-25x. `tiny_vit` CIFAR-10 is close to
  the same limit: L0_H2's crash costs 0.22 and one weight at 1e2 costs 0.26
  (1.2x); on AG News the ratio is 4x.

Not covered: values between 10m and 1e2 (where the single-weight threshold
lies), and PJM (no checkpoints are saved).
