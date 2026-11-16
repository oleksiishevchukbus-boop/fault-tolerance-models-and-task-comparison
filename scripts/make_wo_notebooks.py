"""Generate the W_o weight-fault notebooks from a family's original notebooks.

    python scripts/make_wo_notebooks.py                       # base_transformer, both datasets
    python scripts/make_wo_notebooks.py --family tiny_gpt2
    python scripts/make_wo_notebooks.py --family tiny_vit --dataset cifar10

Each notebook loads the trained checkpoint of the original experiment
(results/<family>/<dataset>/models/best.pt) and never trains.  The
import, global-variable, data-split and model-construction cells are copied
VERBATIM from notebooks/<family>/<dataset>.ipynb, so the fixed
evaluation subset X_eval / y_eval is exactly the one the crash study used.
Everything else is the shared W_o experiment below (docs/wo_faults.md).

Outputs go to results/<family>/<dataset>/wo_faults/; the original
results are never written.  Re-running this script overwrites the generated
notebooks and CLEARS THEIR OUTPUTS.  It refuses to write if a copied cell
cannot be found exactly once, so a change to a base notebook cannot silently
produce a notebook with a different data split.
"""
from __future__ import annotations

import argparse
import pathlib

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell

ROOT = pathlib.Path(__file__).resolve().parents[1]
NOTEBOOKS = ROOT / "notebooks"
FAMILIES = ("base_transformer", "tiny_vit", "tiny_gpt2")

# Per dataset: title, the cells to copy (each matched by one marker string),
# exact-substring edits to the copied cells, and the evaluation chunk size the
# original notebook used.
DATASETS = {
    "ag_news": {
        "title": "AG News",
        "copy": ["from faulttol import config as C", "RESULTS_DIR = ROOT",
                 "TOKEN_RE = re.compile", "def build_vocab", "def balanced_subset",
                 "model = TextClassifier("],
        "edits": [(
            'MAX_LEN = C._env_int("FT_MAX_LEN", 64)',
            '# the sequence length must match the trained checkpoint\n'
            'MAX_LEN = json.loads((RESULTS_DIR / "summary.json")'
            '.read_text(encoding="utf-8"))["max_len"]',
        )],
        "eval_chunk": 1000,
    },
    "cifar10": {
        "title": "CIFAR-10",
        "copy": ["from faulttol import config as C", "RESULTS_DIR = ROOT",
                 "def load_cifar10", "def augment_batch", "model = ImageClassifier("],
        "edits": [],
        "eval_chunk": 500,
    },
}


INTRO = """\
# {title}: attention-head faults in the output projection W_o

The crash fault of the original study zeroes a head's output before the heads
are concatenated and projected by `W_o`. Because

`W_o(concat(head_0..head_H)) = sum_h W_o[:, h*d_k:(h+1)*d_k] @ head_h + b_o`,

zeroing head *h*'s `d_k` columns of `W_o` removes exactly the same term. A crash
is therefore one particular **weight fault in W_o**. This notebook corrupts
those columns in other ways and compares the outcome with the crash.

It loads the checkpoint trained by `notebooks/{family}/{dataset}.ipynb`
(no training here) and uses the same fixed evaluation subset. Results go to
`results/{family}/{dataset}/wo_faults/`; the original results are not
touched. Method and outputs: `docs/wo_faults.md`.

| Step | Question |
|---|---|
| E0 | Does zeroing W_o columns reproduce the crash exactly? (sanity) |
| E1 | Single head, every head: how harmful are bit flips and stuck-at values compared with the crash, and do they hit the same heads? |
| E2 | Random multi-head faults on the same head sets as the crash study |
| D1 | Most critical heads: damage vs the **value** a weight is changed to |
| D2 | Most critical heads: damage vs the **original magnitude** of the changed weight |
| D3 | Most critical heads: damage vs the **float32 bit** that is flipped |
| D4 | Pooled: is damage explained by the size of the change alone? |
"""


SETUP = '''\
from faulttol import wo_faults as W
from faulttol.metrics import (classification_clean_metrics, classification_damage,
                              classification_outputs)
from faulttol.training import load_checkpoint
from scipy.stats import kendalltau, pearsonr, spearmanr

EVAL_CHUNK = {eval_chunk}
OUT = RESULTS_DIR / "wo_faults"
TABLES, FIGURES = OUT / "tables", OUT / "figures"
for d in (TABLES, FIGURES):
    d.mkdir(parents=True, exist_ok=True)
CKPT = RESULTS_DIR / "models" / "best.pt"
SHIPPED = json.loads((RESULTS_DIR / "summary.json").read_text(encoding="utf-8"))

CRIT_COL = "D_pos_mean"   # criticality / damage signal, as in the crash study
DEG_COL = "D_mean"        # degradation variable for the severity analysis
SAFE_COL = "D_pos_mean"   # safe-region degradation variable
TAUS = C.CLS_TAUS
TAU_LABELS = ["tau_%.3f" % t for t in TAUS]

def save(df, name):
    path = TABLES / name
    df.to_csv(path, index=False)
    print("saved", path.relative_to(ROOT), df.shape)
    return path

print("checkpoint:", CKPT.relative_to(ROOT), "| exists:", CKPT.is_file())
print("artifacts ->", OUT.relative_to(ROOT))
print("N_FAULT=%d  WO_REPEATS=%d  WO_K_TRIALS=%d  WO_SAMPLES=%d"
      % (C.N_FAULT, C.WO_REPEATS, C.WO_K_TRIALS, C.WO_DEEPDIVE_SAMPLES))
'''


LOAD = '''\
ckpt = load_checkpoint(CKPT, model)
model.to(DEVICE).eval()
CHECKSUM = W.model_checksum(model)
print("loaded epoch %d, best val score %.4f" % (ckpt["epoch"], ckpt["best_score"]))

def evaluate():
    return classification_outputs(model, X_eval, y_eval, DEVICE, chunk=EVAL_CHUNK)

damage_fn = classification_damage
clean_eval = classification_clean_metrics(evaluate())
print("clean (fixed eval subset, n=%d):" % len(X_eval), clean_eval)
if C.N_FAULT == SHIPPED["n_fault_eval_samples"]:
    for k, v in clean_eval.items():
        if abs(v - SHIPPED["clean_metrics"][k]) > 1e-9:
            print("WARNING: clean %s %.6f differs from summary.json %.6f - "
                  "is the checkpoint from the same run?" % (k, v, SHIPPED["clean_metrics"][k]))
    print("clean metrics checked against summary.json")

t0 = time.time()
for _ in range(5):
    evaluate()
per_eval = (time.time() - t0) / 5.0
print("%.3f s per evaluation" % per_eval)
'''


CRASH_MD = """\
## Crash reference

The single-head crash analysis is re-run on the loaded checkpoint (32
evaluations). It gives the reference criticality, the severity weights `W` and
the critical-head set used by the multi-head study, and picks the deep-dive
heads: the top 3 by crash criticality plus the least critical head.
"""

CRASH = '''\
crash_df = single_head.run_single_head_analysis(
    model=model, evaluate=evaluate, damage_fn=damage_fn,
    criticality_col=CRIT_COL, model_name=MODEL_NAME, verbose=False)
save(crash_df, "crash_single_head.csv")

shipped_sh = pd.read_csv(RESULTS_DIR / "tables" / "single_head_damage.csv")
if C.N_FAULT == SHIPPED["n_fault_eval_samples"]:
    diff = np.abs(shipped_sh["criticality"].to_numpy() - crash_df["criticality"].to_numpy()).max()
    print("max |criticality - shipped| = %.2e" % diff,
          "(OK)" if diff < 1e-6 else "<- WARNING: differs from the shipped crash run")

weight_lookup = single_head.build_head_weight_lookup(crash_df)
critical_heads = single_head.critical_head_set(crash_df, C.CRITICAL_TOP_FRACTION)
DEEP_HEADS = W.select_deepdive_heads(crash_df, C.WO_DEEPDIVE_N_CRITICAL)
print("critical heads:", ["L%d_H%d" % h for h in critical_heads])
print("deep-dive heads:", ["L%d_H%d (%s)" % h for h in DEEP_HEADS])
M_LAYER = W.layer_wo_absmax(model)
print("max|W_o| per layer (m):", ["%.4f" % m for m in M_LAYER])
crash_df.nlargest(5, "criticality")[["head_name", "criticality", DEG_COL]]
'''


E0_MD = """\
## E0: sanity check

Zeroing a head's `W_o` columns must give **bit-identical** outputs to the crash
mask, for every single head and for 50 random multi-head sets (the first 50
configurations of the crash study). Two more checks exercise the bit-flip code:
a bit error rate of 0, and the same bit flipped twice, must both equal the clean
model.
"""

E0 = '''\
sanity = W.run_wo_sanity(model, evaluate, n_multi=C.WO_SANITY_MULTI,
                         seed=C.MULTIHEAD_FAULT_SEED, fault_seed=C.WO_FAULT_SEED)
save(sanity, "sanity_checks.csv")
sanity_summary = sanity.groupby("check").agg(n=("exact", "size"), all_exact=("exact", "all"),
                                             max_abs_diff=("max_abs_diff", "max"))
assert sanity["exact"].all(), "W_o fault does not reproduce the expected model state"
assert W.model_checksum(model) == CHECKSUM, "weights were not restored"
print("all sanity checks exact; weights restored")
sanity_summary
'''


E1_MD = """\
## E1: single-head W_o faults, every head

Each head in turn gets one W_o fault while all other heads stay intact:

- **bit flips** at bit error rate (BER) 1e-6 ... 1e-3: every bit of the head's
  `W_o` columns flips independently with probability BER;
- **stuck-at**: a random fraction of the head's `W_o` weights is stuck at 0,
  at +m or at -m, where m = max|W_o| of the head's layer.

Stochastic levels are repeated `WO_REPEATS` times. The W_o criticality of a head
is the mean `D_pos_mean` over repeats, the same formula as the crash
criticality. `catastrophic_rate` is the share of trials whose outputs became
non-finite (counted as failed predictions, see `docs/wo_faults.md`).
"""

E1 = '''\
fault_grid = W.build_fault_grid(C.WO_BER_LEVELS, C.WO_STUCK_VALUES_REL, C.WO_STUCK_FRACS)
wo_sh = W.run_wo_single_head(model, evaluate, damage_fn, fault_grid, C.WO_REPEATS,
                             seed=C.WO_FAULT_SEED, model_name=MODEL_NAME)
save(wo_sh, "wo_single_head.csv")
assert W.model_checksum(model) == CHECKSUM

wo_crit = W.wo_criticality(wo_sh, CRIT_COL)
save(wo_crit, "wo_single_head_criticality.csv")
crash_cmp = W.compare_with_crash(wo_crit, crash_df)
save(crash_cmp, "wo_vs_crash_single_head.csv")

plots.wo_level_curves(wo_crit, crash_df, "%s - single-head W_o fault criticality" % MODEL_NAME,
                      FIGURES / "wo_single_head_levels.png", ylabel=CRIT_COL)
plots.crash_vs_wo_criticality(wo_crit, crash_df, "%s - crash vs W_o criticality" % MODEL_NAME,
                              FIGURES / "crash_vs_wo_criticality.png")
crash_cmp
'''


E2_MD = """\
## E2: random multi-head W_o faults

The head sets are drawn with the crash study's seed, so trial *k* here faults
exactly the heads that crash trial *k* crashed; only the fault differs. Each
failed head gets the W_o fault of the setting. The analysis is the crash
study's: degradation grid, rank correlations of severity vs degradation (plus
the size of the weight change `rel_change` and the number of flipped bits), and
the 95% safe regions.
"""

E2 = '''\
def severity_correlations(df, deg_col, severities):
    rows = []
    for sev in severities:
        x, y = df[sev].to_numpy(float), df[deg_col].to_numpy(float)
        fin = np.isfinite(x) & np.isfinite(y)
        if np.std(x[fin]) == 0:
            continue
        pr = pearsonr(x[fin], y[fin]).statistic
        sr, sp = spearmanr(x, y)        # ranks: inf is just the largest value
        kr = kendalltau(x, y).statistic
        rows.append({"severity": sev, "degradation": deg_col, "pearson_finite": pr,
                     "spearman": sr, "kendall": kr, "spearman_p": sp})
    return pd.DataFrame(rows)

n_heads = len(get_all_heads(model))
crash_mh = pd.read_csv(RESULTS_DIR / "tables" / "random_multihead.csv")
mh_frames, mh_summary, curves = {}, [], {}
if C.N_FAULT == SHIPPED["n_fault_eval_samples"]:
    k = min(C.WO_K_TRIALS, len(crash_mh))
    curves["crash (shipped, same head sets)"] = (crash_mh["N_fail"][:k], crash_mh[DEG_COL][:k])

for setting, spec in C.WO_MULTIHEAD_SETTINGS:
    print("setting", setting)
    df = W.run_wo_multihead(model, evaluate, damage_fn, spec, setting, weight_lookup,
                            critical_heads, MODEL_NAME, C.WO_K_TRIALS,
                            seed=C.MULTIHEAD_FAULT_SEED, fault_seed=C.WO_FAULT_SEED,
                            num_layers=model.num_layers)
    assert W.model_checksum(model) == CHECKSUM
    mh_frames[setting] = df
    save(df, "wo_multihead_%s.csv" % setting)

    grid = multihead.degradation_grid(df, DEG_COL)
    save(grid.reset_index(), "wo_degradation_grid_%s.csv" % setting)
    plots.degradation_grid_plot(grid, "%s - W_o %s: mean %s" % (MODEL_NAME, setting, DEG_COL),
                                FIGURES / ("wo_degradation_grid_%s.png" % setting),
                                value_label=DEG_COL)
    corr = severity_correlations(df, DEG_COL, ("N_fail", "L_fault", "N_crit_fail", "W_fail",
                                               "W_fail_mean", "rel_change", "n_bitflips",
                                               "n_exp_msb_flips"))
    save(corr, "wo_severity_correlations_%s.csv" % setting)
    grids, safe = saferegion.run_safe_region_analysis(
        df, SAFE_COL, TAUS, n_heads, len(critical_heads),
        min_configs=C.MIN_CONFIGS_PER_BOUNDARY, safe_prob=C.SAFE_PROB, tau_labels=TAU_LABELS)
    save(grids, "wo_safe_region_grid_%s.csv" % setting)
    save(safe, "wo_safe_region_summary_%s.csv" % setting)
    curves["W_o " + setting] = (df["N_fail"], df[DEG_COL])

    row = {"setting": setting, "k_trials": len(df),
           "mean_" + DEG_COL: float(df[DEG_COL].mean()),
           "catastrophic_trial_rate": float((df["nonfinite_rate"] > 0).mean()),
           "mean_nonfinite_sample_rate": float(df["nonfinite_rate"].mean()),
           "spearman_N_fail": float(corr.set_index("severity").loc["N_fail", "spearman"]),
           "safe_regions": safe.to_dict(orient="records")}
    if C.N_FAULT == SHIPPED["n_fault_eval_samples"]:
        paired = crash_mh[DEG_COL].to_numpy()[:len(df)]
        row["crash_mean_" + DEG_COL + "_same_sets"] = float(paired.mean())
        row["share_trials_worse_than_crash"] = float(np.mean(df[DEG_COL].to_numpy() > paired))
    mh_summary.append(row)

plots.degradation_curves(curves, "%s - degradation vs failed heads: crash vs W_o faults" % MODEL_NAME,
                         FIGURES / "wo_multihead_curves.png", ylabel="mean " + DEG_COL)
pd.DataFrame(mh_summary).drop(columns="safe_regions")
'''


DEEP_MD = """\
## Deep-dive on the most critical heads

The heads are the top 3 by crash criticality plus the least critical head as a
baseline. Every trial changes the `W_o` columns of **one** head and nothing
else. `m` is max|W_o| of the head's layer.
"""

D1_MD = """\
### D1: damage vs the value a weight is changed to

Each value in the grid is written to:

- **single**: 50 random single weights, one at a time;
- **row**: 8 random whole rows (the `d_k` weights that write one output feature);
- **column**: 8 random whole columns (the `d_model` weights that read one of the
  head's `d_k` output dimensions);
- **block**: all of the head's `W_o` weights.

The same targets are reused for every value, so values are compared on paired
targets. Expect the column and block curves to be **flat**: a constant column
adds the same number to every output feature, and every LayerNorm that later
reads it subtracts it again, so any constant there acts exactly like zero.
"""

D1 = '''\
d1 = W.run_value_sweep(model, evaluate, damage_fn, DEEP_HEADS, C.WO_VALUE_GRID,
                       n_single=C.WO_SWEEP_SINGLE, n_columns=C.WO_SWEEP_COLUMNS,
                       n_rows=C.WO_SWEEP_ROWS, seed=C.WO_FAULT_SEED, model_name=MODEL_NAME)
assert W.model_checksum(model) == CHECKSUM
save(d1, "deep_value_sweep.csv")
d1_table = d1.pivot_table(index=["granularity", "value_label"], columns="head_name",
                          values=CRIT_COL, aggfunc="mean", sort=False)
save(d1_table.reset_index(), "deep_value_sweep_summary.csv")
plots.value_sweep_plot(d1, CRIT_COL, "%s - D1: damage vs the value W_o weights are set to"
                       % MODEL_NAME, FIGURES / "deep_value_sweep.png")
d1_table
'''

D2_MD = """\
### D2: damage vs the original magnitude of the changed weight

The head's weights are ranked by |w| into deciles (`d01` smallest ... `d10`
largest) plus the top 1% and top 0.1%. From each bin, single weights are
**zeroed**, **negated**, **multiplied by 10** or get their **exponent MSB
flipped**, one weight per trial. Finally each whole bin is zeroed at once: is
removing the 10% largest weights worse than removing the 10% smallest?
"""

D2 = '''\
d2 = W.run_magnitude_study(model, evaluate, damage_fn, DEEP_HEADS, C.WO_DEEPDIVE_SAMPLES,
                           seed=C.WO_FAULT_SEED, model_name=MODEL_NAME)
assert W.model_checksum(model) == CHECKSUM
save(d2, "deep_magnitude.csv")
d2_table = d2.pivot_table(index=["granularity", "op", "magnitude_bin"], columns="head_name",
                          values=CRIT_COL, aggfunc="mean", sort=False)
save(d2_table.reset_index(), "deep_magnitude_summary.csv")
plots.magnitude_bins_plot(d2, CRIT_COL, "%s - D2: damage vs |w| of the changed weight"
                          % MODEL_NAME, FIGURES / "deep_magnitude.png")

# rank correlation between |w| and damage, per head and operation (single weights).
# NaN means the damage was constant, e.g. every exponent-MSB flip is catastrophic.
import warnings
d2_single = d2[d2["granularity"] == "single"].copy()
d2_single["abs_old"] = d2_single["old_value"].abs()
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    d2_corr = (d2_single.groupby(["head_name", "role", "op"])
               .apply(lambda g: spearmanr(g["abs_old"], g[CRIT_COL]).statistic,
                      include_groups=False)
               .rename("spearman_abs_w_vs_damage").reset_index())
save(d2_corr, "deep_magnitude_correlation.csv")
d2_corr.pivot(index=["head_name", "role"], columns="op", values="spearman_abs_w_vs_damage")
'''

D3_MD = """\
### D3: damage vs the float32 bit that is flipped

For every bit position 0..31 (0-22 mantissa, 23-30 exponent with 30 the MSB, 31
sign) the same random weights get that one bit flipped, one weight per trial.
"""

D3 = '''\
d3 = W.run_bit_position_study(model, evaluate, damage_fn, DEEP_HEADS, C.WO_DEEPDIVE_SAMPLES,
                              seed=C.WO_FAULT_SEED, model_name=MODEL_NAME)
assert W.model_checksum(model) == CHECKSUM
save(d3, "deep_bit_position.csv")
d3_table = d3.groupby(["head_name", "role", "bit"]).agg(
    damage_mean=(CRIT_COL, "mean"), damage_max=(CRIT_COL, "max"),
    catastrophic_rate=("nonfinite_rate", lambda s: float(np.mean(s > 0)))).reset_index()
save(d3_table, "deep_bit_position_summary.csv")
plots.bit_position_plot(d3, CRIT_COL, "%s - D3: damage per flipped bit of a W_o weight"
                        % MODEL_NAME, FIGURES / "deep_bit_position.png")
d3.groupby(["bit_field", "role"]).agg(
    damage_mean=(CRIT_COL, "mean"),
    catastrophic_rate=("nonfinite_rate", lambda s: float(np.mean(s > 0)))).unstack()
'''

D4_MD = """\
### D4: is damage explained by the size of the change?

All deep-dive trials pooled, as damage against `||dW||_F`, the Frobenius norm
of the change to the head's `W_o` columns. If the curves of the critical heads
lie above the least critical head at the same `||dW||`, *where* the change
lands matters, not only how large it is.
"""

D4 = '''\
pooled = W.pooled_delta_table({"value_sweep": d1, "magnitude": d2, "bit_position": d3}, CRIT_COL)
binned = W.binned_damage_by_delta(pooled, CRIT_COL)
save(binned, "deep_damage_vs_delta.csv")
plots.delta_damage_plot(pooled, binned, CRIT_COL,
                        "%s - D4: damage vs size of the W_o change" % MODEL_NAME,
                        FIGURES / "deep_damage_vs_delta.png")
fin = pooled[np.isfinite(pooled["delta_norm"])]
d4_corr = fin.groupby("role").apply(
    lambda g: spearmanr(g["delta_norm"], g[CRIT_COL]).statistic, include_groups=False)
print("Spearman(||dW||, damage) by head role (finite changes):")
print(d4_corr.round(3).to_string())
'''

SUMMARY_MD = "## Summary"

SUMMARY = '''\
def _top_findings():
    by_field = d3.groupby("bit_field").agg(damage_mean=(CRIT_COL, "mean"),
        catastrophic_rate=("nonfinite_rate", lambda s: float(np.mean(s > 0))))
    return {
        "bit_position_by_field": by_field.to_dict(orient="index"),
        "damage_vs_delta_spearman_by_role": {k: float(v) for k, v in d4_corr.items()},
        "magnitude_spearman": d2_corr.to_dict(orient="records"),
    }

summary = {
    "model_name": MODEL_NAME,
    "checkpoint": str(CKPT.relative_to(ROOT)),
    "fault_location": "W_o.weight[:, h*d_k:(h+1)*d_k] (the head's columns of the output projection)",
    "n_fault_eval_samples": int(C.N_FAULT),
    "clean_metrics": clean_eval,
    "seconds_per_evaluation": per_eval,
    "parameters": {
        "wo_repeats": C.WO_REPEATS, "wo_k_trials": C.WO_K_TRIALS,
        "ber_levels": list(C.WO_BER_LEVELS), "stuck_values_rel": list(C.WO_STUCK_VALUES_REL),
        "stuck_fracs": list(C.WO_STUCK_FRACS),
        "multihead_settings": [{"setting": s, "spec": sp} for s, sp in C.WO_MULTIHEAD_SETTINGS],
        "value_grid": [list(v) for v in C.WO_VALUE_GRID],
        "sweep_single": C.WO_SWEEP_SINGLE, "sweep_rows": C.WO_SWEEP_ROWS,
        "sweep_columns": C.WO_SWEEP_COLUMNS,
        "deepdive_samples": C.WO_DEEPDIVE_SAMPLES, "fault_seed": C.WO_FAULT_SEED,
        "m_layer": M_LAYER,
    },
    "deep_dive_heads": [{"head": "L%d_H%d" % (l, h), "role": r,
                         "crash_criticality": float(crash_df.loc[(crash_df["layer"] == l)
                                                                 & (crash_df["head"] == h),
                                                                 "criticality"].iloc[0])}
                        for l, h, r in DEEP_HEADS],
    "sanity": sanity_summary.reset_index().to_dict(orient="records"),
    "single_head_vs_crash": crash_cmp.to_dict(orient="records"),
    "multihead": mh_summary,
    **_top_findings(),
}
(OUT / "summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
assert W.model_checksum(model) == CHECKSUM
print("wrote", (OUT / "summary.json").relative_to(ROOT))
'''


def _find(cells, marker):
    hits = [c for c in cells if c.cell_type == "code" and marker in c.source]
    if len(hits) != 1:
        raise SystemExit("marker %r matched %d code cells (expected 1)" % (marker, len(hits)))
    return hits[0]


def build(family: str, dataset: str) -> pathlib.Path:
    spec = DATASETS[dataset]
    base_dir = NOTEBOOKS / family
    base = nbformat.read(base_dir / ("%s.ipynb" % dataset), as_version=4)
    copied = []
    for marker in spec["copy"]:
        src = _find(base.cells, marker).source
        for old, new in spec["edits"]:
            if old in src:
                src = src.replace(old, new)
        copied.append(src)
    joined = "\n".join(copied)
    for old, new in spec["edits"]:
        if new not in joined:
            raise SystemExit("edit not applied in %s/%s: %r" % (family, dataset, old))

    md = lambda s: new_markdown_cell(s)  # noqa: E731
    code = lambda s: new_code_cell(s)    # noqa: E731
    cells = [md(INTRO.format(title=spec["title"], dataset=dataset, family=family)),
             md("## Setup (imports, globals, data split and model copied from the original notebook)")]
    cells += [code(s) for s in copied]
    cells += [code(SETUP.format(eval_chunk=spec["eval_chunk"])),
              md("## Load the trained checkpoint"), code(LOAD),
              md(CRASH_MD), code(CRASH),
              md(E0_MD), code(E0),
              md(E1_MD), code(E1),
              md(E2_MD), code(E2),
              md(DEEP_MD),
              md(D1_MD), code(D1),
              md(D2_MD), code(D2),
              md(D3_MD), code(D3),
              md(D4_MD), code(D4),
              md(SUMMARY_MD), code(SUMMARY)]

    nb = nbformat.v4.new_notebook(cells=cells, metadata=base.metadata)
    out = base_dir / ("%s_wo_faults.ipynb" % dataset)
    nbformat.validate(nb)
    nbformat.write(nb, out)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--family", choices=FAMILIES, default="base_transformer")
    ap.add_argument("--dataset", choices=sorted(DATASETS), action="append")
    args = ap.parse_args()
    for ds in args.dataset or sorted(DATASETS):
        print("wrote", build(args.family, ds).relative_to(ROOT))


if __name__ == "__main__":
    main()
