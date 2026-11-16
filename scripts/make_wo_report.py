"""Build a Word report for each W_o weight-fault experiment.

    python scripts/make_wo_report.py                                # every finished run
    python scripts/make_wo_report.py --family tiny_gpt2
    python scripts/make_wo_report.py --family base_transformer cifar10

Writes reports/<family>/<dataset>_wo_faults_report_EN.docx from what the W_o
notebook wrote to results/<family>/<dataset>/wo_faults/ (summary.json,
tables/*.csv, figures/*.png) plus the crash study's results/<family>/<dataset>/
summary.json. Every number in the text is computed from those files, so the
script can be re-run at any time without touching the GPU. Method:
docs/wo_faults.md.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np
import pandas as pd
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Inches, Pt

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from docx_helpers import add_bullets, add_numbered, add_table, fmt, new_document

ROOT = pathlib.Path(__file__).resolve().parents[1]
FAMILIES = ["base_transformer", "tiny_vit", "tiny_gpt2"]
DATASETS = {"ag_news": "AG News", "cifar10": "CIFAR-10"}
FAMILY_LABEL = {
    "base_transformer": "base Transformer encoder (post-norm)",
    "tiny_vit": "Tiny ViT (pre-norm)",
    "tiny_gpt2": "Tiny GPT-2 (pre-norm, causal)",
}
ROLES = ["top1", "top2", "top3", "least"]
DESTROYED_SHARE = 0.9   # a trial is "destroyed" if its damage is >= 90% of the ceiling


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def sci(x, digits=2) -> str:
    """Compact number for the text: 0.0123 -> 0.012, 3.4e-05 -> 3.4e-5."""
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "-"
    x = float(x)
    if x == 0:
        return "0"
    if abs(x) >= 0.01 and abs(x) < 1000:
        return ("%%.%dg" % (digits + 1)) % x
    m, e = ("%.*e" % (digits - 1, x)).split("e")
    return "%se%d" % (m, int(e))


def rng(lo, hi, f="%.2f") -> str:
    a, b = f % lo, f % hi
    return a if a == b else "%s to %s" % (a, b)


def ber(x) -> str:
    t = "%g" % float(x)
    if "e" in t:
        m, e = t.split("e")
        t = "%se%d" % (m, int(e))
    return t


def pct(x, digits=1) -> str:
    if float(x) == 0:
        return "0%"
    return ("%%.%df%%%%" % digits) % (100.0 * float(x))


def caption(doc, text: str) -> None:
    c = doc.add_paragraph(text)
    c.runs[0].italic = True
    c.runs[0].font.size = Pt(9)
    c.alignment = WD_ALIGN_PARAGRAPH.CENTER


def figure(doc, path: pathlib.Path, text: str, width=6.3) -> None:
    if path.exists():
        doc.add_picture(str(path), width=Inches(width))
        caption(doc, text)


def para(doc, text: str, bold_lead: str | None = None):
    p = doc.add_paragraph()
    if bold_lead:
        r = p.add_run(bold_lead + " ")
        r.bold = True
    p.add_run(text)
    return p


# --------------------------------------------------------------------------
# loading
# --------------------------------------------------------------------------
def load(family: str, dataset: str) -> dict | None:
    res = ROOT / "results" / family / dataset
    wo = res / "wo_faults"
    if not (wo / "summary.json").exists():
        return None
    T = wo / "tables"
    r = {
        "family": family, "dataset": dataset, "res": res, "wo": wo,
        "S": json.loads((wo / "summary.json").read_text(encoding="utf-8")),
        "crash": json.loads((res / "summary.json").read_text(encoding="utf-8")),
        "crash_sh": pd.read_csv(T / "crash_single_head.csv"),
        "sanity": pd.read_csv(T / "sanity_checks.csv"),
        "e1": pd.read_csv(T / "wo_single_head.csv"),
        "cmp": pd.read_csv(T / "wo_vs_crash_single_head.csv"),
        "d1": pd.read_csv(T / "deep_value_sweep_summary.csv"),
        "d2": pd.read_csv(T / "deep_magnitude_summary.csv"),
        "d2corr": pd.read_csv(T / "deep_magnitude_correlation.csv"),
        "d3": pd.read_csv(T / "deep_bit_position.csv"),
        "d4": pd.read_csv(T / "deep_damage_vs_delta.csv"),
    }
    r["mh"] = {}
    for setting, _ in [(m["setting"], m) for m in r["S"]["multihead"]]:
        r["mh"][setting] = {
            "trials": pd.read_csv(T / ("wo_multihead_%s.csv" % setting)),
            "safe": pd.read_csv(T / ("wo_safe_region_summary_%s.csv" % setting)),
        }
    S = r["S"]
    r["heads"] = {d["role"]: d for d in S["deep_dive_heads"]}
    r["role_of"] = {d["head"]: d["role"] for d in S["deep_dive_heads"]}
    d3 = r["d3"]
    r["ceiling"] = float(d3.loc[d3["bit"] == 30, "D_pos_mean"].max())
    r["crash_mean"] = float(r["crash_sh"]["criticality"].mean())
    mc = r["crash"].get("model_config", {})
    r["mc"] = mc
    r["d_k"] = mc["d_model"] // mc["num_heads"]
    r["block"] = mc["d_model"] * r["d_k"]
    r["n_heads"] = mc["num_heads"] * mc["num_layers"]
    return r


def destroyed(df: pd.DataFrame, ceiling: float) -> pd.Series:
    return df["D_pos_mean"] >= DESTROYED_SHARE * ceiling


# --------------------------------------------------------------------------
# sections
# --------------------------------------------------------------------------
def section_intro(doc, r):
    S, mc, ds = r["S"], r["mc"], DATASETS[r["dataset"]]
    doc.add_heading("1. Purpose", level=1)
    doc.add_paragraph(
        "The main study of this repository simulates a failed attention head as a crash: the "
        "head's output is multiplied by 0 before the heads are concatenated and projected by "
        "the output projection W_o. This report replaces the crash by faults in the weights "
        "of W_o itself: random bit flips, weights stuck at a value, and targeted changes of "
        "single weights. It asks how such weight faults compare with the crash, and, for the "
        "most critical heads, how the damage depends on the value a weight is changed to, on "
        "the weight's original magnitude and on the float32 bit that flips.")
    doc.add_paragraph(
        "The model is the trained %s %s checkpoint of the crash study. Nothing is "
        "retrained; the evaluation subset is the crash study's fixed subset of %s samples."
        % (ds, FAMILY_LABEL[r["family"]], "{:,}".format(S["n_fault_eval_samples"])))

    doc.add_heading("2. Questions", level=1)
    add_numbered(doc, [
        "E0. Does zeroing a head's W_o columns reproduce the crash exactly?",
        "E1. For every single head: how harmful are bit flips and stuck-at faults compared "
        "with the crash, and do they hit the same heads?",
        "E2. For random sets of failed heads: how do W_o faults change the degradation and "
        "the 95% safe regions of the crash study?",
        "D1. For the most critical heads: how does the damage depend on the value a weight "
        "is changed to?",
        "D2. How does it depend on the original magnitude of the changed weight?",
        "D3. How does it depend on which float32 bit is flipped?",
        "D4. Is the damage explained by the size of the change alone?",
    ])

    doc.add_heading("3. Model and fault location", level=1)
    m = S["parameters"]["m_layer"]
    rows = [
        ["Architecture", FAMILY_LABEL[r["family"]]],
        ["Layers x heads", "%d x %d = %d heads" % (mc["num_layers"], mc["num_heads"], r["n_heads"])],
        ["d_model / d_k", "%d / %d" % (mc["d_model"], r["d_k"])],
        ["LayerNorm placement", "before each sub-layer (pre-norm)" if mc.get("norm_first")
         else "after each sub-layer (post-norm)"],
        ["Faulted weights per head", "%d x %d = %s (the head's columns of W_o.weight)"
         % (mc["d_model"], r["d_k"], "{:,}".format(r["block"]))],
        ["m = max|W_o| per layer", ", ".join("%.3f" % v for v in m)],
        ["Clean accuracy (evaluation subset)", fmt(S["clean_metrics"]["accuracy"])],
        ["Clean macro F1 / ROC AUC", "%s / %s" % (fmt(S["clean_metrics"]["f1"]),
                                                 fmt(S["clean_metrics"]["roc_auc"]))],
        ["Damage ceiling (every sample failed)", fmt(r["ceiling"])],
    ]
    add_table(doc, ["Property", "Value"], rows, "Table 1. Model and fault location.")
    doc.add_paragraph(
        "W_o computes sum_h W_o[:, h*d_k:(h+1)*d_k] @ head_h. Head h is only ever multiplied "
        "by its own d_k columns of W_o, so those columns are the fault target, and zeroing "
        "them removes exactly the term that the crash mask removes. Every LayerNorm subtracts "
        "the mean over the d_model features, so a change that adds the same number to every "
        "feature is removed again. Setting a whole column (or the whole block) to a constant "
        "does exactly that, and therefore acts like setting it to 0. Single weights and rows "
        "are not cancelled.")
    doc.add_paragraph(
        "Damage is D_pos_mean, the mean positive drop of the true-class probability, as in "
        "the crash study. A trial whose outputs became NaN or inf counts every affected "
        "sample as a failed prediction (uniform probabilities, wrong label). The damage of a "
        "completely failed model is the ceiling in Table 1.")


def section_protocol(doc, r):
    S, P = r["S"], r["S"]["parameters"]
    doc.add_heading("4. Fault models and protocol", level=1)
    add_table(doc, ["Fault", "What changes"], [
        ["zero", "the head's whole W_o block becomes 0 (= the crash)"],
        ["bit flips at BER p", "every bit of the block flips independently with probability p"],
        ["stuck-at v, fraction f", "a random fraction f of the block's weights is set to v"],
        ["set value", "chosen weights (single, row, column, block) are set to a value"],
        ["flip bit b", "bit b of chosen weights flips (0-22 mantissa, 23-30 exponent, "
                       "30 = exponent MSB, 31 = sign)"],
        ["zero / negate / x10", "chosen single weights are zeroed, negated or scaled"],
    ], "Table 2. Fault models.")
    add_table(doc, ["Parameter", "Value"], [
        ["E1 bit error rates", ", ".join(sci(v) for v in P["ber_levels"])],
        ["E1 stuck values / fractions", "0, +m, -m / " + ", ".join(pct(f, 0) for f in P["stuck_fracs"])],
        ["E1 repeats per stochastic level", P["wo_repeats"]],
        ["E2 settings", "; ".join(s["setting"] for s in P["multihead_settings"])],
        ["E2 trials per setting", "{:,}".format(P["wo_k_trials"])],
        ["D1 values", ", ".join(v[0] for v in P["value_grid"])],
        ["D1 targets per value", "%d single weights, %d rows, %d columns, the block"
         % (P["sweep_single"], P["sweep_rows"], P["sweep_columns"])],
        ["D2 / D3 weights per bin / per bit", P["deepdive_samples"]],
        ["Evaluation subset", "{:,}".format(S["n_fault_eval_samples"])],
        ["Seconds per evaluation", fmt(S["seconds_per_evaluation"], 3)],
        ["Fault seed / head-set seed", "%d / crash study's MULTIHEAD_FAULT_SEED" % P["fault_seed"]],
    ], "Table 3. Protocol parameters, as recorded in summary.json.")
    doc.add_paragraph(
        "All faults are applied in place and restored after every trial. Every experiment "
        "block ends by asserting that a SHA-1 checksum of the whole model equals the "
        "checkpoint's; the run would have stopped otherwise.")


def section_crash(doc, r):
    doc.add_heading("5. Results", level=1)
    doc.add_heading("5.1 Crash reference and deep-dive heads", level=2)
    sh = r["crash_sh"].sort_values("criticality", ascending=False)
    doc.add_paragraph(
        "The single-head crash analysis was re-run on the loaded checkpoint. The mean crash "
        "criticality over the %d heads is %s; the most critical head is %s (%s). The deep-dive "
        "heads are the three most critical heads and the least critical one."
        % (r["n_heads"], sci(r["crash_mean"]), sh.iloc[0]["head_name"],
           sci(sh.iloc[0]["criticality"])))
    add_table(doc, ["Role", "Head", "Crash criticality", "Rank"], [
        [d["role"], d["head"], sci(d["crash_criticality"], 3),
         int((sh["head_name"].tolist()).index(d["head"]) + 1)]
        for d in r["S"]["deep_dive_heads"]], "Table 4. Deep-dive heads.")


def section_e0(doc, r):
    doc.add_heading("5.2 E0: the crash is one W_o fault", level=2)
    rows = [[x["check"], x["n"], "yes" if x["all_exact"] else "NO", fmt(x["max_abs_diff"])]
            for x in r["S"]["sanity"]]
    add_table(doc, ["Check", "Configurations", "All bit-identical", "Max |difference|"], rows,
              "Table 5. Sanity checks (exact comparison of the class probabilities).")
    ok = all(x["all_exact"] for x in r["S"]["sanity"])
    doc.add_paragraph(
        ("All checks passed exactly: zeroing a head's W_o columns gives the same output as "
         "the crash mask, for every single head and for 50 multi-head sets of the crash "
         "study, and a bit error rate of 0 or the same bit flipped twice give back the clean "
         "model.") if ok else "At least one sanity check FAILED; see Table 5.")


def bitflip_rows(r):
    e1 = r["e1"]
    b = e1[e1["fault_model"] == "bitflip"].copy()
    b["msb"] = b["n_exp_msb_flips"] > 0
    b["nonfinite"] = b["nonfinite_rate"] > 0
    b["destroyed"] = destroyed(b, r["ceiling"])
    out = []
    for lvl, g in b.groupby("level", sort=False):
        ber = float(lvl.split("=")[1])
        rest = g[~g["msb"]]
        out.append({"ber": ber, "flips": g["n_bitflips"].mean(), "msb": g["msb"].mean(),
                    "nonfinite": g["nonfinite"].mean(), "destroyed": g["destroyed"].mean(),
                    "rest": rest["D_pos_mean"].mean() if len(rest) else np.nan,
                    "rest_max": rest["D_pos_mean"].max() if len(rest) else np.nan})
    out.sort(key=lambda x: x["ber"])
    link = pd.crosstab(b["msb"], b["destroyed"])
    return out, link, b


def section_e1(doc, r):
    doc.add_heading("5.3 E1: single-head W_o faults on every head", level=2)
    cmp = r["cmp"]
    rows = []
    for _, x in cmp.iterrows():
        rows.append([x["fault_model"], x["level"], sci(x["mean_criticality"]),
                     sci(x["ratio_to_crash_mean"]),
                     "-" if pd.isna(x["spearman_vs_crash"]) else "%.2f" % x["spearman_vs_crash"],
                     int(x["top3_overlap_with_crash"]), x["most_critical_head"]])
    add_table(doc, ["Fault", "Level", "Mean criticality", "x crash", "Spearman vs crash",
                    "Top-3 overlap", "Worst head"], rows,
              "Table 6. W_o criticality per fault level, compared with the crash. 'x crash' is "
              "the mean criticality over heads divided by the mean crash criticality (%s)."
              % sci(r["crash_mean"]))

    st = cmp[cmp["fault_model"].str.startswith("stuck")]
    full = st[st["level"] == "frac=1"]
    para(doc,
         "With fraction 1.0, every stuck value reproduces the crash criticality of every head "
         "(ratio to the crash %s, Spearman %s), because a whole block set to a constant is "
         "cancelled by LayerNorm. Partial faults grow with the fraction."
         % (rng(full["ratio_to_crash_mean"].min(), full["ratio_to_crash_mean"].max(), "%.4f"),
            "%.2f" % full["spearman_vs_crash"].min()), "Stuck-at.")
    s0 = st[st["fault_model"] == "stuck_at_0"].set_index("level")
    spm = st[st["fault_model"] != "stuck_at_0"].groupby("level")["spearman_vs_crash"].mean()
    rat = st.groupby(["level", "fault_model"])["ratio_to_crash_mean"].first().unstack()
    rows = []
    for lvl in ["frac=0.01", "frac=0.1", "frac=0.5", "frac=1"]:
        if lvl in rat.index:
            rows.append([lvl.replace("frac=", ""), sci(rat.loc[lvl, "stuck_at_0"]),
                         sci((rat.loc[lvl, "stuck_at_+m"] + rat.loc[lvl, "stuck_at_-m"]) / 2),
                         "%.2f" % s0.loc[lvl, "spearman_vs_crash"], "%.2f" % spm.loc[lvl]])
    add_table(doc, ["Fraction stuck", "Stuck 0: x crash", "Stuck ±m: x crash",
                    "Stuck 0: Spearman vs crash", "Stuck ±m: Spearman vs crash"], rows,
              "Table 7. Stuck-at faults as a share of the crash.")

    bf, link, b = bitflip_rows(r)
    n_msb = int(b["msb"].sum())
    n_msb_dest = int((b["msb"] & b["destroyed"]).sum())
    n_rest_dest = int((~b["msb"] & b["destroyed"]).sum())
    para(doc,
         "In %d of %d bit-flip trials with at least one flip of the exponent MSB (bit 30) the "
         "model was destroyed (damage at least %d%% of the ceiling); in %d of %d trials without "
         "such a flip it was. Every W_o weight has |w| < 2, so that flip multiplies it by "
         "2^128. The other flips together cost little."
         % (n_msb_dest, n_msb, int(100 * DESTROYED_SHARE), n_rest_dest, int((~b["msb"]).sum())),
         "Bit flips are decided by one bit.")
    rows = [[ber(x["ber"]), "%.2f" % x["flips"], pct(x["msb"]), pct(x["nonfinite"]),
             pct(x["destroyed"]), sci(x["rest"]), sci(x["rest_max"])] for x in bf]
    add_table(doc, ["BER", "Flips per head", "Trials with a bit-30 flip", "Non-finite outputs",
                    "Destroyed", "Mean damage, no bit-30 flip", "Max damage, no bit-30 flip"],
              rows, "Table 8. Single-head bit flips.")
    if (b["destroyed"] & ~b["nonfinite"]).any():
        doc.add_paragraph(
            "In this model a bit-30 flip does not always produce NaN or inf: in %s of the "
            "destroyed trials the outputs stayed finite but the predictions collapsed. The "
            "'catastrophic rate' column of the result tables counts only non-finite outputs "
            "and therefore under-counts the failures; the damage and the 'Destroyed' column "
            "count them all." % pct((b["destroyed"] & ~b["nonfinite"]).sum()
                                    / max(1, b["destroyed"].sum())))
    bfc = cmp[cmp["fault_model"] == "bitflip"]
    sp = bfc["spearman_vs_crash"].dropna()
    doc.add_paragraph(
        "The mean bit-flip criticality is therefore close to P(bit 30 is hit) x ceiling; it "
        "exceeds the crash from BER %s on (x%s), and reaches x%s at BER 1e-4. The ranking of "
        "heads under bit flips is unrelated to the crash ranking (Spearman %s to %s): which "
        "head suffers depends on where the bit-30 flips land."
        % (ber(bfc[bfc["ratio_to_crash_mean"] > 1]["level"].str.replace("ber=", "")
                   .astype(float).min()) if (bfc["ratio_to_crash_mean"] > 1).any() else "-",
           sci(bfc[bfc["ratio_to_crash_mean"] > 1]["ratio_to_crash_mean"].min()),
           sci(bfc[bfc["level"] == "ber=0.0001"]["ratio_to_crash_mean"].iloc[0]),
           *("%.2f" % sp.min(), "%.2f" % sp.max())))
    figure(doc, r["wo"] / "figures" / "wo_single_head_levels.png",
           "Figure 1. W_o criticality per head vs fault level, one panel per fault model.")
    figure(doc, r["wo"] / "figures" / "crash_vs_wo_criticality.png",
           "Figure 2. Crash criticality vs W_o criticality per head.")


def section_e2(doc, r):
    doc.add_heading("5.4 E2: random multi-head W_o faults", level=2)
    doc.add_paragraph(
        "The failed-head sets are those of the crash study's first %s trials (same seed), so "
        "each W_o trial is paired with the crash trial on the same heads."
        % "{:,}".format(r["S"]["parameters"]["wo_k_trials"]))
    crash_mean = None
    rows = []
    for m in r["S"]["multihead"]:
        t = r["mh"][m["setting"]]["trials"]
        crash_mean = m.get("crash_mean_D_mean_same_sets", crash_mean)
        rows.append([m["setting"], sci(m["mean_D_mean"], 3),
                     pct(m.get("share_trials_worse_than_crash", np.nan)),
                     pct(m["catastrophic_trial_rate"]),
                     pct(destroyed(t, r["ceiling"]).mean()),
                     "%.2f" % m["spearman_N_fail"]])
    rows.insert(0, ["crash (same head sets)", sci(crash_mean, 3), "-", "0%", "-", "-"])
    add_table(doc, ["Setting", "Mean D_mean", "Trials worse than crash", "Non-finite trials",
                    "Destroyed trials", "Spearman(N_fail, damage)"], rows,
              "Table 9. Multi-head faults vs the crash on the same head sets.")

    taus = [x["tau"] for x in r["crash"]["safe_regions"]]
    def cell(x):
        b = x.get("safe_boundary")
        if b is None or (isinstance(b, float) and np.isnan(b)) or b == "":
            return "none"
        return "%s (%s%%)" % (b, fmt(x["coverage_percent"], 1))
    cols = [("crash", {x["tau"]: x for x in r["crash"]["safe_regions"]})]
    for m in r["S"]["multihead"]:
        cols.append((m["setting"], {x["tau"]: x for x in m["safe_regions"]}))
    rows = [[sci(t)] + [cell(c[t]) for _, c in cols] for t in taus]
    rows.append(["required tau (Q0.95)"] + [sci(c[taus[0]]["smallest_required_tau"], 3)
                                            for _, c in cols])
    add_table(doc, ["tau"] + [n for n, _ in cols], rows,
              "Table 10. 95% safe regions: widest boundary (coverage of fault configurations).")

    ms = {m["setting"]: m for m in r["S"]["multihead"]}
    bf = ms.get("bitflip_ber1e-4")
    st = ms.get("stuck_+m_frac0.10")
    if bf:
        t = r["mh"]["bitflip_ber1e-4"]["trials"]
        spared = t[t["n_exp_msb_flips"] == 0]
        doc.add_paragraph(
            "Bit flips at BER 1e-4 destroy almost every trial: %s were destroyed and only %d "
            "of %d trials had no bit-30 flip (their mean damage was %s). The number of failed "
            "heads hardly matters (Spearman %s), since one failed head is usually enough."
            % (pct(destroyed(t, r["ceiling"]).mean()), len(spared), len(t),
               sci(spared["D_mean"].mean()) if len(spared) else "-",
               "%.2f" % bf["spearman_N_fail"]))
    if st:
        doc.add_paragraph(
            "Stuck-at +m on 10%% of every failed head's weights is far milder than the crash: "
            "mean degradation %s against %s, and worse than the crash on the same heads in "
            "%s of the trials."
            % (sci(st["mean_D_mean"]), sci(crash_mean), pct(st["share_trials_worse_than_crash"])))
    figure(doc, r["wo"] / "figures" / "wo_multihead_curves.png",
           "Figure 3. Mean degradation vs number of failed heads: crash and W_o faults.")


def section_d1(doc, r):
    doc.add_heading("5.5 D1: the value a weight is changed to", level=2)
    d1 = r["d1"].copy()
    h = {role: r["heads"][role]["head"] for role in ROLES}
    rows = []
    for lbl in d1["value_label"].drop_duplicates():
        g = d1[d1["value_label"] == lbl].set_index("granularity")
        row = [lbl]
        for gran in ["single", "row"]:
            row += [sci(g.loc[gran, h["top1"]]), sci(g.loc[gran, h["least"]])]
        row += [sci(g.loc["column", h["top1"]]), sci(g.loc["block", h["top1"]])]
        rows.append(row)
    add_table(doc, ["Value", "Single: top1", "Single: least", "Row: top1", "Row: least",
                    "Column: top1", "Block: top1"], rows,
              "Table 11. Mean damage when the targets are set to the value (top1 = %s, least "
              "= %s; m = max|W_o| of the layer)." % (h["top1"], h["least"]))
    g = d1.set_index(["granularity", "value_label"])
    t1, le = h["top1"], h["least"]
    lin = g.loc[("single", "+10m"), t1] / max(g.loc[("single", "+m"), t1], 1e-12)
    sv = [g.loc[("single", v), c] for v in ("+1e2", "-1e2") for c in h.values()]
    rv = [g.loc[("row", v), c] for v in ("+1e2", "-1e2") for c in h.values()]
    blk = [g.loc[("block", v), t1] for v in d1["value_label"].unique()
           if v not in ("+1e6", "-1e6")]
    add_bullets(doc, [
        "The sign does not matter: +v and -v give the same damage within noise.",
        "Up to 10m the damage grows about linearly with the value (x%s from m to 10m for %s); "
        "one weight at 10m costs %s." % (sci(lin), t1, sci(g.loc[("single", "+10m"), t1])),
        "Between 10m and 1e2 a single weight becomes dangerous: at ±1e2 one weight out of %s "
        "costs %s to %s, which is x%s to x%s the crash of the most critical head (%s). At ±1e4 the "
        "model is destroyed."
        % ("{:,}".format(r["block"]), sci(min(sv)), sci(max(sv)),
           sci(min(sv) / r["heads"]["top1"]["crash_criticality"]),
           sci(max(sv) / r["heads"]["top1"]["crash_criticality"]),
           sci(r["heads"]["top1"]["crash_criticality"])),
        "A row (the d_k weights that write one output feature) at ±1e2 costs %s to %s."
        % (sci(min(rv)), sci(max(rv))),
        "Columns and the whole block are value-invariant up to 1e4: every value gives the "
        "damage of setting them to 0 (block of %s: %s to %s, its crash criticality is %s). Only "
        "at ±1e6 does float32 rounding break the LayerNorm cancellation."
        % (t1, sci(min(blk), 4), sci(max(blk), 4), sci(r["heads"]["top1"]["crash_criticality"], 4)),
    ])
    figure(doc, r["wo"] / "figures" / "deep_value_sweep.png",
           "Figure 4. Damage vs the value the targets are set to, per granularity and head.")


def section_d2(doc, r):
    doc.add_heading("5.6 D2: the original magnitude of the changed weight", level=2)
    d2 = r["d2"]
    h = {role: r["heads"][role]["head"] for role in ROLES}
    single = d2[(d2["granularity"] == "single") & (d2["op"] == "zero")].set_index("magnitude_bin")
    whole = d2[d2["granularity"] == "whole_bin"].set_index("magnitude_bin")
    rows = [[b] + [sci(single.loc[b, h[ro]]) for ro in ROLES]
            + [sci(whole.loc[b, h[ro]]) for ro in ("top1", "least")] for b in single.index]
    add_table(doc, ["|w| bin", "Zero one weight: top1", "top2", "top3", "least",
                    "Zero whole bin: top1", "Zero whole bin: least"], rows,
              "Table 12. Mean damage by magnitude bin of the changed weight (d01 = smallest "
              "10%, d10 = largest 10%).")
    c = r["d2corr"]
    c = c[c["op"] != "flip_exp_msb"].dropna(subset=["spearman_abs_w_vs_damage"])
    piv = c.pivot(index="role", columns="op", values="spearman_abs_w_vs_damage").reindex(ROLES)
    add_table(doc, ["Role"] + list(piv.columns), [[ro] + ["%.2f" % v for v in piv.loc[ro]]
                                                 for ro in piv.index],
              "Table 13. Spearman correlation of |w| with damage, single weights. Exponent-MSB "
              "flips are omitted: they destroy the model in every bin.")
    t1 = h["top1"]
    crash1 = r["heads"]["top1"]["crash_criticality"]
    neg = d2[(d2["granularity"] == "single") & (d2["op"] == "negate")].set_index("magnitude_bin")
    x10 = d2[(d2["granularity"] == "single") & (d2["op"] == "x10")].set_index("magnitude_bin")
    add_bullets(doc, [
        "Larger weights matter more: Spearman(|w|, damage) is %s over heads and operations."
        % rng(c["spearman_abs_w_vs_damage"].min(), c["spearman_abs_w_vs_damage"].max()),
        "Zeroing one weight of %s costs x%s more in the top 0.1%% than in the smallest decile."
        % (t1, sci(single.loc["top0.1%", t1] / single.loc["d01", t1])),
        "Negating costs about x%s of zeroing (the change is twice as large), x10 about x%s."
        % (sci((neg[t1] / single[t1]).median()), sci((x10[t1] / single[t1]).median())),
        "Zeroing the 10%% largest weights of %s at once costs %s, i.e. %s of its crash "
        "damage; the 10%% smallest cost %s (%s)."
        % (t1, sci(whole.loc["d10", t1]), pct(whole.loc["d10", t1] / crash1, 0),
           sci(whole.loc["d01", t1]), pct(whole.loc["d01", t1] / crash1)),
    ])
    figure(doc, r["wo"] / "figures" / "deep_magnitude.png",
           "Figure 5. Damage by |w| bin of the changed weight and by operation.")


def section_d3(doc, r):
    doc.add_heading("5.7 D3: the flipped bit", level=2)
    d3 = r["d3"].copy()
    d3["destroyed"] = destroyed(d3, r["ceiling"])
    d3["nonfinite"] = d3["nonfinite_rate"] > 0
    m = d3.pivot_table(index="bit", columns="role", values="D_pos_mean", aggfunc="mean")
    dst = d3.groupby("bit")["destroyed"].mean()
    nf = d3.groupby("bit")["nonfinite"].mean()
    field = lambda b: "mantissa" if b < 23 else ("sign" if b == 31 else "exponent")  # noqa: E731
    rows = [[b, field(b)] + [sci(m.loc[b, ro]) for ro in ROLES] + [pct(dst[b], 0), pct(nf[b], 0)]
            for b in m.index]
    add_table(doc, ["Bit", "Field", "top1", "top2", "top3", "least", "Destroyed",
                    "Non-finite"], rows, "Table 14. Mean damage per flipped bit, one weight "
              "per trial.")
    non30 = d3[d3["bit"] != 30]
    worst = non30.groupby("bit")["D_pos_mean"].mean()
    mant = d3[d3["bit"] < 23]["D_pos_mean"].mean()
    b27 = [d3[d3["bit"] == b]["D_pos_mean"].mean() for b in (27, 28, 29)]
    add_bullets(doc, [
        "Bit 30 (exponent MSB) destroyed the model in %s of the trials, on every head and "
        "independent of the weight." % pct(dst[30], 0),
        "Every other bit is harmless on its own: the worst, bit %d, costs %s on average. "
        "Mantissa bits cost %s on average, doubling with each bit up."
        % (int(worst.idxmax()), sci(worst.max()), sci(mant)),
        "Bits 27-29 give %s damage (%s, %s, %s): they are 1 in these small weights, and "
        "clearing any of them divides the weight by 2^16 or more, which amounts to zeroing it."
        % ("the same" if max(b27) - min(b27) < 0.05 * max(b27) else "similar",
           sci(b27[0]), sci(b27[1]), sci(b27[2])),
        "The sign bit (31) is a negation and costs %s." % sci(d3[d3["bit"] == 31]["D_pos_mean"].mean()),
    ])
    figure(doc, r["wo"] / "figures" / "deep_bit_position.png",
           "Figure 6. Damage and non-finite rate per flipped bit.")


def section_d4(doc, r):
    doc.add_heading("5.8 D4: does only the size of the change matter?", level=2)
    sp = r["S"]["damage_vs_delta_spearman_by_role"]
    add_table(doc, ["Role", "Head", "Spearman(||dW||_F, damage)"],
              [[ro, r["heads"][ro]["head"], "%.3f" % sp[ro]] for ro in ROLES if ro in sp],
              "Table 15. Rank correlation of the size of the change with the damage, all "
              "deep-dive trials with a finite change.")
    d4 = r["d4"]
    mid = d4[(d4["catastrophic_rate"] == 0) & (d4["damage_median"] < 0.1 * r["ceiling"])
             & (d4["log10_delta_mid"] > -4) & (d4["log10_delta_mid"] < 3)]
    piv = mid.pivot_table(index="delta_bin", columns="role", values="damage_median",
                          sort=False)
    rows, ratios = [], []
    for b, x in piv.iterrows():
        if {"top1", "least"} <= set(x.dropna().index):
            rows.append([b] + [sci(x.get(ro)) for ro in ROLES]
                        + [sci(x["top1"] / x["least"])])
            ratios.append(x["top1"] / x["least"])
    if rows:
        add_table(doc, ["||dW||_F bin (log10)", "top1", "top2", "top3", "least", "top1 / least"],
                  rows, "Table 16. Median damage at the same size of change (moderate changes).")
    ratio = float(np.median(ratios)) if ratios else np.nan
    crash_ratio = r["heads"]["top1"]["crash_criticality"] / r["heads"]["least"]["crash_criticality"]
    doc.add_paragraph(
        "The size of the change explains most of the damage (Spearman %s). At the same "
        "size, the median damage of the most critical head is x%s that of the least critical "
        "head, while their crash criticalities differ by x%s. %s"
        % (rng(min(sp.values()), max(sp.values())), sci(ratio), sci(crash_ratio),
           "Where the change lands therefore matters." if ratio >= 1.5 else
           "Where the change lands matters little: the heads lie on nearly one curve."))
    figure(doc, r["wo"] / "figures" / "deep_damage_vs_delta.png",
           "Figure 7. Damage vs ||dW||_F, all deep-dive trials, with binned medians per head.")


def section_findings(doc, r):
    doc.add_heading("6. Key findings", level=1)
    S = r["S"]
    ms = {m["setting"]: m for m in S["multihead"]}
    cmp = r["cmp"].set_index(["fault_model", "level"])
    bf, _, b = bitflip_rows(r)
    d1 = r["d1"].set_index(["granularity", "value_label"])
    t1 = r["heads"]["top1"]["head"]
    sv = max(d1.loc[("single", "+1e2"), t1], d1.loc[("single", "-1e2"), t1])
    c = r["d2corr"]
    c = c[c["op"] != "flip_exp_msb"]["spearman_abs_w_vs_damage"].dropna()
    whole = r["d2"][r["d2"]["granularity"] == "whole_bin"].set_index("magnitude_bin")
    crash1 = r["heads"]["top1"]["crash_criticality"]
    items = [
        "The crash is one W_o fault. Zeroing a head's W_o columns reproduced it bit-exactly; "
        "setting whole columns or the whole block to any constant up to 1e4 gave the same "
        "damage, because LayerNorm cancels the constant.",
        "Under bit flips one bit decides the outcome: a flip of the exponent MSB of any W_o "
        "weight destroyed the model (%s of such trials), while the other bits cost at most "
        "%s on average. At BER 1e-4 the single-head bit-flip criticality is x%s the crash, "
        "and %s of multi-head trials are worse than the crash on the same heads."
        % (pct(b[b["msb"]]["destroyed"].mean(), 0),
           sci(r["d3"][r["d3"]["bit"] != 30].groupby("bit")["D_pos_mean"].mean().max()),
           sci(cmp.loc[("bitflip", "ber=0.0001"), "ratio_to_crash_mean"]),
           pct(ms["bitflip_ber1e-4"].get("share_trials_worse_than_crash", np.nan))),
        "Stuck-at faults are milder than the crash short of the whole block: stuck-at-0 on "
        "50%% of a head's weights costs x%s of the crash, and +m on 10%% of every failed head's "
        "weights was worse than the crash in only %s of multi-head trials."
        % (sci(cmp.loc[("stuck_at_0", "frac=0.5"), "ratio_to_crash_mean"]),
           pct(ms["stuck_+m_frac0.10"].get("share_trials_worse_than_crash", np.nan))),
        "For a single changed weight the new value dominates: damage grows linearly up to "
        "about 10m, and one weight at 1e2 costs %s on %s, x%s its whole-head crash."
        % (sci(sv), t1, sci(sv / crash1)),
        "Larger original weights matter more (Spearman %s); the 10%% largest weights of %s "
        "carry %s of its crash damage." % (rng(c.min(), c.max()), t1,
                                           pct(whole.loc["d10", t1] / crash1, 0)),
        "Crash criticality predicts the head ranking under faults that remove much of a head "
        "(stuck-at-0 at 50%%: Spearman %s) but not under bit flips (Spearman %s at BER 1e-4)."
        % ("%.2f" % cmp.loc[("stuck_at_0", "frac=0.5"), "spearman_vs_crash"],
           "%.2f" % cmp.loc[("bitflip", "ber=0.0001"), "spearman_vs_crash"]),
    ]
    add_numbered(doc, items)


def section_end(doc, r):
    doc.add_heading("7. Limitations", level=1)
    add_bullets(doc, [
        "Only the W_o columns of a head are faulted. Faults in W_q, W_k, W_v, the shared W_o "
        "bias, the feed-forward layers or the activations are not studied.",
        "The value grid jumps from 10m to 1e2, so the value at which one weight becomes "
        "dangerous is only bracketed, not located.",
        "Each fault is applied to a frozen model at inference time, as in the crash study; "
        "there is no detection or correction (e.g. range checks or ECC), which would remove "
        "most of the bit-30 failures.",
        "One trained model per dataset and architecture; the deep-dive covers four heads.",
    ])
    doc.add_heading("8. Reproducibility", level=1)
    fam, ds = r["family"], r["dataset"]
    add_bullets(doc, [
        "Notebook: notebooks/%s/%s_wo_faults.ipynb, generated by scripts/make_wo_notebooks.py "
        "--family %s. It loads results/%s/%s/models/best.pt and never trains." % (fam, ds, fam, fam, ds),
        "Results: results/%s/%s/wo_faults/ (summary.json, tables/*.csv, figures/*.png)." % (fam, ds),
        "Library: src/faulttol/wo_faults.py; settings WO_* in src/faulttol/config.py; method "
        "in docs/wo_faults.md.",
        "This document was generated from the result files by scripts/make_wo_report.py and "
        "can be regenerated at any time without re-running the experiment.",
    ])


def build(r) -> pathlib.Path:
    mc = r["mc"]
    doc = new_document(
        "%s: attention-head faults in the W_o weights" % DATASETS[r["dataset"]],
        "%s, %d x %d = %d heads. Bit flips, stuck-at faults and targeted weight changes in "
        "the output projection, compared with the head crash."
        % (FAMILY_LABEL[r["family"]], mc["num_layers"], mc["num_heads"], r["n_heads"]))
    for f in (section_intro, section_protocol, section_crash, section_e0, section_e1,
              section_e2, section_d1, section_d2, section_d3, section_d4, section_findings,
              section_end):
        f(doc, r)
    out = ROOT / "reports" / r["family"] / ("%s_wo_faults_report_EN.docx" % r["dataset"])
    out.parent.mkdir(parents=True, exist_ok=True)
    doc.save(str(out))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("datasets", nargs="*", help="subset of %s" % ", ".join(DATASETS))
    ap.add_argument("--family", choices=FAMILIES, action="append")
    args = ap.parse_args()
    for ds in args.datasets:
        if ds not in DATASETS:
            raise SystemExit("unknown dataset %r (known: %s)" % (ds, ", ".join(DATASETS)))
    for fam in args.family or FAMILIES:
        for ds in args.datasets or list(DATASETS):
            r = load(fam, ds)
            if r is None:
                print("%s/%s: no wo_faults/summary.json yet, skipping" % (fam, ds))
                continue
            print("wrote", build(r).relative_to(ROOT))


if __name__ == "__main__":
    main()
