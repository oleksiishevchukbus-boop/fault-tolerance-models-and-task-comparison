"""Random multi-head fault injection (Paper B, Section 3.3.2).

For each of K trials: draw N_fail uniformly from 1..M, pick that many heads
without replacement, crash them all simultaneously, evaluate on the fixed
subset, and record the fault configuration together with its degradation.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from .injection import (
    disable_multiple_heads,
    format_failed_heads,
    get_all_heads,
    get_layer_fault_counts,
    get_number_of_affected_layers,
    reset_head_masks,
)

EPS = 1e-12


def run_random_multihead(model, evaluate, damage_fn, weight_lookup: dict,
                         critical_heads: list[tuple[int, int]], model_name: str,
                         k_trials: int, seed: int, num_layers: int,
                         progress_every: int = 500) -> pd.DataFrame:
    all_heads = get_all_heads(model)
    n_heads = len(all_heads)
    critical_set = {tuple(h) for h in critical_heads}
    rng = np.random.default_rng(seed)

    reset_head_masks(model)
    clean = evaluate()

    rows = []
    t0 = time.time()
    for trial in range(k_trials):
        n_fail = int(rng.integers(1, n_heads + 1))
        idx = rng.choice(n_heads, size=n_fail, replace=False)
        failed = [all_heads[i] for i in idx]

        disable_multiple_heads(model, failed)
        fault = evaluate()

        weights = np.array([weight_lookup[h] for h in failed], dtype=float)
        layer_counts = get_layer_fault_counts(failed, num_layers)

        row = {
            "model": model_name,
            "run_id": trial,
            "N_fail": n_fail,
            "fail_fraction": n_fail / n_heads,
            "L_fault": get_number_of_affected_layers(layer_counts),
            "N_crit_fail": int(sum(1 for h in failed if h in critical_set)),
            "W_fail": float(weights.sum()),
            "W_fail_mean": float(weights.mean()),
            "failed_heads": format_failed_heads(failed),
        }
        for li, c in enumerate(layer_counts):
            row["n_fail_layer_%d" % li] = int(c)
        row.update(damage_fn(clean, fault))
        rows.append(row)

        if progress_every and (trial + 1) % progress_every == 0:
            elapsed = time.time() - t0
            rate = (trial + 1) / elapsed
            print(
                "    trial %d/%d  %.1f trials/s  eta %.1f min"
                % (trial + 1, k_trials, rate, (k_trials - trial - 1) / rate / 60.0),
                flush=True,
            )

    reset_head_masks(model)
    return pd.DataFrame(rows)


def fault_fraction_bins(df: pd.DataFrame) -> pd.Series:
    edges = [0.0, 0.10, 0.25, 0.50, 0.75, 1.0001]
    labels = ["0-10%", "10-25%", "25-50%", "50-75%", "75-100%"]
    return pd.cut(df["fail_fraction"], bins=edges, labels=labels,
                  include_lowest=True, right=True)


def criticality_quartiles(df: pd.DataFrame) -> pd.Series:
    """Q1 = least critical failed heads ... Q4 = most critical."""
    return pd.qcut(df["W_fail_mean"].rank(method="first"), 4,
                   labels=["Q1", "Q2", "Q3", "Q4"])


def degradation_grid(df: pd.DataFrame, degradation_col: str) -> pd.DataFrame:
    """Mean degradation by fault fraction x mean-failed-head-criticality quartile.

    Reproduces Figure 3 / Figure 4 of Paper B.
    """
    work = df.copy()
    work["fault_bin"] = fault_fraction_bins(work)
    work["crit_quartile"] = criticality_quartiles(work)
    grid = work.pivot_table(index="fault_bin", columns="crit_quartile",
                            values=degradation_col, aggfunc="mean", observed=False)
    return grid
