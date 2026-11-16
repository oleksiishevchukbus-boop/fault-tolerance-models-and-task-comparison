"""Figures: vulnerability maps, degradation grids, pseudo-observations, tau curves."""
from __future__ import annotations

import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402


def _save(fig, path: pathlib.Path) -> pathlib.Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def vulnerability_map(single_head_df: pd.DataFrame, title: str,
                      path: pathlib.Path, value_col: str = "criticality"):
    """Layer x head heatmap of single-head criticality (Figure 2 analogue)."""
    pivot = single_head_df.pivot(index="layer", columns="head", values=value_col)
    fig, ax = plt.subplots(figsize=(1.0 + 0.7 * pivot.shape[1], 1.2 + 0.6 * pivot.shape[0]))
    im = ax.imshow(pivot.to_numpy(), cmap="magma_r", aspect="auto")
    ax.set_xticks(range(pivot.shape[1]))
    ax.set_xticklabels(pivot.columns)
    ax.set_yticks(range(pivot.shape[0]))
    ax.set_yticklabels(pivot.index)
    ax.set_xlabel("head")
    ax.set_ylabel("encoder layer")
    ax.set_title(title, fontsize=10)
    vmax = float(np.nanmax(pivot.to_numpy()))
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            val = pivot.to_numpy()[i, j]
            ax.text(j, i, "%.3f" % val, ha="center", va="center", fontsize=6,
                    color="white" if val > 0.55 * vmax else "black")
    fig.colorbar(im, ax=ax, label=value_col)
    return _save(fig, path)


def degradation_grid_plot(grid: pd.DataFrame, title: str, path: pathlib.Path,
                          value_label: str):
    """Mean degradation by fault fraction x criticality quartile (Fig 3/4)."""
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    data = grid.to_numpy(dtype=float)
    im = ax.imshow(data, cmap="viridis", aspect="auto")
    ax.set_xticks(range(grid.shape[1]))
    ax.set_xticklabels(grid.columns)
    ax.set_yticks(range(grid.shape[0]))
    ax.set_yticklabels(grid.index)
    ax.set_xlabel("mean failed-head criticality quartile")
    ax.set_ylabel("fraction of failed heads")
    ax.set_title(title, fontsize=10)
    finite = data[np.isfinite(data)]
    vmax = float(finite.max()) if finite.size else 1.0
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            val = data[i, j]
            if np.isfinite(val):
                ax.text(j, i, "%.3f" % val, ha="center", va="center", fontsize=7,
                        color="white" if val < 0.55 * vmax else "black")
    fig.colorbar(im, ax=ax, label=value_label)
    return _save(fig, path)


def pseudo_observation_panel(pairs: list[tuple[str, np.ndarray, np.ndarray]],
                             title: str, path: pathlib.Path):
    """Pseudo-observation scatter for each severity/degradation pair (Fig 5-8)."""
    fig, axes = plt.subplots(1, len(pairs), figsize=(3.4 * len(pairs), 3.4))
    if len(pairs) == 1:
        axes = [axes]
    for ax, (label, u, v) in zip(axes, pairs):
        ax.scatter(u, v, s=1.5, alpha=0.25, color="#1f4e79", linewidths=0)
        ax.set_xlabel("u  (%s)" % label.split(" vs ")[0])
        ax.set_ylabel("v  (%s)" % label.split(" vs ")[-1])
        ax.set_title(label, fontsize=9)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect("equal")
    fig.suptitle(title, fontsize=10)
    return _save(fig, path)


def required_tau_plot(tau_df: pd.DataFrame, title: str, path: pathlib.Path,
                      ylabel: str):
    """tau_required as the admissible failure boundary widens (Figure 9)."""
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    if tau_df.empty:
        ax.text(0.5, 0.5, "no boundary reached the minimum configuration count",
                ha="center", va="center", fontsize=8)
        ax.set_axis_off()
        return _save(fig, path)
    for c, sub in tau_df.groupby("n_crit_fail_max"):
        sub = sub.sort_values("n_fail_max")
        ax.plot(sub["n_fail_max"], sub["required_tau"], marker="o", markersize=3,
                label="N_crit_fail <= %d" % int(c))
    ax.set_xlabel("maximum total failed heads (n)")
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=10)
    ax.legend(fontsize=7)
    ax.grid(alpha=0.25)
    return _save(fig, path)


def criticality_profile(single_head_df: pd.DataFrame, title: str,
                        path: pathlib.Path, ylabel: str):
    """Sorted criticality profile: how concentrated is head importance."""
    vals = np.sort(single_head_df["criticality"].to_numpy())[::-1]
    fig, ax = plt.subplots(figsize=(5.2, 3.2))
    ax.bar(range(len(vals)), vals, color="#8c2d04")
    ax.set_xlabel("heads, ranked by criticality")
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=10)
    ax.grid(alpha=0.25, axis="y")
    return _save(fig, path)


def degradation_curves(curves: dict, title: str, path: pathlib.Path,
                       ylabel: str, normalise: bool = False,
                       xlabel: str = "number of failed attention heads"):
    """Mean degradation vs number of failed heads, one line per dataset.

    curves: {label: (n_fail array, degradation array)}
    normalise: divide each curve by its own maximum, so datasets measured on
    different scales (probability damage vs delta MAE) can share an axis.
    """
    fig, ax = plt.subplots(figsize=(6.6, 4.0))
    for label, (x, y) in curves.items():
        grouped = pd.DataFrame({"n": np.asarray(x), "y": np.asarray(y)}) \
            .groupby("n")["y"].mean()
        vals = grouped.to_numpy(dtype=float)
        if normalise:
            m = float(np.nanmax(np.abs(vals))) if vals.size else 0.0
            if m > 0:
                vals = vals / m
        ax.plot(grouped.index.to_numpy(), vals, marker="o", markersize=2.5,
                linewidth=1.2, label=label)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=10)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=7, ncol=2)
    return _save(fig, path)


def criticality_comparison(profiles: dict, title: str, path: pathlib.Path,
                           ylabel: str = "criticality / max criticality"):
    """Sorted, self-normalised criticality profiles for several models."""
    fig, ax = plt.subplots(figsize=(6.6, 4.0))
    for label, vals in profiles.items():
        v = np.sort(np.asarray(vals, dtype=float))[::-1]
        m = float(v.max()) if v.size else 0.0
        ax.plot(range(len(v)), v / m if m > 0 else v, marker="o", markersize=2.5,
                linewidth=1.2, label=label)
    ax.set_xlabel("heads, ranked by criticality")
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=10)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=7, ncol=2)
    return _save(fig, path)


# --------------------------------------------------------------------------
# W_o weight-fault figures (docs/wo_faults.md)
# --------------------------------------------------------------------------
ROLE_COLORS = {"top1": "#b2182b", "top2": "#ef8a62", "top3": "#f4a582",
               "least": "#2166ac"}


def _role_color(role):
    return ROLE_COLORS.get(role, "#555555")


def wo_level_curves(crit_df: pd.DataFrame, crash_df: pd.DataFrame, title: str,
                    path: pathlib.Path, ylabel: str):
    """E1: per-head W_o criticality vs fault level, one panel per fault model.

    Grey lines are individual heads, black is the mean over heads, the dashed
    line is the mean crash criticality.
    """
    models = list(dict.fromkeys(crit_df["fault_model"]))
    fig, axes = plt.subplots(1, len(models), figsize=(3.3 * len(models), 3.2),
                             squeeze=False, sharey=True)
    crash_mean = float(crash_df["criticality"].mean())
    for ax, fm in zip(axes[0], models):
        sub = crit_df[crit_df["fault_model"] == fm]
        levels = list(dict.fromkeys(sub["level"]))
        x = np.arange(len(levels))
        piv = sub.pivot_table(index="head_name", columns="level", values="criticality")
        piv = piv[levels]
        for _, row in piv.iterrows():
            ax.plot(x, row.to_numpy(), color="#bbbbbb", linewidth=0.7)
        ax.plot(x, piv.mean().to_numpy(), color="black", marker="o", linewidth=1.6,
                label="mean over heads")
        ax.axhline(crash_mean, color="#b2182b", linestyle="--", linewidth=1,
                   label="crash (mean)")
        ax.set_xticks(x)
        ax.set_xticklabels([lv.split("=")[1] for lv in levels], fontsize=7)
        ax.set_xlabel(levels[0].split("=")[0])
        ax.set_yscale("symlog", linthresh=1e-4)
        ax.set_title(fm, fontsize=9)
        ax.grid(alpha=0.25)
    axes[0][0].set_ylabel(ylabel)
    axes[0][0].legend(fontsize=7)
    fig.suptitle(title, fontsize=10)
    return _save(fig, path)


def crash_vs_wo_criticality(crit_df: pd.DataFrame, crash_df: pd.DataFrame,
                            title: str, path: pathlib.Path):
    """E1: crash criticality vs W_o criticality per head, one panel per fault model."""
    crash = crash_df.set_index("head_name")["criticality"]
    models = list(dict.fromkeys(crit_df["fault_model"]))
    fig, axes = plt.subplots(1, len(models), figsize=(3.1 * len(models), 3.1),
                             squeeze=False)
    for ax, fm in zip(axes[0], models):
        sub = crit_df[crit_df["fault_model"] == fm]
        levels = list(dict.fromkeys(sub["level"]))
        for lv, color in zip(levels, plt.cm.viridis(np.linspace(0, 0.9, len(levels)))):
            g = sub[sub["level"] == lv]
            ax.scatter(crash.loc[g["head_name"]], g["criticality"], s=10, color=color,
                       label=lv)
        ax.set_xlabel("crash criticality")
        ax.set_title(fm, fontsize=9)
        ax.set_yscale("symlog", linthresh=1e-4)
        ax.grid(alpha=0.25)
        ax.legend(fontsize=6)
    axes[0][0].set_ylabel("W_o fault criticality")
    fig.suptitle(title, fontsize=10)
    return _save(fig, path)


def value_sweep_plot(df: pd.DataFrame, col: str, title: str, path: pathlib.Path):
    """D1: mean damage vs the value the weights were set to, per granularity.

    The x axis is the value labels ordered by value (m = max|W_o| of the layer).
    """
    grans = [g for g in ("single", "row", "column", "block")
             if g in set(df["granularity"])]
    order = df.groupby("value_label")["value"].median().sort_values().index.tolist()
    x = np.arange(len(order))
    fig, axes = plt.subplots(1, len(grans), figsize=(3.6 * len(grans), 3.6),
                             squeeze=False, sharey=True)
    for ax, gran in zip(axes[0], grans):
        sub = df[df["granularity"] == gran]
        for (hn, role), g in sub.groupby(["head_name", "role"], sort=False):
            agg = g.groupby("value_label")[col].mean().reindex(order)
            ax.plot(x, agg.to_numpy(), marker="o", markersize=3,
                    color=_role_color(role), label="%s (%s)" % (hn, role))
        ax.set_xticks(x)
        ax.set_xticklabels(order, rotation=70, fontsize=7)
        ax.set_yscale("symlog", linthresh=1e-5)
        ax.set_xlabel("value the weights are set to")
        ax.set_title(gran, fontsize=9)
        ax.grid(alpha=0.25)
    axes[0][0].set_ylabel(col)
    axes[0][0].legend(fontsize=7)
    fig.suptitle(title, fontsize=10)
    return _save(fig, path)


def magnitude_bins_plot(df: pd.DataFrame, col: str, title: str, path: pathlib.Path):
    """D2: mean damage by |w| bin of the changed weight, per operation."""
    single_ops = list(dict.fromkeys(df.loc[df["granularity"] == "single", "op"]))
    panels = [("single", op) for op in single_ops] + [("whole_bin", "zero")]
    bins = list(dict.fromkeys(df["magnitude_bin"]))
    fig, axes = plt.subplots(1, len(panels), figsize=(3.2 * len(panels), 3.4),
                             squeeze=False, sharey=True)
    x = np.arange(len(bins))
    for ax, (gran, op) in zip(axes[0], panels):
        sub = df[(df["granularity"] == gran) & (df["op"] == op)]
        for (hn, role), g in sub.groupby(["head_name", "role"], sort=False):
            agg = g.groupby("magnitude_bin")[col].mean().reindex(bins)
            ax.plot(x, agg.to_numpy(), marker="o", markersize=3,
                    color=_role_color(role), label="%s (%s)" % (hn, role))
        ax.set_xticks(x)
        ax.set_xticklabels(bins, rotation=70, fontsize=7)
        ax.set_yscale("symlog", linthresh=1e-6)
        where = "single weight" if gran == "single" else "whole bin"
        ax.set_title("%s: %s" % (where, op), fontsize=9)
        ax.grid(alpha=0.25)
    axes[0][0].set_ylabel(col)
    axes[0][0].set_xlabel("|w| bin (small to large)")
    axes[0][0].legend(fontsize=7)
    fig.suptitle(title, fontsize=10)
    return _save(fig, path)


def bit_position_plot(df: pd.DataFrame, col: str, title: str, path: pathlib.Path):
    """D3: mean damage and catastrophic-output rate per flipped bit (0..31)."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8.0, 5.4), sharex=True)
    for ax in (ax1, ax2):
        ax.axvspan(-0.5, 22.5, color="#f0f0f0", zorder=0)
        ax.axvspan(22.5, 30.5, color="#fde0dd", zorder=0)
        ax.axvspan(30.5, 31.5, color="#deebf7", zorder=0)
    for (hn, role), g in df.groupby(["head_name", "role"], sort=False):
        agg = g.groupby("bit").agg(
            d=(col, "mean"),
            cat=("nonfinite_rate", lambda s: float(np.mean(s > 0))))
        ax1.plot(agg.index, agg["d"], marker="o", markersize=3, color=_role_color(role),
                 label="%s (%s)" % (hn, role))
        ax2.plot(agg.index, agg["cat"], marker="o", markersize=3, color=_role_color(role))
    ax1.set_yscale("symlog", linthresh=1e-7)
    ax1.set_ylabel(col)
    ax1.legend(fontsize=7)
    ax1.set_title(title + "\n(grey: mantissa, red: exponent, blue: sign)", fontsize=10)
    ax2.set_ylabel("share of flips with\nnon-finite outputs")
    ax2.set_xlabel("flipped bit (0 = mantissa LSB, 30 = exponent MSB, 31 = sign)")
    ax2.set_xticks(range(0, 32, 2))
    for ax in (ax1, ax2):
        ax.grid(alpha=0.25)
    return _save(fig, path)


def delta_damage_plot(pooled: pd.DataFrame, binned: pd.DataFrame, col: str,
                      title: str, path: pathlib.Path):
    """D4: damage vs size of the weight change ||dW||, pooled over the deep-dive."""
    fig, ax = plt.subplots(figsize=(6.8, 4.0))
    finite = pooled[np.isfinite(pooled["delta_norm"]) & (pooled["delta_norm"] > 0)]
    for role, g in finite.groupby("role", sort=False):
        ax.scatter(g["delta_norm"], g[col], s=4, alpha=0.25, color=_role_color(role))
    for role, g in binned.groupby("role", sort=False):
        ax.plot(10 ** g["log10_delta_mid"], g["damage_median"], color=_role_color(role),
                linewidth=1.8, label="%s median" % role)
    ax.set_xscale("log")
    ax.set_yscale("symlog", linthresh=1e-6)
    ax.set_xlabel("||dW||_F (size of the change to the head's W_o columns)")
    ax.set_ylabel(col)
    ax.set_title(title, fontsize=10)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=7)
    return _save(fig, path)
