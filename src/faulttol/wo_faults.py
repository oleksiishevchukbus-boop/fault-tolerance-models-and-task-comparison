"""Attention-head faults injected into the output projection W_o.

The crash fault of Definition 1 zeroes a head's output before the heads are
concatenated and projected by W_o.  Because

    W_o(concat(head_0 .. head_H)) = sum_h W_o[:, h*d_k:(h+1)*d_k] @ head_h + b_o

zeroing head h's d_k columns of W_o.weight removes exactly the same term, so a
crash is one particular W_o weight fault (`zero`).  This module corrupts those
columns in other ways - random bit flips, stuck-at values, a chosen value,
scaling, a flip of one chosen bit - so the crash results can be compared with
realistic weight faults and the deep-dive can ask which weight changes matter.

Every fault is applied in place inside the `wo_fault` context manager and the
original weights are restored on exit, so the model is always clean between
trials.  The W_o bias is shared by all heads and is never touched.

A corrupted weight can be inf/NaN (an exponent bit flip) or large enough to
overflow the forward pass.  `sanitize_outputs` turns every sample whose class
probabilities are not finite into a failed prediction: uniform probabilities
and a wrong predicted label.  `nonfinite_rate` reports how often that happened,
so catastrophic outcomes are visible rather than averaged away as NaN.
"""
from __future__ import annotations

import hashlib
import time
from contextlib import contextmanager

import numpy as np
import pandas as pd
import torch

from .injection import (
    attention_modules,
    disable_multiple_heads,
    format_failed_heads,
    get_all_heads,
    get_layer_fault_counts,
    get_number_of_affected_layers,
    head_name,
    reset_head_masks,
)

EPS = 1e-12
FLOAT_BITS = 32
SIGN_BIT = 31
EXPONENT_BITS = tuple(range(23, 31))   # bit 30 is the exponent MSB
MANTISSA_BITS = tuple(range(0, 23))


# --------------------------------------------------------------------------
# where a head lives in W_o
# --------------------------------------------------------------------------
def wo_block(model, layer_id: int, head_id: int) -> torch.Tensor:
    """View of head (layer_id, head_id)'s columns of W_o.weight, shape (d_model, d_k).

    Writing to the view writes to the model.  Use it under torch.no_grad().
    """
    attn = attention_modules(model)[layer_id]
    if not 0 <= head_id < attn.num_heads:
        raise ValueError("head_id %d out of range for layer %d" % (head_id, layer_id))
    d_k = attn.d_k
    return attn.W_o.weight[:, head_id * d_k:(head_id + 1) * d_k]


def layer_wo_absmax(model) -> list[float]:
    """m = max|W_o.weight| per layer: the scale for the 'rel' fault values."""
    with torch.no_grad():
        return [float(a.W_o.weight.abs().max()) for a in attention_modules(model)]


def column_indices(block_shape, column: int) -> np.ndarray:
    """Flat indices (row-major) of one column of a (d_model, d_k) block.

    A column holds the d_model weights that read one of the head's d_k output
    dimensions.  Setting it to a constant adds the same value to every output
    feature, which a following LayerNorm removes again.
    """
    d_model, d_k = block_shape
    return np.arange(d_model) * d_k + int(column)


def row_indices(block_shape, row: int) -> np.ndarray:
    """Flat indices of one row: the d_k weights that write one output feature."""
    _, d_k = block_shape
    return int(row) * d_k + np.arange(d_k)


def model_checksum(model) -> str:
    """SHA-1 of every tensor in the state dict, to prove the model was restored."""
    h = hashlib.sha1()
    for name, t in model.state_dict().items():
        h.update(name.encode())
        h.update(t.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


# --------------------------------------------------------------------------
# applying one fault to one block
# --------------------------------------------------------------------------
def _resolve_value(spec: dict, m_layer: float) -> float:
    if "value_rel" in spec:
        return float(spec["value_rel"]) * m_layer
    return float(spec["value"])


def _flip_bits(block: torch.Tensor, flat_idx: np.ndarray, bits: np.ndarray) -> None:
    """XOR the given (element, bit) positions of a float32 block, in place."""
    if len(flat_idx) == 0:
        return
    mask = np.zeros(block.numel(), dtype=np.uint32)
    np.bitwise_xor.at(mask, np.asarray(flat_idx, dtype=np.int64),
                      np.left_shift(np.uint32(1), np.asarray(bits, dtype=np.uint32)))
    mask_t = torch.from_numpy(mask.view(np.int32).reshape(tuple(block.shape)))
    block.view(torch.int32).bitwise_xor_(mask_t.to(block.device))


def _set_flat(block: torch.Tensor, flat_idx: np.ndarray, value) -> None:
    d_k = block.shape[1]
    idx = torch.as_tensor(np.asarray(flat_idx, dtype=np.int64), device=block.device)
    rows, cols = idx // d_k, idx % d_k
    if torch.is_tensor(value):
        block[rows, cols] = value.to(block.dtype)
    else:
        block[rows, cols] = torch.tensor(float(value), dtype=block.dtype,
                                         device=block.device)


def apply_spec(block: torch.Tensor, spec: dict, rng: np.random.Generator,
               m_layer: float) -> dict:
    """Corrupt `block` in place according to `spec`; return a record of the change.

    Specs (flat indices are row-major over the (d_model, d_k) block):
      {"kind": "zero"}                                  whole block -> 0 (= crash)
      {"kind": "bitflip_ber", "p": p}                   each bit flips with prob p
      {"kind": "stuck_at", "value"|"value_rel": v, "frac": f}
                                                        random fraction f -> v
      {"kind": "set_value", "idx": idx, "value"|"value_rel": v}
      {"kind": "flip_bit", "idx": idx, "bit": b}        flip bit b of those weights
      {"kind": "scale", "idx": idx, "factor": a}        w -> a*w (a=-1 negates)
    "value_rel" is a multiple of m_layer = max|W_o| of the head's layer.
    """
    kind = spec["kind"]
    n = block.numel()
    old = block.detach().clone()
    rec = {"kind": kind, "n_bitflips": 0, "n_exp_msb_flips": 0, "value": np.nan}

    if kind == "zero":
        block.zero_()
    elif kind == "bitflip_ber":
        n_bits = n * FLOAT_BITS
        k = int(rng.binomial(n_bits, float(spec["p"])))
        pos = (rng.choice(n_bits, size=k, replace=False) if k
               else np.empty(0, dtype=np.int64))
        bits = pos % FLOAT_BITS
        _flip_bits(block, pos // FLOAT_BITS, bits)
        rec["n_bitflips"] = k
        rec["n_exp_msb_flips"] = int(np.sum(bits == 30))
    elif kind == "stuck_at":
        value = _resolve_value(spec, m_layer)
        k = int(round(float(spec["frac"]) * n))
        idx = rng.choice(n, size=k, replace=False) if k else np.empty(0, dtype=np.int64)
        _set_flat(block, idx, value)
        rec["value"] = value
    elif kind == "set_value":
        value = _resolve_value(spec, m_layer)
        _set_flat(block, np.atleast_1d(spec["idx"]), value)
        rec["value"] = value
    elif kind == "flip_bit":
        idx = np.atleast_1d(spec["idx"])
        bits = np.full(len(idx), int(spec["bit"]))
        _flip_bits(block, idx, bits)
        rec["n_bitflips"] = int(len(idx))
        rec["n_exp_msb_flips"] = int(len(idx)) if int(spec["bit"]) == 30 else 0
    elif kind == "scale":
        idx = np.atleast_1d(spec["idx"])
        d_k = block.shape[1]
        it = torch.as_tensor(np.asarray(idx, dtype=np.int64), device=block.device)
        rows, cols = it // d_k, it % d_k
        block[rows, cols] = block[rows, cols] * float(spec["factor"])
    else:
        raise ValueError("unknown W_o fault kind %r" % kind)

    new = block.detach()
    changed = (new.contiguous().view(torch.int32) != old.view(torch.int32))
    old64, new64 = old.double(), new.double()
    finite = torch.isfinite(new64)
    rec["n_changed"] = int(changed.sum())
    rec["nonfinite_weights"] = int((~finite).sum())
    rec["w_sq"] = float((old64 ** 2).sum())
    if bool(finite.all()):
        rec["delta_sq"] = float(((new64 - old64) ** 2).sum())
    else:
        rec["delta_sq"] = float("inf")
    rec["max_abs_new"] = float(new64.abs().nan_to_num(nan=float("inf")).max())
    if "idx" in spec and np.size(spec["idx"]) == 1:
        i = int(np.atleast_1d(spec["idx"])[0])
        rec["old_value"] = float(old.reshape(-1)[i])
        rec["new_value"] = float(new.reshape(-1)[i])
    return rec


@contextmanager
def wo_fault(model, faults, rng: np.random.Generator | None = None):
    """Apply [(layer, head, spec), ...] to W_o, yield the records, then restore.

    The values of m = max|W_o| per layer are read once, before any fault is
    applied, so several faults in one layer all scale to the clean weights.
    """
    rng = rng if rng is not None else np.random.default_rng()
    m = layer_wo_absmax(model)
    saved, records = [], []
    try:
        with torch.no_grad():
            for layer_id, head_id, spec in faults:
                blk = wo_block(model, layer_id, head_id)
                saved.append((blk, blk.detach().clone()))
                rec = apply_spec(blk, spec, rng, m[layer_id])
                rec.update(layer=int(layer_id), head=int(head_id), m_layer=m[layer_id])
                records.append(rec)
        yield records
    finally:
        with torch.no_grad():
            for blk, orig in reversed(saved):
                blk.copy_(orig)


def summarize_records(records: list[dict]) -> dict:
    """One row's worth of fault descriptors over all faulted heads of a trial."""
    w_sq = sum(r["w_sq"] for r in records)
    d_sq = sum(r["delta_sq"] for r in records)
    return {
        "n_changed": int(sum(r["n_changed"] for r in records)),
        "n_bitflips": int(sum(r["n_bitflips"] for r in records)),
        "n_exp_msb_flips": int(sum(r["n_exp_msb_flips"] for r in records)),
        "nonfinite_weights": int(sum(r["nonfinite_weights"] for r in records)),
        "delta_norm": float(np.sqrt(d_sq)),
        "rel_change": float(np.sqrt(d_sq) / (np.sqrt(w_sq) + EPS)),
        "max_abs_new": float(max(r["max_abs_new"] for r in records)) if records else 0.0,
    }


# --------------------------------------------------------------------------
# evaluation that survives inf / NaN weights
# --------------------------------------------------------------------------
def sanitize_outputs(out: dict) -> dict:
    """Samples with non-finite probabilities become failed predictions."""
    p = out["p_all"]
    bad = ~np.isfinite(p).all(axis=1)
    out = dict(out)
    out["nonfinite_rate"] = float(bad.mean())
    if bad.any():
        n_classes = p.shape[1]
        p = p.copy()
        p[bad] = 1.0 / n_classes
        p_true = out["p_true"].copy()
        p_true[bad] = 1.0 / n_classes
        y_pred = out["y_pred"].copy()
        y_pred[bad] = (np.asarray(out["y_true"])[bad] + 1) % n_classes
        out.update(p_all=p, p_true=p_true, y_pred=y_pred)
    return out


def safe_evaluate(evaluate):
    return lambda: sanitize_outputs(evaluate())


# --------------------------------------------------------------------------
# E0: the zero fault must reproduce the crash mask exactly
# --------------------------------------------------------------------------
def sample_head_sets(all_heads, k: int, seed: int):
    """The same N_fail / head draws as multihead.run_random_multihead."""
    rng = np.random.default_rng(seed)
    n = len(all_heads)
    for _ in range(k):
        n_fail = int(rng.integers(1, n + 1))
        idx = rng.choice(n, size=n_fail, replace=False)
        yield [all_heads[i] for i in idx]


def run_wo_sanity(model, evaluate, n_multi: int, seed: int,
                  fault_seed: int = 0) -> pd.DataFrame:
    """Compare W_o faults that must equal a known model state.

    - zero W_o columns vs head_mask crash: every single head + n_multi random sets
    - bitflip at BER 0, and the same bit flipped twice: must equal the clean model
    """
    all_heads = get_all_heads(model)
    rng = np.random.default_rng(fault_seed)
    reset_head_masks(model)
    clean = evaluate()["p_all"]
    rows = []

    def _cmp(check, label, a, b):
        diff = np.abs(a - b)
        rows.append({"check": check, "config": label,
                     "exact": bool(np.array_equal(a, b)),
                     "max_abs_diff": float(np.nanmax(diff)) if diff.size else 0.0})

    sets = [("single", [h]) for h in all_heads]
    sets += [("multi", s) for s in sample_head_sets(all_heads, n_multi, seed)]
    for check, failed in sets:
        disable_multiple_heads(model, failed)
        p_mask = evaluate()["p_all"]
        reset_head_masks(model)
        with wo_fault(model, [(l, h, {"kind": "zero"}) for l, h in failed], rng):
            p_wo = evaluate()["p_all"]
        _cmp("zero_vs_mask_" + check, format_failed_heads(failed), p_mask, p_wo)

    for l, h in all_heads:
        with wo_fault(model, [(l, h, {"kind": "bitflip_ber", "p": 0.0})], rng):
            p = evaluate()["p_all"]
        _cmp("ber0_vs_clean", head_name(l, h), clean, p)
        n = wo_block(model, l, h).numel()
        idx = rng.choice(n, size=16, replace=False)
        bit = int(rng.integers(0, FLOAT_BITS))
        spec = {"kind": "flip_bit", "idx": idx, "bit": bit}
        with wo_fault(model, [(l, h, spec), (l, h, spec)], rng):
            p = evaluate()["p_all"]
        _cmp("double_flip_vs_clean", "%s bit%d" % (head_name(l, h), bit), clean, p)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# E1: single-head W_o faults over every head
# --------------------------------------------------------------------------
def build_fault_grid(ber_levels, stuck_values_rel, stuck_fracs) -> list[dict]:
    grid = []
    for p in ber_levels:
        grid.append({"fault_model": "bitflip", "level": "ber=%g" % p,
                     "spec": {"kind": "bitflip_ber", "p": float(p)},
                     "stochastic": True})
    for v in stuck_values_rel:
        vlabel = "0" if v == 0 else ("%+gm" % v).replace("1m", "m")
        for f in stuck_fracs:
            grid.append({"fault_model": "stuck_at_%s" % vlabel,
                         "level": "frac=%g" % f,
                         "spec": {"kind": "stuck_at", "value_rel": float(v),
                                  "frac": float(f)},
                         "stochastic": f < 1.0})
    return grid


def _progress(i, total, t0, every, label):
    if every and (i + 1) % every == 0:
        rate = (i + 1) / (time.time() - t0)
        print("    %s %d/%d  %.1f evals/s  eta %.1f min"
              % (label, i + 1, total, rate, (total - i - 1) / rate / 60.0), flush=True)


def run_wo_single_head(model, evaluate, damage_fn, fault_grid: list[dict],
                       repeats: int, seed: int, model_name: str,
                       progress_every: int = 500) -> pd.DataFrame:
    ev = safe_evaluate(evaluate)
    reset_head_masks(model)
    clean = ev()
    rng = np.random.default_rng(seed)
    heads = get_all_heads(model)
    jobs = [(l, h, g, r) for l, h in heads for g in fault_grid
            for r in range(repeats if g["stochastic"] else 1)]
    rows, t0 = [], time.time()
    for i, (l, h, g, r) in enumerate(jobs):
        with wo_fault(model, [(l, h, g["spec"])], rng) as recs:
            fault = ev()
        row = {"model": model_name, "layer": l, "head": h, "head_name": head_name(l, h),
               "fault_model": g["fault_model"], "level": g["level"], "repeat": r}
        row.update(summarize_records(recs))
        row["nonfinite_rate"] = fault["nonfinite_rate"]
        row.update(damage_fn(clean, fault))
        rows.append(row)
        _progress(i, len(jobs), t0, progress_every, "single-head")
    return pd.DataFrame(rows)


def wo_criticality(df: pd.DataFrame, crit_col: str) -> pd.DataFrame:
    """Per head x fault level: mean damage over repeats (the W_o criticality)."""
    keys = ["fault_model", "level", "layer", "head", "head_name"]
    out = df.groupby(keys, sort=False).agg(
        criticality=(crit_col, lambda s: float(np.mean(np.maximum(s, 0.0)))),
        criticality_q95=(crit_col, lambda s: float(np.quantile(s, 0.95))),
        criticality_max=(crit_col, "max"),
        D_mean=("D_mean", "mean"),
        catastrophic_rate=("nonfinite_rate", lambda s: float(np.mean(s > 0))),
        mean_rel_change=("rel_change", lambda s: float(np.mean(s[np.isfinite(s)]))
                         if np.isfinite(s).any() else float("inf")),
        mean_bitflips=("n_bitflips", "mean"),
        n_trials=(crit_col, "size"),
    ).reset_index()
    return out


def compare_with_crash(crit_df: pd.DataFrame, crash_df: pd.DataFrame) -> pd.DataFrame:
    """Does the W_o fault rank heads like the crash does?"""
    from scipy.stats import spearmanr

    crash = crash_df.set_index("head_name")["criticality"]
    rows = []
    for (fm, lvl), g in crit_df.groupby(["fault_model", "level"], sort=False):
        c = crash.loc[g["head_name"]].to_numpy()
        w = g["criticality"].to_numpy()
        rho = spearmanr(c, w).statistic if np.std(w) > 0 and np.std(c) > 0 else np.nan
        top_crash = set(crash.nlargest(3).index)
        top_wo = set(g.nlargest(3, "criticality")["head_name"])
        rows.append({
            "fault_model": fm, "level": lvl,
            "mean_criticality": float(np.mean(w)),
            "max_criticality": float(np.max(w)),
            "most_critical_head": g.loc[g["criticality"].idxmax(), "head_name"],
            "ratio_to_crash_mean": float(np.mean(w) / (crash.mean() + EPS)),
            "spearman_vs_crash": float(rho),
            "top3_overlap_with_crash": len(top_crash & top_wo),
            "catastrophic_rate": float(g["catastrophic_rate"].mean()),
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# E2: random multi-head W_o faults (same head sets as the crash study)
# --------------------------------------------------------------------------
def run_wo_multihead(model, evaluate, damage_fn, spec: dict, setting: str,
                     weight_lookup: dict, critical_heads, model_name: str,
                     k_trials: int, seed: int, fault_seed: int, num_layers: int,
                     progress_every: int = 500) -> pd.DataFrame:
    ev = safe_evaluate(evaluate)
    all_heads = get_all_heads(model)
    n_heads = len(all_heads)
    critical_set = {tuple(h) for h in critical_heads}
    rng = np.random.default_rng(fault_seed)
    reset_head_masks(model)
    clean = ev()

    rows, t0 = [], time.time()
    for trial, failed in enumerate(sample_head_sets(all_heads, k_trials, seed)):
        with wo_fault(model, [(l, h, spec) for l, h in failed], rng) as recs:
            fault = ev()
        weights = np.array([weight_lookup[h] for h in failed], dtype=float)
        layer_counts = get_layer_fault_counts(failed, num_layers)
        row = {
            "model": model_name, "setting": setting, "run_id": trial,
            "N_fail": len(failed), "fail_fraction": len(failed) / n_heads,
            "L_fault": get_number_of_affected_layers(layer_counts),
            "N_crit_fail": int(sum(1 for h in failed if h in critical_set)),
            "W_fail": float(weights.sum()), "W_fail_mean": float(weights.mean()),
            "failed_heads": format_failed_heads(failed),
        }
        for li, c in enumerate(layer_counts):
            row["n_fail_layer_%d" % li] = int(c)
        row.update(summarize_records(recs))
        row["nonfinite_rate"] = fault["nonfinite_rate"]
        row.update(damage_fn(clean, fault))
        rows.append(row)
        _progress(trial, k_trials, t0, progress_every, setting)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# deep-dive on the most critical heads
# --------------------------------------------------------------------------
def select_deepdive_heads(single_head_df: pd.DataFrame, n_top: int) -> list[tuple]:
    """Top-n heads by crash criticality plus the least critical one."""
    top = single_head_df.nlargest(n_top, "criticality")
    heads = [(int(r.layer), int(r.head), "top%d" % (i + 1))
             for i, r in enumerate(top.itertuples())]
    low = single_head_df.loc[single_head_df["criticality"].idxmin()]
    heads.append((int(low["layer"]), int(low["head"]), "least"))
    return heads


def _deep_row(model_name, l, h, role, recs, fault, clean, damage_fn, **extra) -> dict:
    row = {"model": model_name, "layer": l, "head": h, "head_name": head_name(l, h),
           "role": role}
    row.update(extra)
    rec = recs[0]
    for k in ("old_value", "new_value"):
        if k in rec:
            row[k] = rec[k]
    row.update(summarize_records(recs))
    row["nonfinite_rate"] = fault["nonfinite_rate"]
    row.update(damage_fn(clean, fault))
    return row


def run_value_sweep(model, evaluate, damage_fn, heads, value_grid, n_single: int,
                    n_columns: int, n_rows: int, seed: int, model_name: str,
                    progress_every: int = 500) -> pd.DataFrame:
    """D1: set weights to each value in value_grid at four granularities.

    single weight, whole row (one output feature), whole column (one head
    output dimension) and the whole block.  The same random targets are reused
    for every value of a head, so the values are compared on paired targets.
    """
    ev = safe_evaluate(evaluate)
    reset_head_masks(model)
    clean = ev()
    rng = np.random.default_rng(seed)
    jobs = []
    for l, h, role in heads:
        shape = tuple(wo_block(model, l, h).shape)
        n = shape[0] * shape[1]
        targets = [("single", int(i), np.array([i]))
                   for i in rng.choice(n, size=min(n_single, n), replace=False)]
        targets += [("row", int(r), row_indices(shape, r))
                    for r in rng.choice(shape[0], size=min(n_rows, shape[0]),
                                        replace=False)]
        targets += [("column", int(c), column_indices(shape, c))
                    for c in rng.choice(shape[1], size=min(n_columns, shape[1]),
                                        replace=False)]
        targets.append(("block", -1, np.arange(n)))
        for label, v, mode in value_grid:
            for gran, tid, idx in targets:
                jobs.append((l, h, role, label, v, mode, gran, tid, idx))

    rows, t0 = [], time.time()
    for i, (l, h, role, label, v, mode, gran, tid, idx) in enumerate(jobs):
        spec = {"kind": "set_value", "idx": idx,
                ("value_rel" if mode == "rel" else "value"): float(v)}
        with wo_fault(model, [(l, h, spec)], rng) as recs:
            fault = ev()
        rows.append(_deep_row(model_name, l, h, role, recs, fault, clean, damage_fn,
                              value_label=label, value=recs[0]["value"],
                              granularity=gran, target=tid))
        _progress(i, len(jobs), t0, progress_every, "value sweep")
    return pd.DataFrame(rows)


MAGNITUDE_OPS = (
    ("zero", {"kind": "set_value", "value": 0.0}),
    ("negate", {"kind": "scale", "factor": -1.0}),
    ("x10", {"kind": "scale", "factor": 10.0}),
    ("flip_exp_msb", {"kind": "flip_bit", "bit": 30}),
)


def magnitude_bins(block: torch.Tensor) -> list[tuple[str, np.ndarray]]:
    """Deciles of |w| plus the top 1% and top 0.1% (by rank, so bins are equal-sized)."""
    absw = block.detach().abs().reshape(-1).double().cpu().numpy()
    order = np.argsort(absw, kind="stable")
    n = len(order)
    bins = [("d%02d" % (d + 1), order[d * n // 10:(d + 1) * n // 10]) for d in range(10)]
    bins.append(("top1%", order[int(np.floor(0.99 * n)):]))
    bins.append(("top0.1%", order[int(np.floor(0.999 * n)):]))
    return bins


def run_magnitude_study(model, evaluate, damage_fn, heads, n_samples: int,
                        seed: int, model_name: str, ops=MAGNITUDE_OPS,
                        progress_every: int = 500) -> pd.DataFrame:
    """D2: does damage depend on the original |w| of the weight that is changed?

    Per |w| bin: n_samples single weights x each op, plus zeroing the whole bin.
    """
    ev = safe_evaluate(evaluate)
    reset_head_masks(model)
    clean = ev()
    rng = np.random.default_rng(seed)
    jobs = []
    for l, h, role in heads:
        blk = wo_block(model, l, h)
        flat = blk.detach().reshape(-1).cpu().numpy()
        for b_label, b_idx in magnitude_bins(blk):
            lo, hi = float(np.abs(flat[b_idx]).min()), float(np.abs(flat[b_idx]).max())
            picks = rng.choice(b_idx, size=min(n_samples, len(b_idx)), replace=False)
            for i in picks:
                for op, base in ops:
                    jobs.append((l, h, role, b_label, lo, hi, "single", op,
                                 dict(base, idx=np.array([i]))))
            jobs.append((l, h, role, b_label, lo, hi, "whole_bin", "zero",
                         {"kind": "set_value", "value": 0.0, "idx": b_idx}))

    rows, t0 = [], time.time()
    for i, (l, h, role, b_label, lo, hi, gran, op, spec) in enumerate(jobs):
        with wo_fault(model, [(l, h, spec)], rng) as recs:
            fault = ev()
        rows.append(_deep_row(model_name, l, h, role, recs, fault, clean, damage_fn,
                              magnitude_bin=b_label, bin_abs_min=lo, bin_abs_max=hi,
                              granularity=gran, op=op, n_in_bin=int(np.size(spec["idx"]))))
        _progress(i, len(jobs), t0, progress_every, "magnitude")
    return pd.DataFrame(rows)


def run_bit_position_study(model, evaluate, damage_fn, heads, n_samples: int,
                           seed: int, model_name: str,
                           progress_every: int = 500) -> pd.DataFrame:
    """D3: flip bit b (0..31) of single weights; the same weights for every bit."""
    ev = safe_evaluate(evaluate)
    reset_head_masks(model)
    clean = ev()
    rng = np.random.default_rng(seed)
    jobs = []
    for l, h, role in heads:
        n = wo_block(model, l, h).numel()
        picks = rng.choice(n, size=min(n_samples, n), replace=False)
        for bit in range(FLOAT_BITS):
            for i in picks:
                jobs.append((l, h, role, bit, int(i)))

    rows, t0 = [], time.time()
    for j, (l, h, role, bit, i) in enumerate(jobs):
        spec = {"kind": "flip_bit", "idx": np.array([i]), "bit": bit}
        with wo_fault(model, [(l, h, spec)], rng) as recs:
            fault = ev()
        field = "sign" if bit == SIGN_BIT else ("exponent" if bit in EXPONENT_BITS
                                                else "mantissa")
        rows.append(_deep_row(model_name, l, h, role, recs, fault, clean, damage_fn,
                              bit=bit, bit_field=field, weight_index=i))
        _progress(j, len(jobs), t0, progress_every, "bit position")
    return pd.DataFrame(rows)


def pooled_delta_table(frames: dict, crit_col: str) -> pd.DataFrame:
    """D4: every deep-dive trial as (study, head, |dW|, damage)."""
    parts = []
    for study, df in frames.items():
        cols = ["head_name", "role", "delta_norm", "rel_change", "n_changed",
                "nonfinite_rate", crit_col, "D_mean"]
        p = df[cols].copy()
        p.insert(0, "study", study)
        parts.append(p)
    return pd.concat(parts, ignore_index=True)


def binned_damage_by_delta(pooled: pd.DataFrame, crit_col: str,
                           n_bins: int = 16) -> pd.DataFrame:
    """Median / q90 damage by log10 |dW| bin, per head role (finite changes only)."""
    work = pooled[np.isfinite(pooled["delta_norm"]) & (pooled["delta_norm"] > 0)].copy()
    work["log10_delta"] = np.log10(work["delta_norm"])
    edges = np.linspace(work["log10_delta"].min(), work["log10_delta"].max() + 1e-9,
                        n_bins + 1)
    work["delta_bin"] = pd.cut(work["log10_delta"], edges, include_lowest=True)
    out = work.groupby(["role", "delta_bin"], observed=True).agg(
        log10_delta_mid=("log10_delta", "median"),
        damage_median=(crit_col, "median"),
        damage_q90=(crit_col, lambda s: float(np.quantile(s, 0.9))),
        catastrophic_rate=("nonfinite_rate", lambda s: float(np.mean(s > 0))),
        n=(crit_col, "size"),
    ).reset_index()
    out["delta_bin"] = out["delta_bin"].astype(str)
    return out
