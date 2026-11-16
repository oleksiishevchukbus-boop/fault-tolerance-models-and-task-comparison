"""Shared configuration for the attention-head fault-tolerance experiments.

The architecture is deliberately held FIXED across all datasets (the "Base"
configuration of Paper B / transformers_EXP_A / transformers_EXP_B) so that
layer-position findings are comparable across modalities.  Only the input
embedding, the task head and the activation follow the source notebook for the
corresponding task family.
"""
from __future__ import annotations

import os


def _env_int(name, default):
    """Allow smoke runs / reduced budgets without editing the file."""
    try:
        return int(os.environ[name])
    except (KeyError, ValueError):
        return default

# Base architecture: 4 encoder layers x 8 heads = 32 attention heads.
BASE_PARAM = {
    "d_model": 256,
    "num_heads": 8,
    "num_layers": 4,
    "d_ff": 1024,
    "dropout": 0.05,
}

MODEL_NAME = "base_transformer"

# Second architecture, used by the tiny_vit family: the Tiny ViT of
# "Dropout and the Outliers" (Hezbri et al.), which is 6 blocks x 6 heads =
# 36 attention heads at width 384 with an MLP expansion factor of 4.  Relative
# to BASE_PARAM this changes depth, width, head count, head width (d_k 64 not
# 32), normalisation placement, positional encoding and the q/k/v biases, so it
# is a bundle of choices rather than a single-factor contrast - see
# docs/architectures.md.
TINY_VIT_PARAM = {
    "d_model": 384,
    "num_heads": 6,
    "num_layers": 6,
    "d_ff": 1536,
    "dropout": 0.05,
    "norm_first": True,     # ViT places the norm inside the residual branch
    "qkv_bias": False,      # ViT: biases everywhere except q, k and v
    "learned_pe": True,     # ViT: learnable absolute positional embedding
}

# Third architecture, used by the tiny_gpt2 family: the Tiny GPT-2
# of the same paper (Table 4), which has the SAME 6 blocks x 6 heads at width
# 384 with MLP expansion 4 as the Tiny ViT above - the paper deliberately gives
# all three of its models one shape.  So relative to TINY_VIT_PARAM only two
# things change, and the first is nearly free in parameter terms:
#   bias=False    GPT-2 has no biases at all, where ViT keeps them everywhere
#                 except q/k/v.  This removes 18,432 of 10,639,872 stack
#                 parameters (0.17%).
#   causal=True   GPT-2 is decoder-only: every query attends to its own
#                 position and to the left only.  This is the substantive
#                 contrast - bidirectional versus autoregressive attention -
#                 and it is what makes this family worth running.
# cls_last follows from causal: see _EncoderBackbone in model.py.
TINY_GPT2_PARAM = {
    "d_model": 384,
    "num_heads": 6,
    "num_layers": 6,
    "d_ff": 1536,
    "dropout": 0.05,
    "norm_first": True,     # GPT-2: pre-layer norms
    "qkv_bias": False,      # GPT-2: no biases anywhere ...
    "bias": False,          # ... including W_o, the FFN and the LayerNorms
    "learned_pe": True,     # GPT-2: learnable absolute positional embedding
    "causal": True,         # GPT-2: decoder-only, left-to-right attention
    "cls_last": True,       # read-out token must sit where it can see the input
}

# Seeds, copied from transformers_EXP_A / transformers_EXP_B.
DATA_SEED = 42
SPLIT_SEED = 42
TRAIN_SEED = 42
FAULT_SPLIT_SEED = 2026
MULTIHEAD_FAULT_SEED = 1024
FAULT_ANALYSIS_SEED = 999
BOOTSTRAP_SEED = 2026

# Fault-injection protocol (Paper B, Section 3.3.2).
K_TRIALS = _env_int("FT_K_TRIALS", 10_000)   # random multi-head fault configurations per model
N_FAULT = _env_int("FT_N_FAULT", 1_000)      # fixed evaluation subset size used for every trial
BATCH_SIZE = 128

# Copula analysis.
COPULA_FAMILIES = ("clayton", "gumbel", "frank", "survival_clayton", "survival_gumbel")
CVM_BOOTSTRAP_REPS = _env_int("FT_BOOTSTRAP", 500)

# Safe-region analysis (Paper B, Section 3.6).
CRITICAL_TOP_FRACTION = 0.10
SAFE_PROB = 0.95
MIN_CONFIGS_PER_BOUNDARY = 100
CLS_TAUS = (0.010, 0.025, 0.050)          # absolute D_pos_mean thresholds
REG_ALPHAS = (0.10, 0.25, 0.50)           # fractions of clean MAPE

# W_o weight-fault experiments (docs/wo_faults.md).  A head's crash is
# equivalent to zeroing its d_k columns of the output projection W_o; these
# experiments corrupt those columns in other ways instead.
WO_FAULT_SEED = 4242
WO_BER_LEVELS = (1e-6, 1e-5, 1e-4, 1e-3)        # per-bit flip probability
WO_STUCK_VALUES_REL = (0.0, 1.0, -1.0)          # x max|W_o| of the head's layer
WO_STUCK_FRACS = (0.01, 0.10, 0.50, 1.00)       # fraction of the head's W_o weights stuck
WO_REPEATS = _env_int("FT_WO_REPEATS", 10)      # random repeats per stochastic fault level
WO_K_TRIALS = _env_int("FT_WO_K_TRIALS", 2_000)  # random multi-head trials per W_o setting
WO_MULTIHEAD_SETTINGS = (                       # (label, spec) for the multi-head study
    ("bitflip_ber1e-4", {"kind": "bitflip_ber", "p": 1e-4}),
    ("stuck_+m_frac0.10", {"kind": "stuck_at", "value_rel": 1.0, "frac": 0.10}),
)
WO_SANITY_MULTI = 50                            # multi-head sets in the zero-vs-mask check
# Deep-dive on the most critical heads.  Values marked "rel" are multiples of
# m = max|W_o| of the head's layer; "abs" values are used as they are.
WO_VALUE_GRID = (
    ("0", 0.0, "abs"),
    ("+m/2", 0.5, "rel"), ("-m/2", -0.5, "rel"),
    ("+m", 1.0, "rel"), ("-m", -1.0, "rel"),
    ("+10m", 10.0, "rel"), ("-10m", -10.0, "rel"),
    ("+1e2", 1e2, "abs"), ("-1e2", -1e2, "abs"),
    ("+1e4", 1e4, "abs"), ("-1e4", -1e4, "abs"),
    ("+1e6", 1e6, "abs"), ("-1e6", -1e6, "abs"),
)
WO_SWEEP_SINGLE = _env_int("FT_WO_SWEEP_SINGLE", 50)   # single weights per value
WO_SWEEP_ROWS = _env_int("FT_WO_SWEEP_ROWS", 8)        # whole W_o rows per value
WO_SWEEP_COLUMNS = _env_int("FT_WO_SWEEP_COLUMNS", 8)  # whole W_o columns per value
WO_DEEPDIVE_SAMPLES = _env_int("FT_WO_SAMPLES", 30)    # weights per magnitude bin / bit
WO_DEEPDIVE_N_CRITICAL = 3                      # top-N critical heads (+ the least critical)
