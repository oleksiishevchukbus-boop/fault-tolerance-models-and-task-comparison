# Study design

New experiments extending *Attention-Head Fault Tolerance in Transformers: A
Copula-Based Study* (the source study's paper and its notebooks
`transformers_EXP_A.ipynb` / `transformers_EXP_B.ipynb` are not part of this
repository) from one synthetic classification task plus one energy series to
**three real datasets covering text, images and numeric time series**, and
then to **three architecture families**.

| Family | Notebooks | Architecture |
|---|---|---|
| `base_transformer` | `notebooks/base_transformer/` | Base encoder below, 4 x 8 = 32 heads |
| `tiny_vit` | `notebooks/tiny_vit/` | Tiny ViT, 6 x 6 = 36 heads - see [architectures.md](architectures.md) |
| `tiny_gpt2` | `notebooks/tiny_gpt2/` | Tiny GPT-2 (causal), 6 x 6 = 36 heads - see [architectures.md](architectures.md) |

Every notebook writes `results/<family>/<dataset>/` (`summary.json`,
`tables/*.csv`, `figures/*.png`; one sub-folder per region for PJM) and a
checkpoint at `results/<family>/<dataset>/models/best.pt` (not committed).

## The model

The `base_transformer` family uses the **Base** configuration of the source
notebooks:

| d_model | encoder layers | heads/layer | total heads | d_ff | dropout |
|---|---|---|---|---|---|
| 256 | 4 | 8 | **32** | 1024 | 0.05 |

This is deliberate. Both source papers vary architecture *and* task at the same
time, so their finding that critical heads sit in layers 0-1 for classification
but layer 3 for forecasting cannot be attributed to either. Holding the
architecture fixed makes the layer-position question answerable.

Only the input adapter and task head change per modality:

| Experiment | Input | Tokens | Head | Activation |
|---|---|---|---|---|
| ag_news | `nn.Embedding` (word vocab) | 48 (+CLS) | `Linear(256, 4)` | ReLU (as EXP_A) |
| cifar10 | `Conv2d` 4x4 patch projection | 64 (+CLS) | `Linear(256, 10)` | ReLU |
| pjm_regions | `Linear(1, 256)` per hour | 20 (+CLS) | `Linear(256, 1)` | SiLU (as EXP_B) |

## The protocol (unchanged from the notebooks)

1. **Crash model** (Definition 1): a head's output tensor is zeroed *after*
   `softmax(QK'/sqrt(d_k))V` and *before* concatenation and `W_o`. Faults are
   injected at inference only; the model is trained clean and frozen.
2. **Single-head injection** over all 32 heads gives criticality
   `C = mean(max(damage, 0))`, redundancy `R = 1 - C/max(C)` and the severity
   weight `W = C / mean(C)`.
3. **Random multi-head injection**: `N_fail ~ Uniform{1..32}`, heads chosen
   without replacement, all crashed simultaneously, evaluated on one fixed
   subset. Records `N_fail`, `L_fault`, `N_crit_fail`, `W_fail`, `W_fail_mean`.
4. **Fault severity vs degradation**: rank correlations (Pearson, Spearman,
   Kendall) between each severity descriptor and the degradation, plus the
   `W_fail`/`N_fail` confound. No copula modelling - the original study's
   copula stage is deliberately out of scope here.
5. **Safe regions**: widest `(n, c)` with
   `P(Y <= tau | N_fail <= n, N_crit_fail <= c) >= 0.95`, at least 100
   configurations per boundary, plus `tau_required = Q_0.95(Y)`.

Degradation variables follow the source notebooks: probability damage
`D = p_clean(true) - p_fault(true)` for classification (`D_mean` for the severity
analysis, `D_pos_mean` for criticality and the safe regions) and `delta MAE` / `delta MAPE` for
regression.

## Implementation differences from the notebooks

These are deliberate and do not change results:

- **No copula analysis.** The source study fits Archimedean copulas to the
  fault-severity / degradation dependence; that stage is out of scope here.
  Dependence is reported with plain rank correlations instead, and the focus is
  the cross-dataset comparison of model results. (`src/faulttol/copula.py` is left
  in the tree, unused, in case it is wanted again.)
- **Attention maps are not cached during fault trials.** The notebooks store
  `last_attn`/`last_head_output` on every forward pass; here caching is enabled
  only while attention statistics are collected.
- **Local paths** replace the Colab Drive mount.

## Compute budget

The notebooks select their device automatically: an NVIDIA GPU when CUDA is
available, CPU otherwise. `FT_DEVICE` overrides the choice - `FT_DEVICE=cpu`
forces the CPU path, `FT_DEVICE=cuda:1` pins one GPU. An explicit CUDA request
that cannot be satisfied raises rather than falling back silently. TF32 is left
**off** on CUDA so matmuls and the CIFAR patch convolution stay full fp32 and
the damage numbers remain comparable with the CPU results; `FT_TF32=1` trades
that comparability for speed.

All 39 models whose results are in `results/` (3 families x (AG News +
CIFAR-10 + 11 PJM regions)) were run at the **full protocol**, as recorded in
every `summary.json` (`k_trials`, `n_fault_eval_samples`):

| Setting | Full protocol (notebook default, shipped results) | Reduced (`scripts/run_family.sh` default, for CPU) | Env var |
|---|---|---|---|
| Random fault configurations K | 10000 | 4000 | `FT_K_TRIALS` |
| Fixed evaluation subset | 1000 | 250 | `FT_N_FAULT` |

Measured on an RTX 4070 Laptop (torch 2.14.0+cu126, TF32 off, `N_FAULT=1000`):
0.100 s per evaluation for AG News and CIFAR-10, 0.034 s for PJM, and 34 / 13 /
4 s per training epoch respectively. Since the run costs
`training + (32 + K) x per-evaluation`, the full protocol is roughly 25 min for
AG News, 26 min for CIFAR-10 and 9 min per PJM region. Peak GPU memory is
under 1 GB.

Reducing `N_FAULT` rather than `K` is the right trade: `K` drives the
safe-region coverage counts and the severity/degradation sample size, while
`N_FAULT` only adds estimation noise to each configuration's degradation value.

Code defaults are the paper's full values, so running the notebooks unchanged on
a GPU box reproduces the original fault-injection protocol.

## Running, metrics and reports

- How to set up the environment and datasets: [setup.md](setup.md).
- How to run the notebooks, the environment variables and the reports: [running.md](running.md).
- Every metric, its formula and the function computing it: [metrics.md](metrics.md).

## Data

| Dataset | File | Source |
|---|---|---|
| AG News | `data/ag_news/{train,test}.csv` | 120000 train / 7600 test, 4 classes |
| CIFAR-10 | `data/cifar-10-python.tar.gz` | md5 `c58f30108f718f92721af3b95e74349a` |
| PJM hourly load | `data/pjm/*_hourly.csv` | 11 regions used; AEP (the paper's region, 121273 rows) is excluded |
