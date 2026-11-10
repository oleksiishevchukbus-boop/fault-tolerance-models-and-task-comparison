# Attention-head fault tolerance across models and tasks

How well does a transformer keep working when some of its attention heads
crash? This repository applies one fault-injection protocol to **three
architecture families** and **three modalities**, giving 39 trained models:

| | AG News (text, 4 classes) | CIFAR-10 (image, 10 classes) | PJM hourly load (time series, 11 regions) |
|---|---|---|---|
| `base_transformer`: encoder, 4 x 8 = 32 heads | 1 model | 1 model | 11 models |
| `tiny_vit`: Tiny ViT, 6 x 6 = 36 heads | 1 model | 1 model | 11 models |
| `tiny_gpt2`: Tiny GPT-2 (causal), 6 x 6 = 36 heads | 1 model | 1 model | 11 models |

A crashed head outputs zeros. Faults are injected at inference only, into a
model that was trained clean. For each model the notebooks measure:

- the **criticality** of every single head;
- the degradation under **10,000 random multi-head failures**;
- how strongly the fault severity correlates with the degradation;
- the empirical **95% safe regions**: how many heads, and how many critical
  heads, may fail while the damage stays under a threshold.

## Where to start

| I want to... | Go to |
|---|---|
| read the findings for one experiment | `reports/<family>/<dataset>_report_EN.docx` |
| look at raw numbers or plots | `results/<family>/<dataset>/`: `summary.json`, `tables/*.csv`, `figures/*.png` |
| see the code and its stored outputs | `notebooks/<family>/<dataset>.ipynb` |
| understand the design and protocol | [docs/study_design.md](docs/study_design.md) |
| understand how the three architectures differ | [docs/architectures.md](docs/architectures.md) |
| know what a column or metric means | [docs/metrics.md](docs/metrics.md) |
| set up and re-run the experiments | [docs/setup.md](docs/setup.md), then [docs/running.md](docs/running.md) |

You don't need to run anything to read the results. The notebooks ship with
their outputs stored.

> **Datasets are not included.** `data/` ships as an empty folder skeleton
> (`data/ag_news/`, `data/pjm/`, each with a placeholder note). To **re-run**
> any notebook you must first download AG News, CIFAR-10 and the PJM hourly-load
> CSVs into it, as described in [docs/setup.md §2](docs/setup.md#2-datasets).
> Without them every notebook stops at its first cell with `FileNotFoundError`.
> Reading `results/`, `reports/` and the stored notebook outputs needs no data.

## Results at a glance

All models were run at the full protocol (K = 10,000 fault configurations,
1,000-sample evaluation subset). "Critical layer" is the layer with the
highest mean head criticality. For PJM it is the count over the 11 regions.

| Family | Dataset | Clean performance | Heads | Most critical head | Critical layer | Heads above half of max criticality |
|---|---|---|---|---|---|---|
| `base_transformer` | ag_news | accuracy 0.907 | 32 | `L0_H4` | L1 | 28 |
| `base_transformer` | cifar10 | accuracy 0.810 | 32 | `L1_H7` | L1 | 6 |
| `base_transformer` | pjm_regions | mean MAPE 1.46% | 32 | per region | L3 (6), L0 (5) | median 3 |
| `tiny_vit` | ag_news | accuracy 0.898 | 36 | `L0_H5` | L0 | 9 |
| `tiny_vit` | cifar10 | accuracy 0.735 | 36 | `L0_H2` | L0 | 4 |
| `tiny_vit` | pjm_regions | mean MAPE 1.49% | 36 | per region | L0 (8), L1 (3) | median 2 |
| `tiny_gpt2` | ag_news | accuracy 0.908 | 36 | `L0_H4` | L0 | 10 |
| `tiny_gpt2` | cifar10 | accuracy 0.749 | 36 | `L0_H4` | L0 | 4 |
| `tiny_gpt2` | pjm_regions | mean MAPE 1.44% | 36 | per region | L0 (5), L1 (4), L3 (2) | median 2 |

The reports interpret these numbers: degradation bands, safe regions and the
verdict on each hypothesis.

## Repository layout

```
README.md                  this file
pyproject.toml             makes src/faulttol installable (pip install -e .)
requirements.txt           pinned environment
docs/
  setup.md                 environment, datasets, troubleshooting
  running.md               running notebooks, parameters, cost, regenerating reports
  study_design.md          model, fault protocol, compute budget
  architectures.md         base_transformer vs tiny_vit vs tiny_gpt2
  metrics.md               every metric: meaning, formula, computing function, caveats
src/faulttol/              shared library, one implementation for every experiment
  config.py                architectures, seeds, fault protocol, thresholds
  model.py                 encoder + head-crash mask
  injection.py             reset / single / multi-head crash helpers
  metrics.py               clean metrics, probability damage, error change
  single_head.py           criticality C, redundancy R, severity weight W
  multihead.py             K random fault configurations, degradation grid
  saferegion.py            empirical 95% fault-tolerance boundaries
  plots.py                 vulnerability maps, degradation grids
  training.py              training loops, early stopping, checkpoints
  runtime.py               device / thread selection
  copula.py                unused here, kept from the source study
notebooks/<family>/        ag_news.ipynb, cifar10.ipynb, pjm_regions.ipynb
results/<family>/<dataset>/
  summary.json             headline numbers
  tables/*.csv             all result tables
  figures/*.png            criticality profile, vulnerability map, degradation grid, required tau
  (pjm_regions: all_regions_summary.json + one such folder per region)
reports/<family>/          <dataset>_report_EN.docx
scripts/
  run_family.sh            run one family's notebooks (or all) headless
  make_experiment_report.py   rebuild the .docx reports from results/
  docx_helpers.py          Word formatting helpers used by the report script
  derive_notebooks.py      regenerate tiny_vit / tiny_gpt2 notebooks from base_transformer
data/                      EMPTY skeleton - you download the datasets into it (see data/README.md)
  ag_news/                 train.csv, test.csv
  pjm/                     11 x <REGION>_hourly.csv
  cifar-10-python.tar.gz   (at data/ root, not unpacked)
```

## Quickstart (re-running)

```bash
python -m venv .venv && source .venv/bin/activate       # Windows: .venv\Scripts\activate
pip install torch==2.14.0                               # or the CUDA wheel, see docs/setup.md
pip install -r requirements.txt && pip install -e .
python -m ipykernel install --user --name ft-python
# download the datasets into data/ (see docs/setup.md, section 2), then:
FT_K_TRIALS=10000 FT_N_FAULT=1000 bash scripts/run_family.sh base_transformer
python scripts/make_experiment_report.py --family base_transformer
```

## Not included

- **Trained checkpoints** (`results/**/models/best.pt`, 39 files, ~4.2 GB):
  re-running a notebook retrains its model.
- **Datasets**: download them as described in [docs/setup.md](docs/setup.md#2-datasets).
- **Cross-dataset / cross-family comparison** documents and the source study's
  paper and notebooks. They belong to the wider study.
