# Running the experiments

You don't have to run anything to read the results. The notebooks ship with
their outputs stored, and every number is also in `results/` and `reports/`.
Run the notebooks only if you want to reproduce the results or change them.

Do the one-time setup in [setup.md](setup.md) first: the environment, the
`ft-python` Jupyter kernel and the datasets.

> **Running a notebook overwrites the shipped results** for that experiment:
> `results/<family>/<dataset>/summary.json`, `tables/*.csv` and `figures/*.png`.
> It also writes a fresh checkpoint to `results/<family>/<dataset>/models/best.pt`.
> Copy the folder first if you want to keep the shipped numbers next to yours.

## One notebook

Open `notebooks/<family>/<dataset>.ipynb` in Jupyter or VS Code with the
`ft-python` kernel and run it top to bottom. To run it headless instead
(the outputs are stored back into the notebook):

```bash
python -m nbconvert --to notebook --execute --inplace \
  --ExecutePreprocessor.timeout=-1 \
  --ExecutePreprocessor.kernel_name=ft-python \
  notebooks/base_transformer/ag_news.ipynb
```

To run only some of the PJM regions:

```bash
FT_REGIONS=PJME,COMED python -m nbconvert --to notebook --execute --inplace \
  --ExecutePreprocessor.timeout=-1 \
  --ExecutePreprocessor.kernel_name=ft-python \
  notebooks/base_transformer/pjm_regions.ipynb
```

## A whole family

```bash
bash scripts/run_family.sh base_transformer      # or tiny_vit, tiny_gpt2
bash scripts/run_family.sh all                   # all three families, one after another
```

The script runs the family's three notebooks in place, one after another. It
writes logs to `logs/<family>/`. By default it uses a **reduced fault budget**
that fits a CPU-only run (`FT_K_TRIALS=4000`, `FT_N_FAULT=250`), along with the
per-dataset training budgets set inside the script.

**The shipped results use the full protocol.** To reproduce them:

```bash
FT_K_TRIALS=10000 FT_N_FAULT=1000 bash scripts/run_family.sh base_transformer
```

On Windows, run the script from Git Bash or WSL, or run the notebooks one at a
time as shown above. To use a specific interpreter, set `PYTHON=.venv/Scripts/python`.

## Parameters (environment variables)

All protocol parameters are read from environment variables, so you can change
a run's budget without editing any cell. Every value actually used is recorded
in the run's `summary.json`.

| Variable | Meaning | Default |
|---|---|---|
| `FT_K_TRIALS` | number of random multi-head fault configurations | 10000 (`run_family.sh`: 4000) |
| `FT_N_FAULT` | size of the fixed evaluation subset | 1000 (`run_family.sh`: 250) |
| `FT_EPOCHS`, `FT_PATIENCE` | training budget | set per notebook |
| `FT_TRAIN_SUBSAMPLE` | caps the training set (AG News, CIFAR-10) | full set |
| `FT_MAX_LEN` | sequence length (AG News) | 64 (`run_family.sh`: 48) |
| `FT_WINDOWS`, `FT_REGIONS` | number of sampled windows / which regions to run (PJM) | all |
| `FT_DEVICE` | `cuda`, `cuda:N` or `cpu` | CUDA if available |
| `FT_TF32` | `1` allows TF32 on CUDA: faster, but not fp32-comparable | off |
| `FT_THREADS` | torch thread count (CPU runs) | torch default |

If you set `FT_DEVICE` to CUDA and CUDA isn't available, the notebook raises an
error instead of silently falling back to CPU. TF32 stays off by default so the
damage numbers remain comparable with CPU results.

## Cost

Each run costs `training + (n_heads + K) x per-evaluation`. The figures below
are for the full protocol (K=10000, N_fault=1000), measured on an RTX 4070
Laptop with TF32 off:

| Experiment | `base_transformer` | `tiny_vit` / `tiny_gpt2` |
|---|---|---|
| AG News | ~25 min | ~55 min |
| CIFAR-10 | ~26 min | ~1 h |
| PJM (per region, x11) | ~9 min | ~20 min |

Peak GPU memory stays under 1 GB. CPU-only runs are much slower, which is why
`run_family.sh` defaults to the reduced budget. For a breakdown, see
[study_design.md](study_design.md#compute-budget) and
[architectures.md](architectures.md).

## Regenerating the reports

`scripts/make_experiment_report.py` reads only `results/<family>/<dataset>/`
(`summary.json`, `tables/*.csv`, `figures/*.png`) and writes
`reports/<family>/<dataset>_report_EN.docx`. It needs no GPU, and you can re-run
it any time after the notebooks:

```bash
python scripts/make_experiment_report.py                        # base_transformer, all three datasets
python scripts/make_experiment_report.py --family tiny_vit
python scripts/make_experiment_report.py --family tiny_gpt2
python scripts/make_experiment_report.py cifar10                # one dataset only
python scripts/make_experiment_report.py --lang en,ua           # also write the Ukrainian version
```

The analysis text is derived from the measured numbers, not fixed prose. For
example, if a model's criticality comes out near-uniform, the report says it
*contradicts* the concentration hypothesis. A regenerated report therefore
follows your numbers.

## Regenerating the derived notebooks

The `tiny_vit` and `tiny_gpt2` notebooks are generated from the
`base_transformer` ones. Only the results directory, the model name and the
architecture config are substituted. If you change a base notebook, propagate
the change with:

```bash
python scripts/derive_notebooks.py                     # both families
python scripts/derive_notebooks.py --family tiny_vit   # one family
```

This **clears the stored outputs** of the derived notebooks, so re-run them
afterwards. The script stops with an error if an expected substitution is
missing, so a changed base notebook can't silently produce a derived notebook
that still points at `base_transformer`.
