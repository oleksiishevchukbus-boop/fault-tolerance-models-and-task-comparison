# Setup

One-time setup: a Python environment and the three datasets. Then see
[running.md](running.md).

## 1. Environment

Python **3.12** (verified on 3.12.10, Windows 11). `requirements.txt` carries
the pins and repeats these commands.

`torch` must come from the CUDA wheel index — the PyPI Windows wheel is
CPU-only and the notebooks pick up a GPU automatically when one is visible.

```powershell
# Windows
python -m venv .venv
.venv\Scripts\python -m pip install --upgrade pip
.venv\Scripts\python -m pip install torch==2.14.0+cu126 --index-url https://download.pytorch.org/whl/cu126
.venv\Scripts\python -m pip install -r requirements.txt
.venv\Scripts\python -m pip install -e .
.venv\Scripts\python -m ipykernel install --user --name ft-python
```

```bash
# Linux / macOS
python3.12 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install torch==2.14.0+cu126 --index-url https://download.pytorch.org/whl/cu126
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m pip install -e .
.venv/bin/python -m ipykernel install --user --name ft-python
```

CPU-only is fine — drop the `--index-url` line and install plain
`torch==2.14.0`.

`pip install -e .` makes `import faulttol` work from anywhere (scripts, your own
notebooks). The shipped notebooks also add `src/` to `sys.path` themselves, so
they run even without it.

The kernel name `ft-python` matters: `scripts/run_family.sh` passes it to
`nbconvert`.

---

## 2. Datasets

> **Required before running anything.** The repository does not ship any data:
> `data/` contains only the folder skeleton (`data/ag_news/`, `data/pjm/`) and
> placeholder notes. You have to download the three datasets yourself and put
> them into those folders. Which notebooks need which dataset:
>
> | Notebook | Needs |
> |---|---|
> | `notebooks/*/ag_news.ipynb` | `data/ag_news/train.csv`, `test.csv` (§2.1) |
> | `notebooks/*/cifar10.ipynb` | `data/cifar-10-python.tar.gz` (§2.2) |
> | `notebooks/*/pjm_regions.ipynb` | the 11 CSVs in `data/pjm/` (§2.3) |
>
> You only need the datasets for the notebooks you intend to re-run. The
> placeholder `PUT_*_HERE.md` files can stay; the notebooks ignore them.
> Downloaded files are git-ignored, so they will never be committed by accident.

The folders already exist; fill them exactly as shown. The notebooks locate the repository root
by walking up from their own directory until they find `pyproject.toml`, then
read from `<root>/data/...`, so nothing needs configuring.

```
<repo>/
  data/
    ag_news/
      train.csv                 29 470 338 bytes, 120 000 rows
      test.csv                   1 857 427 bytes,   7 600 rows
    pjm/
      PJME_hourly.csv
      COMED_hourly.csv
      DAYTON_hourly.csv
      DOM_hourly.csv
      DUQ_hourly.csv
      PJMW_hourly.csv
      FE_hourly.csv
      NI_hourly.csv
      DEOK_hourly.csv
      EKPC_hourly.csv
      PJM_Load_hourly.csv
    cifar-10-python.tar.gz     170 498 071 bytes
```

### 2.1 AG News

Needed by: `notebooks/*/ag_news.ipynb`.

Download either of:

- fast.ai mirror of the original Zhang et al. release:
  `https://s3.amazonaws.com/fast-ai-nlp/ag_news_csv.tgz` — unpack and take
  `train.csv` and `test.csv` from inside `ag_news_csv/`.
- Kaggle, *AG News Classification Dataset*
  (`amananandrai/ag-news-classification-dataset`) — `train.csv` / `test.csv`
  are at the top level of the archive.

Place them at `data/ag_news/train.csv` and `data/ag_news/test.csv`.

Expected format — **no header row**, three columns, read as
`pd.read_csv(..., header=None)`:

```
"3","Wall St. Bears Claw Back Into the Black (Reuters)","Reuters - Short-sellers, ..."
```

`class (1-4), title, description`. If your download has a header line
(`Class Index,Title,Description`), delete it, or the first record is consumed as
column names and the row counts will be off by one.

### 2.2 CIFAR-10

Needed by: `notebooks/*/cifar10.ipynb`.

Download the **Python pickled version** and leave it as a `.tar.gz` — the
notebook reads it straight out of the archive with `tarfile`, it is *not*
unpacked:

```
https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz
```

Place at `data/cifar-10-python.tar.gz`. Verify:

```powershell
# Windows PowerShell
(Get-FileHash data\cifar-10-python.tar.gz -Algorithm MD5).Hash.ToLower()
# expected: c58f30108f718f92721af3b95e74349a
```

```bash
# Linux / macOS
md5sum data/cifar-10-python.tar.gz
```

`torchvision` is deliberately **not** a dependency; the augmentation is
implemented inside the notebook.

### 2.3 PJM hourly load

Needed by: `notebooks/*/pjm_regions.ipynb`.

Kaggle, *Hourly Energy Consumption* (`robikscube/hourly-energy-consumption`).
Copy the eleven `<REGION>_hourly.csv` files listed above into `data/pjm/`.
`AEP_hourly.csv` ships in the same archive but is **not** used here — AEP is the
source paper's own region and is excluded from these runs. Keeping it costs
nothing.

Expected format — with a header, two columns:

```
Datetime,PJME_MW
2002-12-31 01:00:00,26498.0
```

The megawatt column is named after the region (`COMED_MW`, `DAYTON_MW`, ...);
the notebook takes the second column whatever it is called.

---

## 3. Troubleshooting

| Symptom | Cause |
|---|---|
| `FileNotFoundError` on a `data/...` path | a file is missing or misnamed - check the tree in §2. |
| Row counts off by one in AG News | the CSV has a header row; delete it (`header=None` is used). |
| `No such kernel named ft-python` | run the `ipykernel install --name ft-python` step from §1. |
| `RuntimeError` about CUDA at startup | `FT_DEVICE` asks for CUDA on a box without it — unset it or use `FT_DEVICE=cpu`. |
| Numbers differ slightly from the shipped ones | different `FT_K_TRIALS` / `FT_N_FAULT` (the shipped results use 10000 / 1000), or TF32 enabled. Compare the `summary.json` budgets. |
| `ModuleNotFoundError: No module named 'faulttol'` in your own code | run `pip install -e .` from the repository root. |

