# data/

Empty on purpose — the datasets are downloaded, not shipped. **You must download
them before re-running any notebook**; otherwise the notebooks fail with
`FileNotFoundError`. (Reading the stored results needs no data.)

The `ag_news/` and `pjm/` folders are kept with a `PUT_*_HERE.md` placeholder
so the expected layout is visible. Fill this folder as follows (full download links, formats and checksums are in
`../docs/setup.md`, section 2):

```
data/
  ag_news/
    train.csv                 AG News, no header, 120 000 rows
    test.csv                  AG News, no header,   7 600 rows
  pjm/
    PJME_hourly.csv           PJM hourly load, 11 regions
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
  cifar-10-python.tar.gz      left as an archive, md5 c58f30108f718f92721af3b95e74349a
```

The notebooks find the repository root (the folder with `pyproject.toml`) and
read `<root>/data/...`, so keep this folder where it is.
