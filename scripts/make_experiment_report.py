"""Build a per-experiment Word report (English; Ukrainian on request).

Writes one self-contained report per experiment:

    reports/<family>/<experiment>_report_EN.docx
    reports/<family>/<experiment>_report_UA.docx     (only with --lang en,ua)

It reads only what the notebook already wrote - results/<family>/<experiment>/
summary.json, tables/*.csv and figures/*.png - so it can be re-run at any time
without touching the GPU.

    python scripts/make_experiment_report.py                        # base_transformer, all three
    python scripts/make_experiment_report.py --family tiny_vit
    python scripts/make_experiment_report.py cifar10                # one experiment
    python scripts/make_experiment_report.py --lang en,ua           # also the Ukrainian version
"""
from __future__ import annotations

import json
import pathlib
import sys

import numpy as np
import pandas as pd
from docx.shared import Inches, Pt
from docx.enum.text import WD_ALIGN_PARAGRAPH

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from docx_helpers import add_bullets, add_numbered, add_table, fmt, new_document

ROOT = pathlib.Path(__file__).resolve().parents[1]

# Which experiment family to document.  Set by main() from --family; the module
# level default keeps the original behaviour for any existing caller.
FAMILY = "base_transformer"
BASE = ROOT / "results" / FAMILY
REPORTS = ROOT / "reports" / FAMILY
FAMILIES = ["base_transformer", "tiny_vit", "tiny_gpt2"]

EXPERIMENTS = ["ag_news", "cifar10", "pjm_regions"]


# Per-family framing.  Everything else in this script is derived from the
# result files, so a new family only needs an entry here.
FAMILY_META = {
    "base_transformer": {
        "en": {"suffix": "", "note": None},
        "ua": {"suffix": "", "note": None},
    },
    "tiny_vit": {
        "en": {
            "suffix": " (Tiny ViT architecture)",
            "note": "This experiment uses the second of the three architectures in this "
                    "repository: the Tiny ViT of “Dropout and the Outliers” (Hezbri "
                    "et al.), which is six blocks of six heads at width 384 with "
                    "pre-normalisation, learnable absolute positional embeddings, no biases "
                    "on the query, key and value projections, and GELU. The companion family "
                    "in notebooks/base_transformer/ runs the identical dataset, training "
                    "recipe, fault protocol and analysis on four blocks of eight heads at "
                    "width 256 with post-normalisation and fixed sinusoidal encodings, so the "
                    "two differ in architecture and in nothing else. Seven architectural "
                    "choices move together, so any difference between the families localises "
                    "to this architecture as a bundle rather than to depth, width or "
                    "normalisation placement individually.",
        },
        "ua": {
            "suffix": " (архітектура Tiny ViT)",
            "note": "У цьому експерименті використано другу з трьох архітектур цього "
                    "репозиторію — Tiny ViT з праці «Dropout and the Outliers» (Hezbri та "
                    "ін.): шість блоків по шість голів шириною 384 з пре-нормалізацією, "
                    "навчуваними абсолютними позиційними вкладеннями, без зсувів на проєкціях "
                    "запиту, ключа та значення, з активацією GELU. Парна сім’я в "
                    "notebooks/base_transformer/ виконує той самий набір даних, той самий "
                    "рецепт навчання, той самий протокол відмов і той самий аналіз на "
                    "чотирьох блоках по вісім голів шириною 256 з пост-нормалізацією та "
                    "фіксованим синусоїдним кодуванням, тому ці дві сім’ї відрізняються "
                    "архітектурою і більше нічим. Сім архітектурних рішень змінюються разом, "
                    "тому будь-яка різниця між сім’ями стосується цієї архітектури як цілого, "
                    "а не глибини, ширини чи розташування нормалізації окремо.",
        },
    },
    "tiny_gpt2": {
        "en": {
            "suffix": " (Tiny GPT-2 architecture)",
            "note": "This experiment uses the third architecture in this repository: the "
                    "Tiny GPT-2 of “Dropout and the Outliers” (Hezbri et al.), which is six "
                    "blocks of six heads at width 384 with pre-normalisation, learnable "
                    "absolute positional embeddings, GELU, no biases anywhere in the stack, "
                    "and — the point of the family — causal rather than bidirectional "
                    "attention, so every query attends to its own position and to the left "
                    "only. Because the source paper gives its Tiny GPT-2 and Tiny ViT one "
                    "shared shape, the companion family in notebooks/tiny_vit/ differs "
                    "from this one in just two respects: the remaining biases (0.17% of the "
                    "stack parameters) and the causal mask. That pair is therefore close to "
                    "a single-factor test of bidirectional versus autoregressive attention "
                    "under head crash faults, and is the cleanest architectural attribution "
                    "available here; the older notebooks/base_transformer/ family, four "
                    "blocks of eight heads at width 256 with post-normalisation, remains the "
                    "bundled contrast. Dataset, splits, seeds, training recipe, fault model "
                    "and analysis are identical across all three families. One consequence "
                    "of the causal mask is local to this family: the read-out CLS token sits "
                    "at the last position rather than the first, because at position 0 it "
                    "could attend only to itself.",
        },
        "ua": {
            "suffix": " (архітектура Tiny GPT-2)",
            "note": "У цьому експерименті використано третю архітектуру цього репозиторію — "
                    "Tiny GPT-2 з праці «Dropout and the Outliers» (Hezbri та ін.): шість "
                    "блоків по шість голів шириною 384 з пре-нормалізацією, навчуваними "
                    "абсолютними позиційними вкладеннями, активацією GELU, без зсувів у "
                    "всьому стеку і — це і є сенс цієї сім’ї — з причинною (causal), а не "
                    "двонапрямленою увагою, тож кожен запит бачить лише власну позицію та "
                    "позиції ліворуч. Оскільки вихідна праця задає своїм Tiny GPT-2 і Tiny "
                    "ViT однакову форму, парна сім’я в notebooks/tiny_vit/ відрізняється "
                    "від цієї лише двома речами: рештою зсувів (0,17% параметрів стеку) та "
                    "причинною маскою. Тому ця пара — практично однофакторна перевірка "
                    "двонапрямленої проти авторегресивної уваги за відмов голів і "
                    "найчистіше архітектурне віднесення, доступне тут; давніша сім’я "
                    "notebooks/base_transformer/ (чотири блоки по вісім голів шириною 256 "
                    "з пост-нормалізацією) лишається «пакетним» контрастом. Набір даних, "
                    "поділи, зерна, рецепт навчання, модель відмови та аналіз однакові в "
                    "усіх трьох сім’ях. Один наслідок причинної маски стосується лише цієї "
                    "сім’ї: зчитувальний токен CLS стоїть в останній, а не в першій позиції, "
                    "бо на позиції 0 він міг би бачити лише самого себе.",
        },
    },
}

FIGURES = [
    ("vulnerability_map.png", "vuln"),
    ("criticality_profile.png", "profile"),
    ("degradation_grid.png", "grid"),
    ("required_tau.png", "tau"),
]

BANDS = [(1, 1, "1"), (2, 3, "2-3"), (4, 8, "4-8"), (9, 16, "9-16"),
         (17, 24, "17-24"), (25, 32, "25+")]


# --------------------------------------------------------------------------
# reading what the notebooks wrote
# --------------------------------------------------------------------------
def table(d: pathlib.Path, name: str):
    p = d / "tables" / name
    return pd.read_csv(p) if p.exists() else None


def present(v) -> bool:
    """True when a summary field actually holds a value.

    Absent safe boundaries arrive here as NaN rather than None, because the
    summary is built through a pandas DataFrame and pandas 3.0 stores a column
    of mixed None/str as its string dtype, which maps None to NaN.  NaN is
    truthy, so a plain `if v:` test would report a boundary where there is
    none.
    """
    if v is None:
        return False
    if isinstance(v, float) and v != v:
        return False
    return str(v).strip() not in ("", "nan", "None", "<NA>")


def load_units(experiment: str) -> list[dict]:
    """One 'unit' per trained model: 1 for ag_news/cifar10, 11 for pjm_regions."""
    d = BASE / experiment
    units = []
    for sp in ([d / "summary.json"] if (d / "summary.json").exists()
               else sorted(d.glob("*/summary.json"))):
        s = json.loads(sp.read_text(encoding="utf-8"))
        s["_dir"] = sp.parent
        s["_label"] = s.get("region") or sp.parent.name
        units.append(s)
    return units


def bands(s: dict) -> dict:
    df = table(s["_dir"], "random_multihead.csv")
    if df is None or s["degradation_col"] not in df.columns:
        return {}
    deg, out = s["degradation_col"], {}
    for lo, hi, label in BANDS:
        m = (df["N_fail"] >= lo) & (df["N_fail"] <= min(hi, s["n_heads"]))
        out[label] = float(df.loc[m, deg].mean()) if bool(m.any()) else None
    return out


def identity_effect(s: dict):
    """Q4/Q1 mean degradation with the fault count held in the 4-16 band."""
    df = table(s["_dir"], "random_multihead.csv")
    if df is None or s["degradation_col"] not in df.columns:
        return None
    deg = s["degradation_col"]
    sub = df[(df["N_fail"] >= 4) & (df["N_fail"] <= 16)]
    if len(sub) < 40:
        return None
    q = pd.qcut(sub["W_fail_mean"].rank(method="first"), 4,
                labels=["Q1", "Q2", "Q3", "Q4"])
    means = sub.groupby(q, observed=False)[deg].mean()
    q1, q4 = float(means.get("Q1", np.nan)), float(means.get("Q4", np.nan))
    if not q1 or q1 != q1:
        return None
    return {"q1": q1, "q4": q4, "ratio": q4 / q1, "n": int(len(sub))}


def confound(s: dict):
    """Spearman(W_fail, N_fail): how much weighted severity just restates count."""
    df = table(s["_dir"], "random_multihead.csv")
    if df is None or not {"W_fail", "N_fail"} <= set(df.columns):
        return None
    from scipy.stats import spearmanr
    return float(spearmanr(df["W_fail"], df["N_fail"]).statistic)


def layer_monotonicity(s: dict) -> str | None:
    """Is mean criticality monotone in depth? 'down', 'up', or None.

    H2 claims fragility is non-monotonic, so the report has to test it rather
    than assert it: a pre-norm stack can produce a strictly decreasing profile.
    """
    ls = table(s["_dir"], "layer_summary.csv")
    if ls is None or "mean_layer_criticality" not in ls.columns or len(ls) < 3:
        return None
    v = ls.sort_values("layer")["mean_layer_criticality"].to_numpy(dtype=float)
    if all(v[i] > v[i + 1] for i in range(len(v) - 1)):
        return "down"
    if all(v[i] < v[i + 1] for i in range(len(v) - 1)):
        return "up"
    return None


def train_facts(s: dict) -> dict:
    df = table(s["_dir"], "training_history.csv")
    if df is None or df.empty:
        return {}
    secs = float(df["secs"].sum()) if "secs" in df.columns else None
    return {"epochs": int(len(df)), "seconds": secs,
            "best": s.get("best_val_score", s.get("best_val_mae"))}


def facts_for(s: dict) -> dict:
    sh = s["single_head"]
    return {
        "label": s["_label"], "dir": s["_dir"], "summary": s,
        "sh": sh,
        "worst_layer": int(sh["most_critical_head"].split("_")[0][1:]),
        "monotone": layer_monotonicity(s),
        "concentration": s["median_over_max_criticality"],
        "bands": bands(s), "effect": identity_effect(s), "confound": confound(s),
        "train": train_facts(s),
    }


# --------------------------------------------------------------------------
# text
# --------------------------------------------------------------------------
META = {
    "ag_news": {
        "en": {
            "title": "AG News: Attention-Head Fault Tolerance of an Encoder-Only Transformer",
            "task": "four-class news-topic classification",
            "data": [
                ["Corpus", "AG News"],
                ["Task", "topic classification, 4 balanced classes"],
                ["Training items", "120 000 (12 000 held out for validation)"],
                ["Test items", "7 600, split into a calibration part and a fixed fault-evaluation part"],
                ["Classes", "World, Sports, Business, Sci/Tech"],
                ["Tokenisation", "lower-cased word-level regular expression, 30 000-word vocabulary"],
            ],
            "purpose": [
                "This experiment measures how much an encoder-only Transformer trained on real "
                "news text degrades when individual attention heads stop producing output.",
                "It addresses a specific gap in the source study, which validated its "
                "classification findings on a synthetic dataset only. The architecture here is "
                "the fixed Base configuration used by every experiment in this repository, so "
                "the results can be compared with the image and time-series experiments "
                "without the confound of a changing model.",
                "In this dataset a crashed head removes a channel of attention over word "
                "tokens, that is, part of the syntactic and topical evidence the classifier "
                "uses to assign a topic.",
            ],
        },
        "ua": {
            "title": "AG News: відмовостійкість голів уваги трансформера-енкодера",
            "task": "класифікація тем новин на чотири класи",
            "data": [
                ["Корпус", "AG News"],
                ["Завдання", "класифікація тем, 4 збалансовані класи"],
                ["Навчальних прикладів", "120 000 (12 000 відкладено для валідації)"],
                ["Тестових прикладів", "7 600, поділені на калібрувальну частину та фіксовану частину для оцінювання відмов"],
                ["Класи", "World, Sports, Business, Sci/Tech"],
                ["Токенізація", "словесна за регулярним виразом у нижньому регістрі, словник на 30 000 слів"],
            ],
            "purpose": [
                "У цьому експерименті вимірюється, наскільки погіршується робота "
                "трансформера-енкодера, навченого на реальних новинних текстах, коли окремі "
                "голови уваги перестають видавати результат.",
                "Робота закриває конкретну прогалину первинного дослідження, яке перевіряло "
                "свої висновки щодо класифікації лише на синтетичних даних. Архітектура тут — "
                "фіксована конфігурація Base, спільна для всіх експериментів цього "
                "репозиторію, тому результати можна порівнювати з експериментами на "
                "зображеннях і часових рядах без викривлення через зміну моделі.",
                "У цьому наборі даних відмова голови вилучає канал уваги над словесними "
                "токенами, тобто частину синтаксичних і тематичних ознак, які класифікатор "
                "використовує для визначення теми.",
            ],
        },
    },
    "cifar10": {
        "en": {
            "title": "CIFAR-10: Attention-Head Fault Tolerance of a ViT-Style Encoder",
            "task": "ten-class image classification",
            "data": [
                ["Corpus", "CIFAR-10"],
                ["Task", "image classification, 10 balanced classes"],
                ["Training images", "50 000 (5 000 held out for validation)"],
                ["Test images", "10 000, split into a calibration part and a fixed fault-evaluation part"],
                ["Image size", "32x32 RGB, per-channel standardisation"],
                ["Sequence seen by the model", "64 patches of 4x4 pixels plus a CLS token"],
                ["Augmentation", "random crop with 4-pixel reflect padding, random horizontal flip"],
            ],
            "purpose": [
                "This experiment repeats the protocol on a real image corpus, with the token "
                "embedding replaced by a patch projection so the encoder becomes a "
                "Vision-Transformer-style classifier. The fault model, the criticality "
                "definition and the fault-tolerance analysis are unchanged.",
                "It is the strongest test of whether the layer-position findings are a "
                "property of the architecture: a crashed head here removes a channel of "
                "attention over spatial image patches rather than over words or over time "
                "steps, which is a qualitatively different kind of evidence.",
                "CIFAR-10 is also the corpus used by the Vision Transformer of the "
                "Dropout-and-Outliers study, so the two lines of work share a dataset, an "
                "image and patch size, and an encoder-only ViT-style design. They are not "
                "otherwise the same model: that study uses six blocks of six heads at width "
                "384 with pre-normalisation and learnable absolute positional embeddings, "
                "against four blocks of eight heads at width 256 with post-normalisation and "
                "fixed sinusoidal encodings here. It also measures a different thing - the "
                "magnitude of outlier attention, weight and activation values under Dropout - "
                "rather than the effect of crashing a head.",
            ],
        },
        "ua": {
            "title": "CIFAR-10: відмовостійкість голів уваги енкодера у стилі ViT",
            "task": "класифікація зображень на десять класів",
            "data": [
                ["Корпус", "CIFAR-10"],
                ["Завдання", "класифікація зображень, 10 збалансованих класів"],
                ["Навчальних зображень", "50 000 (5 000 відкладено для валідації)"],
                ["Тестових зображень", "10 000, поділені на калібрувальну частину та фіксовану частину для оцінювання відмов"],
                ["Розмір зображення", "32x32 RGB, стандартизація по каналах"],
                ["Послідовність для моделі", "64 фрагменти 4x4 пікселі плюс токен CLS"],
                ["Аугментація", "випадкове обрізання з відбивним доповненням на 4 пікселі, випадкове горизонтальне відображення"],
            ],
            "purpose": [
                "Цей експеримент повторює протокол на реальному корпусі зображень: вкладення "
                "токенів замінено проєкцією фрагментів, тому енкодер стає класифікатором у "
                "стилі Vision Transformer. Модель відмови, означення критичності та аналіз "
                "відмовостійкості залишаються незмінними.",
                "Це найсуворіша перевірка того, чи є висновки про розташування шарів "
                "властивістю архітектури: відмова голови тут вилучає канал уваги над "
                "просторовими фрагментами зображення, а не над словами чи часовими кроками, "
                "тобто якісно інший тип ознак.",
                "CIFAR-10 — також корпус, який використовує Vision Transformer у дослідженні "
                "Dropout and Outliers, тому ці два напрями робіт мають спільний набір даних, "
                "розмір зображення та фрагмента і загальний план енкодера у стилі ViT. Проте "
                "це не та сама модель: там використано шість блоків по шість голів шириною "
                "384 з пре-нормалізацією та навчуваними абсолютними позиційними вкладеннями, "
                "а тут — чотири блоки по вісім голів шириною 256 з пост-нормалізацією та "
                "фіксованим синусоїдним кодуванням. Там також вимірюють інше — величину "
                "викидів уваги, ваг і активацій за наявності Dropout, а не наслідки краху "
                "голови.",
            ],
        },
    },
    "pjm_regions": {
        "en": {
            "title": "PJM Regional Load: Attention-Head Fault Tolerance of a Transformer Forecaster",
            "task": "one-step-ahead hourly electricity-load forecasting",
            "data": [
                ["Corpus", "PJM hourly load, every region except AEP"],
                ["Task", "forecast the next hourly value from the previous 20 hours"],
                ["Regions", "PJME, COMED, DAYTON, DOM, DUQ, PJMW, FE, NI, DEOK, EKPC, PJM_Load"],
                ["Observations", "roughly 33 000 to 145 000 hourly values per region"],
                ["Normalisation", "per-region min-max to the unit interval; MAPE reported in the original megawatt scale"],
                ["Sequence seen by the model", "20 consecutive hours plus a CLS token"],
            ],
            "purpose": [
                "This experiment runs the whole protocol independently on eleven regional "
                "load series, holding one architecture fixed, and then compares the eleven "
                "results.",
                "That comparison answers a question the source study could not: it examined "
                "one region only, so it could not tell whether the layer that holds the "
                "critical head is a property of the architecture or of the particular series. "
                "Eleven series under one fixed architecture separate the two.",
                "AEP is deliberately excluded because it is the region used in the source "
                "paper; these eleven series are therefore new evidence rather than a "
                "re-run of the published case.",
            ],
        },
        "ua": {
            "title": "Регіональне навантаження PJM: відмовостійкість голів уваги трансформера-прогнозувальника",
            "task": "погодинне прогнозування навантаження на крок уперед",
            "data": [
                ["Корпус", "погодинне навантаження PJM, усі регіони, крім AEP"],
                ["Завдання", "прогноз наступного погодинного значення за попередніми 20 годинами"],
                ["Регіони", "PJME, COMED, DAYTON, DOM, DUQ, PJMW, FE, NI, DEOK, EKPC, PJM_Load"],
                ["Спостережень", "приблизно від 33 000 до 145 000 погодинних значень на регіон"],
                ["Нормалізація", "min-max у межах регіону до одиничного відрізка; MAPE подано у вихідному масштабі мегаватів"],
                ["Послідовність для моделі", "20 послідовних годин плюс токен CLS"],
            ],
            "purpose": [
                "У цьому експерименті весь протокол виконується незалежно на одинадцяти "
                "регіональних рядах навантаження за незмінної архітектури, після чого "
                "одинадцять результатів порівнюються між собою.",
                "Таке порівняння відповідає на питання, недоступне первинному дослідженню: у "
                "ньому розглядався лише один регіон, тому не можна було встановити, чи є шар "
                "із критичною головою властивістю архітектури або конкретного ряду. "
                "Одинадцять рядів за однієї фіксованої архітектури розділяють ці дві "
                "можливості.",
                "Регіон AEP свідомо виключено, бо саме він використаний у первинній статті; "
                "отже, ці одинадцять рядів є новими даними, а не повторенням опублікованого "
                "випадку.",
            ],
        },
    },
}

T_EN = {
    # %s placeholders are filled from the result file, so the text follows the
    # architecture that was actually run rather than assuming one.
    "subtitle": "Experiment description, results and analysis. Encoder-only Transformer, "
                "%d layers x %d heads = %d attention heads.",
    "h_purpose": "1. Purpose and scope",
    "h_questions": "2. Research questions and hypotheses",
    "questions": [
        "RQ1. Does this trained Transformer contain a small number of critical heads "
        "alongside many redundant ones?",
        "RQ2. Is the damage explained by the number of failed heads alone, or also by which "
        "heads fail?",
        "RQ3. How does this result compare with the other datasets, given that the "
        "architecture is identical?",
    ],
    "hypotheses": [
        "H1. Attention-head criticality is non-uniform across the network.",
        "H2. Layer-wise fragility is not monotonic in depth and depends on the task.",
        "H3. Degradation grows with the number of simultaneously failed heads.",
        "H4. Criticality-aware fault severity carries information beyond the raw fault count.",
        "H5. Fault configurations containing critical heads exceed an acceptable degradation "
        "threshold more often.",
    ],
    "h_data": "3. Dataset",
    "h_model": "4. Model and training configuration",
    "h_fault": "5. Fault model and protocol",
    "fault_def": "Definition (attention-head crash). A head is considered crashed when its "
                 "output tensor is replaced with zeros after the attention weights have been "
                 "multiplied by the value matrix and before the heads are concatenated and "
                 "passed through the output projection. The crash removes that head's entire "
                 "contribution while leaving every trained parameter untouched.",
    "fault_notes": [
        "Faults are injected at inference only. The model is trained on clean data and then "
        "frozen, so the measured degradation is the resilience of a fixed model rather than a "
        "learned compensation.",
        "Every fault configuration is evaluated on the same fixed subset, drawn once from data "
        "held out of both training and model selection. Clean and faulted results are "
        "therefore directly comparable.",
        "A configuration stays active for exactly one evaluation pass, after which all head "
        "masks are restored to one.",
    ],
    "protocol": [
        "Single-head injection: each of the %d heads is crashed on its own while every other "
        "head stays alive, so each comparison differs in exactly one head. This yields "
        "per-head criticality C, redundancy R = 1 - C/max(C) and severity weight W = C/mean(C).",
        "Random multi-head injection: the number of failed heads is drawn uniformly from 1 to "
        "%d, that many heads are chosen without replacement, and all are crashed at once.",
        "Fault severity against degradation: Pearson, Spearman and Kendall correlations for "
        "each severity descriptor, plus the correlation between weighted severity and the raw "
        "fault count, which shows how much of the former merely restates the latter.",
        "Empirical fault-tolerance regions: the widest boundary on the total number of failed "
        "heads and the number of failed critical heads within which degradation stays below a "
        "threshold with empirical probability at least 0.95.",
    ],
    "h_budget": "6. Compute budget",
    "h_results": "7. Results",
    "h_analysis": "8. Analysis",
    "h_limits": "9. Limitations",
    "limits": [
        "Only an encoder-only architecture is evaluated. Decoder-only and encoder-decoder "
        "models may behave differently, particularly under causal masking.",
        "Failure is modelled as a head's output being zeroed. Bit flips in the projection "
        "weights, or partial corruption of the attention map, may degrade the model "
        "differently.",
        "Fault configurations are drawn uniformly at random, so correlated hardware failures "
        "affecting physically adjacent units are not represented.",
        "The model is trained from scratch on a limited budget, so its clean accuracy is below "
        "what this architecture reaches with longer training. Criticality measured on an "
        "under-trained model partly reflects incomplete specialisation of the heads.",
        "The fault-tolerance regions describe the sampled configurations for this model and "
        "these thresholds. They are empirical statements, not universal guarantees.",
    ],
    "h_repro": "10. Reproduction",
    "captions": {
        "vuln": "Figure 1. Single-head vulnerability map: the criticality of every head, "
                "arranged by layer and head index.",
        "profile": "Figure 2. Heads sorted by criticality. A steep drop means importance is "
                   "concentrated in a few heads.",
        "grid": "Figure 3. Mean degradation by fault fraction and by the mean criticality of "
                "the heads that failed.",
        "tau": "Figure 4. Degradation threshold that must be tolerated for a boundary to be "
               "95% safe.",
    },
}

T_UA = {
    "subtitle": "Опис експерименту, результати та аналіз. Трансформер-енкодер, "
                "%d шари x %d голів = %d голови уваги.",
    "h_purpose": "1. Мета та межі дослідження",
    "h_questions": "2. Дослідницькі питання та гіпотези",
    "questions": [
        "ДП1. Чи містить цей навчений трансформер невелику кількість критичних голів поряд "
        "із багатьма надлишковими?",
        "ДП2. Чи визначається шкода лише кількістю відмовлених голів, або також тим, які саме "
        "голови відмовили?",
        "ДП3. Як цей результат співвідноситься з іншими наборами даних за однакової "
        "архітектури?",
    ],
    "hypotheses": [
        "Г1. Критичність голів уваги нерівномірно розподілена в мережі.",
        "Г2. Крихкість шарів не є монотонною за глибиною та залежить від завдання.",
        "Г3. Деградація зростає з кількістю одночасно відмовлених голів.",
        "Г4. Тяжкість відмови з урахуванням критичності несе інформацію понад просту "
        "кількість відмов.",
        "Г5. Конфігурації відмов, що містять критичні голови, частіше перевищують "
        "припустимий порог деградації.",
    ],
    "h_data": "3. Набір даних",
    "h_model": "4. Модель і конфігурація навчання",
    "h_fault": "5. Модель відмови та протокол",
    "fault_def": "Означення (крах голови уваги). Голова вважається такою, що відмовила, коли "
                 "її вихідний тензор замінюється нулями після множення ваг уваги на матрицю "
                 "значень і до конкатенації голів та вихідної проєкції. Відмова повністю "
                 "вилучає внесок цієї голови, не змінюючи жодного навченого параметра.",
    "fault_notes": [
        "Відмови вводяться лише на етапі інференсу. Модель навчено на чистих даних і потім "
        "заморожено, тому виміряна деградація — це стійкість фіксованої моделі, а не набута "
        "компенсація.",
        "Кожну конфігурацію відмов оцінюють на тій самій фіксованій підмножині, відібраній "
        "один раз із даних, що не використовувалися ні для навчання, ні для відбору моделі. "
        "Тому чисті та відмовні результати прямо порівнювані.",
        "Конфігурація залишається активною рівно на один прохід оцінювання, після чого всі "
        "маски голів відновлюються до одиниці.",
    ],
    "protocol": [
        "Відмова однієї голови: кожну з %d голів вимикають окремо, тоді як усі інші "
        "працюють, тому кожне порівняння відрізняється рівно однією головою. Це дає "
        "критичність C, надлишковість R = 1 - C/max(C) і ваговий коефіцієнт тяжкості "
        "W = C/mean(C) для кожної голови.",
        "Випадкові відмови кількох голів: кількість відмовлених голів обирають рівномірно від "
        "1 до %d, стільку ж голів вибирають без повторень і вимикають одночасно.",
        "Тяжкість відмови проти деградації: кореляції Пірсона, Спірмена і Кендалла для "
        "кожного описника тяжкості, а також кореляція між ваговою тяжкістю та простою "
        "кількістю відмов, яка показує, наскільки перша лише переформулює другу.",
        "Емпіричні області відмовостійкості: найширша межа за загальною кількістю "
        "відмовлених голів і кількістю відмовлених критичних голів, у якій деградація "
        "залишається нижчою за порог з емпіричною ймовірністю не менше 0,95.",
    ],
    "h_budget": "6. Обчислювальний бюджет",
    "h_results": "7. Результати",
    "h_analysis": "8. Аналіз",
    "h_limits": "9. Обмеження",
    "limits": [
        "Оцінено лише архітектуру з енкодером. Моделі з декодером або енкодер-декодером "
        "можуть поводитися інакше, особливо за причинного маскування.",
        "Відмову змодельовано як обнулення виходу голови. Перемикання бітів у вагах проєкцій "
        "або часткове псування карти уваги можуть погіршувати модель інакше.",
        "Конфігурації відмов формуються рівномірно випадково, тому корелювані апаратні "
        "відмови, що зачіпають фізично сусідні блоки, не відтворені.",
        "Модель навчено з нуля за обмеженого бюджету, тому її чиста точність нижча за "
        "досяжну для цієї архітектури при тривалішому навчанні. Критичність, виміряна на "
        "недонавченій моделі, частково відображає неповну спеціалізацію голів.",
        "Області відмовостійкості описують відібрані конфігурації для цієї моделі та цих "
        "порогів. Це емпіричні твердження, а не універсальні гарантії.",
    ],
    "h_repro": "10. Відтворення",
    "captions": {
        "vuln": "Рисунок 1. Карта уразливості окремих голів: критичність кожної голови за "
                "індексами шару та голови.",
        "profile": "Рисунок 2. Голови, впорядковані за критичністю. Крутий спад означає, що "
                   "важливість зосереджена в кількох головах.",
        "grid": "Рисунок 3. Середня деградація залежно від частки відмов і від середньої "
                "критичності відмовлених голів.",
        "tau": "Рисунок 4. Порог деградації, який потрібно допустити, щоб межа була безпечною "
               "на рівні 95%.",
    },
}


# --------------------------------------------------------------------------
# result sections
# --------------------------------------------------------------------------
def add_figures(doc, d: pathlib.Path, captions: dict) -> None:
    for filename, key in FIGURES:
        p = d / "figures" / filename
        if not p.exists():
            continue
        doc.add_picture(str(p), width=Inches(6.0))
        cap = doc.add_paragraph(captions[key])
        cap.runs[0].italic = True
        cap.runs[0].font.size = Pt(9)
        cap.alignment = WD_ALIGN_PARAGRAPH.CENTER


def clean_row(c: dict, is_reg: bool, en: bool) -> list:
    if is_reg:
        return ["MAE " + fmt(c.get("mae")), "RMSE " + fmt(c.get("rmse")),
                "MAPE " + fmt(c.get("mape"), 3) + "%"]
    return ["accuracy " + fmt(c.get("accuracy")), "macro F1 " + fmt(c.get("f1")),
            "macro ROC AUC " + fmt(c.get("roc_auc"))]


def results_section(doc, units: list[dict], F: list[dict], en: bool, multi: bool) -> None:
    s0 = units[0]
    is_reg = s0["degradation_col"] in ("delta_mae", "delta_mape")

    # ---- clean performance
    hdr = (["Model", "Primary", "Secondary", "Tertiary", "Parameters"] if en else
           ["Модель", "Основна", "Друга", "Третя", "Параметри"])
    rows = [[f["label"]] + clean_row(f["summary"]["clean_metrics"], is_reg, en) +
            ["{:,}".format(f["summary"].get("n_params") or 0)] for f in F]
    add_table(doc, hdr, rows,
              "Table 1. Clean performance on the fixed fault-evaluation subset, before any "
              "fault injection." if en else
              "Таблиця 1. Чиста продуктивність на фіксованій підмножині для оцінювання "
              "відмов, до введення відмов.")

    # ---- training
    hdr = (["Model", "Epochs run", "Best validation score", "Training time, s"] if en else
           ["Модель", "Виконано епох", "Найкраща оцінка на валідації", "Час навчання, с"])
    rows = [[f["label"], f["train"].get("epochs", "-"), fmt(f["train"].get("best"), 4),
             fmt(f["train"].get("seconds"), 0)] for f in F if f["train"]]
    if rows:
        add_table(doc, hdr, rows,
                  "Table 2. Training. Early stopping on the validation selection metric means "
                  "the number of epochs run may be below the configured maximum." if en else
                  "Таблиця 2. Навчання. Через ранню зупинку за метрикою відбору на валідації "
                  "кількість виконаних епох може бути меншою за налаштований максимум.")

    # ---- single-head criticality
    hdr = (["Model", "Most critical head", "Max criticality", "Median criticality",
            "Median / max", "Mean redundancy", "Heads above half of max",
            "Heads with no effect"] if en else
           ["Модель", "Найкритичніша голова", "Макс. критичність", "Медіанна критичність",
            "Медіана / макс.", "Середня надлишковість", "Голів понад половину макс.",
            "Голів без впливу"])
    rows = [[f["label"], f["sh"]["most_critical_head"], fmt(f["sh"]["max_criticality"], 6),
             fmt(f["sh"]["median_criticality"], 6), fmt(f["concentration"], 3),
             fmt(f["sh"]["mean_redundancy"], 3), f["sh"]["n_heads_above_half_max"],
             f["summary"].get("n_heads_zero_effect", "-")] for f in F]
    add_table(doc, hdr, rows,
              "Table 3. Single-head criticality. A small median-to-maximum ratio means "
              "importance is concentrated in a few heads." if en else
              "Таблиця 3. Критичність окремих голів. Мале відношення медіани до максимуму "
              "означає, що важливість зосереджена в кількох головах.")

    # ---- critical head location
    hdr = (["Model", "Its layer", "Most fragile layer by mean",
            "Most fragile layer by strongest head", "Critical heads (top 10%)"] if en else
           ["Модель", "Її шар", "Найкрихкіший шар за середнім",
            "Найкрихкіший шар за найсильнішою головою", "Критичні голови (топ 10%)"])
    rows = [[f["label"], f["worst_layer"], f["sh"]["most_fragile_layer_by_mean"],
             f["sh"]["most_fragile_layer_by_max"],
             ", ".join(f["summary"]["critical_heads"])] for f in F]
    add_table(doc, hdr, rows,
              "Table 4. Where the critical heads sit. The architecture is identical in every "
              "row." if en else
              "Таблиця 4. Розташування критичних голів. Архітектура однакова в кожному рядку.")

    # ---- per-layer fragility (single model only)
    if not multi:
        ls = table(s0["_dir"], "layer_summary.csv")
        if ls is not None:
            hdr = (["Layer", "Mean criticality", "Strongest head", "Weakest head",
                    "Mean redundancy", "Most critical head"] if en else
                   ["Шар", "Середня критичність", "Найсильніша голова", "Найслабша голова",
                    "Середня надлишковість", "Найкритичніша голова"])
            rows = [[int(r.layer), fmt(r.mean_layer_criticality, 6),
                     fmt(r.max_head_criticality, 6), fmt(r.min_head_criticality, 6),
                     fmt(r.mean_redundancy, 3), r.most_critical_head]
                    for r in ls.itertuples()]
            add_table(doc, hdr, rows,
                      "Table 5. Fragility of each encoder layer." if en else
                      "Таблиця 5. Крихкість кожного шару енкодера.")

        sh_df = table(s0["_dir"], "single_head_damage.csv")
        if sh_df is not None:
            crit = s0["criticality_col"]
            hdr = (["Rank", "Head", "Criticality", "Redundancy", crit] if en else
                   ["Місце", "Голова", "Критичність", "Надлишковість", crit])
            top = sh_df.nlargest(5, "criticality")
            bot = sh_df.nsmallest(5, "criticality")
            rows = [[i + 1, r.head_name, fmt(r.criticality, 6), fmt(r.redundancy, 3),
                     fmt(getattr(r, crit), 6)] for i, r in enumerate(top.itertuples())]
            rows += [["-", r.head_name, fmt(r.criticality, 6), fmt(r.redundancy, 3),
                      fmt(getattr(r, crit), 6)] for r in bot.itertuples()]
            add_table(doc, hdr, rows,
                      "Table 6. The five most critical heads followed by the five least "
                      "critical." if en else
                      "Таблиця 6. П’ять найкритичніших голів, за якими наведено п’ять "
                      "найменш критичних.")

    # ---- degradation bands
    labels = [b[2] for b in BANDS]
    hdr = (["Model", "Metric"] + ["%s heads" % b for b in labels] if en else
           ["Модель", "Метрика"] + ["%s голів" % b for b in labels])
    rows = [[f["label"], f["summary"]["degradation_col"]] +
            [fmt(f["bands"].get(b), 6) for b in labels] for f in F if f["bands"]]
    if rows:
        add_table(doc, hdr, rows,
                  "Table 7. Mean degradation by the number of simultaneously failed heads."
                  if en else
                  "Таблиця 7. Середня деградація за кількістю одночасно відмовлених голів.")

    # ---- degradation grid (single model only)
    if not multi:
        g = table(s0["_dir"], "degradation_grid.csv")
        if g is not None and not g.empty:
            cols = [c for c in g.columns if c != "fault_bin"]
            hdr = ([("Fault fraction")] + cols if en else [("Частка відмов")] + cols)
            rows = [[r[0]] + [fmt(v, 6) for v in r[1:]]
                    for r in g[["fault_bin"] + cols].itertuples(index=False)]
            add_table(doc, hdr, rows,
                      "Table 8. Mean degradation by fault fraction (rows) and by quartile of "
                      "the mean criticality of the failed heads (columns, Q1 least critical)."
                      if en else
                      "Таблиця 8. Середня деградація за часткою відмов (рядки) та за "
                      "квартилем середньої критичності відмовлених голів (стовпці, Q1 — "
                      "найменш критичні).")

    # ---- severity correlations
    hdr = (["Model", "Severity descriptor", "Pearson", "Spearman", "Kendall"] if en else
           ["Модель", "Описник тяжкості", "Пірсон", "Спірмен", "Кендалл"])
    rows = []
    for f in F:
        for c in f["summary"].get("severity_correlations", []):
            rows.append([f["label"], c["severity"], fmt(c["pearson"], 3),
                         fmt(c["spearman"], 3), fmt(c["kendall"], 3)])
    if rows:
        add_table(doc, hdr, rows,
                  "Table 9. Rank correlations between each fault-severity descriptor and the "
                  "degradation. N_fail is the number of failed heads, L_fault the number of "
                  "affected layers, N_crit_fail the number of failed critical heads, W_fail "
                  "and W_fail_mean the sum and mean of the criticality weights." if en else
                  "Таблиця 9. Рангові кореляції між кожним описником тяжкості відмови та "
                  "деградацією. N_fail — кількість відмовлених голів, L_fault — кількість "
                  "зачеплених шарів, N_crit_fail — кількість відмовлених критичних голів, "
                  "W_fail і W_fail_mean — сума та середнє вагових коефіцієнтів критичності.")

    # ---- identity effect
    hdr = (["Model", "Least critical quartile", "Most critical quartile", "Ratio",
            "Configurations", "Spearman(W_fail, N_fail)"] if en else
           ["Модель", "Квартиль найменш критичних", "Квартиль найкритичніших", "Відношення",
            "Конфігурацій", "Спірмен(W_fail, N_fail)"])
    rows = [[f["label"], fmt(f["effect"]["q1"], 6), fmt(f["effect"]["q4"], 6),
             fmt(f["effect"]["ratio"], 2), f["effect"]["n"], fmt(f["confound"], 3)]
            for f in F if f["effect"]]
    if rows:
        add_table(doc, hdr, rows,
                  "Table 10. Configurations with 4 to 16 failed heads, split into quartiles by "
                  "the mean criticality of the heads that failed. Holding the fault count "
                  "roughly constant isolates the effect of head identity. The last column "
                  "shows how much the weighted severity is a restatement of the fault count."
                  if en else
                  "Таблиця 10. Конфігурації з 4–16 відмовленими головами, розбиті на квартилі "
                  "за середньою критичністю відмовлених голів. Приблизна фіксація кількості "
                  "відмов виділяє вплив саме ідентичності голів. Останній стовпець показує, "
                  "наскільке вагова тяжкість є переформулюванням кількості відмов.")

    # ---- safe regions
    hdr = (["Model", "Threshold", "Safe boundary", "Highest safe probability", "Coverage, %"]
           if en else
           ["Модель", "Порог", "Безпечна межа", "Найвища безпечна ймовірність", "Покриття, %"])
    rows = []
    for f in F:
        for r in f["summary"].get("safe_regions", []):
            rows.append([f["label"], r.get("threshold_name") or fmt(r.get("tau"), 4),
                         r["safe_boundary"] if present(r.get("safe_boundary"))
                         else ("none" if en else "немає"),
                         fmt(r.get("max_p_safe"), 3), fmt(r.get("coverage_percent"), 2)])
    if rows:
        add_table(doc, hdr, rows,
                  "Table 11. Widest boundary within which degradation stays below the "
                  "threshold with empirical probability at least 0.95. A boundary is written "
                  "as a cap on the total number of failed heads together with a cap on the "
                  "number of failed critical heads." if en else
                  "Таблиця 11. Найширша межа, у якій деградація залишається нижчою за порог з "
                  "емпіричною ймовірністю не менше 0,95. Межу записано як обмеження на "
                  "загальну кількість відмовлених голів разом з обмеженням на кількість "
                  "відмовлених критичних голів.")

    caps = (T_EN if en else T_UA)["captions"]
    add_criticality_text(doc, criticality_text(s0, en))
    if not multi:
        add_figures(doc, s0["_dir"], caps)
    else:
        # 11 models x 4 figures is unreadable, and none at all loses the visual
        # evidence: show one representative model and say why it is representative.
        hs = [f["sh"]["most_critical_head"] for f in F]
        modal = max(set(hs), key=hs.count)
        cand = sorted([f for f in F if f["sh"]["most_critical_head"] == modal] or F,
                      key=lambda f: f["concentration"])
        rep = cand[len(cand) // 2]
        note = doc.add_paragraph(
            ("The figures below are for %s, chosen as representative: its most critical head "
             "is the one that recurs across the models, and its criticality concentration is "
             "the median within that group. The same four figures for every other model are "
             "in that model's own figures folder." % rep["label"]) if en else
            ("Рисунки нижче наведено для %s як представницької моделі: її найкритичніша "
             "голова — та, що повторюється в різних моделях, а концентрація критичності є "
             "медіанною в цій групі. Ті самі чотири рисунки для кожної іншої моделі є в її "
             "власній теці figures." % rep["label"]))
        note.runs[0].italic = True
        add_figures(doc, rep["dir"], caps)


# --------------------------------------------------------------------------
# analysis
# --------------------------------------------------------------------------
def analysis(doc, units: list[dict], F: list[dict], en: bool, multi: bool) -> None:
    def para(txt):
        doc.add_paragraph(txt)

    concs = [f["concentration"] for f in F]
    reds = [f["sh"]["mean_redundancy"] for f in F]
    zeros = [f["summary"].get("n_heads_zero_effect") or 0 for f in F]
    n_heads = units[0]["n_heads"]

    # ---------------- H1
    # The verdict has to follow the measured concentration rather than assume it:
    # a median-to-maximum ratio near 1 means the heads are close to
    # interchangeable, which contradicts H1 instead of supporting it.
    def verdict(u):
        return "concentrated" if u < 0.35 else ("mixed" if u < 0.60 else "uniform")

    vs = [verdict(f["concentration"]) for f in F]
    dominant_v = max(set(vs), key=vs.count)
    halves = [f["sh"]["n_heads_above_half_max"] for f in F]
    spans = [(f["sh"]["min_criticality"] / f["sh"]["max_criticality"])
             if f["sh"]["max_criticality"] else float("nan") for f in F]
    maxc = [f["sh"]["max_criticality"] for f in F]
    # how many times worse the whole fault range is, for weighing the count
    # effect against the head-identity effect later
    gfac = [f["bands"]["25+"] / f["bands"]["1"] for f in F
            if f["bands"].get("1") and f["bands"].get("25+")]
    idfac = [f["effect"]["ratio"] for f in F if f["effect"]]

    def rng(vals, digits=3):
        lo, hi = min(vals), max(vals)
        if not multi or abs(hi - lo) < 10 ** -digits:
            return fmt(lo, digits)
        return "%s to %s" % (fmt(lo, digits), fmt(hi, digits))

    def rng_ua(vals, digits=3):
        lo, hi = min(vals), max(vals)
        if not multi or abs(hi - lo) < 10 ** -digits:
            return fmt(lo, digits)
        return "%s–%s" % (fmt(lo, digits), fmt(hi, digits))

    def rng_i(vals):
        lo, hi = min(vals), max(vals)
        return "%d" % lo if lo == hi else "%d to %d" % (lo, hi)

    def rng_i_ua(vals):
        lo, hi = min(vals), max(vals)
        return "%d" % lo if lo == hi else "від %d до %d" % (lo, hi)

    doc.add_heading("H1: is criticality non-uniform?" if en else
                    "Г1: чи є критичність нерівномірною?", level=2)
    zero_en = ("every head had some measurable effect when crashed alone"
               if max(zeros) == 0 else
               "%s heads had no measurable effect at all when crashed alone" % rng_i(zeros))
    zero_ua = ("кожна голова мала певний вимірний вплив при відмові поодинці"
               if max(zeros) == 0 else
               "%s голів не мали жодного вимірного впливу при відмові поодинці"
               % rng_i_ua(zeros))
    if en:
        para("The ratio of the median head criticality to the largest is %s. A value of one "
             "would mean every head matters equally, and a value near zero that a single head "
             "dominates. Alongside it, %s of the %d heads cause more than half the damage of "
             "the worst head, the weakest head still causes %s of what the worst one does, and "
             "%s."
             % (rng(concs), rng_i(halves), n_heads, rng(spans, 2), zero_en))
        if dominant_v == "concentrated":
            para("Importance is therefore concentrated in a small minority of heads, and mean "
                 "redundancy of %s confirms that the typical head can be removed on its own "
                 "with little effect. The consequence is asymmetric: a randomly chosen head is "
                 "usually almost free, while the worst head is expensive, so single-head "
                 "failure cannot be declared safe in general even though the median head very "
                 "nearly is. H1 is supported." % rng(reds))
        elif dominant_v == "mixed":
            para("This is materially non-uniform: only %s of the %d heads exceed half the "
                 "maximum, and the worst head is about %s times as damaging as the weakest, "
                 "yet no single head dominates the model outright. Mean redundancy is %s."
                 % (rng_i(halves), n_heads, fmt(1.0 / max(min(spans), 1e-9), 1), rng(reds)))
            para("H1 is supported in the graded sense the data shows: a minority of heads "
                 "carries a disproportionate share of the exposure, without the model "
                 "depending on one indispensable head. The worst single head costs %s of "
                 "true-class probability against a clean accuracy of %s, so an individual "
                 "crash is survivable even in the worst case, but the spread across heads is "
                 "wide enough that a criticality ranking is worth having."
                 % (rng(maxc, 4),
                    fmt(F[0]["summary"]["clean_metrics"].get("accuracy"), 3)))
        else:
            para("Criticality on this dataset is graded rather than concentrated. It is not "
                 "flat - the worst head is about %s times as damaging as the weakest, so the "
                 "heads are not interchangeable - but the variation is spread smoothly across "
                 "the whole stack instead of being carried by a few outliers. Mean redundancy "
                 "is %s."
                 % (fmt(1.0 / max(min(spans), 1e-9), 1), rng(reds)))
            para("H1 therefore splits by how strongly it is read. In its strong form, that a "
                 "small critical minority accounts for most of the exposure, it is not "
                 "supported here: %s of %d heads exceed half the maximum, so no such minority "
                 "exists. In its weak form, that criticality is not identical across heads, it "
                 "holds. This distinction matters because the strong form is what would justify "
                 "protecting a chosen few heads, and it is the form the source study could not "
                 "test on synthetic data."
                 % (rng_i(halves), n_heads))
            para("The absolute scale reinforces the point: the worst single head costs only %s "
                 "of true-class probability on average, against a clean accuracy of %s. A "
                 "single-head crash anywhere in this model is close to harmless, whichever head "
                 "it is."
                 % (rng(maxc, 4), fmt(F[0]["summary"]["clean_metrics"].get("accuracy"), 3)))
    else:
        para("Відношення медіанної критичності голови до найбільшої становить %s. Значення, "
             "рівне одиниці, означало б, що всі голови однаково важливі, а значення близьке до "
             "нуля — що домінує одна голова. Поряд із цим %s з %d голів завдають понад "
             "половину шкоди найгіршої голови, найслабша голова завдає %s від шкоди найгіршої, "
             "а %s."
             % (rng_ua(concs), rng_i_ua(halves), n_heads, rng_ua(spans, 2), zero_ua))
        if dominant_v == "concentrated":
            para("Отже, важливість зосереджена в невеликій меншості голів, а середня "
                 "надлишковість %s підтверджує, що типову голову можна вилучити окремо майже "
                 "без наслідків. Наслідок асиметричний: випадково обрана голова зазвичай майже "
                 "безкоштовна, тоді як найгірша голова дорога, тому відмову однієї голови не "
                 "можна визнати безпечною загалом, навіть якщо медіанна голова практично "
                 "безпечна. Г1 підтверджено." % rng_ua(reds))
        elif dominant_v == "mixed":
            para("Це суттєва нерівномірність: лише %s з %d голів перевищують половину "
                 "максимуму, а найгірша голова приблизно у %s раза шкідливіша за найслабшу, "
                 "проте жодна окрема голова не домінує в моделі повністю. Середня "
                 "надлишковість — %s."
                 % (rng_i_ua(halves), n_heads, fmt(1.0 / max(min(spans), 1e-9), 1),
                    rng_ua(reds)))
            para("Г1 підтверджено в тому градуйованому сенсі, який показують дані: меншість "
                 "голів несе непропорційно велику частку ризику, при цьому модель не залежить "
                 "від однієї незамінної голови. Найгірша окрема голова коштує %s ймовірності "
                 "істинного класу за чистої точності %s, тому окремий крах можна пережити "
                 "навіть у найгіршому випадку, але розкид між головами достатньо широкий, щоб "
                 "рейтинг критичності мав сенс."
                 % (rng_ua(maxc, 4),
                    fmt(F[0]["summary"]["clean_metrics"].get("accuracy"), 3)))
        else:
            para("Критичність на цьому наборі даних є градуйованою, а не зосередженою. Вона не "
                 "пласка — найгірша голова приблизно у %s раза шкідливіша за найслабшу, тому "
                 "голови не взаємозамінні, — але ця варіація рівномірно розподілена по всьому "
                 "стеку, а не зосереджена в кількох викидах. Середня надлишковість — %s."
                 % (fmt(1.0 / max(min(spans), 1e-9), 1), rng_ua(reds)))
            para("Тому Г1 розділяється залежно від того, наскільки сильно її читати. У сильній "
                 "формі — що невелика критична меншість відповідає за більшу частину ризику — "
                 "вона тут не підтверджується: %s з %d голів перевищують половину максимуму, "
                 "тому такої меншості немає. У слабкій формі — що критичність не однакова для "
                 "всіх голів — вона справджується. Ця відмінність важлива, бо саме сильна форма "
                 "виправдовувала б захист кількох обраних голів, і саме цю форму первинне "
                 "дослідження не могло перевірити на синтетичних даних."
                 % (rng_i_ua(halves), n_heads))
            para("Абсолютний масштаб підсилює цей висновок: найгірша окрема голова коштує в "
                 "середньому лише %s ймовірності істинного класу за чистої точності %s. Крах "
                 "однієї голови будь-де в цій моделі майже безпечний, незалежно від того, яка "
                 "це голова."
                 % (rng_ua(maxc, 4), fmt(F[0]["summary"]["clean_metrics"].get("accuracy"), 3)))

    # ---------------- H2 / RQ3
    doc.add_heading("H2: where the critical heads sit" if en else
                    "Г2: де розташовані критичні голови", level=2)
    layers = [f["worst_layer"] for f in F]
    counts = {l: layers.count(l) for l in sorted(set(layers))}
    desc = ", ".join(("layer %d: %d" if en else "шар %d: %d") % (k, v)
                     for k, v in counts.items())
    dom = max(counts, key=counts.get)
    heads = [f["sh"]["most_critical_head"] for f in F]
    modal_head = max(set(heads), key=heads.count)
    modal_n = heads.count(modal_head)
    if en:
        if multi:
            para("Counting the layer that holds each model's single most critical head gives "
                 "%s. Layer %d accounts for %d of the %d models. More specifically, one and "
                 "the same head, %s, is the most critical head in %d of the %d models."
                 % (desc, dom, counts[dom], len(F), modal_head, modal_n, len(F)))
            if modal_n * 3 >= len(F) * 2:
                para("That recurrence is the result this experiment exists to produce. The "
                     "architecture, the training recipe and the fault protocol are identical "
                     "across all %d models and only the series differs, so a head that comes "
                     "out most critical %d times cannot be getting its status from the data. "
                     "The location of the critical head is therefore a property of the "
                     "architecture together with the task family, not of the individual "
                     "series. The %d exception%s (%s) show that a series can move it, but the "
                     "default position is fixed."
                     % (len(F), modal_n, len(F) - modal_n,
                        "" if len(F) - modal_n == 1 else "s",
                        ", ".join(f["label"] for f in F
                                  if f["sh"]["most_critical_head"] != modal_head) or "none"))
                para("This resolves the question the source study left open. It observed "
                     "critical heads in a late layer for forecasting and in early layers for "
                     "classification, but it varied the model size along with the task and so "
                     "could not say which caused it. Here the critical head for forecasting "
                     "sits in layer %d, the last encoder layer, in %d of %d independent series "
                     "under one fixed architecture, while the classification experiments in "
                     "this repository place it in the first two layers of the very same "
                     "architecture. Depth position tracks the task family, not the network "
                     "shape." % (dom, counts[dom], len(F)))
            else:
                para("The critical location is not stable across series: no single head or "
                     "layer accounts for most of the %d models. Because the architecture, the "
                     "training recipe and the fault protocol are identical and only the series "
                     "differs, that spread is a property of the data rather than of the "
                     "network. This is exactly the comparison the source study could not make "
                     "from a single region." % len(F))
        else:
            by_mean = F[0]["sh"]["most_fragile_layer_by_mean"]
            by_max = F[0]["sh"]["most_fragile_layer_by_max"]
            if by_mean == by_max:
                para("The single most critical head of this model is %s, in layer %d. Both "
                     "rankings of the layers agree: layer %s comes first by mean criticality "
                     "and also by its strongest single head, so that layer is fragile as a "
                     "whole rather than because of one outlier head."
                     % (F[0]["sh"]["most_critical_head"], F[0]["worst_layer"], by_mean))
            else:
                para("The single most critical head of this model is %s, in layer %d. Ranking "
                     "the layers by mean criticality puts layer %s first, while ranking them by "
                     "their strongest single head puts layer %s first. The two disagree here, "
                     "which happens when a layer holds one very critical head among otherwise "
                     "redundant ones."
                     % (F[0]["sh"]["most_critical_head"], F[0]["worst_layer"], by_mean, by_max))
            mono = F[0]["monotone"]
            if mono:
                para("Fragility here is monotone in depth: mean criticality %s steadily from "
                     "layer to layer across the whole stack, so H2, which predicts a "
                     "non-monotonic profile, is not supported for this model. What H2 gets "
                     "right is that the profile is task- and architecture-dependent rather "
                     "than fixed; what it gets wrong here is the shape."
                     % ("falls" if mono == "down" else "rises"))
            else:
                para("Fragility is therefore not monotonic in depth: damage neither grows "
                     "steadily from the input side to the output side nor the reverse, which "
                     "is what H2 predicts.")
            para("Which layers matter is decided by where this model concentrated the "
                 "computation its prediction depends on. Since the architecture is shared with "
                 "the other experiments in this family, any difference in the critical layer "
                 "between them is attributable to the task and the data, not to depth or "
                 "width. Comparisons against a different family in this repository do not have "
                 "that property, because there the architecture changes too.")
        if dominant_v == "uniform":
            para("This layer attribution should be treated with caution for this experiment. "
                 "Because the heads differ so little in criticality, the identity of the "
                 "single 'most critical' head and of the 'most fragile' layer rests on small "
                 "differences that another training seed could reorder. The ranking is "
                 "reported for completeness, but H2 is only weakly identified here: the "
                 "evidence supports the absence of monotonic depth-wise fragility rather than "
                 "a confident claim about which layer is worst.")
        else:
            if not multi:
                para("H2 is supported: the layer that matters is set by the task, not by depth.")
            elif modal_n * 3 >= len(F) * 2:
                para("H2 is supported: fragility is not monotonic in depth, and the layer that "
                     "matters is set by the task family rather than by depth or by the "
                     "individual series.")
            else:
                para("H2 is supported: the layer that matters is set by the series, not by "
                     "depth.")
    else:
        if multi:
            para("Підрахунок шару, який містить найкритичнішу голову кожної моделі, дає %s. На "
                 "шар %d припадає %d з %d моделей. Точніше, одна й та сама голова, %s, є "
                 "найкритичнішою у %d з %d моделей."
                 % (desc, dom, counts[dom], len(F), modal_head, modal_n, len(F)))
            if modal_n * 3 >= len(F) * 2:
                para("Саме заради цього повторення й проводився цей експеримент. Архітектура, "
                     "рецепт навчання та протокол відмов однакові для всіх %d моделей, і "
                     "відрізняється лише ряд даних, тому голова, яка виявляється "
                     "найкритичнішою %d разів, не може отримувати цей статус від даних. Отже, "
                     "розташування критичної голови є властивістю архітектури разом із "
                     "родиною завдань, а не окремого ряду. %d виняток%s (%s) показує, що ряд "
                     "може її змістити, але типове положення є фіксованим."
                     % (len(F), modal_n, len(F) - modal_n,
                        "" if len(F) - modal_n == 1 else "и",
                        ", ".join(f["label"] for f in F
                                  if f["sh"]["most_critical_head"] != modal_head) or "немає"))
                para("Це розв’язує питання, яке первинне дослідження залишило відкритим. Воно "
                     "спостерігало критичні голови в останньому шарі для прогнозування та в "
                     "перших шарах для класифікації, але змінювало розмір моделі разом із "
                     "завданням і тому не могло сказати, що саме є причиною. Тут критична "
                     "голова для прогнозування розташована в шарі %d, останньому шарі "
                     "енкодера, у %d з %d незалежних рядів за однієї фіксованої архітектури, "
                     "тоді як експерименти з класифікації в цьому репозиторії розміщують її у "
                     "перших двох шарах тієї самої архітектури. Положення за глибиною "
                     "відповідає родині завдань, а не формі мережі."
                     % (dom, counts[dom], len(F)))
            else:
                para("Критичне розташування не є стабільним між рядами: жодна окрема голова чи "
                     "шар не охоплює більшості з %d моделей. Оскільки архітектура, рецепт "
                     "навчання та протокол відмов однакові і відрізняється лише ряд даних, цей "
                     "розподіл є властивістю даних, а не мережі. Саме такого порівняння "
                     "первинне дослідження не могло зробити на одному регіоні." % len(F))
        else:
            by_mean = F[0]["sh"]["most_fragile_layer_by_mean"]
            by_max = F[0]["sh"]["most_fragile_layer_by_max"]
            if by_mean == by_max:
                para("Найкритичніша голова цієї моделі — %s, у шарі %d. Обидва впорядкування "
                     "шарів збігаються: шар %s є першим і за середньою критичністю, і за "
                     "найсильнішою окремою головою, тобто цей шар крихкий як ціле, а не через "
                     "одну голову-викид."
                     % (F[0]["sh"]["most_critical_head"], F[0]["worst_layer"], by_mean))
            else:
                para("Найкритичніша голова цієї моделі — %s, у шарі %d. За середньою "
                     "критичністю першим є шар %s, а за найсильнішою окремою головою — шар %s. "
                     "Тут ці впорядкування розходяться, що трапляється, коли шар містить одну "
                     "дуже критичну голову серед решти надлишкових."
                     % (F[0]["sh"]["most_critical_head"], F[0]["worst_layer"], by_mean, by_max))
            mono = F[0]["monotone"]
            if mono:
                para("Крихкість тут монотонна за глибиною: середня критичність стабільно %s "
                     "від шару до шару по всьому стеку, тому Г2, яка передбачає немонотонний "
                     "профіль, для цієї моделі не підтверджується. Г2 має рацію в тому, що "
                     "профіль залежить від завдання та архітектури, а не є сталим; хибним тут "
                     "є твердження про його форму."
                     % ("спадає" if mono == "down" else "зростає"))
            else:
                para("Отже, крихкість не є монотонною за глибиною: шкода не зростає рівномірно "
                     "від входу до виходу і не зменшується у зворотному напрямі, як і "
                     "передбачає Г2.")
            para("Значущі шари визначаються тим, де ця модель зосередила обчислення, від яких "
                 "залежить її прогноз. Оскільки архітектура спільна з іншими експериментами "
                 "цієї сім’ї, будь-яка різниця в критичному шарі між ними зумовлена завданням "
                 "і даними, а не глибиною чи шириною. Порівняння з іншою сім’єю цього "
                 "репозиторію такої властивості не мають, бо там змінюється й архітектура.")
        if dominant_v == "uniform":
            para("Для цього експерименту таке віднесення до шару варто трактувати обережно. "
                 "Оскільки голови різняться за критичністю дуже мало, визначення однієї "
                 "«найкритичнішої» голови та «найкрихкішого» шару спирається на малі "
                 "відмінності, які інший початковий стан навчання міг би переставити. "
                 "Впорядкування наведено для повноти, але Г2 тут ідентифікується лише слабко: "
                 "дані підтверджують відсутність монотонної крихкості за глибиною, а не "
                 "впевнене твердження про те, який шар найгірший.")
        else:
            if not multi:
                para("Г2 підтверджено: значущий шар визначається завданням, а не глибиною.")
            elif modal_n * 3 >= len(F) * 2:
                para("Г2 підтверджено: крихкість не є монотонною за глибиною, а значущий шар "
                     "визначається родиною завдань, а не глибиною чи окремим рядом даних.")
            else:
                para("Г2 підтверджено: значущий шар визначається рядом даних, а не глибиною.")

    # ---------------- H3
    doc.add_heading("H3: degradation grows with the number of failures" if en else
                    "Г3: деградація зростає з кількістю відмов", level=2)
    growth, mono = [], []
    for f in F:
        b = f["bands"]
        vals = [b.get(l) for l in [x[2] for x in BANDS]]
        vals = [v for v in vals if v is not None]
        if len(vals) >= 2:
            mono.append(all(vals[i] <= vals[i + 1] + 1e-12 for i in range(len(vals) - 1)))
            if b.get("1") and b.get("25+"):
                growth.append((f["label"], b["1"], b["25+"], b["25+"] / b["1"]))
    if growth:
        worst = max(growth, key=lambda g: g[3])
        best = min(growth, key=lambda g: g[3])
        n_mono = sum(1 for m in mono if m)
        if en:
            para("Mean degradation rises with the fault count in Table 7. Going from a single "
                 "failed head to more than twenty-four multiplies the degradation by %s%s."
                 % (fmt(best[3], 1) if not multi else fmt(best[3], 1),
                    "" if not multi else
                    " (%s) up to %s (%s)" % (best[0], fmt(worst[3], 1), worst[0])))
            para("The increase is monotonic across every band in %s. The fault count is thus "
                 "the dominant predictor of damage, which the rank correlations of Table 9 "
                 "independently confirm. H3 is supported."
                 % ("this model" if not multi else "%d of the %d models" % (n_mono, len(mono))))
        else:
            para("Середня деградація зростає з кількістю відмов у таблиці 7. Перехід від однієї "
                 "відмовленої голови до понад двадцяти чотирьох збільшує деградацію у %s раза%s."
                 % (fmt(best[3], 1),
                    "" if not multi else
                    " (%s) — до %s раза (%s)" % (best[0], fmt(worst[3], 1), worst[0])))
            para("Зростання монотонне за всіма смугами у %s. Отже, кількість відмов — головний "
                 "предиктор шкоди, що незалежно підтверджують рангові кореляції таблиці 9. Г3 "
                 "підтверджено."
                 % ("цій моделі" if not multi else "%d з %d моделей" % (n_mono, len(mono))))

    # ---------------- H4
    doc.add_heading("H4: does head identity matter beyond head count?" if en else
                    "Г4: чи важлива ідентичність голів понад їхню кількість?", level=2)
    effs = [f for f in F if f["effect"]]
    if effs:
        ratios = [f["effect"]["ratio"] for f in effs]
        matters = [f["label"] for f in effs if f["effect"]["ratio"] >= 1.2]
        neutral = [f["label"] for f in effs if f["effect"]["ratio"] < 1.2]
        cf = [f["confound"] for f in effs if f["confound"] is not None]

        def sev(f, name):
            for c in f["summary"].get("severity_correlations", []):
                if c["severity"] == name:
                    return float(c["spearman"])
            return None

        wmc = [v for v in (sev(f, "W_fail_mean") for f in effs) if v is not None]
        nfc = [v for v in (sev(f, "N_fail") for f in effs) if v is not None]
        if en:
            para("The weighted severity W_fail correlates with degradation strongly, but that "
                 "is not by itself evidence that criticality adds information: W_fail is a sum "
                 "over the failed heads, so it grows with the fault count almost by "
                 "construction. Its rank correlation with the plain fault count is %s. Any "
                 "honest test of H4 has to hold the fault count fixed."
                 % (fmt(cf[0], 3) if len(cf) == 1 else
                    "%s to %s" % (fmt(min(cf), 3), fmt(max(cf), 3))))
        if en and wmc:
            para("W_fail_mean is the cleaner signal, because it is the mean rather than the sum "
                 "of the criticality weights and so does not grow with the fault count. Its "
                 "rank correlation with degradation is %s, against %s for the raw fault count.%s"
                 % (fmt(wmc[0], 3) if len(wmc) == 1 else
                    "%s to %s" % (fmt(min(wmc), 3), fmt(max(wmc), 3)),
                    fmt(nfc[0], 3) if len(nfc) == 1 else
                    "%s to %s" % (fmt(min(nfc), 3), fmt(max(nfc), 3)),
                    (" Measured unconditionally like this, a small value understates the "
                     "identity effect rather than refuting it, because variation in the fault "
                     "count swamps it. Removing that variation is exactly what the quartile "
                     "test does.") if max(wmc) < 0.2 else
                    (" That is a substantial correlation for a descriptor that carries no "
                     "information about how many heads failed, which already points to head "
                     "identity mattering in its own right.")))
        if en:
            para("Table 10 does that, comparing configurations with 4 to 16 failed heads. The "
                 "most critical quartile degrades the model by a factor of %s relative to the "
                 "least critical quartile."
                 % (fmt(ratios[0], 2) if len(ratios) == 1 else
                    "%s to %s" % (fmt(min(ratios), 2), fmt(max(ratios), 2))))
            if matters and neutral:
                para("The effect is clear in %s and absent in %s, where the ratio is close to "
                     "or below one. Head identity therefore carries extra information for some "
                     "series and not for others, so H4 is only partly supported."
                     % (", ".join(matters), ", ".join(neutral)))
            elif matters:
                para("The effect is present%s, so which heads fail matters on top of how many "
                     "fail and H4 is supported."
                     % ("" if not multi else " in every one of the %d models" % len(effs)))
            else:
                para("No clear effect appears: at these fault counts the damage is explained by "
                     "the number of failures alone, and H4 is not supported for this "
                     "experiment.")
        else:
            para("Вагова тяжкість W_fail сильно корелює з деградацією, але це саме собою не є "
                 "доказом, що критичність додає інформацію: W_fail — сума по відмовлених "
                 "головах, тому вона зростає з кількістю відмов майже за побудовою. Її рангова "
                 "кореляція з простою кількістю відмов становить %s. Тому будь-яка коректна "
                 "перевірка Г4 має фіксувати кількість відмов."
                 % (fmt(cf[0], 3) if len(cf) == 1 else
                    "%s–%s" % (fmt(min(cf), 3), fmt(max(cf), 3))))
            if wmc:
                para("W_fail_mean — чистіший показник, бо це середнє, а не сума вагових "
                     "коефіцієнтів критичності, тому воно не зростає з кількістю відмов. Його "
                     "рангова кореляція з деградацією становить %s проти %s для простої "
                     "кількості відмов.%s"
                     % (fmt(wmc[0], 3) if len(wmc) == 1 else
                        "%s–%s" % (fmt(min(wmc), 3), fmt(max(wmc), 3)),
                        fmt(nfc[0], 3) if len(nfc) == 1 else
                        "%s–%s" % (fmt(min(nfc), 3), fmt(max(nfc), 3)),
                        (" Виміряне так, безумовно, мале значення радше занижує ефект "
                         "ідентичності, ніж заперечує його, бо варіація кількості відмов його "
                         "перекриває. Саме цю варіацію й усуває перевірка квартилів.")
                        if max(wmc) < 0.2 else
                        (" Це суттєва кореляція для описника, який не несе жодної інформації "
                         "про те, скільки голів відмовило, і вже сама вказує на те, що "
                         "ідентичність голів має значення сама собою.")))
            para("Таблиця 10 робить саме це, порівнюючи конфігурації з 4–16 відмовленими "
                 "головами. Квартиль найкритичніших погіршує модель у %s раза порівняно з "
                 "квартилем найменш критичних."
                 % (fmt(ratios[0], 2) if len(ratios) == 1 else
                    "%s–%s" % (fmt(min(ratios), 2), fmt(max(ratios), 2))))
            if matters and neutral:
                para("Ефект виразний у %s і відсутній у %s, де відношення близьке до одиниці або "
                     "менше за неї. Отже, ідентичність голів несе додаткову інформацію для "
                     "одних рядів і не несе для інших, тому Г4 підтверджено лише частково."
                     % (", ".join(matters), ", ".join(neutral)))
            elif matters:
                para("Ефект наявний%s, тому важливо не лише скільки голів відмовило, а й які "
                     "саме; Г4 підтверджено."
                     % ("" if not multi else " у кожній з %d моделей" % len(effs)))
            else:
                para("Виразного ефекту не виявлено: за таких кількостей відмов шкода "
                     "пояснюється лише числом відмов, і Г4 для цього експерименту не "
                     "підтверджується.")

    # ---------------- H5
    doc.add_heading("H5: critical heads and the safe operating region" if en else
                    "Г5: критичні голови та безпечна робоча область", level=2)
    regions = [(f, r) for f in F for r in f["summary"].get("safe_regions", [])]
    found = [(f, r) for f, r in regions if present(r.get("safe_boundary"))]
    n_crit = [len(f["summary"].get("critical_heads", [])) for f in F]
    # A cap of "at most c critical failures" only constrains anything when c is
    # strictly below the number of critical heads the model actually has.
    n_binding = sum(1 for f, r in found
                    if present(r.get("n_crit_fail_max"))
                    and float(r["n_crit_fail_max"]) < len(f["summary"].get("critical_heads", [])))
    if regions:
        if en:
            if found:
                para("A 95%%-safe boundary exists in %d of the %d threshold cases examined."
                     % (len(found), len(regions)))
                if n_binding:
                    para("%s the cap on failed critical heads is binding, that is, strictly "
                         "below the %s critical heads the model has. Widening such a boundary "
                         "to admit one more critical failure raises the degradation that must "
                         "be tolerated, which is the content of H5, and to that extent it is "
                         "supported."
                         % ("In that boundary" if len(found) == 1 else
                            ("In every one of those boundaries" if n_binding == len(found)
                             else "In %d of those %d boundaries" % (n_binding, len(found))),
                            rng_i(n_crit)))
                else:
                    para("In none of those boundaries is the cap on failed critical heads "
                         "binding: each one permits all %s of the model's critical heads to "
                         "fail at once, so the only constraint that actually does any work is "
                         "the cap on the total number of failed heads. On this experiment H5 "
                         "is therefore not supported - the safe region is set by how many heads "
                         "fail, not by whether the critical ones are among them, which is the "
                         "same conclusion the quartile test reached for H4."
                         % rng_i(n_crit))
                widest = max(found, key=lambda lr: lr[1].get("coverage_percent") or 0)
                para("The widest safe boundary is %s (%s), covering %s%% of the sampled "
                     "configurations at threshold %s."
                     % (widest[1]["safe_boundary"], widest[0]["label"],
                        fmt(widest[1].get("coverage_percent"), 1),
                        widest[1].get("threshold_name") or fmt(widest[1].get("tau"), 4)))
            else:
                para("No boundary reaches the 0.95 level at any of the thresholds examined. "
                     "Under a degradation budget this tight, even modest fault configurations "
                     "cannot be declared safe, so the safe-region analysis constrains rather "
                     "than reassures.")
            if dominant_v != "concentrated" and gfac and idfac:
                para("Weighing the two effects against each other gives the engineering "
                     "conclusion. Going from one failed head to the top of the range multiplies "
                     "the degradation by about %s, whereas at a fixed fault count the most "
                     "critical quartile of configurations costs only about %s times the least "
                     "critical. The number of simultaneous failures therefore outweighs head "
                     "identity by roughly a factor of %s."
                     % (fmt(max(gfac), 0), fmt(max(idfac), 2),
                        fmt(max(gfac) / max(max(idfac), 1e-9), 0)))
                para("A fault-tolerance budget for this model should therefore buy a limit on "
                     "how many heads can fail at once rather than selective hardening of a "
                     "chosen few. Identity is not irrelevant - it is worth respecting at the "
                     "strictest degradation thresholds, where the safe boundary does exclude "
                     "critical-head failures - but it is a second-order correction to a "
                     "first-order count effect. Selective hardening pays off where criticality "
                     "is concentrated, and whether it is concentrated is a property to be "
                     "measured per model with the single-head sweep used here, not assumed.")
            else:
                para("The engineering consequence is that protection is worth concentrating on "
                     "the small identified set of critical heads rather than spread evenly over "
                     "all %d. Redundancy is abundant but not uniform, and the exceptions are "
                     "identifiable in advance by exactly the single-head sweep used here."
                     % n_heads)
        else:
            if found:
                para("Безпечна на рівні 95%% межа існує у %d з %d розглянутих випадків порогів."
                     % (len(found), len(regions)))
                if n_binding:
                    para("%s обмеження на кількість відмовлених критичних голів є дієвим, тобто "
                         "строго меншим за %s критичних голів моделі. Розширення такої межі "
                         "так, щоб допустити ще одну критичну відмову, підвищує деградацію, яку "
                         "доводиться терпіти; це і є зміст Г5, і в цій мірі він підтверджується."
                         % ("У цій межі" if len(found) == 1 else
                            ("У кожній із цих меж" if n_binding == len(found)
                             else "У %d з цих %d меж" % (n_binding, len(found))),
                            rng_i_ua(n_crit)))
                else:
                    para("Ні в одній із цих меж обмеження на кількість відмовлених критичних "
                         "голів не є дієвим: кожна з них допускає одночасну відмову всіх %s "
                         "критичних голів моделі, тому єдине обмеження, яке справді працює, — "
                         "це обмеження на загальну кількість відмовлених голів. Отже, у цьому "
                         "експерименті Г5 не підтверджується: безпечна область визначається "
                         "тим, скільки голів відмовило, а не тим, чи є серед них критичні, — "
                         "той самий висновок, до якого привела перевірка квартилів для Г4."
                         % rng_i_ua(n_crit))
                widest = max(found, key=lambda lr: lr[1].get("coverage_percent") or 0)
                para("Найширша безпечна межа — %s (%s), що покриває %s%% відібраних "
                     "конфігурацій за порога %s."
                     % (widest[1]["safe_boundary"], widest[0]["label"],
                        fmt(widest[1].get("coverage_percent"), 1),
                        widest[1].get("threshold_name") or fmt(widest[1].get("tau"), 4)))
            else:
                para("Жодна межа не досягає рівня 0,95 за жодного з розглянутих порогів. За "
                     "такого жорсткого бюджету деградації навіть помірні конфігурації відмов "
                     "не можна визнати безпечними, тому аналіз безпечних областей радше "
                     "обмежує, ніж заспокоює.")
            if dominant_v != "concentrated" and gfac and idfac:
                para("Інженерний висновок дає зважування двох ефектів між собою. Перехід від "
                     "однієї відмовленої голови до верхньої межі діапазону збільшує деградацію "
                     "приблизно у %s раза, тоді як за фіксованої кількості відмов квартиль "
                     "найкритичніших конфігурацій коштує лише приблизно у %s раза більше за "
                     "квартиль найменш критичних. Отже, кількість одночасних відмов переважує "
                     "ідентичність голів приблизно у %s раза."
                     % (fmt(max(gfac), 0), fmt(max(idfac), 2),
                        fmt(max(gfac) / max(max(idfac), 1e-9), 0)))
                para("Тому бюджет відмовостійкості для цієї моделі має купувати обмеження на "
                     "те, скільки голів може відмовити одночасно, а не вибіркове зміцнення "
                     "кількох обраних. Ідентичність не є неістотною — її варто враховувати за "
                     "найсуворіших порогів деградації, де безпечна межа справді виключає "
                     "відмови критичних голів, — але це поправка другого порядку до ефекту "
                     "першого порядку, яким є кількість відмов. Вибіркове зміцнення виправдане "
                     "там, де критичність зосереджена, а чи є вона зосередженою — властивість, "
                     "яку слід вимірювати для кожної моделі тим самим перебором одиничних "
                     "відмов, що використано тут, а не припускати.")
            else:
                para("Інженерний висновок: захист варто зосереджувати на невеликому визначеному "
                     "наборі критичних голів, а не розподіляти рівномірно по всіх %d. "
                     "Надлишковість значна, але не однорідна, і винятки можна виявити "
                     "заздалегідь саме тим перебором одиничних відмов, який використано тут."
                     % n_heads)


# --------------------------------------------------------------------------
# build
# --------------------------------------------------------------------------
FAMILY_LABEL = {
    "base_transformer": ("Base Transformer (custom encoder)",
                         "Базовий Transformer (власний енкодер)"),
    "tiny_vit": ("Tiny ViT shape (Hezbri et al., Table 6)",
                 "Форма Tiny ViT (Hezbri et al., табл. 6)"),
    "tiny_gpt2": ("Tiny GPT-2 shape (Hezbri et al., Table 4)",
                  "Форма Tiny GPT-2 (Hezbri et al., табл. 4)"),
}


def architecture(experiment: str, s0: dict, en: bool):
    """Intro paragraph, Table A rows and forward-pass steps for one model.

    Everything is derived from the model_config recorded in summary.json, so the
    text matches what was actually trained (faulttol/model.py) rather than a
    fixed description of one family.
    """
    mc = s0.get("model_config", {})
    d, H, L = mc.get("d_model", 0), mc.get("num_heads", 0), mc.get("num_layers", 0)
    d_ff = mc.get("d_ff", 0)
    dk = d // H if H else 0
    act = mc.get("activation", "-")
    pre = bool(mc.get("norm_first"))
    causal = bool(mc.get("causal"))
    cls_last = bool(mc.get("cls_last", causal))
    learned = bool(mc.get("learned_pe"))
    qkv_bias = mc.get("qkv_bias", True) is not False
    bias = mc.get("bias", True) is not False
    p = mc.get("dropout", "-")

    if experiment == "ag_news":
        n_tok = s0.get("max_len", "-")
        n_out = len(s0.get("class_names") or []) or 4
        adapter = ("token embedding nn.Embedding(%s, %d), one vector per word token"
                   % ("{:,}".format(s0.get("vocab_size") or 0), d) if en else
                   "вкладення токенів nn.Embedding(%s, %d), один вектор на токен слова"
                   % ("{:,}".format(s0.get("vocab_size") or 0), d))
        tok_en, tok_ua = "word tokens", "токенів слів"
        head = "Linear(%d → %d), class logits" % (d, n_out) if en else \
               "Linear(%d → %d), логіти класів" % (d, n_out)
    elif experiment == "cifar10":
        ps = s0.get("patch_size", 4)
        n_tok = s0.get("n_tokens", "-")
        n_out = len(s0.get("class_names") or []) or 10
        adapter = ("patch embedding Conv2d(3, %d, kernel %d, stride %d): non-overlapping "
                   "%dx%d patches, i.e. a linear projection of each patch (ViT stem, not a CNN)"
                   % (d, ps, ps, ps, ps) if en else
                   "вкладення патчів Conv2d(3, %d, ядро %d, крок %d): патчі %dx%d без "
                   "перекриття, тобто лінійна проєкція кожного патча (вхід ViT, не CNN)"
                   % (d, ps, ps, ps, ps))
        tok_en, tok_ua = "image patches", "патчів зображення"
        head = "Linear(%d → %d), class logits" % (d, n_out) if en else \
               "Linear(%d → %d), логіти класів" % (d, n_out)
    else:
        n_tok = s0.get("window_size", "-")
        adapter = ("scalar projection Linear(1, %d) applied to each hourly load value "
                   "(no convolution over time)" % d if en else
                   "скалярна проєкція Linear(1, %d) для кожного погодинного значення "
                   "навантаження (без згортки за часом)" % d)
        tok_en, tok_ua = "hourly time steps", "погодинних кроків"
        head = "Linear(%d → 1), next-hour load" % d if en else \
               "Linear(%d → 1), навантаження наступної години" % d

    cls_pos = ("last" if cls_last else "first") if en else \
              ("остання" if cls_last else "перша")
    if en:
        biases = ("none (no biases anywhere, LayerNorms included)" if not bias else
                  "everywhere except the q, k, v projections" if not qkv_bias else
                  "on every linear layer and LayerNorm")
    else:
        biases = ("відсутні повністю, зокрема в LayerNorm" if not bias else
                  "усюди, крім проєкцій q, k, v" if not qkv_bias else
                  "на кожному лінійному шарі та LayerNorm")

    if en:
        kind = ("a decoder-only (causal, GPT-2-style)" if causal
                else "an encoder-only (bidirectional)")
        intro = (
            "The model is %s Transformer with %d layers of %d attention heads. Each layer "
            "applies multi-head self-attention followed by a position-wise feed-forward "
            "network, a two-layer MLP applied to every token independently. Each of the two "
            "blocks is wrapped in a residual connection with layer normalisation %s. A learned "
            "CLS token, placed %s in the sequence, carries the prediction, and positional "
            "information is a %s encoding. Self-attention is the only operation that mixes "
            "information across positions: there is no recurrent or convolutional layer over "
            "the sequence. Within a family only the input adapter, the output head and the "
            "activation change between datasets."
            % (kind, L, H,
               "applied before the block, plus a final normalisation after the stack (pre-norm)"
               if pre else "applied after the residual sum (post-norm)",
               cls_pos, "learnable absolute" if learned else "fixed sinusoidal"))
        if causal:
            intro += (" Each query attends only to its own position and to earlier ones, so "
                      "the CLS token is appended last, where it can see the whole input. The "
                      "model uses the GPT-2 block but is trained as a classifier or regressor "
                      "read from that token, not as a next-token language model.")
        rows = [
            ["Architecture family", FAMILY_LABEL.get(FAMILY, (FAMILY, FAMILY))[0]],
            ["Attention", "causal (decoder-only)" if causal else "bidirectional (encoder-only)"],
            ["Layers", L],
            ["Heads per layer", H],
            ["Total attention heads", s0["n_heads"]],
            ["Embedding dimension d_model", d],
            ["Head width d_k", dk],
            ["Feed-forward network", "2-layer MLP: Linear(%d → %d) → %s → Dropout → "
                                     "Linear(%d → %d)" % (d, d_ff, act, d_ff, d)],
            ["Feed-forward activation", act],
            ["Normalisation placement", "pre-norm, with a final norm" if pre else "post-norm"],
            ["Positional encoding", "learnable absolute" if learned else "fixed sinusoidal"],
            ["Biases", biases],
            ["Input adapter", adapter],
            ["Sequence length", "%s %s + 1 CLS token" % (n_tok, tok_en)],
            ["Readout", "CLS token, %s position" % cls_pos],
            ["Output head", head],
            ["Dropout", "%s after the embedding, on the attention-block output, inside the "
                        "FFN and on the FFN output; none on the attention weights" % p],
            ["Fault-injection point", "per-head mask on attn·V, before concatenation and W_o"],
            ["Trainable parameters", "{:,}".format(s0.get("n_params") or 0)],
        ]
        mha = ("Q, K, V = x·W_q, x·W_k, x·W_v, split into %d heads of width %d → "
               "softmax(Q·Kᵀ/√%d%s) · V → head mask (fault injection) → concatenate → W_o."
               % (H, dk, dk, " + causal mask" if causal else ""))
        if pre:
            layer = ("Repeat %d times: x = x + Dropout(MHA(LayerNorm(x))); "
                     "x = x + Dropout(FFN(LayerNorm(x))).") % L
        else:
            layer = ("Repeat %d times: x = LayerNorm(x + Dropout(MHA(x))); "
                     "x = LayerNorm(x + Dropout(FFN(x))).") % L
        steps = [
            "Input adapter: %s → (batch, %s, %d)." % (adapter, n_tok, d),
            "Add the CLS token at the %s position, add the positional encoding, apply "
            "Dropout." % cls_pos,
            layer,
            "Multi-head attention (MHA) inside each layer: " + mha,
        ]
        if pre:
            steps.append("Final LayerNorm after the last layer.")
        steps.append("Take the CLS token and apply the output head: %s." % head)
    else:
        kind = ("лише-декодерний (каузальний, як GPT-2)" if causal
                else "лише-енкодерний (двонапрямлений)")
        intro = (
            "Модель — %s Transformer із %d шарів по %d голів уваги. Кожен шар застосовує "
            "багатоголову самоувагу, після якої йде позиційна повнозв’язна мережа — "
            "двошаровий MLP, що діє на кожен токен окремо. Кожен із двох блоків обгорнуто "
            "залишковим зв’язком із шаровою нормалізацією %s. Прогноз переносить навчений "
            "токен CLS, розміщений на %s позиції, а позиційна інформація задана %s "
            "кодуванням. Самоувага — єдина операція, що змішує інформацію між позиціями: "
            "рекурентних чи згорткових шарів над послідовністю немає. У межах однієї родини "
            "між наборами даних змінюються лише вхідний адаптер, вихідна голова та активація."
            % (kind, L, H,
               "перед блоком і фінальною нормалізацією після стеку (пре-норма)"
               if pre else "після залишкової суми (пост-норма)",
               cls_pos, "навчуваним абсолютним" if learned else "фіксованим синусоїдним"))
        if causal:
            intro += (" Кожен запит бачить лише свою та попередні позиції, тому токен CLS "
                      "додано останнім, звідки він бачить увесь вхід. Модель використовує "
                      "блок GPT-2, але навчена як класифікатор або регресор, що читає цей "
                      "токен, а не як мовна модель наступного токена.")
        rows = [
            ["Родина архітектури", FAMILY_LABEL.get(FAMILY, (FAMILY, FAMILY))[1]],
            ["Увага", "каузальна (лише декодер)" if causal else "двонапрямлена (лише енкодер)"],
            ["Шари", L],
            ["Голів на шар", H],
            ["Усього голів уваги", s0["n_heads"]],
            ["Розмірність вкладень d_model", d],
            ["Ширина голови d_k", dk],
            ["Повнозв’язна мережа", "двошаровий MLP: Linear(%d → %d) → %s → Dropout → "
                                    "Linear(%d → %d)" % (d, d_ff, act, d_ff, d)],
            ["Активація повнозв’язного блоку", act],
            ["Розташування нормалізації",
             "пре-норма, з фінальною нормалізацією" if pre else "пост-норма"],
            ["Позиційне кодування", "навчуване абсолютне" if learned else "фіксоване синусоїдне"],
            ["Зсуви", biases],
            ["Вхідний адаптер", adapter],
            ["Довжина послідовності", "%s %s + 1 токен CLS" % (n_tok, tok_ua)],
            ["Зчитування", "токен CLS, %s позиція" % cls_pos],
            ["Вихідна голова", head],
            ["Dropout", "%s після вкладень, на виході блоку уваги, всередині та на виході "
                        "FFN; на вагах уваги — немає" % p],
            ["Точка введення відмов", "маска голови на attn·V, до конкатенації та W_o"],
            ["Навчальних параметрів", "{:,}".format(s0.get("n_params") or 0)],
        ]
        mha = ("Q, K, V = x·W_q, x·W_k, x·W_v, поділ на %d голів ширини %d → "
               "softmax(Q·Kᵀ/√%d%s) · V → маска голів (введення відмов) → конкатенація → W_o."
               % (H, dk, dk, " + каузальна маска" if causal else ""))
        if pre:
            layer = ("Повторити %d разів: x = x + Dropout(MHA(LayerNorm(x))); "
                     "x = x + Dropout(FFN(LayerNorm(x))).") % L
        else:
            layer = ("Повторити %d разів: x = LayerNorm(x + Dropout(MHA(x))); "
                     "x = LayerNorm(x + Dropout(FFN(x))).") % L
        steps = [
            "Вхідний адаптер: %s → (batch, %s, %d)." % (adapter, n_tok, d),
            "Додати токен CLS на %s позицію, додати позиційне кодування, застосувати "
            "Dropout." % ("останню" if cls_last else "першу"),
            layer,
            "Багатоголова увага (MHA) у кожному шарі: " + mha,
        ]
        if pre:
            steps.append("Фінальна LayerNorm після останнього шару.")
        steps.append("Взяти токен CLS і застосувати вихідну голову: %s." % head)
    return intro, rows, steps


def criticality_text(s0: dict, en: bool) -> list[tuple[str, str]]:
    """Definition of head criticality, placed just before Figure 1.

    Returns (kind, text) items: "h" a bold lead-in, "p" a paragraph, "eq" a
    centred formula.  Matches metrics.py / single_head.py (see METRICS.md 4-6).
    """
    reg = s0.get("criticality_col") == "mean_harmful_error_change"
    N = s0.get("n_fault_eval_samples") or 0
    M = s0.get("n_heads") or 0
    k = len(s0.get("critical_heads") or []) or max(1, round(0.10 * M))
    if en:
        out = [
            ("h", "What criticality is."),
            ("p", "The criticality C(l, h) of a head is the average harm the model suffers "
                  "when that one head is crashed while all other %d heads keep working. "
                  "The head in layer l, head index h (written Ll_Hh, both counted from 0) has "
                  "its output set to zero, the fixed evaluation subset of N = %s samples is "
                  "run through the model, and each sample's prediction is compared with the "
                  "fault-free prediction for the same sample. This is repeated for every head, "
                  "so the %d values are directly comparable: each differs from the clean model "
                  "in exactly one head." % (M - 1, "{:,}".format(N), M)),
        ]
        if not reg:
            out += [
                ("p", "For classification the per-sample damage is the probability the model "
                      "loses on the true class y_i:"),
                ("eq", "D_i(l, h) = p_clean(y_i | x_i) − p_(l,h) crashed(y_i | x_i)"),
                ("eq", "C(l, h) = (1/N) · Σ_i max(D_i(l, h), 0)"),
                ("p", "C is a probability, between 0 and 1. C = 0.01 means that crashing this "
                      "head costs the correct class one percentage point of probability on an "
                      "average sample (recorded as D_pos_mean)."),
            ]
        else:
            out += [
                ("p", "For load forecasting the per-sample damage is the increase in absolute "
                      "error of the next-hour prediction, on the min-max normalised series:"),
                ("eq", "e_i(l, h) = |y_i − ŷ_i, (l,h) crashed| − |y_i − ŷ_i, clean|"),
                ("eq", "C(l, h) = (1/N) · Σ_i max(e_i(l, h), 0)"),
                ("p", "C is in normalised load units (0 to 1 spans the series' minimum to "
                      "maximum), so it is not comparable with the classification experiments "
                      "(recorded as mean_harmful_error_change)."),
            ]
        out += [
            ("p", "Only harm is counted. A crash occasionally helps a sample (negative damage), "
                  "and a signed average would let that help cancel the harm, making a "
                  "disruptive head look safe. The max(·, 0) keeps the measure one-sided."),
            ("p", "Two quantities are derived from C. The severity weight "
                  "W(l, h) = C(l, h) / mean C rescales criticality so that an average head "
                  "has W = 1, and the critical heads are the %d heads with the largest C "
                  "(the top 10%% of %d). Figure 1 shows C for every head: rows are layers, "
                  "columns are head indices, and darker cells are more critical." % (k, M)),
        ]
    else:
        out = [
            ("h", "Що таке критичність."),
            ("p", "Критичність C(l, h) голови — це середня шкода для моделі, коли відмовляє "
                  "лише ця одна голова, а решта %d голів працюють. Вихід голови шару l з "
                  "індексом h (позначення Ll_Hh, обидва з 0) замінюється нулями, через модель "
                  "проганяється фіксована підмножина оцінювання з N = %s прикладів, і прогноз "
                  "для кожного прикладу порівнюється з прогнозом без відмови. Це повторюється "
                  "для кожної голови, тож усі %d значень безпосередньо порівнювані: кожне "
                  "відрізняється від чистої моделі рівно однією головою."
                  % (M - 1, "{:,}".format(N).replace(",", " "), M)),
        ]
        if not reg:
            out += [
                ("p", "Для класифікації шкода на прикладі — це ймовірність, яку модель "
                      "втрачає на правильному класі y_i:"),
                ("eq", "D_i(l, h) = p_чиста(y_i | x_i) − p_(l,h) відмова(y_i | x_i)"),
                ("eq", "C(l, h) = (1/N) · Σ_i max(D_i(l, h), 0)"),
                ("p", "C — це ймовірність, від 0 до 1. C = 0,01 означає, що відмова голови "
                      "забирає в правильного класу один процентний пункт ймовірності на "
                      "середньому прикладі (у файлах — D_pos_mean)."),
            ]
        else:
            out += [
                ("p", "Для прогнозу навантаження шкода на прикладі — це приріст абсолютної "
                      "похибки прогнозу на наступну годину на мін-макс нормалізованому ряді:"),
                ("eq", "e_i(l, h) = |y_i − ŷ_i, (l,h) відмова| − |y_i − ŷ_i, чиста|"),
                ("eq", "C(l, h) = (1/N) · Σ_i max(e_i(l, h), 0)"),
                ("p", "C вимірюється в нормалізованих одиницях навантаження (від 0 до 1 між "
                      "мінімумом і максимумом ряду), тому не порівнюється з класифікаційними "
                      "експериментами (у файлах — mean_harmful_error_change)."),
            ]
        out += [
            ("p", "Враховується лише шкода. Іноді відмова допомагає прикладу (від’ємна шкода), "
                  "і знакове середнє дозволило б цій допомозі компенсувати шкоду, через що "
                  "руйнівна голова виглядала б безпечною. Операція max(·, 0) робить міру "
                  "однобічною."),
            ("p", "З C виводяться дві величини. Ваговий коефіцієнт тяжкості "
                  "W(l, h) = C(l, h) / середнє C нормує критичність так, що середня голова має "
                  "W = 1, а критичні голови — це %d голів із найбільшим C (верхні 10%% з %d). "
                  "Рисунок 1 показує C для кожної голови: рядки — шари, стовпці — індекси "
                  "голів, темніші клітинки критичніші." % (k, M)),
        ]
    return out


def add_criticality_text(doc, items) -> None:
    for kind, text in items:
        p = doc.add_paragraph()
        r = p.add_run(text)
        if kind == "h":
            r.bold = True
        elif kind == "eq":
            r.italic = True
            r.font.name = "Cambria Math"
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER


def _n(v: int) -> str:
    """Thousands separated by a space, as in a Keras model summary."""
    return "{:,}".format(v).replace(",", " ")


def layer_summary(experiment: str, s0: dict, en: bool):
    """Keras-style layer table and a breakdown of one Transformer block.

    Returns (header, layer_rows, block_rows).  Parameter counts follow
    faulttol/model.py exactly and are checked against the recorded n_params.
    """
    mc = s0.get("model_config", {})
    d, H, L = mc["d_model"], mc["num_heads"], mc["num_layers"]
    d_ff = mc["d_ff"]
    dk = d // H
    act = {"relu": "ReLU", "gelu": "GELU", "silu": "SiLU (Swish)"}.get(
        mc.get("activation", ""), mc.get("activation", "-"))
    pre = bool(mc.get("norm_first"))
    causal = bool(mc.get("causal"))
    cls_last = bool(mc.get("cls_last", causal))
    learned = bool(mc.get("learned_pe"))
    qkv_bias = mc.get("qkv_bias", True) is not False
    bias = mc.get("bias", True) is not False

    def u(e, a):
        return e if en else a

    # ---- input adapter and output head (always biased, see model.py) ----
    if experiment == "ag_news":
        T = int(s0["max_len"])
        V = int(s0["vocab_size"])
        C = len(s0.get("class_names") or []) or 4
        inp = [u("input", "вхід"), u("token ids (%d)" % T, "ідентифікатори токенів (%d)" % T),
               "(%d)" % T, 0]
        adapter = [u("embedding", "вкладення"),
                   "Embedding(%s → %d)" % (_n(V), d), "(%d, %d)" % (T, d), V * d]
        out = [u("output", "вихід"), u("Dense %d, softmax logits" % C,
                                      "Dense %d, логіти softmax" % C), "(%d)" % C, d * C + C]
    elif experiment == "cifar10":
        p = int(s0.get("patch_size", 4))
        T = int(s0["n_tokens"])
        C = len(s0.get("class_names") or []) or 10
        inp = [u("input", "вхід"), u("image (3, 32, 32)", "зображення (3, 32, 32)"),
               "(3, 32, 32)", 0]
        adapter = [u("patch embedding", "вкладення патчів"),
                   u("Conv2d %dx%d, stride %d (linear patch projection)" % (p, p, p),
                     "Conv2d %dx%d, крок %d (лінійна проєкція патча)" % (p, p, p)),
                   "(%d, %d)" % (T, d), 3 * p * p * d + d]
        out = [u("output", "вихід"), u("Dense %d, softmax logits" % C,
                                      "Dense %d, логіти softmax" % C), "(%d)" % C, d * C + C]
    else:
        T = int(s0["window_size"])
        inp = [u("input", "вхід"), u("window (%d)" % T, "вікно (%d)" % T), "(%d)" % T, 0]
        adapter = [u("input projection", "вхідна проєкція"),
                   u("Dense %d per time step, linear" % d,
                     "Dense %d на кожен крок, лінійна" % d), "(%d, %d)" % (T, d), 2 * d]
        out = [u("output", "вихід"), u("Dense 1, linear", "Dense 1, лінійна"), "(1)", d + 1]
    S = T + 1  # tokens plus CLS

    pe_params = S * d if learned else 0
    cls_row = [u("CLS + position", "CLS + позиція"),
               u("CLS token (%s) + %s positional encoding, Dropout"
                 % ("last" if cls_last else "first",
                    "learnable" if learned else "sinusoidal"),
                 "токен CLS (%s) + %s позиційне кодування, Dropout"
                 % ("останній" if cls_last else "перший",
                    "навчуване" if learned else "синусоїдне")),
               "(%d, %d)" % (S, d), d + pe_params]

    # ---- one Transformer block ----
    attn = 4 * d * d + (3 * d if qkv_bias else 0) + (d if bias else 0)
    ff1 = d * d_ff + (d_ff if bias else 0)
    ff2 = d_ff * d + (d if bias else 0)
    ln = d * (2 if bias else 1)
    blk = attn + ff1 + ff2 + 2 * ln
    shape = "(%d, %d)" % (S, d)
    norm = u("pre-norm" if pre else "post-norm", "пре-норма" if pre else "пост-норма")
    kind = u("causal " if causal else "", "каузальна " if causal else "")
    blk_type = u("Transformer block: %sMHA %dx%d + FFN %d, %s, %s"
                 % (kind, H, dk, d_ff, act, norm),
                 "блок Transformer: %sMHA %dx%d + FFN %d, %s, %s"
                 % (kind, H, dk, d_ff, act, norm))

    rows = [inp, adapter, cls_row]
    rows += [["L%d" % (i + 1), blk_type, shape, blk] for i in range(L)]
    total = adapter[3] + cls_row[3] + L * blk + out[3]
    if pre:
        rows.append([u("final norm", "фінальна норма"), "LayerNorm", shape, ln])
        total += ln
    rows.append([u("readout", "зчитування"), u("take CLS token", "взяти токен CLS"),
                 "(%d)" % d, 0])
    rows.append(out)
    if total != s0.get("n_params"):
        raise ValueError("layer_summary: computed %d parameters, summary.json records %s"
                         % (total, s0.get("n_params")))
    rows = [[a, b, c, _n(n)] for a, b, c, n in rows]
    rows.append([u("Total (all trainable)", "Усього (усі навчальні)"), "", "", _n(total)])

    ln_row = lambda name: [name, "LayerNorm" + ("" if bias else u(" (no bias)", " (без зсуву)")),
                           shape, _n(ln)]
    block = []
    if pre:
        block.append(ln_row("LayerNorm 1"))
    block += [
        ["MHA", u("%sself-attention, %d heads x %d: W_q, W_k, W_v, W_o%s"
                  % (kind, H, dk, ", no biases" if not bias else
                     "" if qkv_bias else ", no q/k/v bias"),
                  "%sсамоувага, %d голів x %d: W_q, W_k, W_v, W_o%s"
                  % (kind, H, dk, ", без зсувів" if not bias else
                     "" if qkv_bias else ", без зсуву q/k/v")),
         shape, _n(attn)],
        [u("residual", "залишок"), u("Dropout + add", "Dropout + додавання"), shape, "0"],
    ]
    if not pre:
        block.append(ln_row("LayerNorm 1"))
    else:
        block.append(ln_row("LayerNorm 2"))
    block += [
        ["FFN 1", "Dense %d, %s" % (d_ff, act), "(%d, %d)" % (S, d_ff), _n(ff1)],
        ["FFN 2", u("Dropout, Dense %d, linear" % d, "Dropout, Dense %d, лінійна" % d),
         shape, _n(ff2)],
        [u("residual", "залишок"), u("Dropout + add", "Dropout + додавання"), shape, "0"],
    ]
    if not pre:
        block.append(ln_row("LayerNorm 2"))
    block.append([u("Block total", "Усього в блоці"), "", "", _n(blk)])

    hdr = (["Layer", "Type, activation", "Output", "Params"] if en else
           ["Шар", "Тип, активація", "Вихід", "Параметри"])
    return hdr, rows, block


def build(experiment: str, lang: str, units: list[dict], F: list[dict],
          path: pathlib.Path) -> None:
    en = lang == "en"
    T = T_EN if en else T_UA
    M = META[experiment][lang]
    s0 = units[0]
    multi = len(units) > 1

    mc0 = s0.get("model_config", {})
    fam = FAMILY_META.get(FAMILY, FAMILY_META["base_transformer"])[lang]
    doc = new_document(M["title"] + fam["suffix"],
                       T["subtitle"] % (mc0.get("num_layers", 0),
                                        mc0.get("num_heads", 0), s0["n_heads"]))

    doc.add_heading(T["h_purpose"], level=1)
    for p in M["purpose"]:
        doc.add_paragraph(p)
    if fam["note"]:
        doc.add_paragraph(fam["note"])

    doc.add_heading(T["h_questions"], level=1)
    add_numbered(doc, T["questions"])
    doc.add_paragraph("Hypotheses tested:" if en else "Перевірені гіпотези:")
    add_bullets(doc, T["hypotheses"])

    doc.add_heading(T["h_data"], level=1)
    add_table(doc, ["Property", "Value"] if en else ["Властивість", "Значення"], M["data"])

    doc.add_heading(T["h_model"], level=1)
    intro, arch, steps = architecture(experiment, s0, en)
    doc.add_paragraph(intro)
    tc = s0.get("train_config", {})
    add_table(doc, ["Property", "Value"] if en else ["Властивість", "Значення"], arch,
              "Table A. Architecture, as recorded in the result file." if en else
              "Таблиця A. Архітектура за записом у файлі результатів.")
    doc.add_paragraph("Forward pass:" if en else "Прямий прохід:")
    add_bullets(doc, steps)
    hdr, layers, block = layer_summary(experiment, s0, en)
    add_table(doc, hdr, layers,
              "Table A2. Layer-by-layer summary. Parameter counts are computed from the "
              "configuration and match the recorded trainable-parameter total." if en else
              "Таблиця A2. Пошарове зведення. Кількість параметрів обчислено з конфігурації; "
              "вона збігається із записаною загальною кількістю навчальних параметрів.")
    add_table(doc, hdr, block,
              "Table A3. Inside one Transformer block (every block is identical)." if en else
              "Таблиця A3. Будова одного блоку Transformer (усі блоки однакові).")
    train_rows =[[k, str(v)] for k, v in tc.items()]
    if train_rows:
        add_table(doc, ["Setting", "Value"] if en else ["Налаштування", "Значення"], train_rows,
                  "Table B. Training configuration." if en else
                  "Таблиця B. Конфігурація навчання.")

    doc.add_heading(T["h_fault"], level=1)
    p = doc.add_paragraph(T["fault_def"])
    p.runs[0].italic = True
    add_bullets(doc, T["fault_notes"])
    doc.add_paragraph("Protocol:" if en else "Протокол:")
    add_numbered(doc, [step % s0["n_heads"] if "%d" in step else step
                       for step in T["protocol"]])

    doc.add_heading(T["h_budget"], level=1)
    spe = s0.get("seconds_per_evaluation")
    budget = [
        ["Random fault configurations per model" if en else
         "Випадкових конфігурацій відмов на модель", "{:,}".format(s0.get("k_trials") or 0)],
        ["Fixed evaluation subset" if en else "Фіксована підмножина оцінювання",
         "{:,}".format(s0.get("n_fault_eval_samples") or 0)],
        ["Single-head evaluations per model" if en else
         "Оцінювань з однією відмовою на модель", s0["n_heads"]],
        ["Models trained" if en else "Навчено моделей", len(units)],
        ["Seconds per evaluation" if en else "Секунд на одне оцінювання", fmt(spe, 4)],
    ]
    add_table(doc, ["Protocol parameter", "Value"] if en else ["Параметр протоколу", "Значення"],
              budget)
    if en:
        doc.add_paragraph(
            "The run used an NVIDIA GPU; the notebooks select CUDA automatically when it is "
            "available and fall back to CPU otherwise. TF32 is disabled so that matmuls stay "
            "in full single precision, because the criticality measure resolves damages down "
            "to about 1e-6 and reduced-precision accumulation would inject noise directly "
            "into the quantity being measured.")
    else:
        doc.add_paragraph(
            "Запуск виконано на графічному процесорі NVIDIA; зошити автоматично обирають CUDA, "
            "коли вона доступна, і повертаються до процесора в іншому разі. TF32 вимкнено, щоб "
            "матричні множення залишалися в повній одинарній точності: міра критичності "
            "розрізняє шкоду до приблизно 1e-6, і накопичення зі зниженою точністю вносило б "
            "шум безпосередньо у величину, яку вимірюють.")

    doc.add_heading(T["h_results"], level=1)
    results_section(doc, units, F, en, multi)

    doc.add_heading(T["h_analysis"], level=1)
    analysis(doc, units, F, en, multi)

    doc.add_heading(T["h_limits"], level=1)
    add_bullets(doc, T["limits"])

    doc.add_heading(T["h_repro"], level=1)
    nb = "notebooks/%s/%s.ipynb" % (FAMILY, experiment)
    res = "results/%s/%s/" % (FAMILY, experiment)
    if en:
        add_bullets(doc, [
            "The experiment is the notebook %s. It runs top to "
            "bottom and writes its tables, figures and a machine-readable summary to "
            "%s." % (nb, res),
            "The fault model, the criticality measures, the fault-tolerance regions and the "
            "plots all come from the shared library src/faulttol/, so every dataset in "
            "this repository is analysed by exactly the same code.",
            "Protocol parameters are read from environment variables, so a run can be "
            "re-budgeted without editing a cell. FT_K_TRIALS and FT_N_FAULT set the fault "
            "budget, FT_EPOCHS and FT_PATIENCE the training budget, and FT_DEVICE forces the "
            "device. The defaults reproduce the full protocol of the source study.",
            "This document was generated from the result files by "
            "scripts/make_experiment_report.py and can be regenerated at any time without "
            "re-running the experiment.",
        ])
    else:
        add_bullets(doc, [
            "Експеримент — це зошит %s. Він виконується від "
            "початку до кінця та записує таблиці, рисунки й машинозчитуване "
            "зведення до %s." % (nb, res),
            "Модель відмови, міри критичності, області відмовостійкості та побудова графіків "
            "походять зі спільної бібліотеки src/faulttol/, тому кожен набір даних у "
            "цьому репозиторії аналізується точно тим самим кодом.",
            "Параметри протоколу читаються зі змінних середовища, тому бюджет запуску можна "
            "змінити, не редагуючи клітинки. FT_K_TRIALS і FT_N_FAULT задають бюджет відмов, "
            "FT_EPOCHS і FT_PATIENCE — бюджет навчання, а FT_DEVICE примусово задає пристрій. "
            "Типові значення відтворюють повний протокол первинного дослідження.",
            "Цей документ згенеровано з файлів результатів скриптом "
            "scripts/make_experiment_report.py; його можна відтворити будь-коли без "
            "повторного запуску експерименту.",
        ])

    doc.save(str(path))
    try:
        shown = path.relative_to(ROOT)
    except ValueError:
        shown = path
    print("wrote %s" % shown)


def main() -> None:
    global FAMILY, BASE, REPORTS
    import argparse
    ap = argparse.ArgumentParser(description="Build per-experiment Word reports.")
    ap.add_argument("experiments", nargs="*", help="subset of %s" % ", ".join(EXPERIMENTS))
    ap.add_argument("--family", default="base_transformer", choices=FAMILIES)
    ap.add_argument("--lang", default="en", help="comma-separated: en, ua (default: en)")
    args = ap.parse_args()
    langs = [l.strip().lower() for l in args.lang.split(",") if l.strip()]
    for l in langs:
        if l not in ("en", "ua"):
            raise SystemExit("unknown language: %s (known: en, ua)" % l)

    FAMILY = args.family
    BASE = ROOT / "results" / FAMILY
    REPORTS = ROOT / "reports" / FAMILY
    REPORTS.mkdir(parents=True, exist_ok=True)
    print("family: %s" % FAMILY)
    for experiment in args.experiments or EXPERIMENTS:
        if experiment not in META:
            print("unknown experiment: %s" % experiment)
            continue
        units = load_units(experiment)
        if not units:
            print("%s: no summary.json yet, skipping" % experiment)
            continue
        F = [facts_for(s) for s in units]
        print("%s: %d model(s)" % (experiment, len(units)))
        for lang in langs:
            out = REPORTS / ("%s_report_%s.docx" % (experiment, lang.upper()))
            build(experiment, lang, units, F, out)


if __name__ == "__main__":
    main()
