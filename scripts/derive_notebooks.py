"""Derive the tiny_vit / tiny_gpt2 notebooks from the base_transformer originals.

These families change the architecture and nothing else, so their notebooks are
not written by hand: they are derived from notebooks/base_transformer/ by
rewriting only the results directory, the model name and the architecture
config, and inserting a short architecture note after the title cell.  Every
data-loading, fault-injection, analysis and plotting cell is therefore
byte-identical to the base family, which is what makes the families directly
comparable.

    python scripts/derive_notebooks.py                     # both families
    python scripts/derive_notebooks.py --family tiny_vit   # one family
    python scripts/derive_notebooks.py --out /tmp/derived  # write elsewhere

Re-running is safe: it overwrites the notebooks with freshly derived copies and
CLEARS ALL OUTPUTS (the shipped notebooks carry their stored outputs, so only
re-run this if you changed a base notebook and intend to re-execute).  It
refuses to write if any expected substitution is missing, so a change to the
base notebooks cannot silently produce a notebook that still points at
base_transformer.
"""
from __future__ import annotations

import argparse
import json
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
BASE = ROOT / "notebooks" / "base_transformer"
DATASETS = ["ag_news", "cifar10", "pjm_regions"]

# Per-family substitution values.
FAMILIES = {
    "tiny_vit": {"param": "TINY_VIT_PARAM", "note_id": "tinyvit-note"},
    "tiny_gpt2": {"param": "TINY_GPT2_PARAM", "note_id": "tinygpt2-note"},
}

# Base model name and activation per dataset.
BASE_MODEL_NAME = {
    "ag_news": ('MODEL_NAME = "base_transformer_agnews"', 'MODEL_NAME = "{f}_agnews"'),
    "cifar10": ('MODEL_NAME = "base_transformer_cifar10"', 'MODEL_NAME = "{f}_cifar10"'),
    "pjm_regions": ('MODEL_NAME = "base_transformer_regressor_%s" % region',
                    'MODEL_NAME = "{f}_regressor_%s" % region'),
}
BASE_ACTIVATION = {"ag_news": "relu", "cifar10": "relu", "pjm_regions": "silu"}


def edits_for(family: str, dataset: str) -> list[tuple[str, str, int]]:
    """(old, new, exact number of occurrences expected across the notebook).

    Order matters: the MODEL_CONFIG line also contains the activation kwarg, so
    it is rewritten before the bare activation kwargs of the model constructors.
    """
    act = BASE_ACTIVATION[dataset]
    old_name, new_name = BASE_MODEL_NAME[dataset]
    return [
        ('ROOT / "results" / "base_transformer" / "%s"' % dataset,
         'ROOT / "results" / "%s" / "%s"' % (family, dataset), 1),
        (old_name, new_name.format(f=family), 1),
        ('dict(C.BASE_PARAM, activation="%s")' % act,
         'dict(C.%s, activation="gelu")' % FAMILIES[family]["param"], 1),
        ('activation="%s")' % act, 'activation="gelu")', 1),
    ]


NOTES = {
    "tiny_vit": """## Architecture: Tiny ViT, not the Base encoder

This notebook is a derived copy of the matching notebook in
`notebooks/base_transformer/`. Every dataset, fault-injection and analysis
cell is identical; the only change is the architecture, taken from
`C.TINY_VIT_PARAM` - the Tiny ViT of *Dropout and the Outliers*
(Hezbri et al.): **6 blocks x 6 heads = 36 attention heads**, `d_model` 384,
`d_ff` 1536, pre-normalisation, learnable absolute positional embeddings, no
biases on the query/key/value projections, and GELU.

| | Base family | This family |
|---|---|---|
| Layers x heads | 4 x 8 = 32 heads | **6 x 6 = 36 heads** |
| Width / FFN | 256 / 1024 | **384 / 1536** |
| Head width `d_k` | 32 | **64** |
| Normalisation | post-norm | **pre-norm + final norm** |
| Positional encoding | fixed sinusoidal | **learnable absolute** |
| q/k/v biases | present | **absent** |
| Activation | ReLU (text, image), SiLU (load) | **GELU** |

**Read this as a bundle, not a single-factor test.** Seven choices change at
once, so a difference against the base family localises to "this architecture"
rather than to depth, or width, or normalisation placement on its own.

The training recipe, the fault protocol, the criticality definitions and the
evaluation subsets are deliberately unchanged, so the degradation numbers are
comparable with the base family run for the same dataset.""",
    "tiny_gpt2": """## Architecture: Tiny GPT-2, decoder-only

This notebook is a derived copy of the matching notebook in
`notebooks/base_transformer/`. Every dataset, fault-injection and analysis
cell is identical; the only change is the architecture, taken from
`C.TINY_GPT2_PARAM` - the Tiny GPT-2 of *Dropout and the Outliers*
(Hezbri et al., Table 4): **6 blocks x 6 heads = 36 attention heads**,
`d_model` 384, `d_ff` 1536, pre-normalisation, learnable absolute positional
embeddings, **no biases anywhere**, **causal (left-to-right) attention**, GELU.

| | Base family | Tiny ViT family | This family |
|---|---|---|---|
| Layers x heads | 4 x 8 = 32 | 6 x 6 = 36 | **6 x 6 = 36** |
| Width / FFN | 256 / 1024 | 384 / 1536 | **384 / 1536** |
| Normalisation | post-norm | pre-norm | **pre-norm** |
| Positional encoding | sinusoidal | learnable | **learnable** |
| q/k/v biases | present | absent | **absent** |
| Other biases | present | present | **absent** |
| Attention | bidirectional | bidirectional | **causal** |
| Read-out token | CLS at position 0 | CLS at position 0 | **CLS at last position** |
| Stack parameters | 3,159,040 | 10,639,872 | **10,621,440** |

**Against the Tiny ViT family this is very nearly a single-factor contrast.**
The paper gives its Tiny GPT-2 and Tiny ViT the same six-block, six-head,
width-384 shape, so only two things differ here, and the first is nearly free:
dropping the remaining biases removes 18,432 of 10,639,872 stack parameters
(0.17%). The substantive variable is **causal versus bidirectional attention**.
That makes the Tiny ViT -> Tiny GPT-2 comparison a much cleaner attribution
than the base -> Tiny ViT one, where seven choices move at once.

Two consequences of causal masking to keep in mind when reading the results:

1. **The read-out token moved.** Under a causal mask a CLS token at position 0
   could attend only to itself, so it is appended at the end instead and
   `encode()` reads the last position. Everything else in the pipeline is
   position-agnostic.
2. **`max_cumulative_attention` is not comparable across families.**
   `attention_stats.csv` sums each key's attention over all queries; under a
   causal mask key `j` is visible only to queries `j` and later, so its ceiling
   falls from `L` to `L - j` and the maximum over keys is structurally biased
   toward early positions. Compare that column within this family only.
   (`mean_cumulative_attention` is exactly 1.0 in every family by construction
   - row sums of 1 over `L` queries and `L` keys - so it carries no
   information either way.) The criticality, degradation and safe-region
   results are unaffected: they are defined on task metrics, not attention mass.

The training recipe, the fault protocol, the criticality definitions and the
evaluation subsets are deliberately unchanged, so the degradation numbers are
comparable with the other two families run for the same dataset.""",
}


def apply_edits(nb: dict, edits: list, tag: str) -> None:
    """Rewrite the notebook in place, asserting every edit lands as expected."""
    counts = {old: 0 for old, _, _ in edits}
    for cell in nb["cells"]:
        for i, original in enumerate(cell["source"]):
            line = original
            for old, new, _ in edits:
                # rebind as we go: the MODEL_CONFIG line matches two edits, and
                # replacing against a stale copy would discard the first one
                if old in line:
                    counts[old] += line.count(old)
                    line = line.replace(old, new)
            if line != original:
                cell["source"][i] = line
        if cell["cell_type"] == "code":
            cell["outputs"] = []
            cell["execution_count"] = None

    for old, _, expected in edits:
        if counts[old] != expected:
            raise SystemExit(
                "%s: expected %d occurrence(s) of %r, found %d - the base "
                "notebook changed, so this generator needs updating before it "
                "can be trusted." % (tag, expected, old, counts[old]))


def derive(family: str, dataset: str, out_root: pathlib.Path) -> pathlib.Path:
    src = BASE / ("%s.ipynb" % dataset)
    nb = json.loads(src.read_text(encoding="utf-8"))
    apply_edits(nb, edits_for(family, dataset), "%s/%s" % (family, dataset))

    # architecture note, straight after the notebook's title cell
    lines = NOTES[family].split("\n")
    note = {"cell_type": "markdown", "id": FAMILIES[family]["note_id"], "metadata": {},
            "source": [l + "\n" for l in lines[:-1]] + [lines[-1]]}
    nb["cells"].insert(1, note)

    dst = out_root / family / ("%s.ipynb" % dataset)
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    return dst


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--family", choices=sorted(FAMILIES), action="append",
                    help="family to derive (repeatable; default: all)")
    ap.add_argument("--out", type=pathlib.Path, default=ROOT / "notebooks",
                    help="output root (default: notebooks/)")
    args = ap.parse_args()
    for family in args.family or list(FAMILIES):
        for dataset in DATASETS:
            print("wrote %s" % derive(family, dataset, args.out.resolve()))


if __name__ == "__main__":
    main()
