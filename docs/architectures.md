# Architectures

The fault protocol is identical in all three families; only the architecture
changes. The notebooks of `tiny_vit` and `tiny_gpt2` are **derived** from the
`base_transformer` ones by `scripts/derive_notebooks.py` (see
[running.md](running.md#regenerating-the-derived-notebooks)), so every data,
fault-injection, analysis and plotting cell is byte-identical across families.
All three configurations are in `src/faulttol/config.py` (`BASE_PARAM`,
`TINY_VIT_PARAM`, `TINY_GPT2_PARAM`).

| | `base_transformer` | `tiny_vit` | `tiny_gpt2` |
|---|---|---|---|
| Layers x heads | 4 x 8 = **32** | 6 x 6 = **36** | 6 x 6 = **36** |
| Width / FFN | 256 / 1024 | 384 / 1536 | 384 / 1536 |
| Head width `d_k` | 32 | 64 | 64 |
| Normalisation | post-norm | pre-norm + final norm | pre-norm + final norm |
| Positional encoding | fixed sinusoidal | learnable absolute | learnable absolute |
| q/k/v biases | present | absent | absent |
| `W_o`, FFN, LayerNorm biases | present | present | **absent** |
| Attention | bidirectional | bidirectional | **causal** |
| Read-out token | CLS at position 0 | CLS at position 0 | **CLS at last position** |
| Activation | ReLU (text, image), SiLU (load) | GELU | GELU |
| Parameters (encoder stack) | 3,159,040 | 10,639,872 | 10,621,440 |

How to read the contrasts:

- **`base_transformer` -> `tiny_vit`** is a *bundled* contrast: seven choices
  change at once, so a difference localises to "that architecture", not to
  depth, width or normalisation individually.
- **`tiny_vit` -> `tiny_gpt2`** is very nearly a *single-factor* test of
  bidirectional versus causal attention (the only other change removes 0.17% of
  the parameters).

The `base_transformer` design itself is described in
[study_design.md](study_design.md#the-model).

---

## Tiny ViT (`tiny_vit`)

The architecture is the Tiny ViT of *Dropout and the Outliers* (Hezbri,
Bareilles & El-Mhamdi, SciForDL 2nd edition; the paper is not
included in this repository), defined in `src/faulttol/config.py` as `TINY_VIT_PARAM`.

### What changes, and what deliberately does not

| | `base_transformer` | this family |
|---|---|---|
| Layers x heads | 4 x 8 = **32** heads | 6 x 6 = **36** heads |
| Width / FFN | 256 / 1024 | **384 / 1536** |
| Head width `d_k` | 32 | **64** |
| Normalisation | post-norm | **pre-norm**, plus a final norm |
| Positional encoding | fixed sinusoidal | **learnable absolute** |
| q/k/v biases | present | **absent** (`W_o` keeps its bias) |
| Activation | ReLU (text, image), SiLU (load) | **GELU** |
| Parameters (encoder stack) | 3,159,040 | **10,639,872** (3.37x) |

Unchanged on purpose: the datasets and splits, the seeds, the training recipe
(optimizer, learning rate, weight decay, epochs, batch size, early stopping,
augmentation, label smoothing, cosine schedule), the fault model, the
criticality and redundancy definitions, the severity descriptors, the
fault-tolerance regions and the evaluation subsets. The degradation numbers are
therefore directly comparable with the base family for the same dataset.

**Read a difference against the base family as a property of this
architecture, not of any one choice within it.** Seven things change together.
Isolating depth, or width, or normalisation placement on its own would need
further runs that vary one key of `TINY_VIT_PARAM` at a time; the machinery
supports that, since every key is read from the config.

One further caveat on the training recipe. Keeping it identical is what makes
the comparison clean, but it was tuned for the base architecture: the learning
rate of 6e-4 for CIFAR-10 and 3e-4 for AG News is not necessarily right for a
wider pre-norm model, and pre-norm stacks usually tolerate more. If a model
here underfits relative to its base counterpart, that is a recipe artefact
rather than an architecture result, and the honest fix is to retune both
architectures rather than only this one. `FT_EPOCHS`, `FT_PATIENCE` and the
other environment variables from [running.md](running.md#parameters-environment-variables) apply here too.

### Cost

Cost at the full protocol, extrapolated from the base family's measured
0.103 s per evaluation and this architecture's 2.24x forward cost per token
(same sequence lengths, 3.37x the stack):

| Notebook | Training | Fault sweep | Total |
|---|---|---|---|
| `ag_news` | ~15 min | ~38 min | ~55 min |
| `cifar10` | ~19 min | ~38 min | ~1 h |
| `pjm_regions` | ~7.5 min/region | ~13 min/region | ~3.7 h (11 regions) |

---

## Tiny GPT-2 (`tiny_gpt2`)

The architecture is the Tiny GPT-2 of *Dropout and the Outliers* (Hezbri,
Bareilles & El-Mhamdi, SciForDL 2nd edition, Table 4; the paper is not
included in this repository), defined in `src/faulttol/config.py` as
`TINY_GPT2_PARAM`.

### What changes

See the comparison table at the top of this page.

Unchanged on purpose: the datasets and splits, the seeds, the training recipe
(optimizer, learning rate, weight decay, epochs, batch size, early stopping,
augmentation, label smoothing, cosine schedule), the fault model, the
criticality and redundancy definitions, the severity descriptors, the
fault-tolerance regions and the evaluation subsets.

### Why this family is the useful one

The Tiny ViT section above has to warn that seven things change at once, so a
difference there localises only to "that architecture". **This family does not
have that problem.** The paper deliberately gives its Tiny GPT-2, Tiny BERT and
Tiny ViT one shared shape - six blocks, six heads, width 384, MLP expansion 4
(Tables 4, 5 and 6) - so against `tiny_vit` only two things move:

1. **All remaining biases are dropped.** This removes 18,432 of 10,639,872
   stack parameters, or **0.17%**. Nearly free, and very unlikely to move any
   degradation metric measurably.
2. **Attention becomes causal.** Every query attends to its own position and to
   the left only.

So `tiny_vit` -> `tiny_gpt2` is, to a very good approximation, a **single-factor
test of bidirectional versus autoregressive attention** under head crash
faults - the cleanest attribution available in this repository. Read it that
way, and read `base_transformer` -> `tiny_vit` as the bundled contrast it is.

### Two consequences of causal masking

**The read-out token moved.** Under a causal mask a CLS token at position 0
could attend only to itself, so `_EncoderBackbone` appends it at the end and
reads the last position instead (`cls_last`, implied by `causal`). The
injection, criticality and safe-region code is position-agnostic, so nothing
else needed changing; `model.py` raises rather than silently building the
broken `causal=True, cls_last=False` combination.

**`max_cumulative_attention` is not comparable across families.**
`attention_stats.csv` reports, per head, the attention each key receives summed
over all queries. Under a causal mask key `j` is visible only to queries `j`
and later, so its attainable ceiling falls from `L` to `L - j`, while key 0
keeps the full `L`. The maximum over keys is therefore structurally biased
toward early positions in a way that has nothing to do with the head. Compare
that column **within this family only**.

Note separately that `mean_cumulative_attention` carries no information in
*any* family, causal or not: every query row of the attention map sums to 1, so
`L` queries spread total mass `L` over `L` keys and the mean column sum is
exactly 1.0 by construction. It is 1.000000 for all 36 heads in
`results/tiny_vit/ag_news/tables/attention_stats.csv`. Only the `max` column varies.

Criticality, degradation and safe-region results are unaffected by both points,
being defined on task metrics rather than on attention mass.

### Caveats

Causal attention over CIFAR-10 patches in raster order is an unusual thing to
do to an image, though it is exactly what iGPT did; for PJM load forecasting it
is the natural choice, and for AG News it is standard. Any accuracy drop on
CIFAR-10 relative to `tiny_vit` should be read as the cost of the causal
constraint on a task with no natural scan order, not as a fault-tolerance
result.

The training recipe is inherited unchanged, which is what makes the comparison
clean, but it was tuned for the base architecture. If a model here underfits
relative to its counterparts, that is a recipe artefact rather than an
architecture result, and the honest fix is to retune all three architectures
rather than only this one. `FT_EPOCHS`, `FT_PATIENCE` and the other environment
variables from [running.md](running.md#parameters-environment-variables) apply here too.

As in `tiny_vit`, the activation is GELU. The paper writes the FFN nonlinearity
generically as "phi (e.g. GELU for GPT-2 models)" (Appendix B.1), so GELU is
stated for GPT-2 there rather than inferred - unlike the ViT case. Dropout is
0.05, sitting between the paper's swept values of 0.0 and 0.2.

### Cost

Cost matches the `tiny_vit` family: same depth, width and head count, 0.17%
fewer parameters, and causal masking adds only a `masked_fill` to each
attention. On the 0.26 s per evaluation measured for `tiny_vit/ag_news`:

| Notebook | Training | Fault sweep | Total |
|---|---|---|---|
| `ag_news` | ~15 min | ~38 min | ~55 min |
| `cifar10` | ~19 min | ~38 min | ~1 h |
| `pjm_regions` | ~7.5 min/region | ~13 min/region | ~3.7 h (11 regions) |
