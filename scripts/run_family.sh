#!/usr/bin/env bash
# Execute one architecture family's three notebooks (or all families) in place,
# so the results and figures are stored inside the notebooks and written to
# results/<family>/<dataset>/.
#
#   bash scripts/run_family.sh base_transformer
#   bash scripts/run_family.sh all
#   FT_K_TRIALS=10000 FT_N_FAULT=1000 bash scripts/run_family.sh tiny_vit   # full protocol
#
# The notebooks pick up an NVIDIA GPU automatically (FT_DEVICE=cpu forces the
# CPU path).  They run one after another: on CPU one process saturates the
# cores, and on one GPU they would contend for the same device.
#
# The fault budget below is reduced from the notebook defaults (K=10000,
# N_fault=1000) to fit a CPU-only run, and is overridable.  The per-dataset
# training budgets are set inline and are the same for every family.  Every
# value used is recorded in each summary.json.
#
# Requires the Jupyter kernel "ft-python" - see docs/setup.md.
set -u

FAMILIES=(base_transformer tiny_vit tiny_gpt2)
target="${1:-}"
if [[ -z "$target" ]]; then
  echo "usage: $0 <base_transformer|tiny_vit|tiny_gpt2|all>" >&2
  exit 2
fi
if [[ "$target" == "all" ]]; then
  selected=("${FAMILIES[@]}")
elif [[ " ${FAMILIES[*]} " == *" $target "* ]]; then
  selected=("$target")
else
  echo "unknown family: $target (known: ${FAMILIES[*]}, all)" >&2
  exit 2
fi

export FT_K_TRIALS="${FT_K_TRIALS:-4000}"        # notebooks: 10000
export FT_N_FAULT="${FT_N_FAULT:-250}"           # notebooks: 1000

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON="${PYTHON:-python}"

run_nb () {          # run_nb <notebook> <log-file>
  "$PYTHON" -m nbconvert --to notebook --execute --inplace \
    --ExecutePreprocessor.timeout=-1 \
    --ExecutePreprocessor.kernel_name=ft-python \
    "$1" > "$2" 2>&1
  echo "  $(basename "$1") exit=$? $(date)"
}

for family in "${selected[@]}"; do
  NB="$ROOT/notebooks/$family"
  LOGS="$ROOT/logs/$family"
  mkdir -p "$LOGS"
  echo "=== $family (K=$FT_K_TRIALS, N_fault=$FT_N_FAULT) ==="

  echo "[1/3] AG News (text)"
  FT_MAX_LEN=48 FT_TRAIN_SUBSAMPLE=50000 FT_EPOCHS=6 FT_PATIENCE=2 \
    run_nb "$NB/ag_news.ipynb" "$LOGS/ag_news.log"

  echo "[2/3] CIFAR-10 (image)"
  FT_TRAIN_SUBSAMPLE=25000 FT_EPOCHS=14 FT_PATIENCE=5 \
    run_nb "$NB/cifar10.ipynb" "$LOGS/cifar10.log"

  echo "[3/3] PJM hourly load, 11 regions (AEP is the source paper's region)"
  FT_WINDOWS=20000 FT_EPOCHS=8 FT_PATIENCE=3 \
    run_nb "$NB/pjm_regions.ipynb" "$LOGS/pjm_regions.log"
done

echo "=== ALL DONE $(date) ==="
