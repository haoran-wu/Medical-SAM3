#!/bin/bash
#SBATCH --job-name=targeted_component_recall
#SBATCH --partition=day_amd
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --output=/home/hw646/Medical-SAM3_remote_runs/visium_hd_exp1/logs/targeted_component_recall_%j.log

set -euo pipefail

SCRIPT=${SCRIPT:?Set SCRIPT to diagnose_targeted_component_recall.py}
GT_MASK=${GT_MASK:?Set GT_MASK}
CURRENT_MASK=${CURRENT_MASK:?Set CURRENT_MASK}
COMPONENT_CANDIDATES=${COMPONENT_CANDIDATES:?Set COMPONENT_CANDIDATES}
HE_IMAGE=${HE_IMAGE:?Set HE_IMAGE}
OUT_DIR=${OUT_DIR:?Set OUT_DIR}
PY=${PY:-/nfs/roberts/project/pi_xy48/hw646/ycrc_conda/envs/medsam3/bin/python}
EXTRA_ARGS=${EXTRA_ARGS:-}

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

mkdir -p "$OUT_DIR"

echo "Start targeted component recall: $(date)"
echo "Host: $(hostname)"
echo "SCRIPT=$SCRIPT"
echo "GT_MASK=$GT_MASK"
echo "CURRENT_MASK=$CURRENT_MASK"
echo "COMPONENT_CANDIDATES=$COMPONENT_CANDIDATES"
echo "HE_IMAGE=$HE_IMAGE"
echo "OUT_DIR=$OUT_DIR"
echo "PY=$PY"
echo "EXTRA_ARGS=$EXTRA_ARGS"

"$PY" "$SCRIPT" \
  --gt-mask "$GT_MASK" \
  --current-mask "$CURRENT_MASK" \
  --component-candidates "$COMPONENT_CANDIDATES" \
  --he-image "$HE_IMAGE" \
  --out-dir "$OUT_DIR" \
  $EXTRA_ARGS

echo "Done targeted component recall: $(date)"
