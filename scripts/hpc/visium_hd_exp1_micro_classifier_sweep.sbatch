#!/bin/bash
#SBATCH --job-name=micro_clf
#SBATCH --partition=day
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=03:00:00
#SBATCH --output=/nfs/roberts/project/pi_xy48/hw646/Medical-SAM3/results/visium_hd_exp1/logs/micro_clf_%j.log

set -euo pipefail

NAME=${1:?usage: sbatch visium_hd_exp1_micro_classifier_sweep.sbatch NAME [args...] }
shift

PROJECT=/nfs/roberts/project/pi_xy48/hw646/Medical-SAM3
PY=/nfs/roberts/project/pi_xy48/hw646/ycrc_conda/envs/medsam3/bin/python
OUT=$PROJECT/results/visium_hd_exp1/micro_classifier_sweep/${NAME}_${SLURM_JOB_ID}

cd "$PROJECT"
export MPLCONFIGDIR="$PROJECT/output/.mplconfig"
mkdir -p "$MPLCONFIGDIR" "$OUT" "$PROJECT/results/visium_hd_exp1/logs"

"$PY" inference/visium_hd_exp1/run_micro_pixel_classifier_sweep.py \
  --output-dir "$OUT" \
  "$@"

echo "Output: $OUT"
