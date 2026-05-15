#!/bin/bash
#SBATCH --job-name=px_split_eval
#SBATCH --partition=day
#SBATCH --cpus-per-task=12
#SBATCH --mem=160G
#SBATCH --time=08:00:00
#SBATCH --output=/nfs/roberts/project/pi_xy48/hw646/Medical-SAM3/results/visium_hd_exp1/logs/px_split_eval_%j.log

set -euo pipefail

NAME=${1:?usage: sbatch visium_hd_exp1_spatial_split_classifier_eval.sbatch NAME [args...] }
shift

PROJECT=/nfs/roberts/project/pi_xy48/hw646/Medical-SAM3
PY=/nfs/roberts/project/pi_xy48/hw646/ycrc_conda/envs/medsam3/bin/python
OUT=$PROJECT/results/visium_hd_exp1/classification_spatial_split_eval/${NAME}_${SLURM_JOB_ID}

cd "$PROJECT"
export MPLCONFIGDIR="$PROJECT/output/.mplconfig"
mkdir -p "$MPLCONFIGDIR" "$OUT" "$PROJECT/results/visium_hd_exp1/logs"

"$PY" inference/visium_hd_exp1/run_unified_pixel_classifier_spatial_split_eval.py \
  --output-dir "$OUT" \
  "$@"

echo "Output: $OUT"
