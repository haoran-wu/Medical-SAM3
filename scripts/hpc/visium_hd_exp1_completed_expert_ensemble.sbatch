#!/bin/bash
#SBATCH --job-name=expert_ens
#SBATCH --partition=day
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=03:00:00
#SBATCH --output=/nfs/roberts/project/pi_xy48/hw646/Medical-SAM3/results/visium_hd_exp1/logs/expert_ens_%j.log

set -euo pipefail

PROJECT=/nfs/roberts/project/pi_xy48/hw646/Medical-SAM3
PY=/nfs/roberts/project/pi_xy48/hw646/ycrc_conda/envs/medsam3/bin/python
OUT=$PROJECT/results/visium_hd_exp1/completed_expert_ensemble/ensemble_${SLURM_JOB_ID}

cd "$PROJECT"
export MPLCONFIGDIR="$PROJECT/output/.mplconfig"
mkdir -p "$MPLCONFIGDIR" "$OUT" "$PROJECT/results/visium_hd_exp1/logs"

"$PY" inference/visium_hd_exp1/ensemble_completed_segmentation_experts.py \
  --output-dir "$OUT" \
  "$@"

echo "Output: $OUT"
