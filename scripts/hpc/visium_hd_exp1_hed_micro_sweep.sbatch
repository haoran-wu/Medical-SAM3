#!/bin/bash
#SBATCH --job-name=hed_micro
#SBATCH --partition=day
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=/nfs/roberts/project/pi_xy48/hw646/Medical-SAM3/results/visium_hd_exp1/logs/hed_micro_%j.log

set -euo pipefail

NAME=${1:?usage: sbatch visium_hd_exp1_hed_micro_sweep.sbatch NAME [args...] }
shift

PROJECT=/nfs/roberts/project/pi_xy48/hw646/Medical-SAM3
PY=/nfs/roberts/project/pi_xy48/hw646/ycrc_conda/envs/medsam3/bin/python
OUT=$PROJECT/results/visium_hd_exp1/hed_micro_sweep/${NAME}_${SLURM_JOB_ID}

cd "$PROJECT"
export MPLCONFIGDIR="$PROJECT/output/.mplconfig"
mkdir -p "$MPLCONFIGDIR" "$OUT" "$PROJECT/results/visium_hd_exp1/logs"

"$PY" inference/visium_hd_exp1/run_hed_micro_object_sweep.py \
  --output-dir "$OUT" \
  "$@"

echo "Output: $OUT"
