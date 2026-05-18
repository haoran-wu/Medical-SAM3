#!/bin/bash
#SBATCH --job-name=fic_best_report
#SBATCH --partition=gpu_devel
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=00:45:00
#SBATCH --output=/nfs/roberts/project/pi_xy48/hw646/Medical-SAM3/results/visium_hd_exp1/logs/fic_best_report_%j.log
#SBATCH --mail-type=NONE

set -euo pipefail

PROJECT=/nfs/roberts/project/pi_xy48/hw646/Medical-SAM3
PY=/nfs/roberts/project/pi_xy48/hw646/ycrc_conda/envs/medsam3/bin/python

cd "$PROJECT"
mkdir -p "$PROJECT/results/visium_hd_exp1/logs"
export MPLCONFIGDIR="$PROJECT/output/.mplconfig"
export PYTHONUNBUFFERED=1

"$PY" inference/visium_hd_exp1/render_official_ficture_candidate_best_report.py \
  --output-dir output/visium_hd_exp1/official_ficture_candidate_best_report \
  --allow-reconstruct-missing \
  --device cuda
