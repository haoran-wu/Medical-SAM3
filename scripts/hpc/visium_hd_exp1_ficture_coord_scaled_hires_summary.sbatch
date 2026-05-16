#!/bin/bash
#SBATCH --job-name=fic_coord_sum
#SBATCH --partition=day
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --time=01:00:00
#SBATCH --output=/nfs/roberts/project/pi_xy48/hw646/Medical-SAM3/results/visium_hd_exp1/logs/fic_coord_sum_%j.log
#SBATCH --mail-type=NONE

set -euo pipefail

PROJECT=/nfs/roberts/project/pi_xy48/hw646/Medical-SAM3
PY=/nfs/roberts/project/pi_xy48/hw646/ycrc_conda/envs/medsam3/bin/python
ROOT=$PROJECT/results/visium_hd_exp1/ficture_coord_scaled_hires_candidate_pool
OUT=$PROJECT/output/visium_hd_exp1/ficture_coord_scaled_hires_best_candidate_5panel
HE_ROI=$PROJECT/output/visium_hd_exp1/ficture_coord_scaled_hires_continuous_candidate_pool_input_roi/he_roi_matching_coord_scaled_continuous_ficture_coverage.png

mkdir -p "$PROJECT/results/visium_hd_exp1/logs" "$OUT"
cd "$PROJECT"

export MPLCONFIGDIR="$PROJECT/output/.mplconfig"
mkdir -p "$MPLCONFIGDIR"

date
"$PY" inference/visium_hd_exp1/render_ficture_roi_best_candidate_5panel.py \
  --output-dir "$OUT" \
  --he-roi "$HE_ROI" \
  --root "$ROOT"
date
echo "Output: $OUT"
