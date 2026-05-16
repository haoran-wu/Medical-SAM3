#!/bin/bash
#SBATCH --job-name=mm_rank
#SBATCH --partition=gpu_devel
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=/nfs/roberts/project/pi_xy48/hw646/Medical-SAM3/results/visium_hd_exp1/logs/mm_rank_%j.log
#SBATCH --mail-type=NONE

set -euo pipefail

PROJECT=/nfs/roberts/project/pi_xy48/hw646/Medical-SAM3
PY=/nfs/roberts/project/pi_xy48/hw646/ycrc_conda/envs/medsam3/bin/python
INPUT_DIR=$PROJECT/output/visium_hd_exp1/ficture_coord_scaled_hires_continuous_candidate_pool_input_roi
OUT=$PROJECT/output/visium_hd_exp1/multimodal_candidate_ranking/coord_scaled_hires_he_plus_ficture_${SLURM_JOB_ID}

cd "$PROJECT"
mkdir -p "$OUT" "$PROJECT/results/visium_hd_exp1/logs"
export MPLCONFIGDIR="$PROJECT/output/.mplconfig"
export PYTHONUNBUFFERED=1

HE_ROOT=$PROJECT/results/visium_hd_exp1/he_coord_scaled_hires_candidate_pool
FICTURE_ROOT=$PROJECT/results/visium_hd_exp1/ficture_coord_scaled_hires_candidate_pool

date
"$PY" inference/visium_hd_exp1/rank_multimodal_sam_candidates.py \
  --candidate-root "he=$HE_ROOT" \
  --candidate-root "ficture=$FICTURE_ROOT" \
  --he-image "$INPUT_DIR/he_roi_matching_coord_scaled_continuous_ficture_coverage.png" \
  --factor-label-npy "$INPUT_DIR/ficture_coord_scaled_hires_continuous_roi_factor_index.npy" \
  --summary-path "$INPUT_DIR/region_summary_coord_scaled_hires_continuous_roi_hpc.json" \
  --factor-annotation "$PROJECT/output/visium_hd_exp1/molecular_prior_experiments/dx-60_dy80/factor_annotation.csv" \
  --output-dir "$OUT" \
  --top-ks 1 2 3 5 \
  --max-overlap 0.72 \
  --max-candidates-per-source-setting 40 \
  --disable-consensus \
  --mask-cache-size 24
date
echo "Output: $OUT"
