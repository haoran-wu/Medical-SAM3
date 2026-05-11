#!/bin/bash
#SBATCH --job-name=mol_rank
#SBATCH --partition=day
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=/nfs/roberts/project/pi_xy48/hw646/Medical-SAM3/results/visium_hd_exp1/logs/mol_rank_%x_%j.log
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=vettel.hwu@gmail.com

set -euo pipefail

GROUP="${1:?Usage: sbatch visium_hd_exp1_molecular_rank_summary.sbatch GROUP}"

PROJECT=/nfs/roberts/project/pi_xy48/hw646/Medical-SAM3
PY=/nfs/roberts/project/pi_xy48/hw646/ycrc_conda/envs/medsam3/bin/python
BASE=$PROJECT/results/visium_hd_exp1/sam3_iamsam_retrial
FACTOR=$PROJECT/output/visium_hd_exp1/molecular_prior_experiments/dx-60_dy80/ficture_factor_label_image_he.npy
ANNOT=$PROJECT/output/visium_hd_exp1/molecular_prior_experiments/dx-60_dy80/factor_annotation.csv
OUT=$PROJECT/results/visium_hd_exp1/molecular_sam3_ranking/${GROUP}_${SLURM_JOB_ID}

cd "$PROJECT"
export MPLCONFIGDIR="$PROJECT/output/.mplconfig"
mkdir -p "$MPLCONFIGDIR" "$PROJECT/results/visium_hd_exp1/logs"

case "$GROUP" in
  medium_small)
    ROOT_ARGS=(
      --candidate-root "$BASE/extra_base_medium_multiscale_11390782"
      --candidate-root "$BASE/extra_medical_medium_multiscale_11398191"
      --candidate-root "$BASE/extra_base_micro_text_11387841"
      --candidate-root "$BASE/parallel_medical_small_text_11387727"
    )
    ;;
  large_context)
    ROOT_ARGS=(
      --candidate-root "$BASE/parallel_base_large_context_11387730"
      --candidate-root "$BASE/parallel_medical_large_context_11387731"
      --candidate-root "$BASE/parallel_medical_point_wide_11387729"
    )
    ;;
  points_and_micro)
    ROOT_ARGS=(
      --candidate-root "$BASE/extra_base_dense_points_11405525"
      --candidate-root "$BASE/parallel_base_small_text_11387726"
      --candidate-root "$BASE/parallel_base_small_notext_11387728"
      --candidate-root "$BASE/parallel_medical_small_text_11387727"
    )
    ;;
  all_ready)
    ROOT_ARGS=(--candidate-root "$BASE")
    ;;
  *)
    echo "Unknown group: $GROUP" >&2
    exit 2
    ;;
esac

"$PY" inference/visium_hd_exp1/rank_sam3_candidates_with_molecular_prior.py \
  "${ROOT_ARGS[@]}" \
  --factor-image "$FACTOR" \
  --factor-annotation "$ANNOT" \
  --output-dir "$OUT" \
  --top-ks 1 2 3 5 8 12 20 32 48 \
  --max-overlap 0.72 \
  --skip-panels

echo "Output: $OUT"
