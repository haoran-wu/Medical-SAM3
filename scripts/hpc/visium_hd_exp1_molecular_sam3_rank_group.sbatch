#!/bin/bash
#SBATCH --job-name=mol_sam_rank
#SBATCH --partition=gpu_devel
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --output=/nfs/roberts/project/pi_xy48/hw646/Medical-SAM3/results/visium_hd_exp1/logs/mol_sam_rank_%x_%j.log
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=vettel.hwu@gmail.com

set -euo pipefail

GROUP="${1:?Usage: sbatch visium_hd_exp1_molecular_sam3_rank_group.sbatch GROUP}"

PROJECT=/nfs/roberts/project/pi_xy48/hw646/Medical-SAM3
PY=/nfs/roberts/project/pi_xy48/hw646/ycrc_conda/envs/medsam3/bin/python
FACTOR=$PROJECT/output/visium_hd_exp1/molecular_prior_experiments/dx-60_dy80/ficture_factor_label_image_he.npy
FACTOR_ANNOTATION=$PROJECT/output/visium_hd_exp1/molecular_prior_experiments/dx-60_dy80/factor_annotation.csv
BASE=$PROJECT/results/visium_hd_exp1/sam3_iamsam_retrial
OUT=$PROJECT/results/visium_hd_exp1/molecular_sam3_ranking/${GROUP}_${SLURM_JOB_ID}

cd "$PROJECT"
export MPLCONFIGDIR="$PROJECT/output/.mplconfig"
mkdir -p "$MPLCONFIGDIR" "$OUT" "$PROJECT/results/visium_hd_exp1/logs"

echo "Job: ${SLURM_JOB_ID}"
echo "Group: $GROUP"
echo "Host: $(hostname)"
date

case "$GROUP" in
  sweep_fine)
    ROOT_ARGS=(
      --candidate-root "$BASE/iamsam_sweep_11369858"
      --candidate-root "$BASE/iamsam_fine_11370590"
    )
    ;;
  preprocess)
    ROOT_ARGS=(
      --candidate-root "$BASE/preprocess_lab_clahe_base_11387767"
      --candidate-root "$BASE/preprocess_lab_clahe_medical_11387768"
      --candidate-root "$BASE/preprocess_hematoxylin_focus_base_11387769"
      --candidate-root "$BASE/preprocess_hematoxylin_focus_medical_11387770"
      --candidate-root "$BASE/preprocess_sat_value_norm_base_11387771"
      --candidate-root "$BASE/preprocess_sat_value_norm_medical_11387772"
      --candidate-root "$BASE/preprocess_autocontrast_sharp_base_11387773"
      --candidate-root "$BASE/preprocess_autocontrast_sharp_medical_11387774"
    )
    ;;
  parallel_done)
    ROOT_ARGS=(
      --candidate-root "$BASE/parallel_base_large_context_11387730"
      --candidate-root "$BASE/parallel_medical_large_context_11387731"
      --candidate-root "$BASE/parallel_medical_point_wide_11387729"
    )
    ;;
  parallel_all)
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
  --factor-annotation "$FACTOR_ANNOTATION" \
  --output-dir "$OUT" \
  --top-ks 1 2 3 5 8 12 20 32 \
  --max-overlap 0.72

date
echo "Output: $OUT"
