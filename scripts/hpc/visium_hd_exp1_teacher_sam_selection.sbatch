#!/bin/bash
#SBATCH --job-name=teach_sam
#SBATCH --partition=day
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --output=/nfs/roberts/project/pi_xy48/hw646/Medical-SAM3/results/visium_hd_exp1/logs/teach_sam_%j.log

set -euo pipefail

GROUP=${1:?usage: sbatch visium_hd_exp1_teacher_sam_selection.sbatch GROUP TEACHER_KIND [extra args...] }
TEACHER_KIND=${2:?usage: sbatch visium_hd_exp1_teacher_sam_selection.sbatch GROUP TEACHER_KIND [extra args...] }
shift 2

PROJECT=/nfs/roberts/project/pi_xy48/hw646/Medical-SAM3
PY=/nfs/roberts/project/pi_xy48/hw646/ycrc_conda/envs/medsam3/bin/python
BASE=$PROJECT/results/visium_hd_exp1/sam3_iamsam_retrial
OUT=$PROJECT/results/visium_hd_exp1/teacher_guided_sam_selection/${GROUP}_${TEACHER_KIND}_${SLURM_JOB_ID}

cd "$PROJECT"
export MPLCONFIGDIR="$PROJECT/output/.mplconfig"
mkdir -p "$MPLCONFIGDIR" "$OUT" "$PROJECT/results/visium_hd_exp1/logs"

case "$GROUP" in
  micro_candidates)
    ROOT_ARGS=(
      --candidate-root "$BASE/extra_base_micro_text_11387841"
      --candidate-root "$BASE/extra_medical_micro_text_11387842"
      --candidate-root "$BASE/extra_base_micro_notext_11387940"
      --candidate-root "$BASE/extra_medical_micro_notext_11388489"
      --candidate-root "$BASE/parallel_base_small_text_11387726"
      --candidate-root "$BASE/parallel_medical_small_text_11387727"
    )
    ;;
  medium_large)
    ROOT_ARGS=(
      --candidate-root "$BASE/extra_base_medium_multiscale_11390782"
      --candidate-root "$BASE/extra_medical_medium_multiscale_11398191"
      --candidate-root "$BASE/extra_base_large_multiscale_11406139"
      --candidate-root "$BASE/extra_medical_large_multiscale_11406160"
      --candidate-root "$BASE/parallel_base_large_context_11387730"
      --candidate-root "$BASE/parallel_medical_large_context_11387731"
    )
    ;;
  all_ready)
    ROOT_ARGS=(--candidate-root "$BASE")
    ;;
  dense_preprocess)
    ROOT_ARGS=(
      --candidate-root "$BASE/iamsam_sweep_11369858"
      --candidate-root "$BASE/iamsam_fine_11370590"
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
  *)
    echo "Unknown group: $GROUP" >&2
    exit 2
    ;;
esac

case "$TEACHER_KIND" in
  ensemble)
    TEACHER_ARGS=(--teacher-root "$PROJECT/results/visium_hd_exp1/completed_expert_ensemble/ensemble_11417101/masks")
    ;;
  ensemble_nonoverlap)
    TEACHER_ARGS=(--teacher-root "$PROJECT/results/visium_hd_exp1/completed_expert_ensemble/ensemble_11417101/masks" --teacher-variant ensemble_11417101_nonoverlap)
    ;;
  micro)
    TEACHER_ARGS=(
      --teacher-root "$PROJECT/results/visium_hd_exp1/micro_classifier_sweep/micro_et_1536_11417039/masks"
      --teacher-root "$PROJECT/results/visium_hd_exp1/micro_classifier_sweep/micro_et_xy_1536_11417040/masks"
      --teacher-root "$PROJECT/results/visium_hd_exp1/micro_classifier_sweep/micro_et_2048_11417043/masks"
    )
    ;;
  unified)
    TEACHER_ARGS=(
      --teacher-root "$PROJECT/results/visium_hd_exp1/unified_pixel_classifier/unified_et_nofactor_xy_1536_11418035/masks"
      --teacher-root "$PROJECT/results/visium_hd_exp1/unified_pixel_classifier/unified_hgb_factor_xy_1536_11418037/masks"
      --teacher-root "$PROJECT/results/visium_hd_exp1/unified_pixel_classifier/unified_et_factor_xy_1536_11418033/masks"
    )
    ;;
  *)
    echo "Unknown teacher: $TEACHER_KIND" >&2
    exit 2
    ;;
esac

"$PY" inference/visium_hd_exp1/rank_sam_candidates_with_teacher_masks.py \
  "${ROOT_ARGS[@]}" \
  "${TEACHER_ARGS[@]}" \
  --output-dir "$OUT" \
  "$@"

echo "Output: $OUT"
