#!/bin/bash
#SBATCH --job-name=build_vlm_pool_x10
#SBATCH --partition=devel
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/home/hw646/codex_salip_outputs/logs/build_vlm_expanded_pool_%j.log
#SBATCH --mail-type=NONE

set -euo pipefail

PROJECT=${PROJECT:-/nfs/roberts/project/pi_xy48/hw646/Medical-SAM3}
PY=${PY:-/nfs/roberts/project/pi_xy48/hw646/ycrc_conda/envs/medsam3/bin/python}
INPUT_DIR=$PROJECT/output/visium_hd_exp1/ficture_official_filtered_candidate_pool_input_roi
SEMANTIC_LEGEND=$PROJECT/output/visium_hd_exp1/ficture_official_filtered_he_aligned/factor_semantic_legend.json

HE_ROOT=${HE_ROOT:-/home/hw646/codex_he_official_roi_candidate_pool_b200/he_official_12445451}
FICTURE_ROOT=${FICTURE_ROOT:-$PROJECT/results/visium_hd_exp1/ficture_official_filtered_candidate_pool}
SOURCE_CLIP_CSV=${SOURCE_CLIP_CSV:-/home/hw646/codex_salip_outputs/salip_clip_source_image/official_same_roi_source_image_salip_clip_48of48_rtx_home_log_debug_12520702/source_image_salip_clip_candidate_scores.csv}
SEMANTIC_CLIP_CSV=${SEMANTIC_CLIP_CSV:-$PROJECT/output/visium_hd_exp1/openai_clip_semantic_merged_pool_20260522/official_same_roi_he_plus_ficture_openai_clip_semantic_48of48_12528553/openai_clip_semantic_candidate_scores.csv}
BUILD_SCRIPT=${BUILD_SCRIPT:-/home/hw646/codex_salip_scripts/build_paired_vlm_hit_test_pool.py}
RENDER_SCRIPT=${RENDER_SCRIPT:-/home/hw646/codex_salip_scripts/render_paired_vlm_pool_plain_crops.py}
PER_BUCKET=${PER_BUCKET:-10}

RUN_NAME=${RUN_NAME:-paired_he_ficture_good_mid_bad_6class_perbucket10_20260523}
BASE_OUT=${BASE_OUT:-/home/hw646/codex_vlm_hit_tests_expanded/${RUN_NAME}_overlay}
PLAIN_OUT=${PLAIN_OUT:-/home/hw646/codex_vlm_hit_tests_expanded_plain/${RUN_NAME}_lightgray}
REVERSE_OUT=${REVERSE_OUT:-/home/hw646/codex_vlm_hit_tests_expanded_reverseblur/${RUN_NAME}_reverseblur}

cd "$PROJECT"
mkdir -p /home/hw646/codex_salip_outputs/logs "$(dirname "$BASE_OUT")" "$(dirname "$PLAIN_OUT")" "$(dirname "$REVERSE_OUT")"

export PYTHONUNBUFFERED=1
export PYTHONPATH="$PROJECT/inference/visium_hd_exp1:$PROJECT/inference:$PROJECT:${PYTHONPATH:-}"
export MPLCONFIGDIR="${MPLCONFIGDIR_OVERRIDE:-$PROJECT/output/.mplconfig}"
mkdir -p "$MPLCONFIGDIR"

if [[ "${OVERWRITE:-0}" == "1" ]]; then
  rm -rf "$BASE_OUT" "$PLAIN_OUT" "$REVERSE_OUT"
fi
mkdir -p "$BASE_OUT" "$PLAIN_OUT" "$REVERSE_OUT"

date
echo "Building expanded paired H&E/FICTURE VLM hit-test pool"
echo "PER_BUCKET=$PER_BUCKET"
echo "BASE_OUT=$BASE_OUT"
echo "PLAIN_OUT=$PLAIN_OUT"
echo "REVERSE_OUT=$REVERSE_OUT"
echo "HE_ROOT=$HE_ROOT"
echo "FICTURE_ROOT=$FICTURE_ROOT"
echo "SOURCE_CLIP_CSV=$SOURCE_CLIP_CSV"
echo "SEMANTIC_CLIP_CSV=$SEMANTIC_CLIP_CSV"

"$PY" "$BUILD_SCRIPT" \
  --candidate-root "he=$HE_ROOT" \
  --candidate-root "ficture=$FICTURE_ROOT" \
  --summary-path "$INPUT_DIR/region_summary_official_filtered_roi_hpc.json" \
  --he-image "$INPUT_DIR/he_roi_matching_official_ficture_coverage.png" \
  --ficture-image "$INPUT_DIR/ficture_official_filtered_roi_rgb.png" \
  --factor-label-npy "$INPUT_DIR/ficture_official_filtered_roi_factor_index.npy" \
  --factor-semantic-legend "$SEMANTIC_LEGEND" \
  --source-clip-csv "$SOURCE_CLIP_CSV" \
  --semantic-clip-csv "$SEMANTIC_CLIP_CSV" \
  --output-dir "$BASE_OUT" \
  --per-bucket "$PER_BUCKET"

"$PY" "$RENDER_SCRIPT" \
  --source-pool "$BASE_OUT" \
  --output-dir "$PLAIN_OUT" \
  --he-image "$INPUT_DIR/he_roi_matching_official_ficture_coverage.png" \
  --ficture-image "$INPUT_DIR/ficture_official_filtered_roi_rgb.png" \
  --mode plain_masked

"$PY" "$RENDER_SCRIPT" \
  --source-pool "$BASE_OUT" \
  --output-dir "$REVERSE_OUT" \
  --he-image "$INPUT_DIR/he_roi_matching_official_ficture_coverage.png" \
  --ficture-image "$INPUT_DIR/ficture_official_filtered_roi_rgb.png" \
  --mode reverse_blur_gray

{
  echo "BASE_OUT=$BASE_OUT"
  echo "PLAIN_OUT=$PLAIN_OUT"
  echo "REVERSE_OUT=$REVERSE_OUT"
  echo "PER_BUCKET=$PER_BUCKET"
} > /home/hw646/codex_vlm_hit_tests_expanded/${RUN_NAME}_paths.env

date
echo "Expanded pool and no-overlay crops ready"
