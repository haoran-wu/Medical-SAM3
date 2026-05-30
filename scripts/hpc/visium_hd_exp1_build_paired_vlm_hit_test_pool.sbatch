#!/bin/bash
#SBATCH --job-name=build_vlm_pool
#SBATCH --partition=devel
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --time=01:00:00
#SBATCH --output=/home/hw646/codex_salip_outputs/logs/build_paired_vlm_pool_%j.log
#SBATCH --mail-type=NONE

set -euo pipefail

PROJECT=/nfs/roberts/project/pi_xy48/hw646/Medical-SAM3
PY=/nfs/roberts/project/pi_xy48/hw646/ycrc_conda/envs/medsam3/bin/python
INPUT_DIR=$PROJECT/output/visium_hd_exp1/ficture_official_filtered_candidate_pool_input_roi
SEMANTIC_LEGEND=$PROJECT/output/visium_hd_exp1/ficture_official_filtered_he_aligned/factor_semantic_legend.json

HE_ROOT=${HE_ROOT:?Set HE_ROOT}
FICTURE_ROOT=${FICTURE_ROOT:-$PROJECT/results/visium_hd_exp1/ficture_official_filtered_candidate_pool}
SOURCE_CLIP_CSV=${SOURCE_CLIP_CSV:?Set SOURCE_CLIP_CSV}
SEMANTIC_CLIP_CSV=${SEMANTIC_CLIP_CSV:?Set SEMANTIC_CLIP_CSV}
RUN_NAME=${RUN_NAME:-paired_he_ficture_vlm_hit_test_pool}
OUT_BASE=${OUT_BASE:-/home/hw646/codex_vlm_hit_tests}
BUILD_SCRIPT=${BUILD_SCRIPT:-/home/hw646/codex_salip_scripts/build_paired_vlm_hit_test_pool.py}
PER_BUCKET=${PER_BUCKET:-5}

OUT=$OUT_BASE/${RUN_NAME}_${SLURM_JOB_ID}

cd "$PROJECT"
mkdir -p "$OUT" "$OUT_BASE" /home/hw646/codex_salip_outputs/logs
export PYTHONUNBUFFERED=1
export PYTHONPATH="$PROJECT/inference/visium_hd_exp1:$PROJECT/inference:$PROJECT:${PYTHONPATH:-}"
export MPLCONFIGDIR="${MPLCONFIGDIR_OVERRIDE:-$PROJECT/output/.mplconfig}"
mkdir -p "$MPLCONFIGDIR"

date
echo "Building paired H&E/FICTURE VLM hit-test pool"
echo "HE_ROOT=$HE_ROOT"
echo "FICTURE_ROOT=$FICTURE_ROOT"
echo "SOURCE_CLIP_CSV=$SOURCE_CLIP_CSV"
echo "SEMANTIC_CLIP_CSV=$SEMANTIC_CLIP_CSV"
echo "PER_BUCKET=$PER_BUCKET"
echo "Output=$OUT"
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
  --output-dir "$OUT" \
  --per-bucket "$PER_BUCKET"
date
echo "Output: $OUT"
