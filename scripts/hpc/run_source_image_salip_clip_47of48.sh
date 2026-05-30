#!/bin/bash
set -euo pipefail

PROJECT=/nfs/roberts/project/pi_xy48/hw646/Medical-SAM3
PY=/nfs/roberts/project/pi_xy48/hw646/ycrc_conda/envs/medsam3/bin/python
INPUT_DIR=$PROJECT/output/visium_hd_exp1/ficture_official_filtered_candidate_pool_input_roi
OFFICIAL_SUMMARY=$PROJECT/output/visium_hd_exp1/ficture_official_filtered_he_aligned/summary_official.json
HE_ROOT=/home/hw646/codex_he_official_roi_candidate_pool_b200/he_official_12445451
FICTURE_ROOT=$PROJECT/results/visium_hd_exp1/ficture_official_filtered_candidate_pool
OUT_BASE=/home/hw646/codex_salip_outputs/salip_clip_source_image
RUN_NAME=official_same_roi_source_image_salip_clip_47of48_wrap
OUT=$OUT_BASE/${RUN_NAME}_${SLURM_JOB_ID}

cd "$PROJECT"
mkdir -p "$OUT"
export MPLCONFIGDIR=$PROJECT/output/.mplconfig
export HF_HOME=/nfs/roberts/project/pi_xy48/hw646/ycrc_conda/hf-home
export TRANSFORMERS_CACHE=$HF_HOME/transformers
export PYTHONUNBUFFERED=1
export PYTHONPATH=$PROJECT/inference/visium_hd_exp1:$PROJECT/inference:$PROJECT:${PYTHONPATH:-}
mkdir -p "$MPLCONFIGDIR" "$HF_HOME"

date
echo "RUN_NAME=$RUN_NAME"
echo "Ranker=source_image_salip_clip only"
echo "Plain meaning: candidate mask -> source-image crop -> CLIP image/text score -> top-k masks."
echo "Output=$OUT"

"$PY" /home/hw646/codex_salip_scripts/rank_salip_clip_source_image_pools.py \
  --candidate-root "he=$HE_ROOT" \
  --candidate-root "ficture=$FICTURE_ROOT" \
  --source-image "he=$INPUT_DIR/he_roi_matching_official_ficture_coverage.png" \
  --source-image "ficture=$INPUT_DIR/ficture_official_filtered_roi_rgb.png" \
  --summary-path "$INPUT_DIR/region_summary_official_filtered_roi_hpc.json" \
  --official-summary "$OFFICIAL_SUMMARY" \
  --output-dir "$OUT" \
  --clip-model openai/clip-vit-base-patch32 \
  --clip-device cuda \
  --clip-batch-size 96 \
  --top-ks 1 2 3 5 8 12 20 \
  --max-overlap 0.86 \
  --mask-cache-size 64

date
echo "Output: $OUT"
