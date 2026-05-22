#!/bin/bash
#SBATCH --job-name=salip_he_hist
#SBATCH --partition=gpu_devel
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=/nfs/roberts/project/pi_xy48/hw646/Medical-SAM3/results/visium_hd_exp1/logs/salip_he_hist_%j.log
#SBATCH --mail-type=NONE

set -euo pipefail

# Historical H&E candidate pool salip_clip baseline. This root is the old
# coord-scaled ROI, so it must be reported separately from official FICTURE.

PROJECT=/nfs/roberts/project/pi_xy48/hw646/Medical-SAM3
PY=/nfs/roberts/project/pi_xy48/hw646/ycrc_conda/envs/medsam3/bin/python
INPUT_DIR=${INPUT_DIR:-$PROJECT/output/visium_hd_exp1/ficture_coord_scaled_hires_continuous_candidate_pool_input_roi}
HE_ROOT=$PROJECT/results/visium_hd_exp1/he_coord_scaled_hires_candidate_pool/he_coord_11800076

RUN_NAME=${RUN_NAME:-historical_he_salip_clip_large_pool}
CLIP_MODEL=${CLIP_MODEL:-openai/clip-vit-base-patch32}
CLIP_BATCH_SIZE=${CLIP_BATCH_SIZE:-64}
TOP_KS_LIST=${TOP_KS_LIST:-"1:2:3:5:8:12:20"}
MAX_OVERLAP=${MAX_OVERLAP:-0.86}
MASK_CACHE_SIZE=${MASK_CACHE_SIZE:-64}
OUT_BASE=${OUT_BASE:-$PROJECT/output/visium_hd_exp1/salip_clip_large_pool}
LOG_DIR=${LOG_DIR:-$PROJECT/results/visium_hd_exp1/logs}
RANK_SCRIPT=${RANK_SCRIPT:-$PROJECT/inference/visium_hd_exp1/rank_salip_clip_large_pool.py}

OUT=$OUT_BASE/${RUN_NAME}_${SLURM_JOB_ID}

cd "$PROJECT"
mkdir -p "$OUT" "$LOG_DIR"
export MPLCONFIGDIR="${MPLCONFIGDIR_OVERRIDE:-$PROJECT/output/.mplconfig}"
export HF_HOME="${HF_HOME_OVERRIDE:-/nfs/roberts/project/pi_xy48/hw646/ycrc_conda/hf-home}"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
export PYTHONUNBUFFERED=1
mkdir -p "$MPLCONFIGDIR" "$HF_HOME"

IFS=':' read -r -a top_ks <<< "$TOP_KS_LIST"

date
echo "RUN_NAME=$RUN_NAME"
echo "Ranker=salip_clip only"
echo "CLIP_MODEL=$CLIP_MODEL"
echo "Candidate root=$HE_ROOT"
echo "WARNING: historical old coord-scaled ROI; do not mix directly with official FICTURE."
echo "Output=$OUT"
"$PY" "$RANK_SCRIPT" \
  --candidate-root "he=$HE_ROOT" \
  --he-image "$INPUT_DIR/he_roi_matching_coord_scaled_continuous_ficture_coverage.png" \
  --summary-path "$INPUT_DIR/region_summary_coord_scaled_hires_continuous_roi_hpc.json" \
  --output-dir "$OUT" \
  --clip-model "$CLIP_MODEL" \
  --clip-device cuda \
  --clip-batch-size "$CLIP_BATCH_SIZE" \
  --top-ks "${top_ks[@]}" \
  --max-overlap "$MAX_OVERLAP" \
  --mask-cache-size "$MASK_CACHE_SIZE"
date
echo "Output: $OUT"
