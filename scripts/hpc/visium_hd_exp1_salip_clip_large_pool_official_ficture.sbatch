#!/bin/bash
#SBATCH --job-name=salip_fic_big
#SBATCH --partition=gpu_devel
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=/nfs/roberts/project/pi_xy48/hw646/Medical-SAM3/results/visium_hd_exp1/logs/salip_fic_big_%j.log
#SBATCH --mail-type=NONE

set -euo pipefail

# Pure CLIP image-text retrieval on the official filtered FICTURE candidate pool.
# This is not molecular, shape, consensus, or VLM. The only ranker reported is
# salip_clip: CLIP similarity between the candidate-masked H&E crop and the label
# text prompts.

PROJECT=/nfs/roberts/project/pi_xy48/hw646/Medical-SAM3
PY=/nfs/roberts/project/pi_xy48/hw646/ycrc_conda/envs/medsam3/bin/python
INPUT_DIR=$PROJECT/output/visium_hd_exp1/ficture_official_filtered_candidate_pool_input_roi
OFFICIAL_SUMMARY=$PROJECT/output/visium_hd_exp1/ficture_official_filtered_he_aligned/summary_official.json
FICTURE_ROOT=$PROJECT/results/visium_hd_exp1/ficture_official_filtered_candidate_pool

RUN_NAME=${RUN_NAME:-official_ficture_salip_clip_large_pool}
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
echo "Candidate root=$FICTURE_ROOT"
echo "Output=$OUT"
"$PY" "$RANK_SCRIPT" \
  --candidate-root "ficture=$FICTURE_ROOT" \
  --he-image "$INPUT_DIR/he_roi_matching_official_ficture_coverage.png" \
  --summary-path "$INPUT_DIR/region_summary_official_filtered_roi_hpc.json" \
  --official-summary "$OFFICIAL_SUMMARY" \
  --output-dir "$OUT" \
  --clip-model "$CLIP_MODEL" \
  --clip-device cuda \
  --clip-batch-size "$CLIP_BATCH_SIZE" \
  --top-ks "${top_ks[@]}" \
  --max-overlap "$MAX_OVERLAP" \
  --mask-cache-size "$MASK_CACHE_SIZE"
date
echo "Output: $OUT"
