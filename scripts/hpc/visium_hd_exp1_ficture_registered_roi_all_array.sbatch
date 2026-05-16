#!/bin/bash
#SBATCH --job-name=fic_reg_all
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=120G
#SBATCH --time=08:00:00
#SBATCH --array=0-7
#SBATCH --output=/nfs/roberts/project/pi_xy48/hw646/Medical-SAM3/results/visium_hd_exp1/logs/fic_reg_all_%A_%a.log
#SBATCH --mail-type=NONE

set -euo pipefail

PROJECT=/nfs/roberts/project/pi_xy48/hw646/Medical-SAM3
PY=/nfs/roberts/project/pi_xy48/hw646/ycrc_conda/envs/medsam3/bin/python
INPUT_DIR=$PROJECT/output/visium_hd_exp1/ficture_html_palette_registered_candidate_pool_input_roi
IMAGE=$INPUT_DIR/ficture_html_palette_registered_roi_rgb.png
SUMMARY=$INPUT_DIR/region_summary_registered_roi_hpc.json
ROOT_OUT=$PROJECT/results/visium_hd_exp1/ficture_registered_candidate_pool_roi/ficture_reg_all_${SLURM_ARRAY_JOB_ID}

mkdir -p "$PROJECT/results/visium_hd_exp1/logs" "$ROOT_OUT"
cd "$PROJECT"

export MPLCONFIGDIR="$PROJECT/output/.mplconfig"
export HF_HOME=/nfs/roberts/project/pi_xy48/hw646/ycrc_conda/hf-home
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
mkdir -p "$MPLCONFIGDIR" "$HF_HOME"

BASE=$PROJECT/checkpoints/SAM3-base/sam3.pt
MED=$PROJECT/checkpoints/Medical-SAM3/checkpoint.pt

case "${SLURM_ARRAY_TASK_ID}" in
  0) NAME=base_ficture_registered_roi_points_step24; CKPT=$BASE; MODE=point; MAXSIDE=2048; EXTRA="--grid-step 24 --border 12 --max-points 3600" ;;
  1) NAME=base_ficture_registered_roi_points_step32; CKPT=$BASE; MODE=point; MAXSIDE=2048; EXTRA="--grid-step 32 --border 16 --max-points 3000" ;;
  2) NAME=base_ficture_registered_roi_boxes_b192_s48; CKPT=$BASE; MODE=box; MAXSIDE=2048; EXTRA="--box-size 192 --box-stride 48" ;;
  3) NAME=base_ficture_registered_roi_boxes_b384_s96; CKPT=$BASE; MODE=box; MAXSIDE=2048; EXTRA="--box-size 384 --box-stride 96" ;;
  4) NAME=medical_ficture_registered_roi_points_step24; CKPT=$MED; MODE=point; MAXSIDE=2048; EXTRA="--grid-step 24 --border 12 --max-points 3600" ;;
  5) NAME=medical_ficture_registered_roi_points_step32; CKPT=$MED; MODE=point; MAXSIDE=2048; EXTRA="--grid-step 32 --border 16 --max-points 3000" ;;
  6) NAME=medical_ficture_registered_roi_boxes_b192_s48; CKPT=$MED; MODE=box; MAXSIDE=2048; EXTRA="--box-size 192 --box-stride 48" ;;
  7) NAME=medical_ficture_registered_roi_boxes_b384_s96; CKPT=$MED; MODE=box; MAXSIDE=2048; EXTRA="--box-size 384 --box-stride 96" ;;
  *) echo "Unknown task ${SLURM_ARRAY_TASK_ID}" >&2; exit 2 ;;
esac

OUT=$ROOT_OUT/$NAME
echo "Job: ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
echo "Run: $NAME"
echo "Image: $IMAGE
echo "Registered FICTURE input: HTML-palette global tissue-outline registration""
echo "Summary: $SUMMARY"
date

# shellcheck disable=SC2086
"$PY" inference/visium_hd_exp1/run_sam3_candidate_proposals.py \
  --image-path "$IMAGE" \
  --summary-path "$SUMMARY" \
  --checkpoint "$CKPT" \
  --device cuda \
  --output-dir "$OUT" \
  --proposal-mode "$MODE" \
  --max-side "$MAXSIDE" \
  --min-area-frac 0.00003 \
  --max-area-frac 0.95 \
  --nms-iou 0.96 \
  --save-mask-limit 320 \
  --contact-sheet-limit 60 \
  $EXTRA

date
echo "Output: $OUT"
