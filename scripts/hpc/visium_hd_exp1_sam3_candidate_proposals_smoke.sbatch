#!/bin/bash
#SBATCH --job-name=sam3_candidates
#SBATCH --partition=gpu_rtx6000
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --output=/nfs/roberts/project/pi_xy48/hw646/Medical-SAM3/results/visium_hd_exp1/logs/sam3_candidates_%j.log

set -euo pipefail

PROJECT=/nfs/roberts/project/pi_xy48/hw646/Medical-SAM3
PY=/nfs/roberts/project/pi_xy48/hw646/ycrc_conda/envs/medsam3/bin/python
OUT=$PROJECT/results/visium_hd_exp1/sam3_candidate_proposals/dense_points_smoke_${SLURM_JOB_ID}

mkdir -p "$PROJECT/results/visium_hd_exp1/logs" "$OUT"
cd "$PROJECT"

echo "Job: ${SLURM_JOB_ID}"
echo "Host: $(hostname)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
date

"$PY" inference/visium_hd_exp1/run_sam3_candidate_proposals.py \
  --image-path /nfs/roberts/project/pi_xy48/hw646/Exp1/spatial/tissue_hires_image.png \
  --summary-path /nfs/roberts/project/pi_xy48/hw646/Medical-SAM3/output_visium_hd_exp1_legacy/visium_hd_exp1/region_masks/region_summary.json \
  --checkpoint /nfs/roberts/project/pi_xy48/hw646/Medical-SAM3/checkpoints/SAM3-base/sam3.pt \
  --device cuda \
  --output-dir "$OUT" \
  --max-side 1024 \
  --grid-step 96 \
  --border 48 \
  --min-area-frac 0.0002 \
  --max-area-frac 0.35 \
  --nms-iou 0.92 \
  --save-mask-limit 80 \
  --contact-sheet-limit 25

date
echo "Output: $OUT"
