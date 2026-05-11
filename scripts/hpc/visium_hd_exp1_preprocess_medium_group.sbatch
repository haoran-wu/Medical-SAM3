#!/bin/bash
#SBATCH --job-name=pre_medbox
#SBATCH --partition=gpu_rtx6000
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=120G
#SBATCH --time=05:00:00
#SBATCH --output=/nfs/roberts/project/pi_xy48/hw646/Medical-SAM3/results/visium_hd_exp1/logs/pre_medbox_%x_%j.log
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=vettel.hwu@gmail.com

set -euo pipefail

VARIANT="${1:?Usage: sbatch visium_hd_exp1_preprocess_medium_group.sbatch VARIANT MODEL}"
MODEL="${2:?Usage: sbatch visium_hd_exp1_preprocess_medium_group.sbatch VARIANT MODEL}"

PROJECT=/nfs/roberts/project/pi_xy48/hw646/Medical-SAM3
PY=/nfs/roberts/project/pi_xy48/hw646/ycrc_conda/envs/medsam3/bin/python
IMAGE=$PROJECT/output/visium_hd_exp1/sam3_preprocessed_images/${VARIANT}.png
SUMMARY=$PROJECT/output/visium_hd_exp1/sam3_local_region_summary/region_summary.json
ROOT_OUT=$PROJECT/results/visium_hd_exp1/sam3_iamsam_retrial/preprocess_medium_${VARIANT}_${MODEL}_${SLURM_JOB_ID}
SUMMARY_OUT=${ROOT_OUT}_summary

cd "$PROJECT"
export MPLCONFIGDIR="$PROJECT/output/.mplconfig"
export HF_HOME=/nfs/roberts/project/pi_xy48/hw646/ycrc_conda/hf-home
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
mkdir -p "$MPLCONFIGDIR" "$PROJECT/results/visium_hd_exp1/logs"

case "$MODEL" in
  base) CKPT=$PROJECT/checkpoints/SAM3-base/sam3.pt ;;
  medical) CKPT=$PROJECT/checkpoints/Medical-SAM3/checkpoint.pt ;;
  *) echo "Unknown model: $MODEL" >&2; exit 2 ;;
esac

if [[ ! -f "$IMAGE" ]]; then
  echo "Missing preprocessed image: $IMAGE" >&2
  exit 2
fi

"$PY" inference/visium_hd_exp1/run_sam3_candidate_proposals.py \
  --image-path "$IMAGE" \
  --summary-path "$SUMMARY" \
  --checkpoint "$CKPT" \
  --device cuda \
  --output-dir "$ROOT_OUT/medium_boxes_b192_s64" \
  --proposal-mode box \
  --max-side 2048 \
  --box-size 192 \
  --box-stride 64 \
  --min-area-frac 0.00001 \
  --max-area-frac 0.450 \
  --nms-iou 0.97 \
  --save-mask-limit 360 \
  --contact-sheet-limit 60

"$PY" inference/visium_hd_exp1/summarize_iamsam_sam3_sweep.py \
  --sweep-root "$ROOT_OUT" \
  --output-dir "$SUMMARY_OUT" \
  --max-masks 48 \
  --min-delta 0.001

echo "Output root: $ROOT_OUT"
echo "Summary: $SUMMARY_OUT"
