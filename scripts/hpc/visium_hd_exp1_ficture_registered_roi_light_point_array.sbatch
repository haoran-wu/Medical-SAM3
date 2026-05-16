#!/bin/bash
#SBATCH --job-name=fic_reg_lpt
#SBATCH --partition=gpu_rtx6000
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=120G
#SBATCH --time=03:00:00
#SBATCH --array=0-11
#SBATCH --output=/nfs/roberts/project/pi_xy48/hw646/Medical-SAM3/results/visium_hd_exp1/logs/fic_reg_lpt_%A_%a.log
#SBATCH --mail-type=NONE

set -euo pipefail

PROJECT=/nfs/roberts/project/pi_xy48/hw646/Medical-SAM3
PY=/nfs/roberts/project/pi_xy48/hw646/ycrc_conda/envs/medsam3/bin/python
INPUT_DIR=$PROJECT/output/visium_hd_exp1/ficture_html_palette_registered_candidate_pool_input_roi
IMAGE=$INPUT_DIR/ficture_html_palette_registered_roi_rgb.png
SUMMARY=$INPUT_DIR/region_summary_registered_roi_hpc.json
ROOT_OUT=$PROJECT/results/visium_hd_exp1/ficture_registered_candidate_pool_roi_light_point/ficture_reg_lpt_${SLURM_ARRAY_JOB_ID}

mkdir -p "$PROJECT/results/visium_hd_exp1/logs" "$ROOT_OUT"
cd "$PROJECT"

export MPLCONFIGDIR="$PROJECT/output/.mplconfig"
export HF_HOME=/nfs/roberts/project/pi_xy48/hw646/ycrc_conda/hf-home
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
mkdir -p "$MPLCONFIGDIR" "$HF_HOME"

BASE=$PROJECT/checkpoints/SAM3-base/sam3.pt
MED=$PROJECT/checkpoints/Medical-SAM3/checkpoint.pt

case "${SLURM_ARRAY_TASK_ID}" in
  0)  NAME=base_point48_m1024_p900;    CKPT=$BASE; MAXSIDE=1024; EXTRA="--grid-step 48 --border 24 --max-points 900" ;;
  1)  NAME=base_point64_m1024_p700;    CKPT=$BASE; MAXSIDE=1024; EXTRA="--grid-step 64 --border 32 --max-points 700" ;;
  2)  NAME=base_point80_m1024_p500;    CKPT=$BASE; MAXSIDE=1024; EXTRA="--grid-step 80 --border 40 --max-points 500" ;;
  3)  NAME=base_point64_m1536_p900;    CKPT=$BASE; MAXSIDE=1536; EXTRA="--grid-step 64 --border 32 --max-points 900" ;;
  4)  NAME=base_point80_m1536_p700;    CKPT=$BASE; MAXSIDE=1536; EXTRA="--grid-step 80 --border 40 --max-points 700" ;;
  5)  NAME=base_point96_m1536_p500;    CKPT=$BASE; MAXSIDE=1536; EXTRA="--grid-step 96 --border 48 --max-points 500" ;;
  6)  NAME=medical_point48_m1024_p900; CKPT=$MED; MAXSIDE=1024; EXTRA="--grid-step 48 --border 24 --max-points 900" ;;
  7)  NAME=medical_point64_m1024_p700; CKPT=$MED; MAXSIDE=1024; EXTRA="--grid-step 64 --border 32 --max-points 700" ;;
  8)  NAME=medical_point80_m1024_p500; CKPT=$MED; MAXSIDE=1024; EXTRA="--grid-step 80 --border 40 --max-points 500" ;;
  9)  NAME=medical_point64_m1536_p900; CKPT=$MED; MAXSIDE=1536; EXTRA="--grid-step 64 --border 32 --max-points 900" ;;
  10) NAME=medical_point80_m1536_p700; CKPT=$MED; MAXSIDE=1536; EXTRA="--grid-step 80 --border 40 --max-points 700" ;;
  11) NAME=medical_point96_m1536_p500; CKPT=$MED; MAXSIDE=1536; EXTRA="--grid-step 96 --border 48 --max-points 500" ;;
  *) echo "Unknown task ${SLURM_ARRAY_TASK_ID}" >&2; exit 2 ;;
esac

OUT=$ROOT_OUT/$NAME
echo "Job: ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
echo "Run: $NAME"
echo "Partition: ${SLURM_JOB_PARTITION:-unknown}"
date

# shellcheck disable=SC2086
"$PY" inference/visium_hd_exp1/run_sam3_candidate_proposals.py \
  --image-path "$IMAGE" \
  --summary-path "$SUMMARY" \
  --checkpoint "$CKPT" \
  --device cuda \
  --output-dir "$OUT" \
  --proposal-mode point \
  --max-side "$MAXSIDE" \
  --min-area-frac 0.00002 \
  --max-area-frac 0.95 \
  --nms-iou 0.97 \
  --save-mask-limit 260 \
  --contact-sheet-limit 60 \
  $EXTRA

date
echo "Output: $OUT"
