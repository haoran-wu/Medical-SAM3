#!/bin/bash
#SBATCH --job-name=fic_roi_box
#SBATCH --partition=gpu_rtx6000
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=120G
#SBATCH --time=04:00:00
#SBATCH --array=0-27
#SBATCH --output=/nfs/roberts/project/pi_xy48/hw646/Medical-SAM3/results/visium_hd_exp1/logs/fic_roi_box_%A_%a.log
#SBATCH --mail-type=NONE

set -euo pipefail

PROJECT=/nfs/roberts/project/pi_xy48/hw646/Medical-SAM3
PY=/nfs/roberts/project/pi_xy48/hw646/ycrc_conda/envs/medsam3/bin/python
INPUT_DIR=$PROJECT/output/visium_hd_exp1/ficture_candidate_pool_input_roi
IMAGE=$INPUT_DIR/ficture_factor_label_roi_rgb.png
SUMMARY=$INPUT_DIR/region_summary_roi_hpc.json
ROOT_OUT=$PROJECT/results/visium_hd_exp1/ficture_candidate_pool_roi_box_sweep/ficture_roi_box_${SLURM_ARRAY_JOB_ID}

mkdir -p "$PROJECT/results/visium_hd_exp1/logs" "$ROOT_OUT"
cd "$PROJECT"

export MPLCONFIGDIR="$PROJECT/output/.mplconfig"
export HF_HOME=/nfs/roberts/project/pi_xy48/hw646/ycrc_conda/hf-home
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
mkdir -p "$MPLCONFIGDIR" "$HF_HOME"

BASE=$PROJECT/checkpoints/SAM3-base/sam3.pt
MED=$PROJECT/checkpoints/Medical-SAM3/checkpoint.pt

case "${SLURM_ARRAY_TASK_ID}" in
  0)  NAME=base_box96_s48_m1536;    CKPT=$BASE; MAXSIDE=1536; EXTRA="--box-size 96 --box-stride 48" ;;
  1)  NAME=base_box128_s48_m1536;   CKPT=$BASE; MAXSIDE=1536; EXTRA="--box-size 128 --box-stride 48" ;;
  2)  NAME=base_box160_s64_m1536;   CKPT=$BASE; MAXSIDE=1536; EXTRA="--box-size 160 --box-stride 64" ;;
  3)  NAME=base_box192_s64_m1536;   CKPT=$BASE; MAXSIDE=1536; EXTRA="--box-size 192 --box-stride 64" ;;
  4)  NAME=base_box256_s96_m1536;   CKPT=$BASE; MAXSIDE=1536; EXTRA="--box-size 256 --box-stride 96" ;;
  5)  NAME=base_box384_s128_m1536;  CKPT=$BASE; MAXSIDE=1536; EXTRA="--box-size 384 --box-stride 128" ;;
  6)  NAME=base_box512_s160_m1536;  CKPT=$BASE; MAXSIDE=1536; EXTRA="--box-size 512 --box-stride 160" ;;
  7)  NAME=base_box96_s48_m2048;    CKPT=$BASE; MAXSIDE=2048; EXTRA="--box-size 96 --box-stride 48" ;;
  8)  NAME=base_box128_s48_m2048;   CKPT=$BASE; MAXSIDE=2048; EXTRA="--box-size 128 --box-stride 48" ;;
  9)  NAME=base_box160_s64_m2048;   CKPT=$BASE; MAXSIDE=2048; EXTRA="--box-size 160 --box-stride 64" ;;
  10) NAME=base_box192_s64_m2048;   CKPT=$BASE; MAXSIDE=2048; EXTRA="--box-size 192 --box-stride 64" ;;
  11) NAME=base_box256_s96_m2048;   CKPT=$BASE; MAXSIDE=2048; EXTRA="--box-size 256 --box-stride 96" ;;
  12) NAME=base_box384_s128_m2048;  CKPT=$BASE; MAXSIDE=2048; EXTRA="--box-size 384 --box-stride 128" ;;
  13) NAME=base_box512_s160_m2048;  CKPT=$BASE; MAXSIDE=2048; EXTRA="--box-size 512 --box-stride 160" ;;
  14) NAME=medical_box96_s48_m1536;   CKPT=$MED; MAXSIDE=1536; EXTRA="--box-size 96 --box-stride 48" ;;
  15) NAME=medical_box128_s48_m1536;  CKPT=$MED; MAXSIDE=1536; EXTRA="--box-size 128 --box-stride 48" ;;
  16) NAME=medical_box160_s64_m1536;  CKPT=$MED; MAXSIDE=1536; EXTRA="--box-size 160 --box-stride 64" ;;
  17) NAME=medical_box192_s64_m1536;  CKPT=$MED; MAXSIDE=1536; EXTRA="--box-size 192 --box-stride 64" ;;
  18) NAME=medical_box256_s96_m1536;  CKPT=$MED; MAXSIDE=1536; EXTRA="--box-size 256 --box-stride 96" ;;
  19) NAME=medical_box384_s128_m1536; CKPT=$MED; MAXSIDE=1536; EXTRA="--box-size 384 --box-stride 128" ;;
  20) NAME=medical_box512_s160_m1536; CKPT=$MED; MAXSIDE=1536; EXTRA="--box-size 512 --box-stride 160" ;;
  21) NAME=medical_box96_s48_m2048;   CKPT=$MED; MAXSIDE=2048; EXTRA="--box-size 96 --box-stride 48" ;;
  22) NAME=medical_box128_s48_m2048;  CKPT=$MED; MAXSIDE=2048; EXTRA="--box-size 128 --box-stride 48" ;;
  23) NAME=medical_box160_s64_m2048;  CKPT=$MED; MAXSIDE=2048; EXTRA="--box-size 160 --box-stride 64" ;;
  24) NAME=medical_box192_s64_m2048;  CKPT=$MED; MAXSIDE=2048; EXTRA="--box-size 192 --box-stride 64" ;;
  25) NAME=medical_box256_s96_m2048;  CKPT=$MED; MAXSIDE=2048; EXTRA="--box-size 256 --box-stride 96" ;;
  26) NAME=medical_box384_s128_m2048; CKPT=$MED; MAXSIDE=2048; EXTRA="--box-size 384 --box-stride 128" ;;
  27) NAME=medical_box512_s160_m2048; CKPT=$MED; MAXSIDE=2048; EXTRA="--box-size 512 --box-stride 160" ;;
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
  --proposal-mode box \
  --max-side "$MAXSIDE" \
  --min-area-frac 0.00002 \
  --max-area-frac 0.95 \
  --nms-iou 0.97 \
  --save-mask-limit 420 \
  --contact-sheet-limit 80 \
  $EXTRA

date
echo "Output: $OUT"
