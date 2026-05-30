#!/bin/bash
#SBATCH --job-name=fic_corr_oldroi
#SBATCH --partition=gpu_rtx6000
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=120G
#SBATCH --time=04:00:00
#SBATCH --array=0-47
#SBATCH --output=/nfs/roberts/project/pi_xy48/hw646/Medical-SAM3/results/visium_hd_exp1/logs/fic_corr_oldroi_%A_%a.log
#SBATCH --mail-type=NONE

set -euo pipefail

# Guardrail: this job must use the direction-corrected, filtered FICTURE ROI pasted into the old control ROI canvas
# derived from ficture_from_folder_he_direction_corrected, but preserving the old 3240 x 3399 ROI. Do not point this
# script back to ficture_coord_scaled_hires_continuous_candidate_pool_input_roi.
# Before reporting downstream rankings, verify FICTURE and H&E ROI dimensions,
# orientation, and background filtering.

PROJECT=/nfs/roberts/project/pi_xy48/hw646/Medical-SAM3
PY=/nfs/roberts/project/pi_xy48/hw646/ycrc_conda/envs/medsam3/bin/python
INPUT_DIR=$PROJECT/output/visium_hd_exp1/ficture_corrected_oldroi_candidate_pool_input_roi
IMAGE=$INPUT_DIR/ficture_corrected_oldroi_rgb.png
SUMMARY=$INPUT_DIR/region_summary_corrected_oldroi_hpc.json
ROOT_OUT=$PROJECT/results/visium_hd_exp1/ficture_corrected_oldroi_candidate_pool/ficture_corrected_oldroi_${SLURM_ARRAY_JOB_ID}

mkdir -p "$PROJECT/results/visium_hd_exp1/logs" "$ROOT_OUT"
cd "$PROJECT"

export MPLCONFIGDIR="$PROJECT/output/.mplconfig"
export HF_HOME=/nfs/roberts/project/pi_xy48/hw646/ycrc_conda/hf-home
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
mkdir -p "$MPLCONFIGDIR" "$HF_HOME"

BASE=$PROJECT/checkpoints/SAM3-base/sam3.pt
MED=$PROJECT/checkpoints/Medical-SAM3/checkpoint.pt

case "${SLURM_ARRAY_TASK_ID}" in
  0)  NAME=base_corrected_oldroi_points_step24; CKPT=$BASE; MODE=point; MAXSIDE=2048; EXTRA="--grid-step 24 --border 12 --max-points 3600"; LIMIT=320 ;;
  1)  NAME=base_corrected_oldroi_points_step32; CKPT=$BASE; MODE=point; MAXSIDE=2048; EXTRA="--grid-step 32 --border 16 --max-points 3000"; LIMIT=320 ;;
  2)  NAME=base_corrected_oldroi_boxes_b192_s48; CKPT=$BASE; MODE=box; MAXSIDE=2048; EXTRA="--box-size 192 --box-stride 48"; LIMIT=320 ;;
  3)  NAME=base_corrected_oldroi_boxes_b384_s96; CKPT=$BASE; MODE=box; MAXSIDE=2048; EXTRA="--box-size 384 --box-stride 96"; LIMIT=320 ;;
  4)  NAME=medical_corrected_oldroi_points_step24; CKPT=$MED; MODE=point; MAXSIDE=2048; EXTRA="--grid-step 24 --border 12 --max-points 3600"; LIMIT=320 ;;
  5)  NAME=medical_corrected_oldroi_points_step32; CKPT=$MED; MODE=point; MAXSIDE=2048; EXTRA="--grid-step 32 --border 16 --max-points 3000"; LIMIT=320 ;;
  6)  NAME=medical_corrected_oldroi_boxes_b192_s48; CKPT=$MED; MODE=box; MAXSIDE=2048; EXTRA="--box-size 192 --box-stride 48"; LIMIT=320 ;;
  7)  NAME=medical_corrected_oldroi_boxes_b384_s96; CKPT=$MED; MODE=box; MAXSIDE=2048; EXTRA="--box-size 384 --box-stride 96"; LIMIT=320 ;;
  8)  NAME=base_box96_s48_m1536; CKPT=$BASE; MODE=box; MAXSIDE=1536; EXTRA="--box-size 96 --box-stride 48"; LIMIT=420 ;;
  9)  NAME=base_box128_s48_m1536; CKPT=$BASE; MODE=box; MAXSIDE=1536; EXTRA="--box-size 128 --box-stride 48"; LIMIT=420 ;;
  10) NAME=base_box160_s64_m1536; CKPT=$BASE; MODE=box; MAXSIDE=1536; EXTRA="--box-size 160 --box-stride 64"; LIMIT=420 ;;
  11) NAME=base_box192_s64_m1536; CKPT=$BASE; MODE=box; MAXSIDE=1536; EXTRA="--box-size 192 --box-stride 64"; LIMIT=420 ;;
  12) NAME=base_box256_s96_m1536; CKPT=$BASE; MODE=box; MAXSIDE=1536; EXTRA="--box-size 256 --box-stride 96"; LIMIT=420 ;;
  13) NAME=base_box384_s128_m1536; CKPT=$BASE; MODE=box; MAXSIDE=1536; EXTRA="--box-size 384 --box-stride 128"; LIMIT=420 ;;
  14) NAME=base_box512_s160_m1536; CKPT=$BASE; MODE=box; MAXSIDE=1536; EXTRA="--box-size 512 --box-stride 160"; LIMIT=420 ;;
  15) NAME=base_box96_s48_m2048; CKPT=$BASE; MODE=box; MAXSIDE=2048; EXTRA="--box-size 96 --box-stride 48"; LIMIT=420 ;;
  16) NAME=base_box128_s48_m2048; CKPT=$BASE; MODE=box; MAXSIDE=2048; EXTRA="--box-size 128 --box-stride 48"; LIMIT=420 ;;
  17) NAME=base_box160_s64_m2048; CKPT=$BASE; MODE=box; MAXSIDE=2048; EXTRA="--box-size 160 --box-stride 64"; LIMIT=420 ;;
  18) NAME=base_box192_s64_m2048; CKPT=$BASE; MODE=box; MAXSIDE=2048; EXTRA="--box-size 192 --box-stride 64"; LIMIT=420 ;;
  19) NAME=base_box256_s96_m2048; CKPT=$BASE; MODE=box; MAXSIDE=2048; EXTRA="--box-size 256 --box-stride 96"; LIMIT=420 ;;
  20) NAME=base_box384_s128_m2048; CKPT=$BASE; MODE=box; MAXSIDE=2048; EXTRA="--box-size 384 --box-stride 128"; LIMIT=420 ;;
  21) NAME=base_box512_s160_m2048; CKPT=$BASE; MODE=box; MAXSIDE=2048; EXTRA="--box-size 512 --box-stride 160"; LIMIT=420 ;;
  22) NAME=medical_box96_s48_m1536; CKPT=$MED; MODE=box; MAXSIDE=1536; EXTRA="--box-size 96 --box-stride 48"; LIMIT=420 ;;
  23) NAME=medical_box128_s48_m1536; CKPT=$MED; MODE=box; MAXSIDE=1536; EXTRA="--box-size 128 --box-stride 48"; LIMIT=420 ;;
  24) NAME=medical_box160_s64_m1536; CKPT=$MED; MODE=box; MAXSIDE=1536; EXTRA="--box-size 160 --box-stride 64"; LIMIT=420 ;;
  25) NAME=medical_box192_s64_m1536; CKPT=$MED; MODE=box; MAXSIDE=1536; EXTRA="--box-size 192 --box-stride 64"; LIMIT=420 ;;
  26) NAME=medical_box256_s96_m1536; CKPT=$MED; MODE=box; MAXSIDE=1536; EXTRA="--box-size 256 --box-stride 96"; LIMIT=420 ;;
  27) NAME=medical_box384_s128_m1536; CKPT=$MED; MODE=box; MAXSIDE=1536; EXTRA="--box-size 384 --box-stride 128"; LIMIT=420 ;;
  28) NAME=medical_box512_s160_m1536; CKPT=$MED; MODE=box; MAXSIDE=1536; EXTRA="--box-size 512 --box-stride 160"; LIMIT=420 ;;
  29) NAME=medical_box96_s48_m2048; CKPT=$MED; MODE=box; MAXSIDE=2048; EXTRA="--box-size 96 --box-stride 48"; LIMIT=420 ;;
  30) NAME=medical_box128_s48_m2048; CKPT=$MED; MODE=box; MAXSIDE=2048; EXTRA="--box-size 128 --box-stride 48"; LIMIT=420 ;;
  31) NAME=medical_box160_s64_m2048; CKPT=$MED; MODE=box; MAXSIDE=2048; EXTRA="--box-size 160 --box-stride 64"; LIMIT=420 ;;
  32) NAME=medical_box192_s64_m2048; CKPT=$MED; MODE=box; MAXSIDE=2048; EXTRA="--box-size 192 --box-stride 64"; LIMIT=420 ;;
  33) NAME=medical_box256_s96_m2048; CKPT=$MED; MODE=box; MAXSIDE=2048; EXTRA="--box-size 256 --box-stride 96"; LIMIT=420 ;;
  34) NAME=medical_box384_s128_m2048; CKPT=$MED; MODE=box; MAXSIDE=2048; EXTRA="--box-size 384 --box-stride 128"; LIMIT=420 ;;
  35) NAME=medical_box512_s160_m2048; CKPT=$MED; MODE=box; MAXSIDE=2048; EXTRA="--box-size 512 --box-stride 160"; LIMIT=420 ;;
  36) NAME=base_point48_m1024_p900; CKPT=$BASE; MODE=point; MAXSIDE=1024; EXTRA="--grid-step 48 --border 24 --max-points 900"; LIMIT=260 ;;
  37) NAME=base_point64_m1024_p700; CKPT=$BASE; MODE=point; MAXSIDE=1024; EXTRA="--grid-step 64 --border 32 --max-points 700"; LIMIT=260 ;;
  38) NAME=base_point80_m1024_p500; CKPT=$BASE; MODE=point; MAXSIDE=1024; EXTRA="--grid-step 80 --border 40 --max-points 500"; LIMIT=260 ;;
  39) NAME=base_point64_m1536_p900; CKPT=$BASE; MODE=point; MAXSIDE=1536; EXTRA="--grid-step 64 --border 32 --max-points 900"; LIMIT=260 ;;
  40) NAME=base_point80_m1536_p700; CKPT=$BASE; MODE=point; MAXSIDE=1536; EXTRA="--grid-step 80 --border 40 --max-points 700"; LIMIT=260 ;;
  41) NAME=base_point96_m1536_p500; CKPT=$BASE; MODE=point; MAXSIDE=1536; EXTRA="--grid-step 96 --border 48 --max-points 500"; LIMIT=260 ;;
  42) NAME=medical_point48_m1024_p900; CKPT=$MED; MODE=point; MAXSIDE=1024; EXTRA="--grid-step 48 --border 24 --max-points 900"; LIMIT=260 ;;
  43) NAME=medical_point64_m1024_p700; CKPT=$MED; MODE=point; MAXSIDE=1024; EXTRA="--grid-step 64 --border 32 --max-points 700"; LIMIT=260 ;;
  44) NAME=medical_point80_m1024_p500; CKPT=$MED; MODE=point; MAXSIDE=1024; EXTRA="--grid-step 80 --border 40 --max-points 500"; LIMIT=260 ;;
  45) NAME=medical_point64_m1536_p900; CKPT=$MED; MODE=point; MAXSIDE=1536; EXTRA="--grid-step 64 --border 32 --max-points 900"; LIMIT=260 ;;
  46) NAME=medical_point80_m1536_p700; CKPT=$MED; MODE=point; MAXSIDE=1536; EXTRA="--grid-step 80 --border 40 --max-points 700"; LIMIT=260 ;;
  47) NAME=medical_point96_m1536_p500; CKPT=$MED; MODE=point; MAXSIDE=1536; EXTRA="--grid-step 96 --border 48 --max-points 500"; LIMIT=260 ;;
  *) echo "Unknown task ${SLURM_ARRAY_TASK_ID}" >&2; exit 2 ;;
esac

OUT=$ROOT_OUT/$NAME
echo "Job: ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
echo "Run: $NAME"
echo "Image: $IMAGE"
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
  --min-area-frac 0.00002 \
  --max-area-frac 0.95 \
  --nms-iou 0.97 \
  --save-mask-limit "$LIMIT" \
  --contact-sheet-limit 80 \
  $EXTRA

date
echo "Output: $OUT"
