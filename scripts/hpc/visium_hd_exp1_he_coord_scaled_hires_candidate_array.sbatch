#!/bin/bash
#SBATCH --job-name=he_coord_sam
#SBATCH --partition=gpu_rtx6000
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=96G
#SBATCH --time=03:00:00
#SBATCH --array=0-23
#SBATCH --output=/nfs/roberts/project/pi_xy48/hw646/Medical-SAM3/results/visium_hd_exp1/logs/he_coord_sam_%A_%a.log
#SBATCH --mail-type=NONE

set -euo pipefail

PROJECT=/nfs/roberts/project/pi_xy48/hw646/Medical-SAM3
PY=/nfs/roberts/project/pi_xy48/hw646/ycrc_conda/envs/medsam3/bin/python
INPUT_DIR=$PROJECT/output/visium_hd_exp1/ficture_coord_scaled_hires_continuous_candidate_pool_input_roi
IMAGE=$INPUT_DIR/he_roi_matching_coord_scaled_continuous_ficture_coverage.png
SUMMARY=$INPUT_DIR/region_summary_coord_scaled_hires_continuous_roi_hpc.json
ROOT_OUT=$PROJECT/results/visium_hd_exp1/he_coord_scaled_hires_candidate_pool/he_coord_${SLURM_ARRAY_JOB_ID}

mkdir -p "$PROJECT/results/visium_hd_exp1/logs" "$ROOT_OUT"
cd "$PROJECT"

export MPLCONFIGDIR="$PROJECT/output/.mplconfig"
export HF_HOME=/nfs/roberts/project/pi_xy48/hw646/ycrc_conda/hf-home
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
mkdir -p "$MPLCONFIGDIR" "$HF_HOME"

BASE=$PROJECT/checkpoints/SAM3-base/sam3.pt
MED=$PROJECT/checkpoints/Medical-SAM3/checkpoint.pt

case "${SLURM_ARRAY_TASK_ID}" in
  0)  NAME=base_box96_s48_m1536; CKPT=$BASE; MODE=box; MAXSIDE=1536; EXTRA="--box-size 96 --box-stride 48"; LIMIT=420 ;;
  1)  NAME=base_box128_s48_m1536; CKPT=$BASE; MODE=box; MAXSIDE=1536; EXTRA="--box-size 128 --box-stride 48"; LIMIT=420 ;;
  2)  NAME=base_box160_s64_m1536; CKPT=$BASE; MODE=box; MAXSIDE=1536; EXTRA="--box-size 160 --box-stride 64"; LIMIT=420 ;;
  3)  NAME=base_box192_s64_m1536; CKPT=$BASE; MODE=box; MAXSIDE=1536; EXTRA="--box-size 192 --box-stride 64"; LIMIT=420 ;;
  4)  NAME=base_box256_s96_m1536; CKPT=$BASE; MODE=box; MAXSIDE=1536; EXTRA="--box-size 256 --box-stride 96"; LIMIT=420 ;;
  5)  NAME=base_box384_s128_m1536; CKPT=$BASE; MODE=box; MAXSIDE=1536; EXTRA="--box-size 384 --box-stride 128"; LIMIT=420 ;;
  6)  NAME=base_box512_s160_m1536; CKPT=$BASE; MODE=box; MAXSIDE=1536; EXTRA="--box-size 512 --box-stride 160"; LIMIT=420 ;;
  7)  NAME=base_point64_m1536_p900; CKPT=$BASE; MODE=point; MAXSIDE=1536; EXTRA="--grid-step 64 --border 32 --max-points 900"; LIMIT=260 ;;
  8)  NAME=base_point80_m1536_p700; CKPT=$BASE; MODE=point; MAXSIDE=1536; EXTRA="--grid-step 80 --border 40 --max-points 700"; LIMIT=260 ;;
  9)  NAME=base_point96_m1536_p500; CKPT=$BASE; MODE=point; MAXSIDE=1536; EXTRA="--grid-step 96 --border 48 --max-points 500"; LIMIT=260 ;;
  10) NAME=base_box384_s128_m2048; CKPT=$BASE; MODE=box; MAXSIDE=2048; EXTRA="--box-size 384 --box-stride 128"; LIMIT=420 ;;
  11) NAME=base_box512_s160_m2048; CKPT=$BASE; MODE=box; MAXSIDE=2048; EXTRA="--box-size 512 --box-stride 160"; LIMIT=420 ;;
  12) NAME=medical_box96_s48_m1536; CKPT=$MED; MODE=box; MAXSIDE=1536; EXTRA="--box-size 96 --box-stride 48"; LIMIT=420 ;;
  13) NAME=medical_box128_s48_m1536; CKPT=$MED; MODE=box; MAXSIDE=1536; EXTRA="--box-size 128 --box-stride 48"; LIMIT=420 ;;
  14) NAME=medical_box160_s64_m1536; CKPT=$MED; MODE=box; MAXSIDE=1536; EXTRA="--box-size 160 --box-stride 64"; LIMIT=420 ;;
  15) NAME=medical_box192_s64_m1536; CKPT=$MED; MODE=box; MAXSIDE=1536; EXTRA="--box-size 192 --box-stride 64"; LIMIT=420 ;;
  16) NAME=medical_box256_s96_m1536; CKPT=$MED; MODE=box; MAXSIDE=1536; EXTRA="--box-size 256 --box-stride 96"; LIMIT=420 ;;
  17) NAME=medical_box384_s128_m1536; CKPT=$MED; MODE=box; MAXSIDE=1536; EXTRA="--box-size 384 --box-stride 128"; LIMIT=420 ;;
  18) NAME=medical_box512_s160_m1536; CKPT=$MED; MODE=box; MAXSIDE=1536; EXTRA="--box-size 512 --box-stride 160"; LIMIT=420 ;;
  19) NAME=medical_point64_m1536_p900; CKPT=$MED; MODE=point; MAXSIDE=1536; EXTRA="--grid-step 64 --border 32 --max-points 900"; LIMIT=260 ;;
  20) NAME=medical_point80_m1536_p700; CKPT=$MED; MODE=point; MAXSIDE=1536; EXTRA="--grid-step 80 --border 40 --max-points 700"; LIMIT=260 ;;
  21) NAME=medical_point96_m1536_p500; CKPT=$MED; MODE=point; MAXSIDE=1536; EXTRA="--grid-step 96 --border 48 --max-points 500"; LIMIT=260 ;;
  22) NAME=medical_box384_s128_m2048; CKPT=$MED; MODE=box; MAXSIDE=2048; EXTRA="--box-size 384 --box-stride 128"; LIMIT=420 ;;
  23) NAME=medical_box512_s160_m2048; CKPT=$MED; MODE=box; MAXSIDE=2048; EXTRA="--box-size 512 --box-stride 160"; LIMIT=420 ;;
  *) echo "Unknown task ${SLURM_ARRAY_TASK_ID}" >&2; exit 2 ;;
esac

OUT=$ROOT_OUT/$NAME
date
echo "Job: ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
echo "Run: $NAME"
echo "Image: $IMAGE"
echo "Summary: $SUMMARY"

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
