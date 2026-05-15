#!/bin/bash
#SBATCH --job-name=ficture_pool
#SBATCH --partition=gpu_rtx6000
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=120G
#SBATCH --time=08:00:00
#SBATCH --array=0-9
#SBATCH --output=/nfs/roberts/project/pi_xy48/hw646/Medical-SAM3/results/visium_hd_exp1/logs/ficture_pool_%A_%a.log
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=vettel.hwu@gmail.com

set -euo pipefail

PROJECT=/nfs/roberts/project/pi_xy48/hw646/Medical-SAM3
PY=/nfs/roberts/project/pi_xy48/hw646/ycrc_conda/envs/medsam3/bin/python
IMAGE=$PROJECT/output/visium_hd_exp1/ficture_candidate_pool_input/ficture_factor_label_image_he_aligned_rgb.png
SUMMARY_DIR=$PROJECT/output/visium_hd_exp1/sam3_local_region_summary
SUMMARY=$SUMMARY_DIR/region_summary.json
ROOT_OUT=$PROJECT/results/visium_hd_exp1/ficture_candidate_pool/ficture_pool_${SLURM_ARRAY_JOB_ID}

mkdir -p "$PROJECT/results/visium_hd_exp1/logs" "$SUMMARY_DIR" "$ROOT_OUT"
cd "$PROJECT"

export MPLCONFIGDIR="$PROJECT/output/.mplconfig"
export HF_HOME=/nfs/roberts/project/pi_xy48/hw646/ycrc_conda/hf-home
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
mkdir -p "$MPLCONFIGDIR" "$HF_HOME"

"$PY" - <<'PY'
import json
from pathlib import Path
from PIL import Image

project = Path("/nfs/roberts/project/pi_xy48/hw646/Medical-SAM3")
summary_dir = project / "output/visium_hd_exp1/sam3_local_region_summary"
summary_dir.mkdir(parents=True, exist_ok=True)
entries = [
    ("lung_bronchiola", "output/visium_hd_exp1/sam3_runs/base_sam3_multipoint_remaining_dev/masks/01_lung_bronchiola_target.png"),
    ("erythorocytes", "output/visium_hd_exp1/sam3_runs/base_sam3_multipoint_remaining_dev/masks/02_erythorocytes_target.png"),
    ("immune_infiltration", "output/visium_hd_exp1/sam3_runs/base_sam3_multipoint/masks/03_immune_infiltration_target.png"),
    ("lung_alveoli_normal_adjacent", "output/visium_hd_exp1/sam3_runs/base_sam3_multipoint_remaining_dev/masks/04_lung_alveoli_normal_adjacent_target.png"),
    ("lung_vessels", "output/visium_hd_exp1/sam3_runs/base_sam3_multipoint_remaining_dev/masks/05_lung_vessels_target.png"),
    ("pigment", "output/visium_hd_exp1/sam3_runs/base_sam3_multipoint_remaining_dev/masks/06_pigment_target.png"),
    ("stroma", "output/visium_hd_exp1/sam3_runs/base_sam3_multipoint/masks/07_stroma_target.png"),
    ("tumor", "output/visium_hd_exp1/sam3_runs/base_sam3_multipoint/masks/08_tumor_target.png"),
]
labels = []
for label, rel in entries:
    path = project / rel
    mask = Image.open(path).convert("L")
    labels.append({"label": label, "slug": label, "mask_path": str(path), "area_pixels": int(sum(v > 127 for v in mask.getdata()))})
(summary_dir / "region_summary.json").write_text(json.dumps({"image_path": str(project / "output/visium_hd_exp1/assets/tissue_hires_image.png"), "labels": labels}, indent=2))
PY

BASE=$PROJECT/checkpoints/SAM3-base/sam3.pt
MED=$PROJECT/checkpoints/Medical-SAM3/checkpoint.pt

case "${SLURM_ARRAY_TASK_ID}" in
  0) NAME=base_ficture_points_step32; CKPT=$BASE; MODE=point; MAXSIDE=1536; EXTRA="--grid-step 32 --border 16 --max-points 2400" ;;
  1) NAME=base_ficture_points_step48; CKPT=$BASE; MODE=point; MAXSIDE=1536; EXTRA="--grid-step 48 --border 24 --max-points 2000" ;;
  2) NAME=base_ficture_boxes_b128_s64; CKPT=$BASE; MODE=box; MAXSIDE=1536; EXTRA="--box-size 128 --box-stride 64" ;;
  3) NAME=base_ficture_boxes_b192_s64; CKPT=$BASE; MODE=box; MAXSIDE=1536; EXTRA="--box-size 192 --box-stride 64" ;;
  4) NAME=base_ficture_boxes_b512_s128; CKPT=$BASE; MODE=box; MAXSIDE=2048; EXTRA="--box-size 512 --box-stride 128" ;;
  5) NAME=medical_ficture_points_step32; CKPT=$MED; MODE=point; MAXSIDE=1536; EXTRA="--grid-step 32 --border 16 --max-points 2400" ;;
  6) NAME=medical_ficture_points_step48; CKPT=$MED; MODE=point; MAXSIDE=1536; EXTRA="--grid-step 48 --border 24 --max-points 2000" ;;
  7) NAME=medical_ficture_boxes_b128_s64; CKPT=$MED; MODE=box; MAXSIDE=1536; EXTRA="--box-size 128 --box-stride 64" ;;
  8) NAME=medical_ficture_boxes_b192_s64; CKPT=$MED; MODE=box; MAXSIDE=1536; EXTRA="--box-size 192 --box-stride 64" ;;
  9) NAME=medical_ficture_boxes_b512_s128; CKPT=$MED; MODE=box; MAXSIDE=2048; EXTRA="--box-size 512 --box-stride 128" ;;
  *) echo "Unknown task ${SLURM_ARRAY_TASK_ID}" >&2; exit 2 ;;
esac

OUT=$ROOT_OUT/$NAME

echo "Job: ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
echo "Host: $(hostname)"
echo "Image: $IMAGE"
echo "Run: $NAME"
echo "Checkpoint: $CKPT"
echo "Mode: $MODE"
echo "Max side: $MAXSIDE"
echo "Extra: $EXTRA"
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
  --save-mask-limit 260 \
  --contact-sheet-limit 50 \
  $EXTRA

date
echo "Output: $OUT"
