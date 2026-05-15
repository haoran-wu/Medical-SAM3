#!/bin/bash
#SBATCH --job-name=fic_bav
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=120G
#SBATCH --time=08:00:00
#SBATCH --array=0-7
#SBATCH --output=/nfs/roberts/project/pi_xy48/hw646/Medical-SAM3/results/visium_hd_exp1/logs/fic_bav_%A_%a.log
#SBATCH --mail-type=NONE

set -euo pipefail

PROJECT=/nfs/roberts/project/pi_xy48/hw646/Medical-SAM3
PY=/nfs/roberts/project/pi_xy48/hw646/ycrc_conda/envs/medsam3/bin/python
IMAGE=$PROJECT/output/visium_hd_exp1/ficture_candidate_pool_input/ficture_factor_label_image_he_aligned_rgb.png
SUMMARY_DIR=$PROJECT/output/visium_hd_exp1/sam3_local_region_summary
SUMMARY=$SUMMARY_DIR/region_summary.json
ROOT_OUT=$PROJECT/results/visium_hd_exp1/ficture_candidate_pool_focus_bav/ficture_focus_bav_${SLURM_ARRAY_JOB_ID}

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
LABELS=lung_bronchiola,lung_alveoli_normal_adjacent,lung_vessels

case "${SLURM_ARRAY_TASK_ID}" in
  0) NAME=base_ficture_focus_points_step24; CKPT=$BASE; MODE=point; MAXSIDE=2048; EXTRA="--grid-step 24 --border 12 --max-points 3600" ;;
  1) NAME=base_ficture_focus_points_step32_hi; CKPT=$BASE; MODE=point; MAXSIDE=2048; EXTRA="--grid-step 32 --border 16 --max-points 3000" ;;
  2) NAME=base_ficture_focus_boxes_b192_s48; CKPT=$BASE; MODE=box; MAXSIDE=2048; EXTRA="--box-size 192 --box-stride 48" ;;
  3) NAME=base_ficture_focus_boxes_b384_s96; CKPT=$BASE; MODE=box; MAXSIDE=2048; EXTRA="--box-size 384 --box-stride 96" ;;
  4) NAME=medical_ficture_focus_points_step24; CKPT=$MED; MODE=point; MAXSIDE=2048; EXTRA="--grid-step 24 --border 12 --max-points 3600" ;;
  5) NAME=medical_ficture_focus_points_step32_hi; CKPT=$MED; MODE=point; MAXSIDE=2048; EXTRA="--grid-step 32 --border 16 --max-points 3000" ;;
  6) NAME=medical_ficture_focus_boxes_b192_s48; CKPT=$MED; MODE=box; MAXSIDE=2048; EXTRA="--box-size 192 --box-stride 48" ;;
  7) NAME=medical_ficture_focus_boxes_b384_s96; CKPT=$MED; MODE=box; MAXSIDE=2048; EXTRA="--box-size 384 --box-stride 96" ;;
  *) echo "Unknown task ${SLURM_ARRAY_TASK_ID}" >&2; exit 2 ;;
esac

OUT=$ROOT_OUT/$NAME

echo "Job: ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
echo "Run: $NAME"
echo "Focus labels: $LABELS"
date

# shellcheck disable=SC2086
"$PY" inference/visium_hd_exp1/run_sam3_candidate_proposals.py \
  --image-path "$IMAGE" \
  --summary-path "$SUMMARY" \
  --checkpoint "$CKPT" \
  --device cuda \
  --output-dir "$OUT" \
  --proposal-mode "$MODE" \
  --labels "$LABELS" \
  --max-side "$MAXSIDE" \
  --min-area-frac 0.00003 \
  --max-area-frac 0.95 \
  --nms-iou 0.96 \
  --save-mask-limit 320 \
  --contact-sheet-limit 60 \
  $EXTRA

date
echo "Output: $OUT"
