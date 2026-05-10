#!/bin/bash
#SBATCH --job-name=iamsam_sam3
#SBATCH --partition=gpu_rtx6000
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=120G
#SBATCH --time=08:00:00
#SBATCH --output=/nfs/roberts/project/pi_xy48/hw646/Medical-SAM3/results/visium_hd_exp1/logs/iamsam_sam3_%j.log
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=vettel.hwu@gmail.com

set -euo pipefail

PROJECT=/nfs/roberts/project/pi_xy48/hw646/Medical-SAM3
PY=/nfs/roberts/project/pi_xy48/hw646/ycrc_conda/envs/medsam3/bin/python
IMAGE=$PROJECT/output/visium_hd_exp1/assets/tissue_hires_image.png
SUMMARY_DIR=$PROJECT/output/visium_hd_exp1/sam3_local_region_summary
SUMMARY=$SUMMARY_DIR/region_summary.json
ROOT_OUT=$PROJECT/results/visium_hd_exp1/sam3_iamsam_retrial/iamsam_sweep_${SLURM_JOB_ID}

mkdir -p "$PROJECT/results/visium_hd_exp1/logs" "$SUMMARY_DIR" "$ROOT_OUT"
cd "$PROJECT"

export MPLCONFIGDIR="$PROJECT/output/.mplconfig"
export HF_HOME=/nfs/roberts/project/pi_xy48/hw646/ycrc_conda/hf-home
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
mkdir -p "$MPLCONFIGDIR" "$HF_HOME"

echo "Job: ${SLURM_JOB_ID}"
echo "Host: $(hostname)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
date

"$PY" - <<'PY'
import json
from pathlib import Path
from PIL import Image

project = Path("/nfs/roberts/project/pi_xy48/hw646/Medical-SAM3")
summary_dir = project / "output/visium_hd_exp1/sam3_local_region_summary"
summary_dir.mkdir(parents=True, exist_ok=True)
entries = [
    ("lung_bronchiola", "output_visium_hd_exp1_legacy/visium_hd_exp1/sam3_runs/base_sam3_multipoint_remaining_dev/masks/01_lung_bronchiola_target.png"),
    ("erythorocytes", "output_visium_hd_exp1_legacy/visium_hd_exp1/sam3_runs/base_sam3_multipoint_remaining_dev/masks/02_erythorocytes_target.png"),
    ("immune_infiltration", "output_visium_hd_exp1_legacy/visium_hd_exp1/sam3_runs/base_sam3_multipoint/masks/03_immune_infiltration_target.png"),
    ("lung_alveoli_normal_adjacent", "output_visium_hd_exp1_legacy/visium_hd_exp1/sam3_runs/base_sam3_multipoint_remaining_dev/masks/04_lung_alveoli_normal_adjacent_target.png"),
    ("lung_vessels", "output_visium_hd_exp1_legacy/visium_hd_exp1/sam3_runs/base_sam3_multipoint_remaining_dev/masks/05_lung_vessels_target.png"),
    ("pigment", "output_visium_hd_exp1_legacy/visium_hd_exp1/sam3_runs/base_sam3_multipoint_remaining_dev/masks/06_pigment_target.png"),
    ("stroma", "output_visium_hd_exp1_legacy/visium_hd_exp1/sam3_runs/base_sam3_multipoint/masks/07_stroma_target.png"),
    ("tumor", "output_visium_hd_exp1_legacy/visium_hd_exp1/sam3_runs/base_sam3_multipoint/masks/08_tumor_target.png"),
]
labels = []
for label, rel in entries:
    path = project / rel
    mask = Image.open(path).convert("L")
    labels.append(
        {
            "label": label,
            "slug": label,
            "mask_path": str(path),
            "area_pixels": int(sum(v > 127 for v in mask.getdata())),
        }
    )
summary = {
    "image_path": str(project / "output/visium_hd_exp1/assets/tissue_hires_image.png"),
    "labels": labels,
}
(summary_dir / "region_summary.json").write_text(json.dumps(summary, indent=2))
print(summary_dir / "region_summary.json")
PY

run_candidates () {
  local name="$1"
  local checkpoint="$2"
  local mode="$3"
  local max_side="$4"
  local extra_args="$5"
  local out="$ROOT_OUT/$name"

  echo "======================================================================"
  echo "Run: $name"
  echo "Checkpoint: $checkpoint"
  echo "Mode: $mode"
  echo "Max side: $max_side"
  echo "Extra: $extra_args"
  echo "Output: $out"

  # shellcheck disable=SC2086
  "$PY" inference/visium_hd_exp1/run_sam3_candidate_proposals.py \
    --image-path "$IMAGE" \
    --summary-path "$SUMMARY" \
    --checkpoint "$checkpoint" \
    --device cuda \
    --output-dir "$out" \
    --proposal-mode "$mode" \
    --max-side "$max_side" \
    --min-area-frac 0.00003 \
    --max-area-frac 0.95 \
    --nms-iou 0.96 \
    --save-mask-limit 160 \
    --contact-sheet-limit 50 \
    $extra_args
}

BASE=$PROJECT/checkpoints/SAM3-base/sam3.pt
MED=$PROJECT/checkpoints/Medical-SAM3/checkpoint.pt

# IAMSAM-like "everything mode": broad candidate pool from the H&E image.
run_candidates "base_dense_boxes_1536_b256_s128" "$BASE" "box" 1536 "--box-size 256 --box-stride 128"
run_candidates "medical_dense_boxes_1536_b256_s128" "$MED" "box" 1536 "--box-size 256 --box-stride 128"

# Complement box prompts with point prompts to catch small morphology islands.
run_candidates "base_dense_points_1536_step96" "$BASE" "point" 1536 "--grid-step 96 --border 48"
run_candidates "medical_dense_points_1536_step96" "$MED" "point" 1536 "--grid-step 96 --border 48"

# Larger-context boxes, useful for tumor/stroma/alveoli regions.
run_candidates "base_dense_boxes_2048_b512_s256" "$BASE" "box" 2048 "--box-size 512 --box-stride 256"
run_candidates "medical_dense_boxes_2048_b512_s256" "$MED" "box" 2048 "--box-size 512 --box-stride 256"

date
echo "Output root: $ROOT_OUT"
