#!/bin/bash
#SBATCH --job-name=iamsam_fine
#SBATCH --partition=gpu_rtx6000
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=120G
#SBATCH --time=10:00:00
#SBATCH --output=/nfs/roberts/project/pi_xy48/hw646/Medical-SAM3/results/visium_hd_exp1/logs/iamsam_fine_%j.log
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=vettel.hwu@gmail.com

set -euo pipefail

PROJECT=/nfs/roberts/project/pi_xy48/hw646/Medical-SAM3
PY=/nfs/roberts/project/pi_xy48/hw646/ycrc_conda/envs/medsam3/bin/python
IMAGE=$PROJECT/output/visium_hd_exp1/assets/tissue_hires_image.png
SUMMARY_DIR=$PROJECT/output/visium_hd_exp1/sam3_local_region_summary
SUMMARY=$SUMMARY_DIR/region_summary.json
ROOT_OUT=$PROJECT/results/visium_hd_exp1/sam3_iamsam_retrial/iamsam_fine_${SLURM_JOB_ID}
SUMMARY_OUT=$PROJECT/results/visium_hd_exp1/sam3_iamsam_retrial/iamsam_fine_${SLURM_JOB_ID}_summary

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
    labels.append({"label": label, "slug": label, "mask_path": str(path), "area_pixels": int(sum(v > 127 for v in mask.getdata()))})
summary = {"image_path": str(project / "output/visium_hd_exp1/assets/tissue_hires_image.png"), "labels": labels}
(summary_dir / "region_summary.json").write_text(json.dumps(summary, indent=2))
print(summary_dir / "region_summary.json")
PY

run_candidates () {
  local name="$1"
  local checkpoint="$2"
  local mode="$3"
  local max_side="$4"
  local max_area="$5"
  local extra_args="$6"
  local out="$ROOT_OUT/$name"

  echo "======================================================================"
  echo "Run: $name"
  echo "Checkpoint: $checkpoint"
  echo "Mode: $mode"
  echo "Max side: $max_side"
  echo "Max area: $max_area"
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
    --min-area-frac 0.00001 \
    --max-area-frac "$max_area" \
    --nms-iou 0.97 \
    --save-mask-limit 260 \
    --contact-sheet-limit 60 \
    $extra_args
}

BASE=$PROJECT/checkpoints/SAM3-base/sam3.pt
MED=$PROJECT/checkpoints/Medical-SAM3/checkpoint.pt

# Finer generic candidate pools than the previous sweep.
run_candidates "base_fine_boxes_2048_b192_s96" "$BASE" "box" 2048 0.45 "--box-size 192 --box-stride 96"
run_candidates "medical_fine_boxes_2048_b192_s96" "$MED" "box" 2048 0.45 "--box-size 192 --box-stride 96"
run_candidates "base_fine_points_2048_step64" "$BASE" "point" 2048 0.45 "--grid-step 64 --border 32"
run_candidates "medical_fine_points_2048_step64" "$MED" "point" 2048 0.45 "--grid-step 64 --border 32"

# Label-specific text + box prompting, closer to IAMSAM prompt mode plus semantic hints.
labels=(
  "lung_bronchiola"
  "erythorocytes"
  "immune_infiltration"
  "lung_alveoli_normal_adjacent"
  "lung_vessels"
  "pigment"
  "stroma"
  "tumor"
)
prompts=(
  "bronchiolar airway epithelium in lung H&E"
  "small red blood cell erythrocyte clusters in H&E"
  "dense immune lymphocyte infiltration in H&E"
  "normal adjacent lung alveoli airspace morphology"
  "blood vessel vascular lumen wall in lung H&E"
  "dark pigment deposits in H&E"
  "fibrotic stromal connective tissue in H&E"
  "tumor epithelial region in lung H&E"
)
box_sizes=(192 96 160 256 192 96 256 256)
strides=(96 48 80 128 96 48 128 128)
max_areas=(0.35 0.08 0.20 0.45 0.25 0.06 0.45 0.45)

for model_name in base medical; do
  if [[ "$model_name" == "base" ]]; then
    ckpt="$BASE"
  else
    ckpt="$MED"
  fi
  for i in "${!labels[@]}"; do
    label="${labels[$i]}"
    prompt="${prompts[$i]}"
    box_size="${box_sizes[$i]}"
    stride="${strides[$i]}"
    max_area="${max_areas[$i]}"
    out="$ROOT_OUT/${model_name}_text_box_${label}"

    echo "======================================================================"
    echo "Labelwise run: $model_name / $label"
    echo "Prompt: $prompt"
    echo "Box size/stride/max area: $box_size / $stride / $max_area"

    "$PY" inference/visium_hd_exp1/run_sam3_candidate_proposals.py \
      --image-path "$IMAGE" \
      --summary-path "$SUMMARY" \
      --checkpoint "$ckpt" \
      --device cuda \
      --output-dir "$out" \
      --labels "$label" \
      --text-prompt "$prompt" \
      --proposal-mode box \
      --max-side 2048 \
      --box-size "$box_size" \
      --box-stride "$stride" \
      --min-area-frac 0.000005 \
      --max-area-frac "$max_area" \
      --nms-iou 0.95 \
      --save-mask-limit 180 \
      --contact-sheet-limit 40
  done
done

"$PY" inference/visium_hd_exp1/summarize_iamsam_sam3_sweep.py \
  --sweep-root "$ROOT_OUT" \
  --output-dir "$SUMMARY_OUT" \
  --max-masks 32 \
  --min-delta 0.001

date
echo "Output root: $ROOT_OUT"
echo "Summary: $SUMMARY_OUT"
