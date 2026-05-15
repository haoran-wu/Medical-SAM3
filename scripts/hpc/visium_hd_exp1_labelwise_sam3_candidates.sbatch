#!/bin/bash
#SBATCH --job-name=sam3_labelcand
#SBATCH --partition=gpu_devel
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/nfs/roberts/project/pi_xy48/hw646/Medical-SAM3/results/visium_hd_exp1/logs/sam3_labelwise_candidates_%j.log

set -euo pipefail

PROJECT=/nfs/roberts/project/pi_xy48/hw646/Medical-SAM3
PY=/nfs/roberts/project/pi_xy48/hw646/ycrc_conda/envs/medsam3/bin/python
IMAGE=/nfs/roberts/project/pi_xy48/hw646/Exp1/spatial/tissue_hires_image.png
SUMMARY=$PROJECT/output_visium_hd_exp1_legacy/visium_hd_exp1/region_masks/region_summary.json
CHECKPOINT=$PROJECT/checkpoints/Medical-SAM3/checkpoint.pt
ROOT_OUT=$PROJECT/results/visium_hd_exp1/sam3_candidate_proposals/labelwise_text_box_${SLURM_JOB_ID}

mkdir -p "$PROJECT/results/visium_hd_exp1/logs" "$ROOT_OUT"
cd "$PROJECT"

echo "Job: ${SLURM_JOB_ID}"
echo "Host: $(hostname)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
date

# Label-specific search: one target at a time, with larger boxes than the
# earlier dense-box smoke test so the prompt is not just selecting a small tile.
labels=(
  "Lung Bronchiola"
  "erythorocytes"
  "immune infiltration"
  "lung alveoli (normal adjacent)"
  "lung vessels"
  "pigment"
  "stroma"
  "tumor"
)

prompts=(
  "lung bronchiola airway region"
  "erythrocytes red blood cell region"
  "immune infiltration lymphocyte rich region"
  "normal adjacent lung alveoli airspace region"
  "lung blood vessel region"
  "dark pigment region"
  "stroma connective tissue region"
  "tumor epithelial tumor region"
)

slugs=(
  "lung_bronchiola"
  "erythorocytes"
  "immune_infiltration"
  "lung_alveoli_normal_adjacent"
  "lung_vessels"
  "pigment"
  "stroma"
  "tumor"
)

box_sizes=(384 256 384 512 384 256 512 512)
strides=(192 128 192 256 192 128 256 256)
max_area_fracs=(0.35 0.22 0.35 0.50 0.30 0.15 0.50 0.50)

for i in "${!labels[@]}"; do
  label="${labels[$i]}"
  prompt="${prompts[$i]}"
  slug="${slugs[$i]}"
  box_size="${box_sizes[$i]}"
  stride="${strides[$i]}"
  max_area="${max_area_fracs[$i]}"
  out="$ROOT_OUT/$slug"

  echo "======================================================================"
  echo "Label: $label"
  echo "Prompt: $prompt"
  echo "Box size/stride/max area: $box_size / $stride / $max_area"
  echo "Output: $out"

  "$PY" inference/visium_hd_exp1/run_sam3_candidate_proposals.py \
    --image-path "$IMAGE" \
    --summary-path "$SUMMARY" \
    --checkpoint "$CHECKPOINT" \
    --device cuda \
    --output-dir "$out" \
    --labels "$label" \
    --text-prompt "$prompt" \
    --proposal-mode box \
    --max-side 1024 \
    --box-size "$box_size" \
    --box-stride "$stride" \
    --min-area-frac 0.0002 \
    --max-area-frac "$max_area" \
    --nms-iou 0.90 \
    --save-mask-limit 80 \
    --contact-sheet-limit 25
done

date
echo "Output root: $ROOT_OUT"
