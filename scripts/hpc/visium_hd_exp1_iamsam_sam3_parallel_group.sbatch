#!/bin/bash
#SBATCH --job-name=iamsam_group
#SBATCH --partition=gpu_rtx6000
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=120G
#SBATCH --time=06:00:00
#SBATCH --output=/nfs/roberts/project/pi_xy48/hw646/Medical-SAM3/results/visium_hd_exp1/logs/iamsam_group_%x_%j.log
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=vettel.hwu@gmail.com

set -euo pipefail

GROUP="${1:?Usage: sbatch visium_hd_exp1_iamsam_sam3_parallel_group.sbatch GROUP}"

PROJECT=/nfs/roberts/project/pi_xy48/hw646/Medical-SAM3
PY=/nfs/roberts/project/pi_xy48/hw646/ycrc_conda/envs/medsam3/bin/python
IMAGE=$PROJECT/output/visium_hd_exp1/assets/tissue_hires_image.png
SUMMARY_DIR=$PROJECT/output/visium_hd_exp1/sam3_local_region_summary
SUMMARY=$SUMMARY_DIR/region_summary.json
ROOT_OUT=$PROJECT/results/visium_hd_exp1/sam3_iamsam_retrial/parallel_${GROUP}_${SLURM_JOB_ID}
SUMMARY_OUT=${ROOT_OUT}_summary

mkdir -p "$PROJECT/results/visium_hd_exp1/logs" "$SUMMARY_DIR" "$ROOT_OUT"
cd "$PROJECT"

export MPLCONFIGDIR="$PROJECT/output/.mplconfig"
export HF_HOME=/nfs/roberts/project/pi_xy48/hw646/ycrc_conda/hf-home
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
mkdir -p "$MPLCONFIGDIR" "$HF_HOME"

echo "Job: ${SLURM_JOB_ID}"
echo "Group: $GROUP"
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
PY

BASE=$PROJECT/checkpoints/SAM3-base/sam3.pt
MED=$PROJECT/checkpoints/Medical-SAM3/checkpoint.pt

run_candidates () {
  local name="$1"
  local checkpoint="$2"
  local mode="$3"
  local max_side="$4"
  local max_area="$5"
  shift 5
  local out="$ROOT_OUT/$name"

  echo "======================================================================"
  echo "Run: $name"
  echo "Checkpoint: $checkpoint"
  echo "Mode: $mode; max_side=$max_side; max_area=$max_area"
  echo "Extra: $*"

  "$PY" inference/visium_hd_exp1/run_sam3_candidate_proposals.py \
    --image-path "$IMAGE" \
    --summary-path "$SUMMARY" \
    --checkpoint "$checkpoint" \
    --device cuda \
    --output-dir "$out" \
    --proposal-mode "$mode" \
    --max-side "$max_side" \
    --min-area-frac 0.000005 \
    --max-area-frac "$max_area" \
    --nms-iou 0.96 \
    --save-mask-limit 260 \
    --contact-sheet-limit 50 \
    "$@"
}

case "$GROUP" in
  base_small_text)
    run_candidates "base_text_box_erythorocytes_fine" "$BASE" "box" 2048 0.055 --labels erythorocytes --text-prompt "small red blood cell erythrocyte clusters in H&E" --box-size 80 --box-stride 40
    run_candidates "base_text_box_pigment_fine" "$BASE" "box" 2048 0.040 --labels pigment --text-prompt "dark pigment deposits in H&E" --box-size 80 --box-stride 40
    ;;
  medical_small_text)
    run_candidates "medical_text_box_erythorocytes_fine" "$MED" "box" 2048 0.055 --labels erythorocytes --text-prompt "small red blood cell erythrocyte clusters in H&E" --box-size 80 --box-stride 40
    run_candidates "medical_text_box_pigment_fine" "$MED" "box" 2048 0.040 --labels pigment --text-prompt "dark pigment deposits in H&E" --box-size 80 --box-stride 40
    ;;
  base_small_notext)
    run_candidates "base_small_boxes_no_text" "$BASE" "box" 2048 0.080 --box-size 96 --box-stride 48
    ;;
  medical_point_wide)
    run_candidates "medical_points_wide_step64" "$MED" "point" 1536 0.950 --grid-step 64 --border 32
    run_candidates "medical_points_wide_step48" "$MED" "point" 1536 0.950 --grid-step 48 --border 24
    ;;
  base_large_context)
    run_candidates "base_large_text_tumor_stroma_alveoli" "$BASE" "box" 2048 0.650 --labels tumor,stroma,lung_alveoli_normal_adjacent --text-prompt "large tumor stroma and alveoli tissue region in H&E" --box-size 384 --box-stride 192
    run_candidates "base_large_boxes_no_text" "$BASE" "box" 2048 0.700 --box-size 384 --box-stride 192
    ;;
  medical_large_context)
    run_candidates "medical_large_text_tumor_stroma_alveoli" "$MED" "box" 2048 0.650 --labels tumor,stroma,lung_alveoli_normal_adjacent --text-prompt "large tumor stroma and alveoli tissue region in H&E" --box-size 384 --box-stride 192
    run_candidates "medical_large_boxes_no_text" "$MED" "box" 2048 0.700 --box-size 384 --box-stride 192
    ;;
  *)
    echo "Unknown group: $GROUP" >&2
    exit 2
    ;;
esac

"$PY" inference/visium_hd_exp1/summarize_iamsam_sam3_sweep.py \
  --sweep-root "$ROOT_OUT" \
  --output-dir "$SUMMARY_OUT" \
  --max-masks 40 \
  --min-delta 0.001

date
echo "Output root: $ROOT_OUT"
echo "Summary: $SUMMARY_OUT"
