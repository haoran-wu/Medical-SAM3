#!/bin/bash
#SBATCH --job-name=sam3_extra
#SBATCH --partition=gpu_rtx6000
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=120G
#SBATCH --time=06:00:00
#SBATCH --output=/nfs/roberts/project/pi_xy48/hw646/Medical-SAM3/results/visium_hd_exp1/logs/sam3_extra_%x_%j.log
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=vettel.hwu@gmail.com

set -euo pipefail

GROUP="${1:?Usage: sbatch visium_hd_exp1_sam3_extra_group.sbatch GROUP}"

PROJECT=/nfs/roberts/project/pi_xy48/hw646/Medical-SAM3
PY=/nfs/roberts/project/pi_xy48/hw646/ycrc_conda/envs/medsam3/bin/python
IMAGE=$PROJECT/output/visium_hd_exp1/assets/tissue_hires_image.png
SUMMARY_DIR=$PROJECT/output/visium_hd_exp1/sam3_local_region_summary
SUMMARY=$SUMMARY_DIR/region_summary.json
ROOT_OUT=$PROJECT/results/visium_hd_exp1/sam3_iamsam_retrial/extra_${GROUP}_${SLURM_JOB_ID}
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
(summary_dir / "region_summary.json").write_text(json.dumps({"image_path": str(project / "output/visium_hd_exp1/assets/tissue_hires_image.png"), "labels": labels}, indent=2))
PY

BASE=$PROJECT/checkpoints/SAM3-base/sam3.pt
MED=$PROJECT/checkpoints/Medical-SAM3/checkpoint.pt

run_candidates () {
  local name="$1"
  local checkpoint="$2"
  local mode="$3"
  local max_side="$4"
  local min_area="$5"
  local max_area="$6"
  local save_limit="$7"
  shift 7
  local out="$ROOT_OUT/$name"

  echo "======================================================================"
  echo "Run: $name"
  echo "Checkpoint: $checkpoint"
  echo "Mode: $mode; max_side=$max_side; area=$min_area-$max_area"
  echo "Extra: $*"

  "$PY" inference/visium_hd_exp1/run_sam3_candidate_proposals.py \
    --image-path "$IMAGE" \
    --summary-path "$SUMMARY" \
    --checkpoint "$checkpoint" \
    --device cuda \
    --output-dir "$out" \
    --proposal-mode "$mode" \
    --max-side "$max_side" \
    --min-area-frac "$min_area" \
    --max-area-frac "$max_area" \
    --nms-iou 0.97 \
    --save-mask-limit "$save_limit" \
    --contact-sheet-limit 60 \
    "$@"
}

case "$GROUP" in
  base_micro_text)
    run_candidates "base_text_ery_micro_b64_s32" "$BASE" box 2048 0.000003 0.045 300 --labels erythorocytes --text-prompt "tiny erythrocyte red blood cell islands in H&E" --box-size 64 --box-stride 32
    run_candidates "base_text_pigment_micro_b64_s32" "$BASE" box 2048 0.000003 0.035 300 --labels pigment --text-prompt "dark pigment granules and deposits in H&E" --box-size 64 --box-stride 32
    run_candidates "base_text_immune_micro_b96_s48" "$BASE" box 2048 0.000006 0.090 260 --labels immune_infiltration --text-prompt "dense immune infiltration lymphocyte clusters in H&E" --box-size 96 --box-stride 48
    ;;
  medical_micro_text)
    run_candidates "medical_text_ery_micro_b64_s32" "$MED" box 2048 0.000003 0.045 300 --labels erythorocytes --text-prompt "tiny erythrocyte red blood cell islands in H&E" --box-size 64 --box-stride 32
    run_candidates "medical_text_pigment_micro_b64_s32" "$MED" box 2048 0.000003 0.035 300 --labels pigment --text-prompt "dark pigment granules and deposits in H&E" --box-size 64 --box-stride 32
    run_candidates "medical_text_immune_micro_b96_s48" "$MED" box 2048 0.000006 0.090 260 --labels immune_infiltration --text-prompt "dense immune infiltration lymphocyte clusters in H&E" --box-size 96 --box-stride 48
    ;;
  base_micro_notext)
    run_candidates "base_micro_boxes_b64_s32" "$BASE" box 2048 0.000003 0.070 360 --box-size 64 --box-stride 32
    run_candidates "base_micro_boxes_b96_s48" "$BASE" box 2048 0.000006 0.110 320 --box-size 96 --box-stride 48
    ;;
  medical_micro_notext)
    run_candidates "medical_micro_boxes_b64_s32" "$MED" box 2048 0.000003 0.070 360 --box-size 64 --box-stride 32
    run_candidates "medical_micro_boxes_b96_s48" "$MED" box 2048 0.000006 0.110 320 --box-size 96 --box-stride 48
    ;;
  base_medium_multiscale)
    run_candidates "base_medium_boxes_b128_s64" "$BASE" box 2048 0.00001 0.350 320 --box-size 128 --box-stride 64
    run_candidates "base_medium_boxes_b192_s64" "$BASE" box 2048 0.00001 0.450 320 --box-size 192 --box-stride 64
    ;;
  medical_medium_multiscale)
    run_candidates "medical_medium_boxes_b128_s64" "$MED" box 2048 0.00001 0.350 320 --box-size 128 --box-stride 64
    run_candidates "medical_medium_boxes_b192_s64" "$MED" box 2048 0.00001 0.450 320 --box-size 192 --box-stride 64
    ;;
  base_dense_points)
    run_candidates "base_points_step40" "$BASE" point 2048 0.00001 0.550 320 --grid-step 40 --border 20
    run_candidates "base_points_step32_cap1800" "$BASE" point 2048 0.00001 0.550 320 --grid-step 32 --border 16 --max-points 1800
    ;;
  medical_dense_points)
    run_candidates "medical_points_step40" "$MED" point 2048 0.00001 0.550 320 --grid-step 40 --border 20
    run_candidates "medical_points_step32_cap1800" "$MED" point 2048 0.00001 0.550 320 --grid-step 32 --border 16 --max-points 1800
    ;;
  base_large_multiscale)
    run_candidates "base_large_boxes_b512_s128" "$BASE" box 2048 0.00003 0.850 260 --box-size 512 --box-stride 128
    run_candidates "base_large_boxes_b768_s256" "$BASE" box 2048 0.00003 0.950 220 --box-size 768 --box-stride 256
    ;;
  medical_large_multiscale)
    run_candidates "medical_large_boxes_b512_s128" "$MED" box 2048 0.00003 0.850 260 --box-size 512 --box-stride 128
    run_candidates "medical_large_boxes_b768_s256" "$MED" box 2048 0.00003 0.950 220 --box-size 768 --box-stride 256
    ;;
  *)
    echo "Unknown group: $GROUP" >&2
    exit 2
    ;;
esac

"$PY" inference/visium_hd_exp1/summarize_iamsam_sam3_sweep.py \
  --sweep-root "$ROOT_OUT" \
  --output-dir "$SUMMARY_OUT" \
  --max-masks 48 \
  --min-delta 0.001

date
echo "Output root: $ROOT_OUT"
echo "Summary: $SUMMARY_OUT"
