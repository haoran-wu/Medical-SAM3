#!/bin/bash
#SBATCH --job-name=iamsam_pre
#SBATCH --partition=gpu_rtx6000
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --output=/nfs/roberts/project/pi_xy48/hw646/Medical-SAM3/results/visium_hd_exp1/logs/iamsam_pre_%x_%j.log
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=vettel.hwu@gmail.com

set -euo pipefail

VARIANT="${1:?Usage: sbatch visium_hd_exp1_iamsam_sam3_preprocess_group.sbatch VARIANT MODEL}"
MODEL="${2:?Usage: sbatch visium_hd_exp1_iamsam_sam3_preprocess_group.sbatch VARIANT MODEL}"

PROJECT=/nfs/roberts/project/pi_xy48/hw646/Medical-SAM3
PY=/nfs/roberts/project/pi_xy48/hw646/ycrc_conda/envs/medsam3/bin/python
IMAGE_ORIG=$PROJECT/output/visium_hd_exp1/assets/tissue_hires_image.png
IMAGE_DIR=$PROJECT/output/visium_hd_exp1/sam3_preprocessed_images
IMAGE=$IMAGE_DIR/${VARIANT}.png
SUMMARY=$PROJECT/output/visium_hd_exp1/sam3_local_region_summary/region_summary.json
ROOT_OUT=$PROJECT/results/visium_hd_exp1/sam3_iamsam_retrial/preprocess_${VARIANT}_${MODEL}_${SLURM_JOB_ID}
SUMMARY_OUT=${ROOT_OUT}_summary

mkdir -p "$PROJECT/results/visium_hd_exp1/logs" "$IMAGE_DIR" "$(dirname "$SUMMARY")"
cd "$PROJECT"

export MPLCONFIGDIR="$PROJECT/output/.mplconfig"
export HF_HOME=/nfs/roberts/project/pi_xy48/hw646/ycrc_conda/hf-home
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
mkdir -p "$MPLCONFIGDIR" "$HF_HOME"

echo "Job: ${SLURM_JOB_ID}"
echo "Variant: $VARIANT"
echo "Model: $MODEL"
echo "Host: $(hostname)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
date

"$PY" - "$VARIANT" "$IMAGE_ORIG" "$IMAGE" "$SUMMARY" <<'PY'
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageOps

variant = sys.argv[1]
image_orig = Path(sys.argv[2])
image_out = Path(sys.argv[3])
summary_path = Path(sys.argv[4])
project = Path("/nfs/roberts/project/pi_xy48/hw646/Medical-SAM3")

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
summary_path.write_text(json.dumps({"image_path": str(image_orig), "labels": labels}, indent=2))

img = Image.open(image_orig).convert("RGB")
if variant == "autocontrast_sharp":
    out = ImageOps.autocontrast(img, cutoff=0.4)
    out = ImageEnhance.Contrast(out).enhance(1.25)
    out = out.filter(ImageFilter.UnsharpMask(radius=1.2, percent=120, threshold=3))
elif variant in {"lab_clahe", "hematoxylin_focus", "sat_value_norm"}:
    from skimage import color, exposure

    arr = np.asarray(img).astype(np.float32) / 255.0
    if variant == "lab_clahe":
        lab = color.rgb2lab(arr)
        l = lab[..., 0] / 100.0
        lab[..., 0] = exposure.equalize_adapthist(l, clip_limit=0.02) * 100.0
        out_arr = np.clip(color.lab2rgb(lab) * 255, 0, 255).astype(np.uint8)
    elif variant == "hematoxylin_focus":
        hed = color.rgb2hed(arr)
        h = exposure.rescale_intensity(-hed[..., 0], out_range=(0, 1))
        h = exposure.equalize_adapthist(h, clip_limit=0.02)
        out_arr = np.dstack([h, h, h])
        out_arr = np.clip(out_arr * 255, 0, 255).astype(np.uint8)
    else:
        hsv = color.rgb2hsv(arr)
        hsv[..., 1] = exposure.equalize_adapthist(hsv[..., 1], clip_limit=0.02)
        hsv[..., 2] = exposure.equalize_adapthist(hsv[..., 2], clip_limit=0.02)
        out_arr = np.clip(color.hsv2rgb(hsv) * 255, 0, 255).astype(np.uint8)
    out = Image.fromarray(out_arr)
else:
    raise SystemExit(f"Unknown variant: {variant}")

out.save(image_out)
print(image_out)
PY

case "$MODEL" in
  base) CKPT=$PROJECT/checkpoints/SAM3-base/sam3.pt ;;
  medical) CKPT=$PROJECT/checkpoints/Medical-SAM3/checkpoint.pt ;;
  *) echo "Unknown model: $MODEL" >&2; exit 2 ;;
esac

"$PY" inference/visium_hd_exp1/run_sam3_candidate_proposals.py \
  --image-path "$IMAGE" \
  --summary-path "$SUMMARY" \
  --checkpoint "$CKPT" \
  --device cuda \
  --output-dir "$ROOT_OUT/dense_boxes_1536_b256_s128" \
  --proposal-mode box \
  --max-side 1536 \
  --box-size 256 \
  --box-stride 128 \
  --min-area-frac 0.00003 \
  --max-area-frac 0.95 \
  --nms-iou 0.96 \
  --save-mask-limit 160 \
  --contact-sheet-limit 50

"$PY" inference/visium_hd_exp1/summarize_iamsam_sam3_sweep.py \
  --sweep-root "$ROOT_OUT" \
  --output-dir "$SUMMARY_OUT" \
  --max-masks 40 \
  --min-delta 0.001

date
echo "Output root: $ROOT_OUT"
echo "Summary: $SUMMARY_OUT"
