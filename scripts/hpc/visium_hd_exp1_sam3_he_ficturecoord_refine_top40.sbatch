#!/bin/bash
#SBATCH --job-name=sam3_ficturecoord
#SBATCH --partition=gpu_devel
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/nfs/roberts/project/pi_xy48/hw646/Medical-SAM3/results/visium_hd_exp1/logs/sam3_ficturecoord_%j.log
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=vettel.hwu@gmail.com

set -euo pipefail

PROJECT=/nfs/roberts/project/pi_xy48/hw646/Medical-SAM3
PY=/nfs/roberts/project/pi_xy48/hw646/ycrc_conda/envs/medsam3/bin/python

cd "$PROJECT"
export MPLCONFIGDIR="$PROJECT/output/.mplconfig"
export HF_HOME=/nfs/roberts/project/pi_xy48/hw646/ycrc_conda/hf-home
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
mkdir -p "$MPLCONFIGDIR"

$PY inference/visium_hd_exp1/run_sam3_refine_reference_candidates.py \
  --candidate-root output/visium_hd_exp1/reference_map_candidates/hex12_k12_original_smooth \
  --reference-map output/visium_hd_exp1/assets/hex_12.k12.pixel.png \
  --target-image output/visium_hd_exp1/assets/tissue_hires_image.png \
  --output-dir results/visium_hd_exp1/sam3_reference_candidate_refine/he_ficturecoord_smooth_top40_${SLURM_JOB_ID} \
  --checkpoint checkpoints/SAM3-base/sam3.pt \
  --device cuda \
  --max-side 2048 \
  --limit 40 \
  --box-margin 0.06 \
  --transform-json output/visium_hd_exp1/registration_debug/ficture_coordinate_ref_to_he.json \
  --use-affine-prior-mask \
  --clip-to-tissue \
  --prior-dilate-radius 72
