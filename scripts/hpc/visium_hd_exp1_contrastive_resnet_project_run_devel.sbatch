#!/bin/bash
#SBATCH --job-name=cm_resnet64_devel_project_v2_devel
#SBATCH --partition=gpu_devel
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=72G
#SBATCH --time=04:00:00
#SBATCH --output=/nfs/roberts/project/pi_xy48/hw646/Medical-SAM3/results/visium_hd_exp1/logs/cm_resnet64_devel_project_v2_devel_%j.log

set -euo pipefail

PROJECT=/nfs/roberts/project/pi_xy48/hw646/Medical-SAM3
DATASET=${PROJECT}/output_visium_hd_exp1_legacy/visium_hd_exp1/patch_dataset
RESULTS=${PROJECT}/results/visium_hd_exp1
PY=/nfs/roberts/project/pi_xy48/hw646/ycrc_conda/envs/medsam3/bin/python

export HF_HOME=/nfs/roberts/project/pi_xy48/hw646/ycrc_conda/hf-home
export TRANSFORMERS_CACHE=${HF_HOME}/hub
export HF_HUB_CACHE=${HF_HOME}/hub
export MPLCONFIGDIR=/tmp/hw646_mpl_${SLURM_JOB_ID}
export PYTHONUNBUFFERED=1

mkdir -p "${RESULTS}/logs" "${RESULTS}/training_runs" "${MPLCONFIGDIR}"
cd "${PROJECT}"

"${PY}" "${PROJECT}/inference/visium_hd_exp1/train_contrastive.py" \
  --items-csv "${DATASET}/items.csv" \
  --expr-npz "${DATASET}/expr_log1p.npz" \
  --scaler-npz "${DATASET}/scaler.npz" \
  --image-path /nfs/roberts/project/pi_xy48/hw646/Exp1/spatial/tissue_hires_image.png \
  --output-dir "${RESULTS}/training_runs/contrastive_resnet50_crop64_devel_project_v2_devel_${SLURM_JOB_ID}" \
  --image-backbone resnet50 \
  --crop-size 64 \
  --input-size 64 \
  --batch-size 256 \
  --total-epochs 20 \
  --freeze-epochs 20 \
  --max-train-samples 50000 \
  --max-val-samples 10000 \
  --num-workers 4
