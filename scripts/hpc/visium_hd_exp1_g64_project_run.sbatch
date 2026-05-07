#!/bin/bash
#SBATCH --job-name=g64_rtx6000_project_v2
#SBATCH --partition=gpu_rtx6000
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=72G
#SBATCH --time=04:00:00
#SBATCH --output=/nfs/roberts/project/pi_xy48/hw646/Medical-SAM3/results/visium_hd_exp1/logs/g64_rtx6000_project_v2_%j.log

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

"${PY}" "${PROJECT}/inference/visium_hd_exp1/train_classification.py" \
  --items-csv "${DATASET}/items.csv" \
  --expr-npz "${DATASET}/expr_log1p.npz" \
  --scaler-npz "${DATASET}/scaler.npz" \
  --image-path /nfs/roberts/project/pi_xy48/hw646/Exp1/spatial/tissue_hires_image.png \
  --output-dir "${RESULTS}/training_runs/cls_gigapath_crop64_metric_gpu_rtx6000_project_v2_${SLURM_JOB_ID}" \
  --image-backbone gigapath \
  --crop-size 64 \
  --input-size 224 \
  --batch-size 32 \
  --total-epochs 8 \
  --freeze-epochs 8 \
  --max-train-samples 10000 \
  --max-val-samples 2500 \
  --class-weight-mode sqrt_inv \
  --sampler balanced \
  --monitor val_macro_f1_expr \
  --early-stop-patience 2 \
  --num-workers 4
