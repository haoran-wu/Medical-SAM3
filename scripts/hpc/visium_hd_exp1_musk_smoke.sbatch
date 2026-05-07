#!/bin/bash
#SBATCH --job-name=musk_smoke
#SBATCH --partition=gpu_rtx6000
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=01:00:00
#SBATCH --output=/nfs/roberts/project/pi_xy48/hw646/Medical-SAM3/results/visium_hd_exp1/logs/musk_smoke_%j.log

set -euo pipefail

PROJECT=/nfs/roberts/project/pi_xy48/hw646/Medical-SAM3
DATASET=${PROJECT}/output_visium_hd_exp1_legacy/visium_hd_exp1/patch_dataset
RESULTS=${PROJECT}/results/visium_hd_exp1
PY=/nfs/roberts/project/pi_xy48/hw646/ycrc_conda/envs/medsam3/bin/python

export HF_HOME=/nfs/roberts/project/pi_xy48/hw646/ycrc_conda/hf-home
export TRANSFORMERS_CACHE=${HF_HOME}/hub
export HF_HUB_CACHE=${HF_HOME}/hub
export PYTHONPATH="${PROJECT}/third_party/MUSK:${PYTHONPATH:-}"
export MPLCONFIGDIR=/tmp/hw646_mpl_${SLURM_JOB_ID}
export PYTHONUNBUFFERED=1

mkdir -p "${RESULTS}/logs" "${RESULTS}/training_runs/musk_smoke" "${MPLCONFIGDIR}"
cd "${PROJECT}"

OUT="${RESULTS}/training_runs/musk_smoke/musk_crop128_input384_3class_${SLURM_JOB_ID}"

echo "MUSK smoke test"
echo "Output: ${OUT}"
echo "Goal: verify MUSK HF access, adapter output, GPU memory, and a tiny 3-class training pass."

"${PY}" "${PROJECT}/inference/visium_hd_exp1/train_classification.py" \
  --items-csv "${DATASET}/items.csv" \
  --expr-npz "${DATASET}/expr_log1p.npz" \
  --scaler-npz "${DATASET}/scaler.npz" \
  --image-path /nfs/roberts/project/pi_xy48/hw646/Exp1/spatial/tissue_hires_image.png \
  --output-dir "${OUT}" \
  --image-backbone musk \
  --crop-size 128 \
  --input-size 384 \
  --batch-size 8 \
  --total-epochs 1 \
  --freeze-epochs 1 \
  --max-train-samples 256 \
  --max-val-samples 128 \
  --include-labels "tumor,stroma,immune infiltration" \
  --class-weight-mode sqrt_inv \
  --sampler balanced \
  --monitor val_macro_f1_expr \
  --align-loss infonce \
  --align-weight 0.05 \
  --contrastive-temperature 0.07 \
  --img-ce-weight 1.0 \
  --expr-ce-weight 1.0 \
  --num-workers 2
