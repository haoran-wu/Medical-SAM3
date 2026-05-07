#!/bin/bash
#SBATCH --job-name=giga3_embed512_infonce
#SBATCH --partition=gpu_rtx6000
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=72G
#SBATCH --time=04:00:00
#SBATCH --array=0-3%4
#SBATCH --output=/nfs/roberts/project/pi_xy48/hw646/Medical-SAM3/results/visium_hd_exp1/logs/giga3_embed512_infonce_%A_%a.log

set -euo pipefail

PROJECT=/nfs/roberts/project/pi_xy48/hw646/Medical-SAM3
DATASET=${PROJECT}/output_visium_hd_exp1_legacy/visium_hd_exp1/patch_dataset
RESULTS=${PROJECT}/results/visium_hd_exp1
PY=/nfs/roberts/project/pi_xy48/hw646/ycrc_conda/envs/medsam3/bin/python

export HF_HOME=/nfs/roberts/project/pi_xy48/hw646/ycrc_conda/hf-home
export TRANSFORMERS_CACHE=${HF_HOME}/hub
export HF_HUB_CACHE=${HF_HOME}/hub
export MPLCONFIGDIR=/tmp/hw646_mpl_embed512_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}
export PYTHONUNBUFFERED=1

mkdir -p "${RESULTS}/logs" "${RESULTS}/training_runs/loss_ablation" "${MPLCONFIGDIR}"
cd "${PROJECT}"

case "${SLURM_ARRAY_TASK_ID}" in
  0) NAME=infonce_l005; ALIGN_WEIGHT=0.05 ;;
  1) NAME=infonce_l010; ALIGN_WEIGHT=0.10 ;;
  2) NAME=infonce_l020; ALIGN_WEIGHT=0.20 ;;
  3) NAME=ce_only;      ALIGN_WEIGHT=0.00 ;;
  *) echo "Unknown task id: ${SLURM_ARRAY_TASK_ID}" >&2; exit 2 ;;
esac

ALIGN_LOSS=infonce
if [[ "${NAME}" == "ce_only" ]]; then
  ALIGN_LOSS=none
fi

OUT="${RESULTS}/training_runs/loss_ablation/gigapath_crop128_3class_embed512_${NAME}_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"

"${PY}" "${PROJECT}/inference/visium_hd_exp1/train_classification.py" \
  --items-csv "${DATASET}/items.csv" \
  --expr-npz "${DATASET}/expr_log1p.npz" \
  --scaler-npz "${DATASET}/scaler.npz" \
  --image-path /nfs/roberts/project/pi_xy48/hw646/Exp1/spatial/tissue_hires_image.png \
  --output-dir "${OUT}" \
  --image-backbone gigapath \
  --embed-dim 512 \
  --crop-size 128 \
  --input-size 224 \
  --batch-size 32 \
  --total-epochs 8 \
  --freeze-epochs 8 \
  --max-train-samples 10000 \
  --max-val-samples 2500 \
  --include-labels "tumor,stroma,immune infiltration" \
  --class-weight-mode sqrt_inv \
  --sampler balanced \
  --monitor val_macro_f1_expr \
  --early-stop-patience 2 \
  --align-loss "${ALIGN_LOSS}" \
  --align-weight "${ALIGN_WEIGHT}" \
  --contrastive-temperature 0.07 \
  --img-ce-weight 1.0 \
  --expr-ce-weight 1.0 \
  --num-workers 4
