#!/bin/bash
#SBATCH --job-name=giga3_train_strategy
#SBATCH --partition=gpu_rtx6000
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=72G
#SBATCH --time=04:00:00
#SBATCH --array=0-3
#SBATCH --output=/nfs/roberts/project/pi_xy48/hw646/Medical-SAM3/results/visium_hd_exp1/logs/giga3_train_strategy_%A_%a.log

set -euo pipefail

PROJECT=/nfs/roberts/project/pi_xy48/hw646/Medical-SAM3
DATASET=${PROJECT}/output_visium_hd_exp1_legacy/visium_hd_exp1/patch_dataset
RESULTS=${PROJECT}/results/visium_hd_exp1
PY=/nfs/roberts/project/pi_xy48/hw646/ycrc_conda/envs/medsam3/bin/python

export HF_HOME=/nfs/roberts/project/pi_xy48/hw646/ycrc_conda/hf-home
export TRANSFORMERS_CACHE=${HF_HOME}/hub
export HF_HUB_CACHE=${HF_HOME}/hub
export MPLCONFIGDIR=/tmp/hw646_mpl_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}
export PYTHONUNBUFFERED=1

mkdir -p "${RESULTS}/logs" "${RESULTS}/training_runs/training_strategy" "${MPLCONFIGDIR}"
cd "${PROJECT}"

case "${SLURM_ARRAY_TASK_ID}" in
  0) NAME=frozen_constant_l005; SCHEDULE=constant;      FREEZE_EPOCHS=8; TRAINABLE_BLOCKS=0; WARMUP_RATIO=0.0 ;;
  1) NAME=frozen_warmcos_l005;  SCHEDULE=warmup_cosine; FREEZE_EPOCHS=8; TRAINABLE_BLOCKS=0; WARMUP_RATIO=0.1 ;;
  2) NAME=last1_warmcos_l005;   SCHEDULE=warmup_cosine; FREEZE_EPOCHS=1; TRAINABLE_BLOCKS=1; WARMUP_RATIO=0.1 ;;
  3) NAME=last2_warmcos_l005;   SCHEDULE=warmup_cosine; FREEZE_EPOCHS=1; TRAINABLE_BLOCKS=2; WARMUP_RATIO=0.1 ;;
  *) echo "Unknown task id: ${SLURM_ARRAY_TASK_ID}" >&2; exit 2 ;;
esac

OUT="${RESULTS}/training_runs/training_strategy/gigapath_crop128_3class_${NAME}_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"

echo "Training strategy task ${SLURM_ARRAY_TASK_ID}: ${NAME}"
echo "Output: ${OUT}"
echo "Schedule=${SCHEDULE}; freeze_epochs=${FREEZE_EPOCHS}; trainable_blocks=${TRAINABLE_BLOCKS}; warmup_ratio=${WARMUP_RATIO}"

"${PY}" "${PROJECT}/inference/visium_hd_exp1/train_classification.py" \
  --items-csv "${DATASET}/items.csv" \
  --expr-npz "${DATASET}/expr_log1p.npz" \
  --scaler-npz "${DATASET}/scaler.npz" \
  --image-path /nfs/roberts/project/pi_xy48/hw646/Exp1/spatial/tissue_hires_image.png \
  --output-dir "${OUT}" \
  --image-backbone gigapath \
  --embed-dim 128 \
  --crop-size 128 \
  --input-size 224 \
  --batch-size 32 \
  --total-epochs 8 \
  --freeze-epochs "${FREEZE_EPOCHS}" \
  --trainable-backbone-blocks "${TRAINABLE_BLOCKS}" \
  --lr 1e-4 \
  --lr-backbone 1e-5 \
  --lr-schedule "${SCHEDULE}" \
  --warmup-ratio "${WARMUP_RATIO}" \
  --max-train-samples 10000 \
  --max-val-samples 2500 \
  --include-labels "tumor,stroma,immune infiltration" \
  --class-weight-mode sqrt_inv \
  --sampler balanced \
  --monitor val_macro_f1_expr \
  --early-stop-patience 2 \
  --align-loss infonce \
  --align-weight 0.05 \
  --contrastive-temperature 0.07 \
  --img-ce-weight 1.0 \
  --expr-ce-weight 1.0 \
  --num-workers 4
