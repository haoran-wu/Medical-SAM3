#!/bin/bash
#SBATCH --job-name=train_strategy_retrieval
#SBATCH --partition=gpu_rtx6000
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --array=0-3
#SBATCH --output=/nfs/roberts/project/pi_xy48/hw646/Medical-SAM3/results/visium_hd_exp1/logs/train_strategy_retrieval_%A_%a.log

set -euo pipefail

PROJECT=/nfs/roberts/project/pi_xy48/hw646/Medical-SAM3
DATASET=${PROJECT}/output_visium_hd_exp1_legacy/visium_hd_exp1/patch_dataset
RESULTS=${PROJECT}/results/visium_hd_exp1
PY=/nfs/roberts/project/pi_xy48/hw646/ycrc_conda/envs/medsam3/bin/python

export HF_HOME=/nfs/roberts/project/pi_xy48/hw646/ycrc_conda/hf-home
export TRANSFORMERS_CACHE=${HF_HOME}/hub
export HF_HUB_CACHE=${HF_HOME}/hub
export MPLCONFIGDIR=/tmp/hw646_mpl_strategy_ret_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}
export PYTHONUNBUFFERED=1

mkdir -p "${RESULTS}/logs" "${RESULTS}/retrieval_eval/training_strategy" "${MPLCONFIGDIR}"
cd "${PROJECT}"

case "${SLURM_ARRAY_TASK_ID}" in
  0) RUN_DIR=gigapath_crop128_3class_frozen_constant_l005_11038859_0 ;;
  1) RUN_DIR=gigapath_crop128_3class_frozen_warmcos_l005_11038859_1 ;;
  2) RUN_DIR=gigapath_crop128_3class_last1_warmcos_l005_11038859_2 ;;
  3) RUN_DIR=gigapath_crop128_3class_last2_warmcos_l005_11038859_3 ;;
  *) echo "Unknown task id: ${SLURM_ARRAY_TASK_ID}" >&2; exit 2 ;;
esac

TRAIN_DIR="${RESULTS}/training_runs/training_strategy/${RUN_DIR}"
CKPT="${TRAIN_DIR}/best.pt"
OUT="${RESULTS}/retrieval_eval/training_strategy/${RUN_DIR}_retrieval_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"

echo "Training-strategy retrieval eval for ${RUN_DIR}"
echo "Checkpoint: ${CKPT}"
echo "Output: ${OUT}"

"${PY}" "${PROJECT}/inference/visium_hd_exp1/evaluate_retrieval.py" \
  --checkpoint "${CKPT}" \
  --model-type classification \
  --items-csv "${DATASET}/items.csv" \
  --expr-npz "${DATASET}/expr_log1p.npz" \
  --scaler-npz "${DATASET}/scaler.npz" \
  --image-path /nfs/roberts/project/pi_xy48/hw646/Exp1/spatial/tissue_hires_image.png \
  --output-dir "${OUT}" \
  --batch-size 64 \
  --num-workers 2 \
  --retrieval-chunk-size 512 \
  --fusion-alpha-sweep
