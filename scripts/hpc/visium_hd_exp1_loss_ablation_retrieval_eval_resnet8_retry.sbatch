#!/bin/bash
#SBATCH --job-name=retrieval_resnet8_retry
#SBATCH --partition=gpu_rtx6000
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --array=18-23
#SBATCH --output=/nfs/roberts/project/pi_xy48/hw646/Medical-SAM3/results/visium_hd_exp1/logs/retrieval_resnet8_retry_%A_%a.log

set -euo pipefail

PROJECT=/nfs/roberts/project/pi_xy48/hw646/Medical-SAM3
DATASET=${PROJECT}/output_visium_hd_exp1_legacy/visium_hd_exp1/patch_dataset
RESULTS=${PROJECT}/results/visium_hd_exp1
PY=/nfs/roberts/project/pi_xy48/hw646/ycrc_conda/envs/medsam3/bin/python

export HF_HOME=/nfs/roberts/project/pi_xy48/hw646/ycrc_conda/hf-home
export TRANSFORMERS_CACHE=${HF_HOME}/hub
export HF_HUB_CACHE=${HF_HOME}/hub
export MPLCONFIGDIR=/tmp/hw646_mpl_resnet8_retry_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}
export PYTHONUNBUFFERED=1

mkdir -p "${RESULTS}/logs" "${RESULTS}/retrieval_eval/loss_ablation" "${MPLCONFIGDIR}"
cd "${PROJECT}"

case "${SLURM_ARRAY_TASK_ID}" in
  18) RUN_DIR=resnet50_crop128_8class_ce_only_11032366_6 ;;
  19) RUN_DIR=resnet50_crop128_8class_infonce_l005_11032366_7 ;;
  20) RUN_DIR=resnet50_crop128_8class_infonce_l010_11032366_8 ;;
  21) RUN_DIR=resnet50_crop128_8class_infonce_l020_11032366_9 ;;
  22) RUN_DIR=resnet50_crop128_8class_infonce_l050_11032366_10 ;;
  23) RUN_DIR=resnet50_crop128_8class_infonce_only_11032366_11 ;;
  *) echo "Unknown task id: ${SLURM_ARRAY_TASK_ID}" >&2; exit 2 ;;
esac

TRAIN_DIR="${RESULTS}/training_runs/loss_ablation/${RUN_DIR}"
CKPT="${TRAIN_DIR}/best.pt"
OUT="${RESULTS}/retrieval_eval/loss_ablation/${RUN_DIR}_retrieval_11037672_${SLURM_ARRAY_TASK_ID}"

echo "Retrying chunked retrieval eval for ${RUN_DIR}"
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
  --batch-size 128 \
  --num-workers 2 \
  --retrieval-chunk-size 512 \
  --fusion-alpha-sweep
