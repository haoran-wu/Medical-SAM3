#!/bin/bash
#SBATCH --job-name=visium_retrieval_eval
#SBATCH --partition=gpu_rtx6000
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --array=0-2
#SBATCH --output=/nfs/roberts/project/pi_xy48/hw646/Medical-SAM3/results/visium_hd_exp1/logs/retrieval_eval_%A_%a.log

set -euo pipefail

PROJECT=/nfs/roberts/project/pi_xy48/hw646/Medical-SAM3
DATASET=${PROJECT}/output_visium_hd_exp1_legacy/visium_hd_exp1/patch_dataset
RESULTS=${PROJECT}/results/visium_hd_exp1
PY=/nfs/roberts/project/pi_xy48/hw646/ycrc_conda/envs/medsam3/bin/python

export HF_HOME=/nfs/roberts/project/pi_xy48/hw646/ycrc_conda/hf-home
export TRANSFORMERS_CACHE=${HF_HOME}/hub
export HF_HUB_CACHE=${HF_HOME}/hub
export MPLCONFIGDIR=/tmp/hw646_mpl_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}
export PYTHONUNBUFFERED=1

mkdir -p "${RESULTS}/logs" "${RESULTS}/retrieval_eval" "${MPLCONFIGDIR}"
cd "${PROJECT}"

NAMES=(
  "g64_8class"
  "g128_8class"
  "g128_3class"
)

CHECKPOINTS=(
  "${RESULTS}/training_runs/cls_gigapath_crop64_metric_gpu_devel_project_v2_devel_11016957/best.pt"
  "${RESULTS}/training_runs/cls_gigapath_crop128_metric_gpu_rtx6000_project_v2_11016973/best.pt"
  "${RESULTS}/training_runs/cls_gigapath_crop128_3class_gpu_rtx6000_project_v2_11016974/best.pt"
)

NAME=${NAMES[$SLURM_ARRAY_TASK_ID]}
CHECKPOINT=${CHECKPOINTS[$SLURM_ARRAY_TASK_ID]}
OUTDIR="${RESULTS}/retrieval_eval/${NAME}_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"

echo "Retrieval eval: ${NAME}"
echo "Checkpoint: ${CHECKPOINT}"
echo "Output: ${OUTDIR}"

"${PY}" "${PROJECT}/inference/visium_hd_exp1/evaluate_retrieval.py" \
  --checkpoint "${CHECKPOINT}" \
  --model-type classification \
  --items-csv "${DATASET}/items.csv" \
  --expr-npz "${DATASET}/expr_log1p.npz" \
  --scaler-npz "${DATASET}/scaler.npz" \
  --image-path /nfs/roberts/project/pi_xy48/hw646/Exp1/spatial/tissue_hires_image.png \
  --output-dir "${OUTDIR}" \
  --batch-size 64 \
  --num-workers 4 \
  --save-examples
