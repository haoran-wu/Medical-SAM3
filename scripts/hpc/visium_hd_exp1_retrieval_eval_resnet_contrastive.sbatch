#!/bin/bash
#SBATCH --job-name=visium_ret_resnet
#SBATCH --partition=gpu_rtx6000
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=96G
#SBATCH --time=03:00:00
#SBATCH --output=/nfs/roberts/project/pi_xy48/hw646/Medical-SAM3/results/visium_hd_exp1/logs/retrieval_eval_resnet_contrastive_%j.log

set -euo pipefail

PROJECT=/nfs/roberts/project/pi_xy48/hw646/Medical-SAM3
DATASET=${PROJECT}/output_visium_hd_exp1_legacy/visium_hd_exp1/patch_dataset
RESULTS=${PROJECT}/results/visium_hd_exp1
PY=/nfs/roberts/project/pi_xy48/hw646/ycrc_conda/envs/medsam3/bin/python

export MPLCONFIGDIR=/tmp/hw646_mpl_ret_resnet_${SLURM_JOB_ID}
export PYTHONUNBUFFERED=1

mkdir -p "${RESULTS}/logs" "${RESULTS}/retrieval_eval" "${MPLCONFIGDIR}"
cd "${PROJECT}"

OUTDIR="${RESULTS}/retrieval_eval/resnet_contrastive_crop64_${SLURM_JOB_ID}"

"${PY}" "${PROJECT}/inference/visium_hd_exp1/evaluate_retrieval.py" \
  --checkpoint "${RESULTS}/training_runs/contrastive_resnet50_crop64_rtx6000_project_v2_11016975/best.pt" \
  --model-type contrastive \
  --items-csv "${DATASET}/items.csv" \
  --expr-npz "${DATASET}/expr_log1p.npz" \
  --scaler-npz "${DATASET}/scaler.npz" \
  --image-path /nfs/roberts/project/pi_xy48/hw646/Exp1/spatial/tissue_hires_image.png \
  --output-dir "${OUTDIR}" \
  --batch-size 128 \
  --num-workers 4
