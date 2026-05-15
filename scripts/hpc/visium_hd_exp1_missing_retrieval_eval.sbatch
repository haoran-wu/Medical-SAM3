#!/bin/bash
#SBATCH --job-name=missing_retrieval
#SBATCH --partition=gpu_rtx6000
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --array=0-4
#SBATCH --output=/nfs/roberts/project/pi_xy48/hw646/Medical-SAM3/results/visium_hd_exp1/logs/missing_retrieval_%A_%a.log

set -euo pipefail

PROJECT=/nfs/roberts/project/pi_xy48/hw646/Medical-SAM3
DATASET=${PROJECT}/output_visium_hd_exp1_legacy/visium_hd_exp1/patch_dataset
RESULTS=${PROJECT}/results/visium_hd_exp1
PY=/nfs/roberts/project/pi_xy48/hw646/ycrc_conda/envs/medsam3/bin/python

export HF_HOME=/nfs/roberts/project/pi_xy48/hw646/ycrc_conda/hf-home
export TRANSFORMERS_CACHE=${HF_HOME}/hub
export HF_HUB_CACHE=${HF_HOME}/hub
export MPLCONFIGDIR=/tmp/hw646_mpl_missing_ret_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}
export PYTHONUNBUFFERED=1

mkdir -p "${RESULTS}/logs" "${RESULTS}/retrieval_eval/missing" "${MPLCONFIGDIR}"
cd "${PROJECT}"

case "${SLURM_ARRAY_TASK_ID}" in
  0)
    GROUP=loss_ablation
    RUN_DIR=gigapath_crop128_3class_embed512_infonce_l005_11037673_0
    ;;
  1)
    GROUP=loss_ablation
    RUN_DIR=gigapath_crop128_3class_embed512_infonce_l010_11037673_1
    ;;
  2)
    GROUP=loss_ablation
    RUN_DIR=gigapath_crop128_3class_embed512_infonce_l020_11037673_2
    ;;
  3)
    GROUP=loss_ablation
    RUN_DIR=gigapath_crop128_3class_embed512_ce_only_11037673_3
    ;;
  4)
    GROUP=training_strategy
    RUN_DIR=gigapath_crop128_8class_last2_warmcos_l005_11040134
    ;;
  *)
    echo "Unknown task id: ${SLURM_ARRAY_TASK_ID}" >&2
    exit 2
    ;;
esac

TRAIN_DIR="${RESULTS}/training_runs/${GROUP}/${RUN_DIR}"
CKPT="${TRAIN_DIR}/best.pt"
OUT="${RESULTS}/retrieval_eval/missing/${RUN_DIR}_retrieval_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"

echo "Missing retrieval eval for ${RUN_DIR}"
echo "Checkpoint: ${CKPT}"
echo "Output: ${OUT}"

if [[ ! -s "${CKPT}" ]]; then
  echo "Checkpoint not found: ${CKPT}" >&2
  exit 3
fi

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
