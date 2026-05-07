#!/bin/bash
#SBATCH --job-name=loss_ablation_retrieval
#SBATCH --partition=gpu_rtx6000
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=72G
#SBATCH --time=04:00:00
#SBATCH --array=0-23
#SBATCH --output=/nfs/roberts/project/pi_xy48/hw646/Medical-SAM3/results/visium_hd_exp1/logs/loss_ablation_retrieval_%A_%a.log

set -euo pipefail

PROJECT=/nfs/roberts/project/pi_xy48/hw646/Medical-SAM3
DATASET=${PROJECT}/output_visium_hd_exp1_legacy/visium_hd_exp1/patch_dataset
RESULTS=${PROJECT}/results/visium_hd_exp1
PY=/nfs/roberts/project/pi_xy48/hw646/ycrc_conda/envs/medsam3/bin/python

export HF_HOME=/nfs/roberts/project/pi_xy48/hw646/ycrc_conda/hf-home
export TRANSFORMERS_CACHE=${HF_HOME}/hub
export HF_HUB_CACHE=${HF_HOME}/hub
export MPLCONFIGDIR=/tmp/hw646_mpl_retrieval_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}
export PYTHONUNBUFFERED=1

mkdir -p "${RESULTS}/logs" "${RESULTS}/retrieval_eval/loss_ablation" "${MPLCONFIGDIR}"
cd "${PROJECT}"

case "${SLURM_ARRAY_TASK_ID}" in
  0)  RUN_DIR=gigapath_crop128_3class_ce_only_11028639_0 ;;
  1)  RUN_DIR=gigapath_crop128_3class_infonce_l005_11028639_1 ;;
  2)  RUN_DIR=gigapath_crop128_3class_infonce_l010_11028639_2 ;;
  3)  RUN_DIR=gigapath_crop128_3class_infonce_l020_11028639_3 ;;
  4)  RUN_DIR=gigapath_crop128_3class_infonce_l050_11028639_4 ;;
  5)  RUN_DIR=gigapath_crop128_3class_infonce_only_11028639_5 ;;
  6)  RUN_DIR=gigapath_crop128_8class_ce_only_11028639_6 ;;
  7)  RUN_DIR=gigapath_crop128_8class_infonce_l005_11028639_7 ;;
  8)  RUN_DIR=gigapath_crop128_8class_infonce_l010_11028639_8 ;;
  9)  RUN_DIR=gigapath_crop128_8class_infonce_l020_11028639_9 ;;
  10) RUN_DIR=gigapath_crop128_8class_infonce_l050_11028639_10 ;;
  11) RUN_DIR=gigapath_crop128_8class_infonce_only_11028639_11 ;;
  12) RUN_DIR=resnet50_crop128_3class_ce_only_11032366_0 ;;
  13) RUN_DIR=resnet50_crop128_3class_infonce_l005_11032366_1 ;;
  14) RUN_DIR=resnet50_crop128_3class_infonce_l010_11032366_2 ;;
  15) RUN_DIR=resnet50_crop128_3class_infonce_l020_11032366_3 ;;
  16) RUN_DIR=resnet50_crop128_3class_infonce_l050_11032366_4 ;;
  17) RUN_DIR=resnet50_crop128_3class_infonce_only_11032366_5 ;;
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
OUT="${RESULTS}/retrieval_eval/loss_ablation/${RUN_DIR}_retrieval_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"

echo "Retrieval eval for ${RUN_DIR}"
echo "Waiting for checkpoint: ${CKPT}"
for _ in $(seq 1 240); do
  if [[ -s "${CKPT}" && -s "${TRAIN_DIR}/summary.json" ]]; then
    break
  fi
  sleep 30
done

if [[ ! -s "${CKPT}" ]]; then
  echo "Checkpoint not found after waiting: ${CKPT}" >&2
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
  --batch-size 256 \
  --num-workers 4 \
  --fusion-alpha-sweep
