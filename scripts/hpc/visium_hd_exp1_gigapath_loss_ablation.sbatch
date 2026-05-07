#!/bin/bash
#SBATCH --job-name=gigapath_loss_ablation
#SBATCH --partition=gpu_rtx6000
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=72G
#SBATCH --time=04:00:00
#SBATCH --array=0-11%2
#SBATCH --output=/nfs/roberts/project/pi_xy48/hw646/Medical-SAM3/results/visium_hd_exp1/logs/loss_ablation_%A_%a.log

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

mkdir -p "${RESULTS}/logs" "${RESULTS}/training_runs/loss_ablation" "${MPLCONFIGDIR}"
cd "${PROJECT}"

case "${SLURM_ARRAY_TASK_ID}" in
  0)  CLASS_SET=3class; INCLUDE_LABELS="tumor,stroma,immune infiltration"; NAME=ce_only;       ALIGN_LOSS=none;    ALIGN_WEIGHT=0.0; IMG_CE=1.0; EXPR_CE=1.0; MONITOR=val_macro_f1_expr ;;
  1)  CLASS_SET=3class; INCLUDE_LABELS="tumor,stroma,immune infiltration"; NAME=infonce_l005;  ALIGN_LOSS=infonce; ALIGN_WEIGHT=0.05; IMG_CE=1.0; EXPR_CE=1.0; MONITOR=val_macro_f1_expr ;;
  2)  CLASS_SET=3class; INCLUDE_LABELS="tumor,stroma,immune infiltration"; NAME=infonce_l010;  ALIGN_LOSS=infonce; ALIGN_WEIGHT=0.10; IMG_CE=1.0; EXPR_CE=1.0; MONITOR=val_macro_f1_expr ;;
  3)  CLASS_SET=3class; INCLUDE_LABELS="tumor,stroma,immune infiltration"; NAME=infonce_l020;  ALIGN_LOSS=infonce; ALIGN_WEIGHT=0.20; IMG_CE=1.0; EXPR_CE=1.0; MONITOR=val_macro_f1_expr ;;
  4)  CLASS_SET=3class; INCLUDE_LABELS="tumor,stroma,immune infiltration"; NAME=infonce_l050;  ALIGN_LOSS=infonce; ALIGN_WEIGHT=0.50; IMG_CE=1.0; EXPR_CE=1.0; MONITOR=val_macro_f1_expr ;;
  5)  CLASS_SET=3class; INCLUDE_LABELS="tumor,stroma,immune infiltration"; NAME=infonce_only;  ALIGN_LOSS=infonce; ALIGN_WEIGHT=1.00; IMG_CE=0.0; EXPR_CE=0.0; MONITOR=val_loss ;;
  6)  CLASS_SET=8class; INCLUDE_LABELS="";                                     NAME=ce_only;       ALIGN_LOSS=none;    ALIGN_WEIGHT=0.0; IMG_CE=1.0; EXPR_CE=1.0; MONITOR=val_macro_f1_expr ;;
  7)  CLASS_SET=8class; INCLUDE_LABELS="";                                     NAME=infonce_l005;  ALIGN_LOSS=infonce; ALIGN_WEIGHT=0.05; IMG_CE=1.0; EXPR_CE=1.0; MONITOR=val_macro_f1_expr ;;
  8)  CLASS_SET=8class; INCLUDE_LABELS="";                                     NAME=infonce_l010;  ALIGN_LOSS=infonce; ALIGN_WEIGHT=0.10; IMG_CE=1.0; EXPR_CE=1.0; MONITOR=val_macro_f1_expr ;;
  9)  CLASS_SET=8class; INCLUDE_LABELS="";                                     NAME=infonce_l020;  ALIGN_LOSS=infonce; ALIGN_WEIGHT=0.20; IMG_CE=1.0; EXPR_CE=1.0; MONITOR=val_macro_f1_expr ;;
  10) CLASS_SET=8class; INCLUDE_LABELS="";                                     NAME=infonce_l050;  ALIGN_LOSS=infonce; ALIGN_WEIGHT=0.50; IMG_CE=1.0; EXPR_CE=1.0; MONITOR=val_macro_f1_expr ;;
  11) CLASS_SET=8class; INCLUDE_LABELS="";                                     NAME=infonce_only;  ALIGN_LOSS=infonce; ALIGN_WEIGHT=1.00; IMG_CE=0.0; EXPR_CE=0.0; MONITOR=val_loss ;;
  *) echo "Unknown task id: ${SLURM_ARRAY_TASK_ID}" >&2; exit 2 ;;
esac

OUT="${RESULTS}/training_runs/loss_ablation/gigapath_crop128_${CLASS_SET}_${NAME}_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"

COMMON_ARGS=(
  --items-csv "${DATASET}/items.csv"
  --expr-npz "${DATASET}/expr_log1p.npz"
  --scaler-npz "${DATASET}/scaler.npz"
  --image-path /nfs/roberts/project/pi_xy48/hw646/Exp1/spatial/tissue_hires_image.png
  --output-dir "${OUT}"
  --image-backbone gigapath
  --crop-size 128
  --input-size 224
  --batch-size 32
  --total-epochs 8
  --freeze-epochs 8
  --max-train-samples 10000
  --max-val-samples 2500
  --class-weight-mode sqrt_inv
  --sampler balanced
  --monitor "${MONITOR}"
  --early-stop-patience 2
  --align-loss "${ALIGN_LOSS}"
  --align-weight "${ALIGN_WEIGHT}"
  --contrastive-temperature 0.07
  --img-ce-weight "${IMG_CE}"
  --expr-ce-weight "${EXPR_CE}"
  --num-workers 4
)

if [[ -n "${INCLUDE_LABELS}" ]]; then
  COMMON_ARGS+=(--include-labels "${INCLUDE_LABELS}")
fi

echo "Loss ablation task ${SLURM_ARRAY_TASK_ID}: ${CLASS_SET} ${NAME}"
echo "Output: ${OUT}"

"${PY}" "${PROJECT}/inference/visium_hd_exp1/train_classification.py" "${COMMON_ARGS[@]}"
