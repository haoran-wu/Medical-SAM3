#!/bin/bash
#SBATCH --job-name=jun11_hegw3
#SBATCH --partition=gpu_b200,gpu_h200,gpu_rtx6000
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=160G
#SBATCH --time=02:00:00
#SBATCH --output=/nfs/roberts/project/pi_xy48/hw646/codex_salip_outputs/logs/jun11_hegraywhite_evidence3_%j.log
#SBATCH --mail-type=NONE

set -euo pipefail

PROJECT=${PROJECT:-/nfs/roberts/project/pi_xy48/hw646/Medical-SAM3}
PY=${PY:-/nfs/roberts/project/pi_xy48/hw646/ycrc_conda/envs/medsam3/bin/python}
SCRIPT_DIR=${SCRIPT_DIR:-/nfs/roberts/project/pi_xy48/hw646/codex_salip_scripts_jun11}
RUN_SCRIPT=${RUN_SCRIPT:-$SCRIPT_DIR/run_local_vlm_evidence_first_he_graywhite.py}
POOL_CSV=${POOL_CSV:-/nfs/roberts/project/pi_xy48/hw646/Medical-SAM3_remote_runs/visium_hd_exp1/jun11_he_graywhite_evidence3/corrected_pool/public_vlm_requests.csv}
MODEL_NAME=${MODEL_NAME:-Qwen/Qwen3-VL-32B-Instruct}
MODEL_SLUG=${MODEL_SLUG:-qwen3vl32b}
OUT_BASE=${OUT_BASE:-/nfs/roberts/project/pi_xy48/hw646/Medical-SAM3_remote_runs/visium_hd_exp1/jun11_he_graywhite_evidence3/local_vlm_outputs}
RUN_NAME=${RUN_NAME:-Jun11_hegraywhite_evidence3}
DEVICE=${DEVICE:-cuda}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-2200}

OUT="$OUT_BASE/${RUN_NAME}_${MODEL_SLUG}_${SLURM_JOB_ID}"

cd "$PROJECT"
mkdir -p "$OUT" "$OUT_BASE" /nfs/roberts/project/pi_xy48/hw646/codex_salip_outputs/logs

export MPLCONFIGDIR="${MPLCONFIGDIR_OVERRIDE:-$PROJECT/output/.mplconfig}"
export HF_HOME="${HF_HOME_OVERRIDE:-/nfs/roberts/project/pi_xy48/hw646/hf-home}"
export HF_HUB_CACHE="$HF_HOME/hub"
export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
export PYTHONUNBUFFERED=1
export PYTHONPATH="$SCRIPT_DIR:$PROJECT/inference/visium_hd_exp1:$PROJECT/inference:$PROJECT:${PYTHONPATH:-}"
export VLM_DEVICE_MAP="${VLM_DEVICE_MAP:-auto}"
export VLM_ALLOW_AUTOMODEL_FALLBACK="${VLM_ALLOW_AUTOMODEL_FALLBACK:-1}"
export VLM_OFFLOAD_FOLDER="${VLM_OFFLOAD_FOLDER:-/tmp/hw646_jun11_hegw3_offload_${SLURM_JOB_ID}}"
export VLM_MAX_MEMORY_CUDA="${VLM_MAX_MEMORY_CUDA:-130GiB}"
export VLM_MAX_MEMORY_CPU="${VLM_MAX_MEMORY_CPU:-180GiB}"
mkdir -p "$MPLCONFIGDIR" "$HF_HOME" "$VLM_OFFLOAD_FOLDER"

date
echo "Running Jun11 H&E grayscale-white evidence-first local VLM"
echo "MODEL_NAME=$MODEL_NAME"
echo "POOL_CSV=$POOL_CSV"
echo "OUT=$OUT"
echo "RUN_SCRIPT=$RUN_SCRIPT"

"$PY" "$RUN_SCRIPT" \
  --pool-csv "$POOL_CSV" \
  --output-dir "$OUT" \
  --model "$MODEL_NAME" \
  --device "$DEVICE" \
  --max-new-tokens "$MAX_NEW_TOKENS" \
  --resume

date
echo "Output: $OUT"
