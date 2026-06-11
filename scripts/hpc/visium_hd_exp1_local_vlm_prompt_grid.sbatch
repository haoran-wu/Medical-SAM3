#!/bin/bash
#SBATCH --job-name=vlm_pgrid
#SBATCH --partition=gpu_h200,gpu_b200,gpu_rtx6000
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=120G
#SBATCH --time=04:00:00
#SBATCH --output=/home/hw646/codex_salip_outputs/logs/vlm_prompt_grid_%j.log
#SBATCH --mail-type=NONE

set -euo pipefail

PROJECT=${PROJECT:-/nfs/roberts/project/pi_xy48/hw646/Medical-SAM3}
PY=${PY:-/nfs/roberts/project/pi_xy48/hw646/ycrc_conda/envs/medsam3/bin/python}
RUN_SCRIPT=${RUN_SCRIPT:-/home/hw646/codex_salip_scripts/run_local_vlm_multiimage_prompt_grid.py}
POOL_CSV=${POOL_CSV:?Set POOL_CSV}
OUT_BASE=${OUT_BASE:?Set OUT_BASE}
RUN_NAME=${RUN_NAME:?Set RUN_NAME}
PROMPT_GRID=${PROMPT_GRID:?Set PROMPT_GRID}
MODEL_NAME=${MODEL_NAME:?Set MODEL_NAME}
DEVICE=${DEVICE:-cuda}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-192}
LIMIT=${LIMIT:-0}

cd "$PROJECT"
mkdir -p "$OUT_BASE" /home/hw646/codex_salip_outputs/logs

export MPLCONFIGDIR="${MPLCONFIGDIR_OVERRIDE:-$PROJECT/output/.mplconfig}"
export HF_HOME="${HF_HOME_OVERRIDE:-/tmp/hw646_hf_${SLURM_JOB_ID}}"
export HF_HUB_CACHE="$HF_HOME/hub"
export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
DEFAULT_HF_TOKEN_PATH=${DEFAULT_HF_TOKEN_PATH:-$HOME/.cache/huggingface/token}
if [[ -z "${HF_TOKEN:-}" && -s "$DEFAULT_HF_TOKEN_PATH" ]]; then
  export HF_TOKEN
  HF_TOKEN="$(cat "$DEFAULT_HF_TOKEN_PATH")"
fi
export PYTHONUNBUFFERED=1
export PYTHONPATH="$PROJECT/inference/visium_hd_exp1:$PROJECT/inference:$PROJECT:${PYTHONPATH:-}"
export VLM_DEVICE_MAP="${VLM_DEVICE_MAP:-auto}"
export VLM_OFFLOAD_FOLDER="${VLM_OFFLOAD_FOLDER:-/tmp/hw646_vlm_pgrid_offload_${SLURM_JOB_ID}}"
export VLM_ALLOW_AUTOMODEL_FALLBACK="${VLM_ALLOW_AUTOMODEL_FALLBACK:-1}"
export VLM_MAX_MEMORY_CUDA="${VLM_MAX_MEMORY_CUDA:-70GiB}"
export VLM_MAX_MEMORY_CPU="${VLM_MAX_MEMORY_CPU:-120GiB}"
mkdir -p "$MPLCONFIGDIR" "$HF_HOME" "$VLM_OFFLOAD_FOLDER"

date
echo "Running local VLM prompt grid"
echo "MODEL_NAME=$MODEL_NAME"
echo "POOL_CSV=$POOL_CSV"
echo "OUT_BASE=$OUT_BASE"
echo "RUN_NAME=$RUN_NAME"
echo "PROMPT_GRID=$PROMPT_GRID"
echo "LIMIT=$LIMIT"
if [[ -n "${HF_TOKEN:-}" ]]; then
  echo "HF_TOKEN=present"
else
  echo "HF_TOKEN=missing"
fi

"$PY" "$RUN_SCRIPT" \
  --pool-csv "$POOL_CSV" \
  --output-base "$OUT_BASE" \
  --run-name "$RUN_NAME" \
  --prompt-grid "$PROMPT_GRID" \
  --model "$MODEL_NAME" \
  --device "$DEVICE" \
  --max-new-tokens "$MAX_NEW_TOKENS" \
  --limit "$LIMIT" \
  --resume

date
echo "Prompt grid output base: $OUT_BASE"
