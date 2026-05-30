#!/bin/bash
#SBATCH --job-name=vlm_pairwise
#SBATCH --partition=gpu_h200,gpu_b200,gpu_rtx6000
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=140G
#SBATCH --time=06:00:00
#SBATCH --output=/home/hw646/codex_salip_outputs/logs/vlm_pairwise_rank_%j.log
#SBATCH --mail-type=NONE

set -euo pipefail

PROJECT=${PROJECT:-/nfs/roberts/project/pi_xy48/hw646/Medical-SAM3}
PY=${PY:-/nfs/roberts/project/pi_xy48/hw646/ycrc_conda/envs/medsam3/bin/python}
RUN_SCRIPT=${RUN_SCRIPT:-/home/hw646/codex_salip_scripts/run_paired_vlm_pairwise_rank_test.py}
POOL_DIR=${POOL_DIR:?Set POOL_DIR}
MODEL_NAME=${MODEL_NAME:?Set MODEL_NAME}
MODEL_SLUG=${MODEL_SLUG:-$(echo "$MODEL_NAME" | tr '/:' '__')}
OUT_BASE=${OUT_BASE:-/home/hw646/codex_vlm_pairwise_rank_20260525}
RUN_NAME=${RUN_NAME:-pairwise_reverseblur90}
DEVICE=${DEVICE:-cuda}
ROUNDS=${ROUNDS:-5}
SEED=${SEED:-17}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-8}
LIMIT_LABELS=${LIMIT_LABELS:-0}

OUT=$OUT_BASE/${RUN_NAME}_${MODEL_SLUG}_${SLURM_JOB_ID}

cd "$PROJECT"
mkdir -p "$OUT" "$OUT_BASE" /home/hw646/codex_salip_outputs/logs

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
export VLM_OFFLOAD_FOLDER="${VLM_OFFLOAD_FOLDER:-$OUT/offload}"
export VLM_ALLOW_AUTOMODEL_FALLBACK="${VLM_ALLOW_AUTOMODEL_FALLBACK:-1}"
export VLM_MAX_MEMORY_CUDA="${VLM_MAX_MEMORY_CUDA:-70GiB}"
export VLM_MAX_MEMORY_CPU="${VLM_MAX_MEMORY_CPU:-140GiB}"
mkdir -p "$MPLCONFIGDIR" "$HF_HOME" "$VLM_OFFLOAD_FOLDER"

date
echo "Running pairwise VLM direct retrieval"
echo "MODEL_NAME=$MODEL_NAME"
echo "POOL_DIR=$POOL_DIR"
echo "ROUNDS=$ROUNDS"
echo "SEED=$SEED"
echo "LIMIT_LABELS=$LIMIT_LABELS"
if [[ -n "${HF_TOKEN:-}" ]]; then
  echo "HF_TOKEN=present"
else
  echo "HF_TOKEN=missing"
fi
echo "OUT=$OUT"

"$PY" "$RUN_SCRIPT" \
  --pool-dir "$POOL_DIR" \
  --output-dir "$OUT" \
  --model "$MODEL_NAME" \
  --device "$DEVICE" \
  --rounds "$ROUNDS" \
  --seed "$SEED" \
  --max-new-tokens "$MAX_NEW_TOKENS" \
  --limit-labels "$LIMIT_LABELS"

date
echo "Output: $OUT"
