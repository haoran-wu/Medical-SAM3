#!/bin/bash
#SBATCH --job-name=vlm_pair_ext
#SBATCH --partition=gpu_h200,gpu_b200,gpu_rtx6000
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=120G
#SBATCH --time=08:00:00
#SBATCH --output=/home/hw646/codex_salip_outputs/logs/vlm_direct_retrieval_pair_%j.log
#SBATCH --mail-type=NONE

set -euo pipefail

MODEL_NAME=${MODEL_NAME:?Set MODEL_NAME}
MODEL_SLUG=${MODEL_SLUG:?Set MODEL_SLUG}
PROJECT=${PROJECT:-/nfs/roberts/project/pi_xy48/hw646/Medical-SAM3}
PY=${PY:-/nfs/roberts/project/pi_xy48/hw646/ycrc_conda/envs/medsam3/bin/python}
RUN_SCRIPT=${RUN_SCRIPT:-/home/hw646/codex_salip_scripts/run_paired_vlm_hit_test.py}
PROMPT_MODE=${PROMPT_MODE:-description_only}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-256}
DEVICE=${DEVICE:-cuda}
LIMIT=${LIMIT:-0}

PLAIN_POOL=${PLAIN_POOL:-/home/hw646/codex_vlm_hit_tests_nohighlight/paired_he_ficture_good_mid_bad_6class_plain_masked_20260523}
REVERSE_POOL=${REVERSE_POOL:-/home/hw646/codex_vlm_hit_tests_reverseblur/paired_he_ficture_good_mid_bad_6class_reverse_blur_gray_20260523}
PLAIN_OUT_BASE=${PLAIN_OUT_BASE:-/home/hw646/codex_vlm_hit_tests_simple_ext_plain}
REVERSE_OUT_BASE=${REVERSE_OUT_BASE:-/home/hw646/codex_vlm_hit_tests_simple_ext_reverseblur}

cd "$PROJECT"
mkdir -p "$PLAIN_OUT_BASE" "$REVERSE_OUT_BASE" /home/hw646/codex_salip_outputs/logs

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
export VLM_OFFLOAD_FOLDER="${VLM_OFFLOAD_FOLDER:-/tmp/hw646_vlm_offload_${SLURM_JOB_ID}}"
export VLM_ALLOW_AUTOMODEL_FALLBACK="${VLM_ALLOW_AUTOMODEL_FALLBACK:-1}"
export VLM_MAX_MEMORY_CUDA="${VLM_MAX_MEMORY_CUDA:-70GiB}"
export VLM_MAX_MEMORY_CPU="${VLM_MAX_MEMORY_CPU:-140GiB}"
mkdir -p "$MPLCONFIGDIR" "$HF_HOME" "$VLM_OFFLOAD_FOLDER"

run_mode() {
  local mode=$1
  local pool=$2
  local out_base=$3
  local out=$out_base/vlm_simple_${mode}_${MODEL_SLUG}_${SLURM_JOB_ID}
  mkdir -p "$out"
  echo "============================================================"
  date
  echo "MODEL_NAME=$MODEL_NAME"
  echo "MODEL_SLUG=$MODEL_SLUG"
  echo "MODE=$mode"
  echo "POOL=$pool"
  echo "OUT=$out"
  "$PY" "$RUN_SCRIPT" \
    --pool-dir "$pool" \
    --output-dir "$out" \
    --model "$MODEL_NAME" \
    --device "$DEVICE" \
    --max-new-tokens "$MAX_NEW_TOKENS" \
    --limit "$LIMIT" \
    --prompt-mode "$PROMPT_MODE"
}

date
echo "Running paired light-gray and reverse-blur VLM direct retrieval"
echo "HF_HOME=$HF_HOME"
echo "HF_HUB_CACHE=$HF_HUB_CACHE"
echo "TRANSFORMERS_CACHE=$TRANSFORMERS_CACHE"
if [[ -n "${HF_TOKEN:-}" ]]; then
  echo "HF_TOKEN=present"
else
  echo "HF_TOKEN=missing"
fi
run_mode plain "$PLAIN_POOL" "$PLAIN_OUT_BASE"
run_mode reverseblur "$REVERSE_POOL" "$REVERSE_OUT_BASE"
date
