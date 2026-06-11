#!/bin/bash
#SBATCH --job-name=jun10_fic_filter
#SBATCH --partition=gpu_h200
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=160G
#SBATCH --time=06:00:00
#SBATCH --output=/nfs/roberts/project/pi_xy48/hw646/codex_salip_outputs/logs/jun10_ficture_filter_%j.log
#SBATCH --error=/nfs/roberts/project/pi_xy48/hw646/codex_salip_outputs/logs/jun10_ficture_filter_%j.log
#SBATCH --mail-type=NONE

set -euo pipefail

PROJECT=${PROJECT:-/nfs/roberts/project/pi_xy48/hw646/Medical-SAM3}
PY=${PY:-/nfs/roberts/project/pi_xy48/hw646/ycrc_conda/envs/medsam3/bin/python}
RUN_SCRIPT=${RUN_SCRIPT:-$PROJECT/inference/visium_hd_exp1/run_qwen3_ficture_image_hypothesis_verification.py}

REQUESTS_CSV=${REQUESTS_CSV:-$PROJECT/output/visium_hd_exp1/final_deliverables/Jun10_CellTypeFirst_Qwen3_Skill/cell_type_first_requests.csv}
STEP1_CSV=${STEP1_CSV:-/nfs/roberts/project/pi_xy48/hw646/Medical-SAM3_remote_runs/visium_hd_exp1/jun10_cell_type_first_qwen3/step1_qwen3vl32b_full167_limit167_14252238/step1_hypotheses.csv}
IMAGE_ROOT=${IMAGE_ROOT:-$PROJECT/output/visium_hd_exp1/final_deliverables/Jun09_SkillRanker_ComponentAwareMaskSelection/corrected_pool}
OUT_BASE=${OUT_BASE:-/nfs/roberts/project/pi_xy48/hw646/Medical-SAM3_remote_runs/visium_hd_exp1/jun10_ficture_image_filter_qwen3}
MODEL_NAME=${MODEL_NAME:-Qwen/Qwen3-VL-32B-Instruct}
LIMIT=${LIMIT:-0}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-128}
PROMPT_STYLE=${PROMPT_STYLE:-strict_spatial}

OUT="$OUT_BASE/ficture_filter_qwen3vl32b_${PROMPT_STYLE}_full167_${SLURM_JOB_ID}"

mkdir -p "$OUT" /nfs/roberts/project/pi_xy48/hw646/codex_salip_outputs/logs
cd "$PROJECT"

export PYTHONUNBUFFERED=1
export PYTHONPATH="$PROJECT/inference/visium_hd_exp1:$PROJECT/inference:$PROJECT:${PYTHONPATH:-}"
export MPLCONFIGDIR="${MPLCONFIGDIR_OVERRIDE:-/tmp/${USER}_mpl_${SLURM_JOB_ID}}"
export HF_HOME="${HF_HOME_OVERRIDE:-/nfs/roberts/project/pi_xy48/hw646/hf-home}"
export HF_HUB_CACHE="$HF_HOME/hub"
export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"
export TRANSFORMERS_CACHE="$HF_HOME/hub"
export VLM_DEVICE_MAP="${VLM_DEVICE_MAP:-auto}"
export VLM_ALLOW_AUTOMODEL_FALLBACK="${VLM_ALLOW_AUTOMODEL_FALLBACK:-1}"
export VLM_MAX_MEMORY_CUDA="${VLM_MAX_MEMORY_CUDA:-150GiB}"
export VLM_MAX_MEMORY_CPU="${VLM_MAX_MEMORY_CPU:-160GiB}"
export VLM_OFFLOAD_FOLDER="${VLM_OFFLOAD_FOLDER:-/tmp/hw646_jun10_ficture_filter_${SLURM_JOB_ID}}"
mkdir -p "$MPLCONFIGDIR" "$HF_HOME" "$VLM_OFFLOAD_FOLDER"

TOKEN_PATH=${TOKEN_PATH:-$HOME/.cache/huggingface/token}
if [[ -z "${HF_TOKEN:-}" && -s "$TOKEN_PATH" ]]; then
  export HF_TOKEN
  HF_TOKEN="$(cat "$TOKEN_PATH")"
fi

echo "[jun10_ficture_filter] job=${SLURM_JOB_ID} node=$(hostname)"
echo "[jun10_ficture_filter] model=${MODEL_NAME}"
echo "[jun10_ficture_filter] requests=${REQUESTS_CSV}"
echo "[jun10_ficture_filter] step1=${STEP1_CSV}"
echo "[jun10_ficture_filter] image_root=${IMAGE_ROOT}"
echo "[jun10_ficture_filter] output=${OUT}"
echo "[jun10_ficture_filter] prompt_style=${PROMPT_STYLE}"

"$PY" "$RUN_SCRIPT" \
  --requests-csv "$REQUESTS_CSV" \
  --step1-csv "$STEP1_CSV" \
  --image-root "$IMAGE_ROOT" \
  --output-dir "$OUT" \
  --model "$MODEL_NAME" \
  --device cuda \
  --max-new-tokens "$MAX_NEW_TOKENS" \
  --prompt-style "$PROMPT_STYLE" \
  --limit "$LIMIT" \
  --resume

echo "[jun10_ficture_filter] done: $OUT"
