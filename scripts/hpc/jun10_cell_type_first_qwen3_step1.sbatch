#!/bin/bash
#SBATCH --job-name=jun10_celltype_qwen3
#SBATCH --partition=gpu_b200,gpu_h200,gpu_rtx6000
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=160G
#SBATCH --time=03:00:00
#SBATCH --output=/nfs/roberts/project/pi_xy48/hw646/codex_salip_outputs/logs/jun10_celltype_qwen3_%j.log
#SBATCH --error=/nfs/roberts/project/pi_xy48/hw646/codex_salip_outputs/logs/jun10_celltype_qwen3_%j.log
#SBATCH --mail-type=NONE

set -euo pipefail

PROJECT=${PROJECT:-/nfs/roberts/project/pi_xy48/hw646/Medical-SAM3}
PY=${PY:-/nfs/roberts/project/pi_xy48/hw646/ycrc_conda/envs/medsam3/bin/python}
RUN_SCRIPT=${RUN_SCRIPT:-$PROJECT/inference/visium_hd_exp1/run_qwen3_cell_type_first_hypothesis.py}
REQUESTS_CSV=${REQUESTS_CSV:-$PROJECT/output/visium_hd_exp1/final_deliverables/Jun10_CellTypeFirst_Qwen3_Skill/cell_type_first_requests.csv}
OUT_BASE=${OUT_BASE:-/nfs/roberts/project/pi_xy48/hw646/Medical-SAM3_remote_runs/visium_hd_exp1/jun10_cell_type_first_qwen3}
MODEL_NAME=${MODEL_NAME:-Qwen/Qwen3-VL-32B-Instruct}
MODEL_SLUG=${MODEL_SLUG:-qwen3vl32b}
LIMIT=${LIMIT:-18}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-384}

mkdir -p "$OUT_BASE" /nfs/roberts/project/pi_xy48/hw646/codex_salip_outputs/logs
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
export VLM_OFFLOAD_FOLDER="${VLM_OFFLOAD_FOLDER:-/tmp/hw646_jun10_celltype_offload_${SLURM_JOB_ID}}"
mkdir -p "$MPLCONFIGDIR" "$HF_HOME" "$VLM_OFFLOAD_FOLDER"

TOKEN_PATH=${TOKEN_PATH:-$HOME/.cache/huggingface/token}
if [[ -z "${HF_TOKEN:-}" && -s "$TOKEN_PATH" ]]; then
  export HF_TOKEN
  HF_TOKEN="$(cat "$TOKEN_PATH")"
fi

OUT="$OUT_BASE/step1_${MODEL_SLUG}_limit${LIMIT}_${SLURM_JOB_ID}"

echo "[jun10_celltype_qwen3] job=${SLURM_JOB_ID} node=$(hostname)"
echo "[jun10_celltype_qwen3] model=${MODEL_NAME}"
echo "[jun10_celltype_qwen3] requests=${REQUESTS_CSV}"
echo "[jun10_celltype_qwen3] output=${OUT}"
echo "[jun10_celltype_qwen3] limit=${LIMIT}"
if [[ -n "${HF_TOKEN:-}" ]]; then
  echo "[jun10_celltype_qwen3] HF_TOKEN=present"
else
  echo "[jun10_celltype_qwen3] HF_TOKEN=missing"
fi

if [[ ! -f "$REQUESTS_CSV" ]]; then
  echo "Missing requests CSV: $REQUESTS_CSV" >&2
  exit 2
fi

"$PY" "$RUN_SCRIPT" \
  --requests-csv "$REQUESTS_CSV" \
  --output-dir "$OUT" \
  --model "$MODEL_NAME" \
  --device cuda \
  --max-new-tokens "$MAX_NEW_TOKENS" \
  --limit "$LIMIT" \
  --resume

echo "[jun10_celltype_qwen3] done: $OUT"
