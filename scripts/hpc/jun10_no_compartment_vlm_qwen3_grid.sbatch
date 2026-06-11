#!/bin/bash
#SBATCH --job-name=jun10_nocomp_vlm
#SBATCH --partition=gpu_b200,gpu_h200
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=180G
#SBATCH --time=04:00:00
#SBATCH --output=/nfs/roberts/project/pi_xy48/hw646/codex_salip_outputs/logs/jun10_nocomp_vlm_%j.log
#SBATCH --error=/nfs/roberts/project/pi_xy48/hw646/codex_salip_outputs/logs/jun10_nocomp_vlm_%j.log

set -euo pipefail

PROJECT=${PROJECT:-/nfs/roberts/project/pi_xy48/hw646/Medical-SAM3}
PY=${PY:-/nfs/roberts/project/pi_xy48/hw646/ycrc_conda/envs/medsam3/bin/python}
SCRIPT=${SCRIPT:-$PROJECT/inference/visium_hd_exp1/run_local_vlm_multiimage_prompt_grid.py}
POOL=${POOL:-$PROJECT/output/visium_hd_exp1/final_deliverables/Jun09_SkillRanker_ComponentAwareMaskSelection/corrected_pool/public_vlm_requests.csv}
LEGEND=${LEGEND:-$PROJECT/data/visium_hd_exp1/current_ficture_vlm_inputs/ficture_factor_prompt_legend_from_html.csv}
OUTBASE=${OUTBASE:-/nfs/roberts/project/pi_xy48/hw646/Medical-SAM3_remote_runs/visium_hd_exp1/jun10_no_compartment_vlm_outputs}
RUN_NAME=${RUN_NAME:-qwen3vl32b_rgb_celltype_only_${SLURM_JOB_ID}}
MODEL_NAME=${MODEL_NAME:-Qwen/Qwen3-VL-32B-Instruct}
LIMIT=${LIMIT:-0}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-192}

mkdir -p "$OUTBASE" /nfs/roberts/project/pi_xy48/hw646/codex_salip_outputs/logs
cd "$PROJECT"

export PYTHONUNBUFFERED=1
export PYTHONPATH="$PROJECT/inference/visium_hd_exp1:$PROJECT/inference:$PROJECT:${PYTHONPATH:-}"
export HF_HOME="${HF_HOME_OVERRIDE:-/nfs/roberts/project/pi_xy48/hw646/hf-home}"
export TRANSFORMERS_CACHE="$HF_HOME/hub"
export HF_HUB_CACHE="$HF_HOME/hub"
export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"
export MPLCONFIGDIR="${MPLCONFIGDIR_OVERRIDE:-/tmp/${USER}_mpl_${SLURM_JOB_ID}}"
export VLM_DEVICE_MAP="${VLM_DEVICE_MAP:-auto}"
export VLM_ALLOW_AUTOMODEL_FALLBACK="${VLM_ALLOW_AUTOMODEL_FALLBACK:-1}"
export VLM_MAX_MEMORY_CUDA="${VLM_MAX_MEMORY_CUDA:-150GiB}"
export VLM_MAX_MEMORY_CPU="${VLM_MAX_MEMORY_CPU:-160GiB}"
export VLM_OFFLOAD_FOLDER="${VLM_OFFLOAD_FOLDER:-/tmp/hw646_jun10_nocomp_vlm_offload_${SLURM_JOB_ID}}"
mkdir -p "$MPLCONFIGDIR" "$HF_HOME" "$VLM_OFFLOAD_FOLDER"

TOKEN_PATH=${TOKEN_PATH:-$HOME/.cache/huggingface/token}
if [[ -z "${HF_TOKEN:-}" && -s "$TOKEN_PATH" ]]; then
  export HF_TOKEN
  HF_TOKEN="$(cat "$TOKEN_PATH")"
fi

PROMPT_GRID=${PROMPT_GRID:-"full:piece_only:HE_FICTURE_RGB_celltype_only"}

echo "[jun10_nocomp_vlm] job=${SLURM_JOB_ID} node=$(hostname)"
echo "[jun10_nocomp_vlm] model=${MODEL_NAME}"
echo "[jun10_nocomp_vlm] pool=${POOL}"
echo "[jun10_nocomp_vlm] legend=${LEGEND}"
echo "[jun10_nocomp_vlm] outbase=${OUTBASE}"
echo "[jun10_nocomp_vlm] prompt_grid=${PROMPT_GRID}"
echo "[jun10_nocomp_vlm] limit=${LIMIT}"

"$PY" "$SCRIPT" \
  --pool-csv "$POOL" \
  --output-base "$OUTBASE" \
  --run-name "$RUN_NAME" \
  --prompt-grid "$PROMPT_GRID" \
  --factor-legend-csv "$LEGEND" \
  --model "$MODEL_NAME" \
  --device cuda \
  --max-new-tokens "$MAX_NEW_TOKENS" \
  --limit "$LIMIT" \
  --resume

echo "[jun10_nocomp_vlm] done ${OUTBASE}/${RUN_NAME}_*"
