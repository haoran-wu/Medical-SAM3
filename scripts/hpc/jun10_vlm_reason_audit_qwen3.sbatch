#!/bin/bash
#SBATCH --job-name=jun10_vlm_reason
#SBATCH --partition=gpu_b200,gpu_h200
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=180G
#SBATCH --time=04:00:00
#SBATCH --output=/nfs/roberts/project/pi_xy48/hw646/codex_salip_outputs/logs/jun10_vlm_reason_%j.log
#SBATCH --error=/nfs/roberts/project/pi_xy48/hw646/codex_salip_outputs/logs/jun10_vlm_reason_%j.log

set -euo pipefail

PROJECT=${PROJECT:-/nfs/roberts/project/pi_xy48/hw646/Medical-SAM3}
PY=${PY:-/nfs/roberts/project/pi_xy48/hw646/ycrc_conda/envs/medsam3/bin/python}
SCRIPT=${SCRIPT:-$PROJECT/inference/visium_hd_exp1/run_local_vlm_reason_audit.py}
POOL=${POOL:-$PROJECT/output/visium_hd_exp1/final_deliverables/Jun09_SkillRanker_ComponentAwareMaskSelection/corrected_pool/public_vlm_requests.csv}
LEGEND=${LEGEND:-$PROJECT/data/visium_hd_exp1/current_ficture_vlm_inputs/ficture_factor_prompt_legend_from_html.csv}
OUTBASE=${OUTBASE:-/nfs/roberts/project/pi_xy48/hw646/Medical-SAM3_remote_runs/visium_hd_exp1/jun10_vlm_reason_audit_outputs}
RUN_NAME=${RUN_NAME:-qwen3vl32b_full_descriptions_visible_reason_${SLURM_JOB_ID}}
MODEL_NAME=${MODEL_NAME:-Qwen/Qwen3-VL-32B-Instruct}
LIMIT=${LIMIT:-0}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-512}

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
export VLM_OFFLOAD_FOLDER="${VLM_OFFLOAD_FOLDER:-/tmp/hw646_jun10_vlm_reason_offload_${SLURM_JOB_ID}}"
mkdir -p "$MPLCONFIGDIR" "$HF_HOME" "$VLM_OFFLOAD_FOLDER"

TOKEN_PATH=${TOKEN_PATH:-$HOME/.cache/huggingface/token}
if [[ -z "${HF_TOKEN:-}" && -s "$TOKEN_PATH" ]]; then
  export HF_TOKEN
  HF_TOKEN="$(cat "$TOKEN_PATH")"
fi

OUTDIR="$OUTBASE/$RUN_NAME"

echo "[jun10_vlm_reason] job=${SLURM_JOB_ID} node=$(hostname)"
echo "[jun10_vlm_reason] model=${MODEL_NAME}"
echo "[jun10_vlm_reason] pool=${POOL}"
echo "[jun10_vlm_reason] legend=${LEGEND}"
echo "[jun10_vlm_reason] outdir=${OUTDIR}"
echo "[jun10_vlm_reason] limit=${LIMIT}"

"$PY" "$SCRIPT" \
  --pool-csv "$POOL" \
  --output-dir "$OUTDIR" \
  --factor-legend-csv "$LEGEND" \
  --model "$MODEL_NAME" \
  --device cuda \
  --max-new-tokens "$MAX_NEW_TOKENS" \
  --limit "$LIMIT" \
  --resume

echo "[jun10_vlm_reason] done ${OUTDIR}"
