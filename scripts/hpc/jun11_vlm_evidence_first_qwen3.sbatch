#!/bin/bash
#SBATCH --job-name=jun11_vlm_evidence
#SBATCH --partition=gpu_b200,gpu_h200
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=180G
#SBATCH --time=05:00:00
#SBATCH --output=/nfs/roberts/project/pi_xy48/hw646/codex_salip_outputs/logs/jun11_vlm_evidence_%j.log
#SBATCH --error=/nfs/roberts/project/pi_xy48/hw646/codex_salip_outputs/logs/jun11_vlm_evidence_%j.log

set -euo pipefail

PROJECT=${PROJECT:-/nfs/roberts/project/pi_xy48/hw646/Medical-SAM3}
PY=${PY:-/nfs/roberts/project/pi_xy48/hw646/ycrc_conda/envs/medsam3/bin/python}
SCRIPT=${SCRIPT:-$PROJECT/inference/visium_hd_exp1/run_local_vlm_evidence_first_audit.py}
POOL=${POOL:-$PROJECT/output/visium_hd_exp1/final_deliverables/Jun09_SkillRanker_ComponentAwareMaskSelection/corrected_pool/public_vlm_requests.csv}
LEGEND=${LEGEND:-$PROJECT/data/visium_hd_exp1/current_ficture_vlm_inputs/ficture_factor_prompt_legend_from_html.csv}
OUTBASE=${OUTBASE:-/nfs/roberts/project/pi_xy48/hw646/Medical-SAM3_remote_runs/visium_hd_exp1/jun11_vlm_evidence_first_outputs}
RUN_NAME=${RUN_NAME:-qwen3vl32b_evidence_first_${SLURM_JOB_ID}}
MODEL_NAME=${MODEL_NAME:-Qwen/Qwen3-VL-32B-Instruct}
LIMIT=${LIMIT:-0}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-1600}
CANDIDATE_UIDS=${CANDIDATE_UIDS:-}

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
export VLM_OFFLOAD_FOLDER="${VLM_OFFLOAD_FOLDER:-/tmp/hw646_jun11_vlm_evidence_offload_${SLURM_JOB_ID}}"
mkdir -p "$MPLCONFIGDIR" "$HF_HOME" "$VLM_OFFLOAD_FOLDER"

TOKEN_PATH=${TOKEN_PATH:-$HOME/.cache/huggingface/token}
if [[ -z "${HF_TOKEN:-}" && -s "$TOKEN_PATH" ]]; then
  export HF_TOKEN
  HF_TOKEN="$(cat "$TOKEN_PATH")"
fi

OUTDIR="$OUTBASE/$RUN_NAME"
EXTRA_ARGS=()
if [[ -n "$CANDIDATE_UIDS" ]]; then
  IFS=',' read -r -a UID_ARRAY <<< "$CANDIDATE_UIDS"
  for uid in "${UID_ARRAY[@]}"; do
    EXTRA_ARGS+=(--candidate-uid "$uid")
  done
fi

echo "[jun11_vlm_evidence] job=${SLURM_JOB_ID} node=$(hostname)"
echo "[jun11_vlm_evidence] model=${MODEL_NAME}"
echo "[jun11_vlm_evidence] pool=${POOL}"
echo "[jun11_vlm_evidence] legend=${LEGEND}"
echo "[jun11_vlm_evidence] outdir=${OUTDIR}"
echo "[jun11_vlm_evidence] limit=${LIMIT}"
echo "[jun11_vlm_evidence] candidate_uids=${CANDIDATE_UIDS}"

"$PY" "$SCRIPT" \
  --pool-csv "$POOL" \
  --output-dir "$OUTDIR" \
  --factor-legend-csv "$LEGEND" \
  --model "$MODEL_NAME" \
  --device cuda \
  --max-new-tokens "$MAX_NEW_TOKENS" \
  --limit "$LIMIT" \
  --resume \
  "${EXTRA_ARGS[@]}"

echo "[jun11_vlm_evidence] done ${OUTDIR}"
