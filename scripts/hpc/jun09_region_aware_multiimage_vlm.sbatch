#!/bin/bash
#SBATCH --job-name=jun09_region_vlm
#SBATCH --partition=gpu_b200,gpu_h200
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=180G
#SBATCH --time=04:00:00
#SBATCH --output=/nfs/roberts/project/pi_xy48/hw646/codex_salip_outputs/logs/jun09_region_vlm_%j.log
#SBATCH --error=/nfs/roberts/project/pi_xy48/hw646/codex_salip_outputs/logs/jun09_region_vlm_%j.log
#SBATCH --mail-type=NONE

set -euo pipefail

PROJECT=${PROJECT:-/nfs/roberts/project/pi_xy48/hw646/Medical-SAM3}
PY=${PY:-/nfs/roberts/project/pi_xy48/hw646/ycrc_conda/envs/medsam3/bin/python}
RUN_SCRIPT=${RUN_SCRIPT:-$PROJECT/inference/visium_hd_exp1/run_local_vlm_multiimage_cross_label.py}
OUT_BASE=${OUT_BASE:-/nfs/roberts/project/pi_xy48/hw646/Medical-SAM3_remote_runs/visium_hd_exp1/jun09_region_aware_vlm_outputs}
PACK_DIR=${PACK_DIR:-$PROJECT/output/visium_hd_exp1/final_deliverables/Jun09_SkillRanker_ComponentAwareMaskSelection/region_aware_piece_context_locator_pack}
POOL_CSV=${POOL_CSV:-$PACK_DIR/public_region_aware_requests.csv}
PROMPT_FILE=${PROMPT_FILE:-$PACK_DIR/prompt_region_aware_he_ficture.txt}
MODEL_NAME=${MODEL_NAME:-Qwen/Qwen3-VL-32B-Instruct}
RUN_NAME=${RUN_NAME:-jun09_region_all6_qwen3vl32b_${SLURM_JOB_ID}}
LIMIT=${LIMIT:-18}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-256}

mkdir -p "$OUT_BASE" /nfs/roberts/project/pi_xy48/hw646/codex_salip_outputs/logs
cd "$PROJECT"

export PYTHONUNBUFFERED=1
export PYTHONPATH="$PROJECT/inference/visium_hd_exp1:$PROJECT/inference:$PROJECT:${PYTHONPATH:-}"
export CONTEXT_IMAGE_MODE=all6
export MPLCONFIGDIR="${MPLCONFIGDIR_OVERRIDE:-/tmp/${USER}_mpl_${SLURM_JOB_ID}}"
export HF_HOME="${HF_HOME_OVERRIDE:-/nfs/roberts/project/pi_xy48/hw646/hf-home}"
export HF_HUB_CACHE="$HF_HOME/hub"
export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"
export TRANSFORMERS_CACHE="$HF_HOME/hub"
export VLM_DEVICE_MAP="${VLM_DEVICE_MAP:-auto}"
export VLM_ALLOW_AUTOMODEL_FALLBACK="${VLM_ALLOW_AUTOMODEL_FALLBACK:-1}"
export VLM_MAX_MEMORY_CUDA="${VLM_MAX_MEMORY_CUDA:-150GiB}"
export VLM_MAX_MEMORY_CPU="${VLM_MAX_MEMORY_CPU:-160GiB}"
export VLM_OFFLOAD_FOLDER="${VLM_OFFLOAD_FOLDER:-/tmp/hw646_jun09_region_offload_${SLURM_JOB_ID}}"
mkdir -p "$MPLCONFIGDIR" "$HF_HOME" "$VLM_OFFLOAD_FOLDER"

TOKEN_PATH=${TOKEN_PATH:-$HOME/.cache/huggingface/token}
if [[ -z "${HF_TOKEN:-}" && -s "$TOKEN_PATH" ]]; then
  export HF_TOKEN
  HF_TOKEN="$(cat "$TOKEN_PATH")"
fi

echo "[jun09_region_vlm] job=${SLURM_JOB_ID} node=$(hostname)"
echo "[jun09_region_vlm] model=${MODEL_NAME}"
echo "[jun09_region_vlm] pool=${POOL_CSV}"
echo "[jun09_region_vlm] prompt=${PROMPT_FILE}"
echo "[jun09_region_vlm] limit=${LIMIT}"
echo "[jun09_region_vlm] out=${OUT_BASE}/${RUN_NAME}_${SLURM_JOB_ID}"
if [[ -n "${HF_TOKEN:-}" ]]; then
  echo "[jun09_region_vlm] HF_TOKEN=present"
else
  echo "[jun09_region_vlm] HF_TOKEN=missing"
fi

if [[ ! -f "$POOL_CSV" ]]; then
  echo "Missing pool CSV: $POOL_CSV" >&2
  exit 2
fi
if [[ ! -f "$PROMPT_FILE" ]]; then
  echo "Missing prompt file: $PROMPT_FILE" >&2
  exit 2
fi

"$PY" "$RUN_SCRIPT" \
  --pool-csv "$POOL_CSV" \
  --output-dir "$OUT_BASE/${RUN_NAME}_${SLURM_JOB_ID}" \
  --model "$MODEL_NAME" \
  --device cuda \
  --max-new-tokens "$MAX_NEW_TOKENS" \
  --prompt-template-file "$PROMPT_FILE" \
  --limit "$LIMIT" \
  --resume

echo "[jun09_region_vlm] done: $OUT_BASE/${RUN_NAME}_${SLURM_JOB_ID}"
