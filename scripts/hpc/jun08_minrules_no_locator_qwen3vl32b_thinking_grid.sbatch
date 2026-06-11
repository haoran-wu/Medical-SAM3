#!/bin/bash
#SBATCH --job-name=jun08_qwen_think
#SBATCH --partition=gpu_b200,gpu_h200
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=220G
#SBATCH --time=04:00:00
#SBATCH --output=/nfs/roberts/project/pi_xy48/hw646/codex_salip_outputs/logs/jun08_qwen_thinking_%j.log
#SBATCH --error=/nfs/roberts/project/pi_xy48/hw646/codex_salip_outputs/logs/jun08_qwen_thinking_%j.log

set -euo pipefail

PY=/nfs/roberts/project/pi_xy48/hw646/ycrc_conda/envs/medsam3/bin/python
SCRIPT=/nfs/roberts/project/pi_xy48/hw646/codex_salip_scripts_jun05/run_local_vlm_multiimage_prompt_grid.py
POOL=${POOL:-/home/hw646/Medical-SAM3_remote_runs/visium_hd_exp1/piece_first_context_20260605/context167_multiscale_pool/public_vlm_requests.csv}
LEGEND=/nfs/roberts/project/pi_xy48/hw646/codex_salip_scripts_jun05/ficture_factor_prompt_legend_from_html.csv
OUTBASE=/nfs/roberts/project/pi_xy48/hw646/Medical-SAM3_remote_runs/visium_hd_exp1/jun08_qwen3vl32b_thinking_nolegend_finaljson_20260608/local_vlm_outputs
RUN_NAME=${RUN_NAME:-qwen3vl32b_thinking_nolegend_finaljson_no_locator_${SLURM_JOB_ID}}
LIMIT=${LIMIT:-6}

export HF_HOME=/nfs/roberts/project/pi_xy48/hw646/hf-home
export TRANSFORMERS_CACHE=${HF_HOME}/hub
export HF_HUB_CACHE=${HF_HOME}/hub
export VLM_DEVICE_MAP=auto
export VLM_MAX_MEMORY_CUDA=130GiB
export VLM_MAX_MEMORY_CPU=80GiB
export VLM_OFFLOAD_FOLDER=/tmp/${USER}_vlm_offload_${SLURM_JOB_ID}
export CONTEXT_FORCE_FINAL_JSON=1
export MPLCONFIGDIR=/tmp/${USER}_mpl_${SLURM_JOB_ID}
mkdir -p "${OUTBASE}" "${MPLCONFIGDIR}" "${VLM_OFFLOAD_FOLDER}"

PROMPT_GRID=${PROMPT_GRID:-"no_legend_nodesc:he_piece_local:HE_only_no_locator_minrules;no_legend_nodesc:ficture_piece_local:FICTURE_only_no_locator_minrules;no_legend_nodesc:piece_local:HE_FICTURE_no_locator_minrules"}

echo "[jun08_qwen_thinking] job=${SLURM_JOB_ID} node=$(hostname) limit=${LIMIT} outbase=${OUTBASE}"
echo "[jun08_qwen_thinking] model=Qwen/Qwen3-VL-32B-Thinking"
echo "[jun08_qwen_thinking] VLM_DEVICE_MAP=${VLM_DEVICE_MAP} VLM_MAX_MEMORY_CUDA=${VLM_MAX_MEMORY_CUDA}"
echo "[jun08_qwen_thinking] prompt_grid=${PROMPT_GRID}"

"${PY}" "${SCRIPT}" \
  --pool-csv "${POOL}" \
  --output-base "${OUTBASE}" \
  --run-name "${RUN_NAME}" \
  --prompt-grid "${PROMPT_GRID}" \
  --factor-legend-csv "${LEGEND}" \
  --model Qwen/Qwen3-VL-32B-Thinking \
  --device cuda \
  --max-new-tokens 2048 \
  --limit "${LIMIT}" \
  --resume

echo "[jun08_qwen_thinking] done ${OUTBASE}/${RUN_NAME}_*"
