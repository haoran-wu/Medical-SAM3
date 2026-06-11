#!/bin/bash
#SBATCH --job-name=jun08_minrules
#SBATCH --partition=gpu_b200,gpu_h200
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=180G
#SBATCH --time=03:00:00
#SBATCH --output=/nfs/roberts/project/pi_xy48/hw646/codex_salip_outputs/logs/jun08_minrules_%j.log
#SBATCH --error=/nfs/roberts/project/pi_xy48/hw646/codex_salip_outputs/logs/jun08_minrules_%j.log

set -euo pipefail

PY=/nfs/roberts/project/pi_xy48/hw646/ycrc_conda/envs/medsam3/bin/python
SCRIPT=/nfs/roberts/project/pi_xy48/hw646/codex_salip_scripts_jun05/run_local_vlm_multiimage_prompt_grid.py
POOL=/home/hw646/Medical-SAM3_remote_runs/visium_hd_exp1/piece_first_context_20260605/context167_multiscale_pool/public_vlm_requests.csv
LEGEND=/nfs/roberts/project/pi_xy48/hw646/codex_salip_scripts_jun05/ficture_factor_prompt_legend_from_html.csv
OUTBASE=/nfs/roberts/project/pi_xy48/hw646/Medical-SAM3_remote_runs/visium_hd_exp1/jun08_minimal_rules_no_locator_20260608/local_vlm_outputs
RUN_NAME=${RUN_NAME:-qwen3vl32b_minrules_no_locator_${SLURM_JOB_ID}}
LIMIT=${LIMIT:-0}

export HF_HOME=/nfs/roberts/project/pi_xy48/hw646/hf-home
export TRANSFORMERS_CACHE=${HF_HOME}/hub
export HF_HUB_CACHE=${HF_HOME}/hub
export MPLCONFIGDIR=/tmp/${USER}_mpl_${SLURM_JOB_ID}
mkdir -p "${OUTBASE}" "${MPLCONFIGDIR}"

PROMPT_GRID="no_legend_nodesc:he_piece_local:HE_only_no_locator_minrules;full:ficture_piece_local:FICTURE_only_no_locator_minrules;full:piece_local:HE_FICTURE_no_locator_minrules"

echo "[jun08_minrules] job=${SLURM_JOB_ID} node=$(hostname) limit=${LIMIT} outbase=${OUTBASE}"
echo "[jun08_minrules] prompt_grid=${PROMPT_GRID}"

"${PY}" "${SCRIPT}" \
  --pool-csv "${POOL}" \
  --output-base "${OUTBASE}" \
  --run-name "${RUN_NAME}" \
  --prompt-grid "${PROMPT_GRID}" \
  --factor-legend-csv "${LEGEND}" \
  --model Qwen/Qwen3-VL-32B-Instruct \
  --device cuda \
  --max-new-tokens 192 \
  --limit "${LIMIT}" \
  --resume

echo "[jun08_minrules] done ${OUTBASE}/${RUN_NAME}_*"
