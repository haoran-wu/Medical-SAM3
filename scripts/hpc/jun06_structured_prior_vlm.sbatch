#!/bin/bash
#SBATCH --job-name=jun06_struct_prior
#SBATCH --partition=gpu_b200,gpu_h200
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=160G
#SBATCH --time=02:00:00
#SBATCH --output=/nfs/roberts/project/pi_xy48/hw646/codex_salip_outputs/logs/jun06_struct_prior_%j.log
#SBATCH --error=/nfs/roberts/project/pi_xy48/hw646/codex_salip_outputs/logs/jun06_struct_prior_%j.log

set -euo pipefail

PY=/nfs/roberts/project/pi_xy48/hw646/ycrc_conda/envs/medsam3/bin/python
SCRIPT=/nfs/roberts/project/pi_xy48/hw646/codex_salip_scripts_jun05/run_local_vlm_multiimage_cross_label_structured.py
POOL=/nfs/roberts/project/pi_xy48/hw646/Medical-SAM3_remote_runs/visium_hd_exp1/jun06_prompt_structured_prior_20260606/structured_prior_pool/public_vlm_requests.csv
LEGEND=/nfs/roberts/project/pi_xy48/hw646/codex_salip_scripts_jun05/ficture_factor_prompt_legend_from_html.csv
OUTBASE=/nfs/roberts/project/pi_xy48/hw646/Medical-SAM3_remote_runs/visium_hd_exp1/jun06_prompt_structured_prior_20260606/local_vlm_outputs
LIMIT=${LIMIT:-18}
TAG=${TAG:-smoke18}

export HF_HOME=/nfs/roberts/project/pi_xy48/hw646/hf-home
export TRANSFORMERS_CACHE=${HF_HOME}/hub
export HF_HUB_CACHE=${HF_HOME}/hub
export MPLCONFIGDIR=/tmp/${USER}_mpl_${SLURM_JOB_ID}
export CONTEXT_IMAGE_MODE=he_piece_local
export CONTEXT_PROMPT_STYLE=structured_prior

OUT=${OUTBASE}/qwen3vl32b_HE_structured_FICTURE_prior_${TAG}_${SLURM_JOB_ID}
mkdir -p "${OUT}" "${MPLCONFIGDIR}"

echo "[jun06_struct_prior] job=${SLURM_JOB_ID} node=$(hostname) limit=${LIMIT} out=${OUT}"
"${PY}" "${SCRIPT}" \
  --pool-csv "${POOL}" \
  --output-dir "${OUT}" \
  --factor-legend-csv "${LEGEND}" \
  --model Qwen/Qwen3-VL-32B-Instruct \
  --device cuda \
  --max-new-tokens 192 \
  --limit "${LIMIT}" \
  --resume

echo "[jun06_struct_prior] done ${OUT}"
