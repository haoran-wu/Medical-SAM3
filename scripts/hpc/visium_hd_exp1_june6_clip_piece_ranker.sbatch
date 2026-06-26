#!/bin/bash
#SBATCH --job-name=june6_clip_piece_ranker
#SBATCH --partition=gpu_devel
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=/nfs/roberts/project/pi_xy48/hw646/codex_salip_outputs/logs/june6_clip_piece_ranker_%j.log

set -euo pipefail

PY=/nfs/roberts/project/pi_xy48/hw646/ycrc_conda/envs/medsam3/bin/python
SCRIPT_DIR=/nfs/roberts/project/pi_xy48/hw646/codex_salip_scripts_jun05
RUN_SCRIPT=${SCRIPT_DIR}/run_june6_clip_piece_ranker.py
ASSEMBLE_SCRIPT=${SCRIPT_DIR}/assemble_piece_first_vlm_results.py

export HF_HOME=/nfs/roberts/project/pi_xy48/hw646/hf-home
export TRANSFORMERS_CACHE=${HF_HOME}/hub
export HF_HUB_CACHE=${HF_HOME}/hub
export MPLCONFIGDIR=/tmp/${USER}_mpl_${SLURM_JOB_ID}

POOL=/home/hw646/Medical-SAM3_remote_runs/visium_hd_exp1/piece_first_prompt_ablation_20260604/compact167_reverse_blur_pool
INPUT=/home/hw646/Medical-SAM3_remote_runs/visium_hd_exp1/current_project_inputs/ficture_official_filtered_candidate_pool_input_roi
FACTOR_LABELS=/nfs/roberts/project/pi_xy48/hw646/Medical-SAM3/output/visium_hd_exp1/ficture_official_filtered_candidate_pool_input_roi/ficture_official_filtered_roi_factor_index.npy
LEGEND=${SCRIPT_DIR}/ficture_factor_prompt_legend_from_html.csv
OUT=/nfs/roberts/project/pi_xy48/hw646/Medical-SAM3_remote_runs/visium_hd_exp1/june6_clip_piece_ranker_20260606

mkdir -p "${OUT}/score_outputs" "${OUT}/assembly_outputs" "${MPLCONFIGDIR}"

echo "[june6_clip] node=$(hostname) start=$(date)"
echo "[june6_clip] pool=${POOL}"

"${PY}" "${RUN_SCRIPT}" \
  --pool-dir "${POOL}" \
  --factor-label-image "${FACTOR_LABELS}" \
  --prompt-legend-csv "${LEGEND}" \
  --output-base "${OUT}/score_outputs" \
  --clip-model openai/clip-vit-base-patch32 \
  --device cuda \
  --batch-size 32

for method_dir in "${OUT}"/score_outputs/*; do
  if [[ ! -f "${method_dir}/per_candidate_predictions.csv" ]]; then
    continue
  fi
  method=$(basename "${method_dir}")
  echo "[june6_clip] assembling ${method}"
  "${PY}" "${ASSEMBLE_SCRIPT}" \
    --vlm-output-dir "${method_dir}" \
    --pool-dir "${POOL}" \
    --annotation-dir "${INPUT}/cropped_annotation_masks" \
    --he-roi "${INPUT}/he_roi_matching_official_ficture_coverage.png" \
    --ficture-roi "${INPUT}/ficture_official_filtered_roi_rgb.png" \
    --output-dir "${OUT}/assembly_outputs/${method}" \
    --model-name "June6 Clip ${method}"
done

echo "[june6_clip] done=$(date)"
echo "${OUT}"
