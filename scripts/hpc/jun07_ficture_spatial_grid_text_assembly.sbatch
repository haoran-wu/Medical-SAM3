#!/bin/bash
#SBATCH --job-name=jun07_gridtxt_asm
#SBATCH --partition=devel
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=/nfs/roberts/project/pi_xy48/hw646/codex_salip_outputs/logs/jun07_gridtxt_asm_%j.log
set -euo pipefail

export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

PY=/nfs/roberts/project/pi_xy48/hw646/ycrc_conda/envs/medsam3/bin/python
SCRIPT=/nfs/roberts/project/pi_xy48/hw646/codex_salip_scripts_jun05/assemble_piece_first_vlm_results.py
BASE=/nfs/roberts/project/pi_xy48/hw646/Medical-SAM3_remote_runs/visium_hd_exp1
RUN=${RUN_NAME:-qwen3vl32b_he3_ficture_gridtext_full167}
VLM_OUT=$BASE/jun07_ficture_spatial_grid_text_20260607/local_vlm_outputs/$RUN
POOL=$BASE/jun07_ficture_spatial_grid_text_20260607/he_located_ficture_spatial_grid_pool
ANN=/home/hw646/Medical-SAM3_remote_runs/visium_hd_exp1/current_project_inputs/ficture_official_filtered_candidate_pool_input_roi/cropped_annotation_masks
HE=/home/hw646/Medical-SAM3_remote_runs/visium_hd_exp1/current_project_inputs/ficture_official_filtered_candidate_pool_input_roi/he_roi_matching_official_ficture_coverage.png
FIC=/home/hw646/Medical-SAM3_remote_runs/visium_hd_exp1/current_project_inputs/ficture_official_filtered_candidate_pool_input_roi/ficture_official_filtered_roi_rgb.png
OUT=$BASE/jun07_ficture_spatial_grid_text_20260607/assembly_outputs/$RUN

"$PY" "$SCRIPT" \
  --vlm-output-dir "$VLM_OUT" \
  --pool-dir "$POOL" \
  --annotation-dir "$ANN" \
  --he-roi "$HE" \
  --ficture-roi "$FIC" \
  --output-dir "$OUT" \
  --model-name "$RUN"
