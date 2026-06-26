#!/bin/bash
#SBATCH --job-name=alv_box384_vlm
#SBATCH --partition=gpu_h200,gpu_b200,gpu_rtx6000
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=120G
#SBATCH --time=06:00:00
#SBATCH --output=/nfs/roberts/project/pi_xy48/hw646/codex_salip_outputs/logs/jun07_alveoli_box384_vlm_%j.log
#SBATCH --mail-type=NONE

set -euo pipefail

PROJECT=${PROJECT:-/nfs/roberts/project/pi_xy48/hw646/Medical-SAM3}
PY=${PY:-/nfs/roberts/project/pi_xy48/hw646/ycrc_conda/envs/medsam3/bin/python}
RUN_ROOT=${RUN_ROOT:-/nfs/roberts/project/pi_xy48/hw646/Medical-SAM3_remote_runs/visium_hd_exp1/jun07_alveoli_box384_setting_rank_20260607}
SCRIPT_DIR=${SCRIPT_DIR:-/nfs/roberts/project/pi_xy48/hw646/codex_salip_scripts_jun07}
INPUT_DIR=${INPUT_DIR:-$PROJECT/output/visium_hd_exp1/ficture_official_filtered_candidate_pool_input_roi}
MASK_DIR=${MASK_DIR:-/home/hw646/codex_he_official_roi_candidate_pool_b200/he_official_12445451/base_box384_s128_m1536/candidate_masks}
MODEL_NAME=${MODEL_NAME:-Qwen/Qwen3-VL-32B-Instruct}
MODEL_SLUG=${MODEL_SLUG:-qwen3vl32b}
LIMIT=${LIMIT:-100}

SOURCE_POOL=$RUN_ROOT/alveoli_box384_source_pool
REVERSE_POOL=$RUN_ROOT/alveoli_box384_reverse_blur_pool
CONTEXT_POOL=$RUN_ROOT/alveoli_box384_context_pool
VLM_OUT=$RUN_ROOT/local_vlm_outputs

mkdir -p "$RUN_ROOT" "$SCRIPT_DIR" "$VLM_OUT" /nfs/roberts/project/pi_xy48/hw646/codex_salip_outputs/logs
cd "$PROJECT"

export PYTHONUNBUFFERED=1
export PYTHONPATH="$SCRIPT_DIR:$PROJECT/inference/visium_hd_exp1:$PROJECT/inference:$PROJECT:${PYTHONPATH:-}"
export MPLCONFIGDIR="${MPLCONFIGDIR_OVERRIDE:-$PROJECT/output/.mplconfig}"
export HF_HOME="${HF_HOME_OVERRIDE:-/nfs/roberts/project/pi_xy48/hw646/hf-home}"
export HF_HUB_CACHE="$HF_HOME/hub"
export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
export VLM_DEVICE_MAP="${VLM_DEVICE_MAP:-auto}"
export VLM_OFFLOAD_FOLDER="${VLM_OFFLOAD_FOLDER:-$RUN_ROOT/offload_${SLURM_JOB_ID}}"
export VLM_ALLOW_AUTOMODEL_FALLBACK="${VLM_ALLOW_AUTOMODEL_FALLBACK:-1}"
export VLM_MAX_MEMORY_CUDA="${VLM_MAX_MEMORY_CUDA:-70GiB}"
export VLM_MAX_MEMORY_CPU="${VLM_MAX_MEMORY_CPU:-140GiB}"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
mkdir -p "$MPLCONFIGDIR" "$HF_HOME" "$VLM_OFFLOAD_FOLDER"

DEFAULT_HF_TOKEN_PATH=${DEFAULT_HF_TOKEN_PATH:-$HOME/.cache/huggingface/token}
if [[ -z "${HF_TOKEN:-}" && -s "$DEFAULT_HF_TOKEN_PATH" ]]; then
  export HF_TOKEN
  HF_TOKEN="$(cat "$DEFAULT_HF_TOKEN_PATH")"
fi

date
echo "Jun07 alveoli single-setting VLM rank test"
echo "MODEL_NAME=$MODEL_NAME"
echo "MASK_DIR=$MASK_DIR"
echo "RUN_ROOT=$RUN_ROOT"

"$PY" "$SCRIPT_DIR/build_target_setting_vlm_source_pool.py" \
  --mask-dir "$MASK_DIR" \
  --output-dir "$SOURCE_POOL" \
  --target-class alveoli \
  --annotation-dir "$INPUT_DIR/cropped_annotation_masks" \
  --he-image "$INPUT_DIR/he_roi_matching_official_ficture_coverage.png" \
  --ficture-image "$INPUT_DIR/ficture_official_filtered_roi_rgb.png" \
  --source he \
  --run he_official_12445451 \
  --setting base_box384_s128_m1536

"$PY" "$SCRIPT_DIR/render_paired_vlm_pool_plain_crops.py" \
  --source-pool "$SOURCE_POOL" \
  --output-dir "$REVERSE_POOL" \
  --he-image "$INPUT_DIR/he_roi_matching_official_ficture_coverage.png" \
  --ficture-image "$INPUT_DIR/ficture_official_filtered_roi_rgb.png" \
  --mode reverse_blur_gray \
  --pad 96 \
  --max-side 768

"$PY" "$SCRIPT_DIR/build_piece_first_context_vlm_pool.py" \
  --source-pool "$REVERSE_POOL" \
  --output-dir "$CONTEXT_POOL" \
  --he-roi "$INPUT_DIR/he_roi_matching_official_ficture_coverage.png" \
  --ficture-roi "$INPUT_DIR/ficture_official_filtered_roi_rgb.png" \
  --context-scale 5.0 \
  --context-min-side 768 \
  --context-max-side-fullres 1800 \
  --output-max-side 768 \
  --locator-max-side 900

PROMPT_GRID=${PROMPT_GRID:-"no_description:he_only:he_only_nodesc;no_description:all5:all5_nodesc"}

"$PY" "$SCRIPT_DIR/run_local_vlm_multiimage_prompt_grid.py" \
  --pool-csv "$CONTEXT_POOL/public_vlm_requests.csv" \
  --output-base "$VLM_OUT" \
  --run-name "jun07_alveoli_box384_${MODEL_SLUG}" \
  --prompt-grid "$PROMPT_GRID" \
  --factor-legend-csv "$INPUT_DIR/html_palette/html_palette_factor_legend.csv" \
  --model "$MODEL_NAME" \
  --device cuda \
  --max-new-tokens 192 \
  --limit "$LIMIT" \
  --resume

date
echo "Output root: $RUN_ROOT"
