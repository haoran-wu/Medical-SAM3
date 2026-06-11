#!/bin/bash
#SBATCH --job-name=piece_asm_strong
#SBATCH --partition=devel
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=01:00:00
#SBATCH --output=/home/hw646/Medical-SAM3_remote_runs/visium_hd_exp1/logs/piece_prompt_ablation_strong_assemble_%j.log
#SBATCH --mail-type=NONE

set -euo pipefail

PROJECT=${PROJECT:-/nfs/roberts/project/pi_xy48/hw646/Medical-SAM3}
PY=${PY:-/nfs/roberts/project/pi_xy48/hw646/ycrc_conda/envs/medsam3/bin/python}
ASSEMBLE_SCRIPT=${ASSEMBLE_SCRIPT:-/home/hw646/codex_salip_scripts/assemble_piece_first_vlm_results.py}
BASE=${BASE:-/home/hw646/Medical-SAM3_remote_runs/visium_hd_exp1/piece_first_prompt_ablation_20260604}
POOL_DIR=${POOL_DIR:-$BASE/compact167_reverse_blur_pool}
OUT_BASE=${OUT_BASE:-$BASE/local_vlm_outputs}
ASSEMBLY_BASE=${ASSEMBLY_BASE:-$BASE/piece_assembly_outputs_strong}
INPUT_DIR=${INPUT_DIR:-/home/hw646/Medical-SAM3_remote_runs/visium_hd_exp1/current_project_inputs/ficture_official_filtered_candidate_pool_input_roi}
ANNOTATION_DIR=${ANNOTATION_DIR:-$INPUT_DIR/cropped_annotation_masks}
HE_ROI=${HE_ROI:-$INPUT_DIR/he_roi_matching_official_ficture_coverage.png}
FICTURE_ROI=${FICTURE_ROI:-$INPUT_DIR/ficture_official_filtered_roi_rgb.png}

mkdir -p "$ASSEMBLY_BASE" /home/hw646/Medical-SAM3_remote_runs/visium_hd_exp1/logs
cd "$PROJECT"

export PYTHONUNBUFFERED=1
export PYTHONPATH="/home/hw646/codex_salip_scripts:$PROJECT/inference/visium_hd_exp1:$PROJECT/inference:$PROJECT:${PYTHONPATH:-}"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

date
echo "Piece-first strong-model prompt ablation assembly"
echo "BASE=$BASE"
echo "POOL_DIR=$POOL_DIR"
echo "OUT_BASE=$OUT_BASE"
echo "ASSEMBLY_BASE=$ASSEMBLY_BASE"

declare -a RUNS=(
  "Qwen3-VL-32B mainline Jun04_compact167_prompt_ablation_strong_qwen3vl32b_mainline_13759681"
  "Qwen3-VL-32B mainline_nodesc Jun04_compact167_prompt_ablation_strong_qwen3vl32b_mainline_nodesc_13759682"
  "Qwen3-VL-32B compact Jun04_compact167_prompt_ablation_strong_qwen3vl32b_compact_13759683"
  "Qwen3-VL-32B compact_nodesc Jun04_compact167_prompt_ablation_strong_qwen3vl32b_compact_nodesc_13759684"
  "InternVL3.5-38B mainline Jun04_compact167_prompt_ablation_strong_internvl35_38b_mainline_13759685"
  "InternVL3.5-38B mainline_nodesc Jun04_compact167_prompt_ablation_strong_internvl35_38b_mainline_nodesc_13759686"
  "InternVL3.5-38B compact Jun04_compact167_prompt_ablation_strong_internvl35_38b_compact_13759687"
  "InternVL3.5-38B compact_nodesc Jun04_compact167_prompt_ablation_strong_internvl35_38b_compact_nodesc_13759688"
)

for spec in "${RUNS[@]}"; do
  read -r MODEL STYLE RUN_DIR <<<"$spec"
  VLM_OUT="$OUT_BASE/$RUN_DIR"
  if [[ ! -f "$VLM_OUT/cross_label_scores.csv" ]]; then
    echo "Skip missing scores: $VLM_OUT"
    continue
  fi
  SAFE_MODEL=${MODEL//[^A-Za-z0-9._-]/_}
  ASM_OUT="$ASSEMBLY_BASE/${SAFE_MODEL}_${STYLE}"
  echo "Assembling $MODEL $STYLE -> $ASM_OUT"
  "$PY" "$ASSEMBLE_SCRIPT" \
    --vlm-output-dir "$VLM_OUT" \
    --pool-dir "$POOL_DIR" \
    --annotation-dir "$ANNOTATION_DIR" \
    --he-roi "$HE_ROI" \
    --ficture-roi "$FICTURE_ROI" \
    --output-dir "$ASM_OUT" \
    --model-name "$MODEL / $STYLE"
done

date
echo "Done: $ASSEMBLY_BASE"
