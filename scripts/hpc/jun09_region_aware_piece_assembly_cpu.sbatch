#!/bin/bash
#SBATCH --job-name=jun09_region_asm
#SBATCH --partition=devel
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/nfs/roberts/project/pi_xy48/hw646/codex_salip_outputs/logs/jun09_region_asm_%j.log
#SBATCH --error=/nfs/roberts/project/pi_xy48/hw646/codex_salip_outputs/logs/jun09_region_asm_%j.log
#SBATCH --mail-type=NONE

set -euo pipefail

PROJECT=${PROJECT:-/nfs/roberts/project/pi_xy48/hw646/Medical-SAM3}
PY=${PY:-/nfs/roberts/project/pi_xy48/hw646/ycrc_conda/envs/medsam3/bin/python}
ASSEMBLE_SCRIPT=${ASSEMBLE_SCRIPT:-$PROJECT/inference/visium_hd_exp1/assemble_piece_first_vlm_results.py}
PACK_DIR=${PACK_DIR:-$PROJECT/output/visium_hd_exp1/final_deliverables/Jun09_SkillRanker_ComponentAwareMaskSelection/region_aware_piece_context_locator_pack}
INPUT_DIR=${INPUT_DIR:-$PROJECT/output/visium_hd_exp1/ficture_official_filtered_candidate_pool_input_roi}
ANNOTATION_DIR=${ANNOTATION_DIR:-$INPUT_DIR/cropped_annotation_masks}
HE_ROI=${HE_ROI:-$INPUT_DIR/he_roi_matching_official_ficture_coverage.png}
FICTURE_ROI=${FICTURE_ROI:-$INPUT_DIR/ficture_official_filtered_roi_rgb.png}
VLM_OUTPUT_DIR=${VLM_OUTPUT_DIR:?Set VLM_OUTPUT_DIR to one completed jun09_region_aware VLM output directory}
ASSEMBLY_BASE=${ASSEMBLY_BASE:-/nfs/roberts/project/pi_xy48/hw646/Medical-SAM3_remote_runs/visium_hd_exp1/jun09_region_aware_vlm_assembly}
MODEL_LABEL=${MODEL_LABEL:-$(basename "$VLM_OUTPUT_DIR")}
OUT_DIR=${OUT_DIR:-$ASSEMBLY_BASE/${MODEL_LABEL}_${SLURM_JOB_ID}}

mkdir -p "$ASSEMBLY_BASE" /nfs/roberts/project/pi_xy48/hw646/codex_salip_outputs/logs
cd "$PROJECT"

export PYTHONUNBUFFERED=1
export PYTHONPATH="$PROJECT/inference/visium_hd_exp1:$PROJECT/inference:$PROJECT:${PYTHONPATH:-}"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export MPLCONFIGDIR="${MPLCONFIGDIR_OVERRIDE:-/tmp/${USER}_mpl_${SLURM_JOB_ID}}"
mkdir -p "$MPLCONFIGDIR"

echo "[jun09_region_asm] job=${SLURM_JOB_ID} node=$(hostname)"
echo "[jun09_region_asm] vlm=${VLM_OUTPUT_DIR}"
echo "[jun09_region_asm] pack=${PACK_DIR}"
echo "[jun09_region_asm] input=${INPUT_DIR}"
echo "[jun09_region_asm] out=${OUT_DIR}"

for required in \
  "$VLM_OUTPUT_DIR/per_candidate_predictions.csv" \
  "$PACK_DIR/hidden_candidate_truth.csv" \
  "$ANNOTATION_DIR" \
  "$HE_ROI" \
  "$FICTURE_ROI"; do
  if [[ ! -e "$required" ]]; then
    echo "Missing required assembly input: $required" >&2
    exit 2
  fi
done

"$PY" "$ASSEMBLE_SCRIPT" \
  --vlm-output-dir "$VLM_OUTPUT_DIR" \
  --pool-dir "$PACK_DIR" \
  --annotation-dir "$ANNOTATION_DIR" \
  --he-roi "$HE_ROI" \
  --ficture-roi "$FICTURE_ROI" \
  --output-dir "$OUT_DIR" \
  --model-name "$MODEL_LABEL"

echo "[jun09_region_asm] done: $OUT_DIR"
