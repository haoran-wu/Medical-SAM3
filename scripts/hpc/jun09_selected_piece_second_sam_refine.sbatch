#!/bin/bash
#SBATCH --job-name=jun09_2sam_refine
#SBATCH --partition=gpu_b200,gpu_h200,gpu_devel
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=120G
#SBATCH --time=03:00:00
#SBATCH --output=/nfs/roberts/project/pi_xy48/hw646/codex_salip_outputs/logs/jun09_2sam_refine_%j.log
#SBATCH --error=/nfs/roberts/project/pi_xy48/hw646/codex_salip_outputs/logs/jun09_2sam_refine_%j.log
#SBATCH --mail-type=NONE

set -euo pipefail

PROJECT=${PROJECT:-/nfs/roberts/project/pi_xy48/hw646/Medical-SAM3}
PY=${PY:-/nfs/roberts/project/pi_xy48/hw646/ycrc_conda/envs/medsam3/bin/python}
OUT_BASE=${OUT_BASE:-/nfs/roberts/project/pi_xy48/hw646/Medical-SAM3_remote_runs/visium_hd_exp1/jun09_selected_piece_second_sam_refine}
RUN_NAME=${RUN_NAME:-jun09_selected_piece_second_sam_medical_${SLURM_JOB_ID}}
CHECKPOINT_KIND=${CHECKPOINT_KIND:-medical}
LABELS=${LABELS:-bronchiola,alveoli,vessels,tumor,stroma,immune_infiltration}
BOX_MARGIN=${BOX_MARGIN:-0.08}
PRIOR_RADIUS=${PRIOR_RADIUS:-96}
MAX_PIECES_PER_CLASS=${MAX_PIECES_PER_CLASS:-0}

case "$CHECKPOINT_KIND" in
  base)
    CHECKPOINT="$PROJECT/checkpoints/SAM3-base/sam3.pt"
    ;;
  medical)
    CHECKPOINT="$PROJECT/checkpoints/Medical-SAM3/checkpoint.pt"
    ;;
  *)
    echo "Unknown CHECKPOINT_KIND=$CHECKPOINT_KIND" >&2
    exit 2
    ;;
esac

mkdir -p "$OUT_BASE" /nfs/roberts/project/pi_xy48/hw646/codex_salip_outputs/logs
cd "$PROJECT"

export PYTHONUNBUFFERED=1
export PYTHONPATH="$PROJECT/inference/visium_hd_exp1:$PROJECT/inference:$PROJECT:${PYTHONPATH:-}"
export MPLCONFIGDIR="${MPLCONFIGDIR_OVERRIDE:-/tmp/${USER}_mpl_${SLURM_JOB_ID}}"
export HF_HOME="${HF_HOME_OVERRIDE:-/nfs/roberts/project/pi_xy48/hw646/hf-home}"
export HF_HUB_CACHE="$HF_HOME/hub"
export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"
export TRANSFORMERS_CACHE="$HF_HOME/hub"
mkdir -p "$MPLCONFIGDIR" "$HF_HOME"

BASE_DIR="$PROJECT/output/visium_hd_exp1/final_deliverables/Jun09_SkillRanker_ComponentAwareMaskSelection"
SELECTED="$BASE_DIR/refined_diagnostics/refined_policy_selected_pieces.csv"
HIDDEN="$BASE_DIR/corrected_pool/hidden_candidate_truth.csv"
HE_ROI="$PROJECT/output/visium_hd_exp1/ficture_official_filtered_candidate_pool_input_roi/he_roi_matching_official_ficture_coverage.png"
FICTURE_ROI="$PROJECT/output/visium_hd_exp1/ficture_official_filtered_candidate_pool_input_roi/ficture_official_filtered_roi_rgb.png"
ANNOTATION_DIR="$PROJECT/output/visium_hd_exp1/ficture_official_filtered_candidate_pool_input_roi/cropped_annotation_masks"
OUT_DIR="$OUT_BASE/${RUN_NAME}"

echo "[jun09_2sam] job=${SLURM_JOB_ID} node=$(hostname)"
echo "[jun09_2sam] checkpoint_kind=${CHECKPOINT_KIND}"
echo "[jun09_2sam] labels=${LABELS}"
echo "[jun09_2sam] selected=${SELECTED}"
echo "[jun09_2sam] out=${OUT_DIR}"

for path in "$SELECTED" "$HIDDEN" "$HE_ROI" "$FICTURE_ROI" "$ANNOTATION_DIR" "$CHECKPOINT"; do
  if [[ ! -e "$path" ]]; then
    echo "Missing required path: $path" >&2
    exit 2
  fi
done

"$PY" inference/visium_hd_exp1/run_jun09_selected_piece_second_sam_refine.py \
  --selected-pieces-csv "$SELECTED" \
  --hidden-truth-csv "$HIDDEN" \
  --he-roi "$HE_ROI" \
  --ficture-roi "$FICTURE_ROI" \
  --annotation-dir "$ANNOTATION_DIR" \
  --output-dir "$OUT_DIR" \
  --checkpoint "$CHECKPOINT" \
  --device cuda \
  --labels "$LABELS" \
  --box-margin "$BOX_MARGIN" \
  --prior-dilate-radius "$PRIOR_RADIUS" \
  --max-pieces-per-class "$MAX_PIECES_PER_CLASS"

echo "[jun09_2sam] done: $OUT_DIR"
