#!/bin/bash
#SBATCH --job-name=sam3_tcomp
#SBATCH --partition=gpu_devel
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --output=/nfs/roberts/project/pi_xy48/hw646/Medical-SAM3/results/visium_hd_exp1/logs/sam3_tcomp_%j.log

set -euo pipefail

CHECKPOINT_KIND=${1:?usage: sbatch visium_hd_exp1_teacher_component_sam3_refine.sbatch CHECKPOINT_KIND TEACHER_SET VARIANT MIN_AREA TOP_COMPONENTS MARGIN PRIOR_RADIUS MAX_SIDE [extra args...] }
TEACHER_SET=${2:?usage: sbatch visium_hd_exp1_teacher_component_sam3_refine.sbatch CHECKPOINT_KIND TEACHER_SET VARIANT MIN_AREA TOP_COMPONENTS MARGIN PRIOR_RADIUS MAX_SIDE [extra args...] }
VARIANT=${3:?usage: sbatch visium_hd_exp1_teacher_component_sam3_refine.sbatch CHECKPOINT_KIND TEACHER_SET VARIANT MIN_AREA TOP_COMPONENTS MARGIN PRIOR_RADIUS MAX_SIDE [extra args...] }
MIN_AREA=${4:?usage: sbatch visium_hd_exp1_teacher_component_sam3_refine.sbatch CHECKPOINT_KIND TEACHER_SET VARIANT MIN_AREA TOP_COMPONENTS MARGIN PRIOR_RADIUS MAX_SIDE [extra args...] }
TOP_COMPONENTS=${5:?usage: sbatch visium_hd_exp1_teacher_component_sam3_refine.sbatch CHECKPOINT_KIND TEACHER_SET VARIANT MIN_AREA TOP_COMPONENTS MARGIN PRIOR_RADIUS MAX_SIDE [extra args...] }
MARGIN=${6:?usage: sbatch visium_hd_exp1_teacher_component_sam3_refine.sbatch CHECKPOINT_KIND TEACHER_SET VARIANT MIN_AREA TOP_COMPONENTS MARGIN PRIOR_RADIUS MAX_SIDE [extra args...] }
PRIOR_RADIUS=${7:?usage: sbatch visium_hd_exp1_teacher_component_sam3_refine.sbatch CHECKPOINT_KIND TEACHER_SET VARIANT MIN_AREA TOP_COMPONENTS MARGIN PRIOR_RADIUS MAX_SIDE [extra args...] }
MAX_SIDE=${8:?usage: sbatch visium_hd_exp1_teacher_component_sam3_refine.sbatch CHECKPOINT_KIND TEACHER_SET VARIANT MIN_AREA TOP_COMPONENTS MARGIN PRIOR_RADIUS MAX_SIDE [extra args...] }
shift 8

PROJECT=/nfs/roberts/project/pi_xy48/hw646/Medical-SAM3
PY=/nfs/roberts/project/pi_xy48/hw646/ycrc_conda/envs/medsam3/bin/python

case "$CHECKPOINT_KIND" in
  base)
    CHECKPOINT=checkpoints/SAM3-base/sam3.pt
    ;;
  medical)
    CHECKPOINT=checkpoints/Medical-SAM3/checkpoint.pt
    ;;
  *)
    echo "Unknown CHECKPOINT_KIND: $CHECKPOINT_KIND" >&2
    exit 2
    ;;
esac

case "$TEACHER_SET" in
  unified_top3)
    TEACHER_ARGS=(
      --teacher-root results/visium_hd_exp1/unified_pixel_classifier/unified_et_nofactor_xy_1536_11418035/masks
      --teacher-root results/visium_hd_exp1/unified_pixel_classifier/unified_hgb_factor_xy_1536_11418037/masks
      --teacher-root results/visium_hd_exp1/unified_pixel_classifier/unified_et_factor_xy_1536_11418033/masks
    )
    ;;
  unified_best)
    TEACHER_ARGS=(--teacher-root results/visium_hd_exp1/unified_pixel_classifier/unified_et_nofactor_xy_1536_11418035/masks)
    ;;
  ensemble)
    TEACHER_ARGS=(--teacher-root results/visium_hd_exp1/completed_expert_ensemble/ensemble_11417101/masks)
    ;;
  *)
    echo "Unknown TEACHER_SET: $TEACHER_SET" >&2
    exit 2
    ;;
esac

cd "$PROJECT"
export MPLCONFIGDIR="$PROJECT/output/.mplconfig"
export HF_HOME=/nfs/roberts/project/pi_xy48/hw646/ycrc_conda/hf-home
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
mkdir -p "$MPLCONFIGDIR" "$PROJECT/results/visium_hd_exp1/logs"

OUT=results/visium_hd_exp1/teacher_component_sam3_refine/${CHECKPOINT_KIND}_${TEACHER_SET}_${VARIANT}_a${MIN_AREA}_n${TOP_COMPONENTS}_m${MARGIN}_r${PRIOR_RADIUS}_hi${MAX_SIDE}_${SLURM_JOB_ID}

"$PY" inference/visium_hd_exp1/run_teacher_component_sam3_refine.py \
  "${TEACHER_ARGS[@]}" \
  --teacher-variant "$VARIANT" \
  --he-image output/visium_hd_exp1/assets/tissue_hires_image.png \
  --summary-path output/visium_hd_exp1/sam3_local_region_summary/region_summary.json \
  --output-dir "$OUT" \
  --checkpoint "$CHECKPOINT" \
  --device cuda \
  --max-side "$MAX_SIDE" \
  --min-component-area "$MIN_AREA" \
  --max-components-per-label "$TOP_COMPONENTS" \
  --box-margin "$MARGIN" \
  --prior-dilate-radius "$PRIOR_RADIUS" \
  --clip-to-tissue \
  "$@"

echo "Output: $OUT"
