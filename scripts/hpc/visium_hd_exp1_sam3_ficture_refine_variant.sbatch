#!/bin/bash
#SBATCH --job-name=sam3_fictxv
#SBATCH --partition=gpu_devel
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --output=/nfs/roberts/project/pi_xy48/hw646/Medical-SAM3/results/visium_hd_exp1/logs/sam3_fictxv_%j.log

set -euo pipefail

CANDIDATE_SET=${1:?usage: sbatch visium_hd_exp1_sam3_ficture_refine_variant.sbatch CANDIDATE_SET CHECKPOINT_KIND LIMIT SORT MARGIN PRIOR_RADIUS MAX_SIDE [extra args...] }
CHECKPOINT_KIND=${2:?usage: sbatch visium_hd_exp1_sam3_ficture_refine_variant.sbatch CANDIDATE_SET CHECKPOINT_KIND LIMIT SORT MARGIN PRIOR_RADIUS MAX_SIDE [extra args...] }
LIMIT=${3:?usage: sbatch visium_hd_exp1_sam3_ficture_refine_variant.sbatch CANDIDATE_SET CHECKPOINT_KIND LIMIT SORT MARGIN PRIOR_RADIUS MAX_SIDE [extra args...] }
SORT=${4:?usage: sbatch visium_hd_exp1_sam3_ficture_refine_variant.sbatch CANDIDATE_SET CHECKPOINT_KIND LIMIT SORT MARGIN PRIOR_RADIUS MAX_SIDE [extra args...] }
MARGIN=${5:?usage: sbatch visium_hd_exp1_sam3_ficture_refine_variant.sbatch CANDIDATE_SET CHECKPOINT_KIND LIMIT SORT MARGIN PRIOR_RADIUS MAX_SIDE [extra args...] }
PRIOR_RADIUS=${6:?usage: sbatch visium_hd_exp1_sam3_ficture_refine_variant.sbatch CANDIDATE_SET CHECKPOINT_KIND LIMIT SORT MARGIN PRIOR_RADIUS MAX_SIDE [extra args...] }
MAX_SIDE=${7:?usage: sbatch visium_hd_exp1_sam3_ficture_refine_variant.sbatch CANDIDATE_SET CHECKPOINT_KIND LIMIT SORT MARGIN PRIOR_RADIUS MAX_SIDE [extra args...] }
shift 7

PROJECT=/nfs/roberts/project/pi_xy48/hw646/Medical-SAM3
PY=/nfs/roberts/project/pi_xy48/hw646/ycrc_conda/envs/medsam3/bin/python

case "$CANDIDATE_SET" in
  quality12)
    CANDIDATE_ROOT=output/visium_hd_exp1/reference_map_candidates/hex12_k12_original_smooth_quality_top12
    ;;
  smooth)
    CANDIDATE_ROOT=output/visium_hd_exp1/reference_map_candidates/hex12_k12_original_smooth
    ;;
  raw)
    CANDIDATE_ROOT=output/visium_hd_exp1/reference_map_candidates/hex12_k12
    ;;
  *)
    echo "Unknown CANDIDATE_SET: $CANDIDATE_SET" >&2
    exit 2
    ;;
esac

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

cd "$PROJECT"
export MPLCONFIGDIR="$PROJECT/output/.mplconfig"
export HF_HOME=/nfs/roberts/project/pi_xy48/hw646/ycrc_conda/hf-home
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
mkdir -p "$MPLCONFIGDIR" "$PROJECT/results/visium_hd_exp1/logs"

OUT=results/visium_hd_exp1/sam3_reference_candidate_refine/${CANDIDATE_SET}_${CHECKPOINT_KIND}_l${LIMIT}_${SORT}_m${MARGIN}_r${PRIOR_RADIUS}_hi${MAX_SIDE}_${SLURM_JOB_ID}

"$PY" inference/visium_hd_exp1/run_sam3_refine_reference_candidates.py \
  --candidate-root "$CANDIDATE_ROOT" \
  --reference-map output/visium_hd_exp1/assets/hex_12.k12.pixel.png \
  --target-image output/visium_hd_exp1/assets/tissue_hires_image.png \
  --output-dir "$OUT" \
  --checkpoint "$CHECKPOINT" \
  --device cuda \
  --max-side "$MAX_SIDE" \
  --limit "$LIMIT" \
  --candidate-sort "$SORT" \
  --box-margin "$MARGIN" \
  --transform-json output/visium_hd_exp1/registration_debug/ficture_coordinate_ref_to_he.json \
  --use-affine-prior-mask \
  --clip-to-tissue \
  --prior-dilate-radius "$PRIOR_RADIUS" \
  "$@"

"$PY" inference/visium_hd_exp1/summarize_sam3_refine_candidates.py "$OUT" --top-panels 12
echo "Output: $OUT"
