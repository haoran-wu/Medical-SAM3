#!/bin/bash
#SBATCH --job-name=sam3_fictxa
#SBATCH --partition=gpu_devel,scavenge_gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --array=0-9
#SBATCH --output=/nfs/roberts/project/pi_xy48/hw646/Medical-SAM3/results/visium_hd_exp1/logs/sam3_fictxa_%A_%a.log

set -euo pipefail

CONFIGS=(
  "quality12 medical 48 csv 0.010 24 4096"
  "quality12 medical 80 csv 0.015 32 4096"
  "smooth medical 80 csv 0.010 24 4096"
  "smooth medical 120 csv 0.015 32 4096"
  "raw medical 240 csv 0.006 12 4096"
  "raw medical 240 csv 0.012 20 4096"
  "quality12 base 48 csv 0.010 24 4096"
  "smooth base 80 csv 0.015 32 4096"
  "raw base 240 csv 0.006 12 4096"
  "raw base 240 csv 0.012 20 4096"
)

read -r CANDIDATE_SET CHECKPOINT_KIND LIMIT SORT MARGIN PRIOR_RADIUS MAX_SIDE <<< "${CONFIGS[$SLURM_ARRAY_TASK_ID]}"

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

OUT=results/visium_hd_exp1/sam3_reference_candidate_refine/${CANDIDATE_SET}_${CHECKPOINT_KIND}_l${LIMIT}_${SORT}_m${MARGIN}_r${PRIOR_RADIUS}_hi${MAX_SIDE}_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}

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
  --prior-dilate-radius "$PRIOR_RADIUS"

"$PY" inference/visium_hd_exp1/summarize_sam3_refine_candidates.py "$OUT" --top-panels 12
echo "Output: $OUT"
