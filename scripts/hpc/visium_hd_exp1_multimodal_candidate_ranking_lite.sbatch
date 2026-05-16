#!/bin/bash
#SBATCH --job-name=mm_rank_lite
#SBATCH --partition=gpu_devel
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=/nfs/roberts/project/pi_xy48/hw646/Medical-SAM3/results/visium_hd_exp1/logs/mm_rank_lite_%j.log
#SBATCH --mail-type=NONE

set -euo pipefail

PROJECT=/nfs/roberts/project/pi_xy48/hw646/Medical-SAM3
PY=/nfs/roberts/project/pi_xy48/hw646/ycrc_conda/envs/medsam3/bin/python
INPUT_DIR=$PROJECT/output/visium_hd_exp1/ficture_coord_scaled_hires_continuous_candidate_pool_input_roi

RUN_NAME=${RUN_NAME:-both_prior_lite}
CANDIDATE_MODE=${CANDIDATE_MODE:-both}
CAND_LIMIT=${CAND_LIMIT:-40}
TOP_KS_LIST=${TOP_KS_LIST:-"1:2:3:5"}
MAX_OVERLAP=${MAX_OVERLAP:-0.72}
DISABLE_CONSENSUS=${DISABLE_CONSENSUS:-1}
CONSENSUS_SAMPLE_LIMIT=${CONSENSUS_SAMPLE_LIMIT:-80}
MASK_CACHE_SIZE=${MASK_CACHE_SIZE:-24}
FACTOR_ANNOTATION=${FACTOR_ANNOTATION:-$PROJECT/output/visium_hd_exp1/molecular_prior_experiments/dx-60_dy80/factor_annotation.csv}
CLIP_MODEL=${CLIP_MODEL:-}
CLIP_BATCH_SIZE=${CLIP_BATCH_SIZE:-32}

OUT=$PROJECT/output/visium_hd_exp1/multimodal_candidate_ranking/${RUN_NAME}_${SLURM_JOB_ID}
HE_ROOT=$PROJECT/results/visium_hd_exp1/he_coord_scaled_hires_candidate_pool
FICTURE_ROOT=$PROJECT/results/visium_hd_exp1/ficture_coord_scaled_hires_candidate_pool

cd "$PROJECT"
mkdir -p "$OUT" "$PROJECT/results/visium_hd_exp1/logs"
export MPLCONFIGDIR="$PROJECT/output/.mplconfig"
export PYTHONUNBUFFERED=1

candidate_args=()
case "$CANDIDATE_MODE" in
  both)
    candidate_args+=(--candidate-root "he=$HE_ROOT")
    candidate_args+=(--candidate-root "ficture=$FICTURE_ROOT")
    ;;
  he)
    candidate_args+=(--candidate-root "he=$HE_ROOT")
    ;;
  ficture)
    candidate_args+=(--candidate-root "ficture=$FICTURE_ROOT")
    ;;
  *)
    echo "Unknown CANDIDATE_MODE=$CANDIDATE_MODE" >&2
    exit 2
    ;;
esac

consensus_args=()
if [[ "$DISABLE_CONSENSUS" == "1" ]]; then
  consensus_args+=(--disable-consensus)
else
  consensus_args+=(--consensus-sample-limit "$CONSENSUS_SAMPLE_LIMIT")
fi

clip_args=()
if [[ -n "$CLIP_MODEL" ]]; then
  clip_args+=(--clip-model "$CLIP_MODEL" --clip-batch-size "$CLIP_BATCH_SIZE" --clip-device cuda)
fi

IFS=':' read -r -a top_ks <<< "$TOP_KS_LIST"

date
echo "RUN_NAME=$RUN_NAME"
echo "CANDIDATE_MODE=$CANDIDATE_MODE CAND_LIMIT=$CAND_LIMIT TOP_KS=$TOP_KS_LIST CONSENSUS_DISABLED=$DISABLE_CONSENSUS"
echo "CLIP_MODEL=${CLIP_MODEL:-none}"
"$PY" inference/visium_hd_exp1/rank_multimodal_sam_candidates.py \
  "${candidate_args[@]}" \
  --he-image "$INPUT_DIR/he_roi_matching_coord_scaled_continuous_ficture_coverage.png" \
  --factor-label-npy "$INPUT_DIR/ficture_coord_scaled_hires_continuous_roi_factor_index.npy" \
  --summary-path "$INPUT_DIR/region_summary_coord_scaled_hires_continuous_roi_hpc.json" \
  --factor-annotation "$FACTOR_ANNOTATION" \
  --output-dir "$OUT" \
  --top-ks "${top_ks[@]}" \
  --max-overlap "$MAX_OVERLAP" \
  --max-candidates-per-source-setting "$CAND_LIMIT" \
  --mask-cache-size "$MASK_CACHE_SIZE" \
  "${consensus_args[@]}" \
  "${clip_args[@]}"
date
echo "Output: $OUT"
