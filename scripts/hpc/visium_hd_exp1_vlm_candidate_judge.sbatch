#!/bin/bash
#SBATCH --job-name=vlm_judge
#SBATCH --partition=gpu_h200
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --output=/nfs/roberts/project/pi_xy48/hw646/Medical-SAM3/results/visium_hd_exp1/logs/vlm_judge_%j.log
#SBATCH --mail-type=NONE

set -euo pipefail

PROJECT=/nfs/roberts/project/pi_xy48/hw646/Medical-SAM3
PY=/nfs/roberts/project/pi_xy48/hw646/ycrc_conda/envs/medsam3/bin/python
INPUT_DIR=$PROJECT/output/visium_hd_exp1/ficture_coord_scaled_hires_continuous_candidate_pool_input_roi
OUTBASE=$PROJECT/output/visium_hd_exp1/vlm_candidate_judge

RUN_NAME=${RUN_NAME:-qwen25vl7b_bav}
MODEL_NAME=${MODEL_NAME:-Qwen/Qwen2.5-VL-7B-Instruct}
PER_LABEL=${PER_LABEL:-24}
TOP_KS_LIST=${TOP_KS_LIST:-"1:2:3:5:8:12"}
MAX_OVERLAP=${MAX_OVERLAP:-0.82}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-96}

OUT=$OUTBASE/${RUN_NAME}_${SLURM_JOB_ID}

cd "$PROJECT"
mkdir -p "$OUT" "$PROJECT/results/visium_hd_exp1/logs" "$PROJECT/output/.hf_cache"
export MPLCONFIGDIR="$PROJECT/output/.mplconfig"
export HF_HOME=/nfs/roberts/project/pi_xy48/hw646/ycrc_conda/hf-home
export TRANSFORMERS_CACHE=$HF_HOME/hub
export PYTHONUNBUFFERED=1

IFS=':' read -r -a top_ks <<< "$TOP_KS_LIST"

score_args=(
  --score-csv "$PROJECT/output/visium_hd_exp1/multimodal_candidate_ranking/coord_scaled_hires_he_plus_ficture_11827087/candidate_multimodal_scores.csv"
  --score-csv "$PROJECT/output/visium_hd_exp1/multimodal_candidate_ranking/mm_ficture_prior80_rtx_11827833/candidate_multimodal_scores.csv"
  --score-csv "$PROJECT/output/visium_hd_exp1/multimodal_candidate_ranking/salip_clip_smoke20_scav_fix_11828393/candidate_multimodal_scores.csv"
  --score-csv "$PROJECT/output/visium_hd_exp1/multimodal_candidate_ranking/salip_clip_ficture40_h200_fix_11828394/candidate_multimodal_scores.csv"
  --score-csv "$PROJECT/output/visium_hd_exp1/multimodal_candidate_ranking/salip_clip_he40_b200_11828820/candidate_multimodal_scores.csv"
)

date
echo "RUN_NAME=$RUN_NAME"
echo "MODEL_NAME=$MODEL_NAME"
echo "PER_LABEL=$PER_LABEL TOP_KS=$TOP_KS_LIST MAX_OVERLAP=$MAX_OVERLAP"
nvidia-smi || true
"$PY" inference/visium_hd_exp1/vlm_candidate_judge.py \
  "${score_args[@]}" \
  --he-image "$INPUT_DIR/he_roi_matching_coord_scaled_continuous_ficture_coverage.png" \
  --ficture-image "$INPUT_DIR/ficture_coord_scaled_hires_continuous_roi_rgb.png" \
  --factor-label-npy "$INPUT_DIR/ficture_coord_scaled_hires_continuous_roi_factor_index.npy" \
  --factor-annotation "$PROJECT/output/visium_hd_exp1/molecular_prior_experiments/dx-60_dy80/factor_annotation.csv" \
  --summary-path "$INPUT_DIR/region_summary_coord_scaled_hires_continuous_roi_hpc.json" \
  --output-dir "$OUT" \
  --model "$MODEL_NAME" \
  --device cuda \
  --per-label "$PER_LABEL" \
  --top-ks "${top_ks[@]}" \
  --max-overlap "$MAX_OVERLAP" \
  --max-new-tokens "$MAX_NEW_TOKENS"
date
echo "Output: $OUT"
