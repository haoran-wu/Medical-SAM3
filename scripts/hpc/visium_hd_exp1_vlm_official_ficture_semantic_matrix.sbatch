#!/bin/bash
#SBATCH --job-name=vlm_fic_sem_matrix
#SBATCH --partition=gpu_h200
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=10:00:00
#SBATCH --output=/nfs/roberts/project/pi_xy48/hw646/Medical-SAM3/results/visium_hd_exp1/logs/vlm_fic_sem_matrix_%j.log
#SBATCH --mail-type=NONE

set -u

# Sequential official filtered FICTURE VLM matrix. Run after semantic CLIP
# ranking has produced candidate_multimodal_scores.csv files.

PROJECT=/nfs/roberts/project/pi_xy48/hw646/Medical-SAM3
RANK_BASE=$PROJECT/output/visium_hd_exp1/multimodal_candidate_ranking
JUDGE_SCRIPT=$PROJECT/scripts/hpc/visium_hd_exp1_vlm_candidate_judge_official_ficture.sbatch
GROUND_SCRIPT=$PROJECT/scripts/hpc/visium_hd_exp1_vlm_direct_grounding_official_ficture.sbatch

cd "$PROJECT" || exit 2
mkdir -p "$PROJECT/results/visium_hd_exp1/logs"

status=0

has_scores() {
  local glob=$1
  find "$RANK_BASE" -maxdepth 2 -path "$RANK_BASE/$glob/candidate_multimodal_scores.csv" -type f | grep -q .
}

run_judge() {
  local run_name=$1
  local score_glob=$2
  if ! has_scores "$score_glob"; then
    echo "SKIP $run_name: no scores for SCORE_GLOB=$score_glob"
    return 0
  fi
  echo "================================================================"
  date
  echo "Starting VLM judge $run_name SCORE_GLOB=$score_glob"
  RUN_NAME="$run_name" \
  SCORE_GLOB="$score_glob" \
  PER_LABEL=16 \
  TOP_KS_LIST="1:2:3:5:8:12" \
  MAX_OVERLAP=0.82 \
  MAX_NEW_TOKENS=96 \
    bash "$JUDGE_SCRIPT"
  rc=$?
  if [[ $rc -ne 0 ]]; then
    echo "FAILED $run_name rc=$rc" >&2
    status=1
  else
    echo "COMPLETED $run_name"
  fi
}

run_ground() {
  local run_name=$1
  local score_glob=$2
  if ! has_scores "$score_glob"; then
    echo "SKIP $run_name: no scores for SCORE_GLOB=$score_glob"
    return 0
  fi
  echo "================================================================"
  date
  echo "Starting VLM direct grounding $run_name SCORE_GLOB=$score_glob"
  RUN_NAME="$run_name" \
  SCORE_GLOB="$score_glob" \
  PER_LABEL=40 \
  TOP_KS_LIST="1:2:3:5:8:12:20" \
  MAX_OVERLAP=0.88 \
  MAX_NEW_TOKENS=256 \
    bash "$GROUND_SCRIPT"
  rc=$?
  if [[ $rc -ne 0 ]]; then
    echo "FAILED $run_name rc=$rc" >&2
    status=1
  else
    echo "COMPLETED $run_name"
  fi
}

run_judge official_pubmedclip_semantic_fixed_qwen25vl_judge official_ficture_pubmedclip_semantic_fixed_*
run_ground official_pubmedclip_semantic_fixed_qwen25vl_ground official_ficture_pubmedclip_semantic_fixed_*
run_judge official_plip_semantic_fixed_qwen25vl_judge official_ficture_plip_semantic_fixed_*
run_ground official_plip_semantic_fixed_qwen25vl_ground official_ficture_plip_semantic_fixed_*

exit "$status"
