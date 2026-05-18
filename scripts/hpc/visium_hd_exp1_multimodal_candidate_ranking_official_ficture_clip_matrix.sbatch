#!/bin/bash
#SBATCH --job-name=mm_rank_fic_clip_matrix
#SBATCH --partition=gpu_rtx6000
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=/nfs/roberts/project/pi_xy48/hw646/Medical-SAM3/results/visium_hd_exp1/logs/mm_rank_fic_clip_matrix_%j.log
#SBATCH --mail-type=NONE

set -u

# Sequential official filtered FICTURE CLIP matrix. This stays within QOS
# submit-count limits by running several CLIP/ranking designs inside one job.

PROJECT=/nfs/roberts/project/pi_xy48/hw646/Medical-SAM3
cd "$PROJECT" || exit 2
mkdir -p "$PROJECT/results/visium_hd_exp1/logs"

RUN_SCRIPT=$PROJECT/scripts/hpc/visium_hd_exp1_multimodal_candidate_ranking_official_ficture.sbatch
if [[ ! -f "$RUN_SCRIPT" ]]; then
  echo "Missing ranking script: $RUN_SCRIPT" >&2
  exit 2
fi

status=0

run_one() {
  local run_name=$1
  local cand_limit=$2
  local top_ks=$3
  local max_overlap=$4
  local clip_model=$5
  local consensus_disabled=${6:-1}
  local consensus_limit=${7:-80}

  echo "================================================================"
  date
  echo "Starting $run_name"
  echo "CAND_LIMIT=$cand_limit TOP_KS_LIST=$top_ks MAX_OVERLAP=$max_overlap CLIP_MODEL=$clip_model"
  RUN_NAME="$run_name" \
  CAND_LIMIT="$cand_limit" \
  TOP_KS_LIST="$top_ks" \
  MAX_OVERLAP="$max_overlap" \
  DISABLE_CONSENSUS="$consensus_disabled" \
  CONSENSUS_SAMPLE_LIMIT="$consensus_limit" \
  MASK_CACHE_SIZE=24 \
  CLIP_MODEL="$clip_model" \
  CLIP_BATCH_SIZE=32 \
    bash "$RUN_SCRIPT"
  rc=$?
  if [[ $rc -ne 0 ]]; then
    echo "FAILED $run_name rc=$rc" >&2
    status=1
  else
    echo "COMPLETED $run_name"
  fi
  date
}

run_one official_ficture_pubmedclip_semantic_fixed 80 "1:2:3:5:8:12:20" 0.86 flaviagiammarino/pubmed-clip-vit-base-patch32 1
run_one official_ficture_pubmedclip_semantic_precision 60 "1:2:3:5:8:12" 0.70 flaviagiammarino/pubmed-clip-vit-base-patch32 1
run_one official_ficture_plip_semantic_fixed 80 "1:2:3:5:8:12:20" 0.86 vinid/plip 1
run_one official_ficture_openai_clip_semantic 80 "1:2:3:5:8:12:20" 0.86 openai/clip-vit-base-patch32 1

exit "$status"
