#!/bin/bash
#SBATCH --job-name=mm_rank_fic_cpu_matrix
#SBATCH --partition=devel
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --output=/nfs/roberts/project/pi_xy48/hw646/Medical-SAM3/results/visium_hd_exp1/logs/mm_rank_fic_cpu_matrix_%j.log
#SBATCH --mail-type=NONE

set -u

# Sequential CPU-only semantic ranking matrix. It avoids GPU use and keeps the
# submit count low by running several top-k/overlap/consensus variants in one job.

PROJECT=/nfs/roberts/project/pi_xy48/hw646/Medical-SAM3
cd "$PROJECT" || exit 2
mkdir -p "$PROJECT/results/visium_hd_exp1/logs"

RUN_SCRIPT=$PROJECT/scripts/hpc/visium_hd_exp1_multimodal_candidate_ranking_official_ficture_cpu.sbatch
if [[ ! -f "$RUN_SCRIPT" ]]; then
  echo "Missing CPU ranking script: $RUN_SCRIPT" >&2
  exit 2
fi

status=0

run_one() {
  local run_name=$1
  local cand_limit=$2
  local top_ks=$3
  local max_overlap=$4
  local consensus_disabled=$5
  local consensus_limit=${6:-80}
  local mask_cache=${7:-32}

  echo "================================================================"
  date
  echo "Starting $run_name"
  echo "CAND_LIMIT=$cand_limit TOP_KS_LIST=$top_ks MAX_OVERLAP=$max_overlap DISABLE_CONSENSUS=$consensus_disabled"
  RUN_NAME="$run_name" \
  CAND_LIMIT="$cand_limit" \
  TOP_KS_LIST="$top_ks" \
  MAX_OVERLAP="$max_overlap" \
  DISABLE_CONSENSUS="$consensus_disabled" \
  CONSENSUS_SAMPLE_LIMIT="$consensus_limit" \
  MASK_CACHE_SIZE="$mask_cache" \
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

run_one official_ficture_semantic_precision_cpu 80 "1:2:3:5:8:12" 0.68 1 80 24
run_one official_ficture_semantic_lowoverlap_cpu 120 "1:2:3:5:8:12:20" 0.55 1 80 24
run_one official_ficture_semantic_highrecall_cpu 180 "1:2:3:5:8:12:20:32" 0.92 0 120 40
run_one official_ficture_semantic_consensus_wide_cpu 160 "1:2:3:5:8:12:20" 0.86 0 140 40

exit "$status"
