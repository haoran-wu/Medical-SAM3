#!/bin/bash
#SBATCH --job-name=mm_rank_fic_cpu_par
#SBATCH --partition=devel
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --output=/nfs/roberts/project/pi_xy48/hw646/Medical-SAM3/results/visium_hd_exp1/logs/mm_rank_fic_cpu_par_%j.log
#SBATCH --mail-type=NONE

set -u

# Parallel CPU-only semantic ranking matrix. The ranking code is mostly
# single-process Python/NumPy, so asking for more CPUs only helps if we run
# independent ranking designs concurrently inside one Slurm allocation.

PROJECT=/nfs/roberts/project/pi_xy48/hw646/Medical-SAM3
RUN_SCRIPT=$PROJECT/scripts/hpc/visium_hd_exp1_multimodal_candidate_ranking_official_ficture_cpu.sbatch

cd "$PROJECT" || exit 2
mkdir -p "$PROJECT/results/visium_hd_exp1/logs"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

run_one() {
  local run_name=$1
  local cand_limit=$2
  local top_ks=$3
  local max_overlap=$4
  local consensus_disabled=$5
  local consensus_limit=${6:-80}
  local mask_cache=${7:-32}

  {
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
    else
      echo "COMPLETED $run_name"
    fi
    date
    return "$rc"
  } 2>&1 | sed "s/^/[$run_name] /"
}

pids=()
run_one official_ficture_semantic_lowoverlap_cpu_par 120 "1:2:3:5:8:12:20" 0.55 1 80 24 &
pids+=($!)
run_one official_ficture_semantic_highrecall_cpu_par 180 "1:2:3:5:8:12:20:32" 0.92 0 120 40 &
pids+=($!)
run_one official_ficture_semantic_consensus_wide_cpu_par 160 "1:2:3:5:8:12:20" 0.86 0 140 40 &
pids+=($!)

status=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    status=1
  fi
done

exit "$status"
