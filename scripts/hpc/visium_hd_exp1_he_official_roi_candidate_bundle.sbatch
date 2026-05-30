#!/bin/bash
#SBATCH --job-name=he_roi_bundle
#SBATCH --partition=gpu_devel
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=120G
#SBATCH --time=03:00:00
#SBATCH --output=/nfs/roberts/project/pi_xy48/hw646/Medical-SAM3/results/visium_hd_exp1/logs/he_roi_bundle_%j.log
#SBATCH --mail-type=NONE

set -euo pipefail

# Run several same-ROI H&E candidate settings sequentially inside one Slurm job.
# This avoids array submit-count limits while writing into the same final pool.

RUN_SCRIPT=${RUN_SCRIPT:-/nfs/roberts/project/pi_xy48/hw646/Medical-SAM3/scripts/hpc/visium_hd_exp1_he_official_roi_candidate_array.sbatch}
TASK_IDS=${TASK_IDS:?Set TASK_IDS, e.g. "36 37 38 39"}
SOURCE_ARRAY_JOB_ID=${SOURCE_ARRAY_JOB_ID:-$SLURM_JOB_ID}

echo "Bundle job: $SLURM_JOB_ID"
echo "Source array id for logging: $SOURCE_ARRAY_JOB_ID"
echo "Task ids: $TASK_IDS"
echo "Root override: ${ROOT_OUT_OVERRIDE:-}"
date

for task_id in $TASK_IDS; do
  echo "=== Running same-ROI H&E candidate task ${task_id} ==="
  export SLURM_ARRAY_JOB_ID="$SOURCE_ARRAY_JOB_ID"
  export SLURM_ARRAY_TASK_ID="$task_id"
  bash "$RUN_SCRIPT"
done

date
echo "Bundle complete."
