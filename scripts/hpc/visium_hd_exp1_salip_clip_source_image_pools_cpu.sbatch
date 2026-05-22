#!/bin/bash
#SBATCH --job-name=salip_src_cpu
#SBATCH --partition=devel
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=06:00:00
#SBATCH --output=/nfs/roberts/project/pi_xy48/hw646/Medical-SAM3/results/visium_hd_exp1/logs/salip_src_cpu_%j.log
#SBATCH --mail-type=NONE

set -euo pipefail

export CLIP_DEVICE=cpu
export CLIP_BATCH_SIZE=${CLIP_BATCH_SIZE:-16}
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}

bash /home/hw646/codex_salip_scripts/visium_hd_exp1_salip_clip_source_image_pools.sbatch
