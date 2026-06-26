#!/bin/bash
#SBATCH --job-name=jun10_he_view_ablation
#SBATCH --partition=gpu_rtx6000
#SBATCH --exclude=a1112u01n01
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=160G
#SBATCH --time=06:00:00
#SBATCH --output=/nfs/roberts/project/pi_xy48/hw646/codex_salip_outputs/logs/jun10_he_view_ablation_%j.log
#SBATCH --error=/nfs/roberts/project/pi_xy48/hw646/codex_salip_outputs/logs/jun10_he_view_ablation_%j.log
#SBATCH --mail-type=NONE

export VLM_MAX_MEMORY_CUDA="${VLM_MAX_MEMORY_CUDA:-80GiB}"
exec /bin/bash /nfs/roberts/project/pi_xy48/hw646/Medical-SAM3/scripts/hpc/jun10_he_view_ablation_qwen3_seq.sbatch
