#!/bin/bash
#SBATCH -J check_cuda
#SBATCH -p scavenge_gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH -t 00:10:00
#SBATCH -o /nfs/roberts/project/pi_xy48/hw646/Medical-SAM3-tma24/slurm-%j.out

set -euo pipefail

module load miniconda/24.11.3
source "$(conda info --base)/etc/profile.d/conda.sh"

export HOME=/nfs/roberts/project/pi_xy48/hw646/ycrc_conda/home
export CONDA_PKGS_DIRS=/nfs/roberts/project/pi_xy48/hw646/ycrc_conda/pkgs
export PIP_CACHE_DIR=/nfs/roberts/project/pi_xy48/hw646/ycrc_conda/pip-cache
export HF_HOME=/nfs/roberts/project/pi_xy48/hw646/ycrc_conda/hf-home

conda activate /nfs/roberts/project/pi_xy48/hw646/ycrc_conda/envs/medsam3

nvidia-smi

python - <<'PY'
import torch
print("torch", torch.__version__)
print("torch.version.cuda", torch.version.cuda)
print("cuda_available", torch.cuda.is_available())
print("device_count", torch.cuda.device_count())
if torch.cuda.is_available():
    print("device_name", torch.cuda.get_device_name(0))
PY
