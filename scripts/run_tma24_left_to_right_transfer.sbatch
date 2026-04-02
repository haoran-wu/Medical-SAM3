#!/bin/bash
#SBATCH -J tma24_l2r_transfer
#SBATCH -p scavenge_gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH -t 02:00:00
#SBATCH -o /nfs/roberts/project/pi_xy48/hw646/Medical-SAM3-haoran/slurm-%j.out

set -euo pipefail

module load miniconda/24.11.3
source "$(conda info --base)/etc/profile.d/conda.sh"

export HOME=/nfs/roberts/project/pi_xy48/hw646/ycrc_conda/home
export CONDA_PKGS_DIRS=/nfs/roberts/project/pi_xy48/hw646/ycrc_conda/pkgs
export PIP_CACHE_DIR=/nfs/roberts/project/pi_xy48/hw646/ycrc_conda/pip-cache
export HF_HOME=/nfs/roberts/project/pi_xy48/hw646/ycrc_conda/hf-home
export MPLCONFIGDIR=/tmp/mpl

conda activate /nfs/roberts/project/pi_xy48/hw646/ycrc_conda/envs/medsam3

REPO=/nfs/roberts/project/pi_xy48/hw646/Medical-SAM3-haoran
CHECKPOINT=/nfs/roberts/project/pi_xy48/hw646/checkpoint.pt

cd "$REPO"

python - <<'PY'
import torch
print("torch", torch.__version__)
print("cuda", torch.cuda.is_available())
print("device_count", torch.cuda.device_count())
PY

python inference/render_tma24_cell_masks.py

python inference/run_tma24_left_to_right_transfer.py \
  --checkpoint "$CHECKPOINT"
