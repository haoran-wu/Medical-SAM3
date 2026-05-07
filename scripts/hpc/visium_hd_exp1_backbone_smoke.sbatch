#!/bin/bash
#SBATCH --job-name=backbone_smoke
#SBATCH --partition=gpu_rtx6000
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=80G
#SBATCH --time=01:00:00
#SBATCH --array=0-3
#SBATCH --output=/nfs/roberts/project/pi_xy48/hw646/Medical-SAM3/results/visium_hd_exp1/logs/backbone_smoke_%A_%a.log

set -euo pipefail

PROJECT=/nfs/roberts/project/pi_xy48/hw646/Medical-SAM3
PY=/nfs/roberts/project/pi_xy48/hw646/ycrc_conda/envs/medsam3/bin/python
RESULTS=${PROJECT}/results/visium_hd_exp1

export HF_HOME=/nfs/roberts/project/pi_xy48/hw646/ycrc_conda/hf-home
export TRANSFORMERS_CACHE=${HF_HOME}/hub
export HF_HUB_CACHE=${HF_HOME}/hub
export PYTHONUNBUFFERED=1

mkdir -p "${RESULTS}/logs"
cd "${PROJECT}"

BACKBONES=(
  "uni"
  "conch"
  "virchow"
  "virchow2"
)

BACKBONE=${BACKBONES[$SLURM_ARRAY_TASK_ID]}
echo "Backbone smoke: ${BACKBONE}"

"${PY}" - <<PY
import sys
from pathlib import Path
import torch

sys.path.insert(0, str(Path("${PROJECT}") / "inference" / "visium_hd_exp1"))
from image_backbones import create_image_backbone

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device", device)
model, dim = create_image_backbone("${BACKBONE}")
model = model.to(device).eval()
x = torch.zeros(1, 3, 224, 224, device=device)
with torch.inference_mode():
    try:
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=device.type == "cuda"):
            y = model(x)
    except RuntimeError:
        y = model(x)
print("feature_dim", dim)
print("output_shape", tuple(y.shape))
print("ok")
PY
