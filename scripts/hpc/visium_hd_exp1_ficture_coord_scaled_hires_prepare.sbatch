#!/bin/bash
#SBATCH --job-name=fic_coord_prep
#SBATCH --partition=day
#SBATCH --cpus-per-task=2
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=/nfs/roberts/project/pi_xy48/hw646/Medical-SAM3/results/visium_hd_exp1/logs/fic_coord_prep_%j.log
#SBATCH --mail-type=NONE

set -euo pipefail

PROJECT=/nfs/roberts/project/pi_xy48/hw646/Medical-SAM3
PY=/nfs/roberts/project/pi_xy48/hw646/ycrc_conda/envs/medsam3/bin/python

mkdir -p "$PROJECT/results/visium_hd_exp1/logs"
cd "$PROJECT"

export MPLCONFIGDIR="$PROJECT/output/.mplconfig"
mkdir -p "$MPLCONFIGDIR"

date
"$PY" inference/visium_hd_exp1/render_ficture_coord_scaled_hires_continuous.py
date
