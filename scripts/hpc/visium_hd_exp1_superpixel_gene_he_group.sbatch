#!/bin/bash
#SBATCH --job-name=spx_gene_he
#SBATCH --partition=day
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/nfs/roberts/project/pi_xy48/hw646/Medical-SAM3/results/visium_hd_exp1/logs/spx_gene_he_%j.log

set -euo pipefail

NAME=${1:?usage: sbatch visium_hd_exp1_superpixel_gene_he_group.sbatch NAME [args...] }
shift

PROJECT=/nfs/roberts/project/pi_xy48/hw646/Medical-SAM3
PY=/nfs/roberts/project/pi_xy48/hw646/ycrc_conda/envs/medsam3/bin/python
OUT=$PROJECT/results/visium_hd_exp1/superpixel_gene_he/${NAME}_${SLURM_JOB_ID}

mkdir -p "$OUT"
cd "$PROJECT"

echo "Output: $OUT"
echo "Args: $*"
"$PY" inference/visium_hd_exp1/run_superpixel_gene_he_segmentation.py \
  --output-dir "$OUT" \
  "$@"
