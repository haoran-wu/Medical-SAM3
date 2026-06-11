#!/bin/bash
#SBATCH --job-name=jun07_semfic_vlm
#SBATCH --partition=gpu_b200
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=02:00:00
#SBATCH --output=/nfs/roberts/project/pi_xy48/hw646/codex_salip_outputs/logs/jun07_semfic_vlm_%j.log
set -euo pipefail

export HF_HOME=/nfs/roberts/project/pi_xy48/hw646/hf-home
export TRANSFORMERS_CACHE=$HF_HOME/hub
export HUGGINGFACE_HUB_CACHE=$HF_HOME/hub
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

cd /nfs/roberts/project/pi_xy48/hw646/codex_salip_scripts_jun05
PY=/nfs/roberts/project/pi_xy48/hw646/ycrc_conda/envs/medsam3/bin/python
POOL=${POOL:-/nfs/roberts/project/pi_xy48/hw646/Medical-SAM3_remote_runs/visium_hd_exp1/jun07_semantic_ficture_position_20260607/semantic_ficture_position_pool/public_vlm_requests.csv}
OUTBASE=${OUTBASE:-/nfs/roberts/project/pi_xy48/hw646/Medical-SAM3_remote_runs/visium_hd_exp1/jun07_semantic_ficture_position_20260607/local_vlm_outputs}
MODEL=${MODEL:-Qwen/Qwen3-VL-32B-Instruct}
RUN=${RUN_NAME:-qwen3vl32b_semantic_ficture_position_smoke18_${SLURM_JOB_ID}}
LIMIT=${LIMIT:-18}

CONTEXT_PROMPT_STYLE=he_semantic_ficture_map CONTEXT_IMAGE_MODE=he_semantic_ficture \
"$PY" run_local_vlm_multiimage_cross_label.py \
  --pool-csv "$POOL" \
  --output-dir "$OUTBASE/$RUN" \
  --model "$MODEL" \
  --device cuda \
  --max-new-tokens 320 \
  --limit "$LIMIT" \
  --resume
