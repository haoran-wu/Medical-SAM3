#!/bin/bash
set -euo pipefail

# Build and submit the Jun10 Step 1 cell-type-first Qwen3 smoke run.
# This is text-only: no H&E or FICTURE images are sent to the model in Step 1.

REMOTE_HOST="${REMOTE_HOST:-bouchet}"
REMOTE_PROJECT="${REMOTE_PROJECT:-/nfs/roberts/project/pi_xy48/hw646/Medical-SAM3}"
LOCAL_OUT="${LOCAL_OUT:-output/visium_hd_exp1/final_deliverables/Jun10_CellTypeFirst_Qwen3_Skill}"
REMOTE_OUT="$REMOTE_PROJECT/$LOCAL_OUT"
LIMIT="${LIMIT:-18}"
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3-VL-32B-Instruct}"
MODEL_SLUG="${MODEL_SLUG:-qwen3vl32b}"
PROMPT_STYLE="${PROMPT_STYLE:-baseline}"

echo "[Jun10] building local Step 1 cell-type-first prompt pool..."
python3 inference/visium_hd_exp1/build_cell_type_first_prompt_pool.py \
  --output-dir "$LOCAL_OUT" \
  --prompt-style "$PROMPT_STYLE"

echo "[Jun10] checking Bouchet SSH..."
HPC_HOST="$REMOTE_HOST" bash scripts/hpc_ssh_check.sh

echo "[Jun10] syncing scripts and request CSV..."
ssh -o BatchMode=yes "$REMOTE_HOST" "mkdir -p '$REMOTE_PROJECT/inference/visium_hd_exp1' '$REMOTE_PROJECT/scripts/hpc' '$REMOTE_OUT'"
scp inference/visium_hd_exp1/build_cell_type_first_prompt_pool.py \
  "$REMOTE_HOST:$REMOTE_PROJECT/inference/visium_hd_exp1/build_cell_type_first_prompt_pool.py"
scp inference/visium_hd_exp1/run_qwen3_cell_type_first_hypothesis.py \
  "$REMOTE_HOST:$REMOTE_PROJECT/inference/visium_hd_exp1/run_qwen3_cell_type_first_hypothesis.py"
scp scripts/hpc/jun10_cell_type_first_qwen3_step1.sbatch \
  "$REMOTE_HOST:$REMOTE_PROJECT/scripts/hpc/jun10_cell_type_first_qwen3_step1.sbatch"
scp "$LOCAL_OUT/cell_type_first_requests.csv" \
  "$REMOTE_HOST:$REMOTE_OUT/cell_type_first_requests.csv"
scp "$LOCAL_OUT/run_config.json" \
  "$REMOTE_HOST:$REMOTE_OUT/run_config.json"

echo "[Jun10] submitting Qwen3 Step 1 smoke..."
ssh -o BatchMode=yes "$REMOTE_HOST" \
  "cd '$REMOTE_PROJECT' && LIMIT='$LIMIT' MODEL_NAME='$MODEL_NAME' MODEL_SLUG='$MODEL_SLUG' REQUESTS_CSV='$REMOTE_OUT/cell_type_first_requests.csv' sbatch scripts/hpc/jun10_cell_type_first_qwen3_step1.sbatch"

echo "[Jun10] submitted. Monitor with:"
echo "ssh -o BatchMode=yes $REMOTE_HOST 'squeue -u hw646 -o \"%.18i %.12P %.40j %.2t %.12M %.12l %.6C %.10m %.24R\"'"
