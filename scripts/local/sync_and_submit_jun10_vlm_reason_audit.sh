#!/bin/bash
set -euo pipefail

# Submit Qwen3-VL-32B full-description VLM audit with concise visible reasons.

REMOTE_HOST="${REMOTE_HOST:-bouchet}"
REMOTE_PROJECT="${REMOTE_PROJECT:-/nfs/roberts/project/pi_xy48/hw646/Medical-SAM3}"
LOCAL_POOL="${LOCAL_POOL:-output/visium_hd_exp1/final_deliverables/Jun09_SkillRanker_ComponentAwareMaskSelection/corrected_pool}"
REMOTE_POOL="$REMOTE_PROJECT/$LOCAL_POOL"
LIMIT="${LIMIT:-0}"
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3-VL-32B-Instruct}"
RUN_NAME="${RUN_NAME:-qwen3vl32b_full_descriptions_visible_reason}"

echo "[Jun10 VLM reason audit] checking Bouchet SSH..."
HPC_HOST="$REMOTE_HOST" bash scripts/hpc_ssh_check.sh

echo "[Jun10 VLM reason audit] syncing code, prompt contract, pool CSV, and crop images..."
ssh -o BatchMode=yes "$REMOTE_HOST" "mkdir -p '$REMOTE_PROJECT/inference/visium_hd_exp1' '$REMOTE_PROJECT/scripts/hpc' '$REMOTE_POOL'"
scp inference/visium_hd_exp1/vlm_prompt_contract.py \
  "$REMOTE_HOST:$REMOTE_PROJECT/inference/visium_hd_exp1/vlm_prompt_contract.py"
scp inference/visium_hd_exp1/run_paired_vlm_hit_test.py \
  "$REMOTE_HOST:$REMOTE_PROJECT/inference/visium_hd_exp1/run_paired_vlm_hit_test.py"
scp inference/visium_hd_exp1/run_local_vlm_cross_label_top1_accuracy.py \
  "$REMOTE_HOST:$REMOTE_PROJECT/inference/visium_hd_exp1/run_local_vlm_cross_label_top1_accuracy.py"
scp inference/visium_hd_exp1/run_local_vlm_reason_audit.py \
  "$REMOTE_HOST:$REMOTE_PROJECT/inference/visium_hd_exp1/run_local_vlm_reason_audit.py"
scp scripts/hpc/jun10_vlm_reason_audit_qwen3.sbatch \
  "$REMOTE_HOST:$REMOTE_PROJECT/scripts/hpc/jun10_vlm_reason_audit_qwen3.sbatch"
scp "$LOCAL_POOL/public_vlm_requests.csv" \
  "$REMOTE_HOST:$REMOTE_POOL/public_vlm_requests.csv"
scp -r "$LOCAL_POOL/candidate_pair_crops" \
  "$REMOTE_HOST:$REMOTE_POOL/"

echo "[Jun10 VLM reason audit] submitting Qwen3-VL-32B full 167 reason-audit run..."
ssh -o BatchMode=yes "$REMOTE_HOST" \
  "cd '$REMOTE_PROJECT' && LIMIT='$LIMIT' MODEL_NAME='$MODEL_NAME' RUN_NAME='$RUN_NAME' sbatch scripts/hpc/jun10_vlm_reason_audit_qwen3.sbatch"

echo "[Jun10 VLM reason audit] submitted."
