#!/bin/bash
set -euo pipefail

REMOTE_HOST="${REMOTE_HOST:-bouchet}"
REMOTE_PROJECT="${REMOTE_PROJECT:-/nfs/roberts/project/pi_xy48/hw646/Medical-SAM3}"
LOCAL_BASE="${LOCAL_BASE:-output/visium_hd_exp1/final_deliverables/Jun09_SkillRanker_ComponentAwareMaskSelection}"
LOCAL_JUN10="${LOCAL_JUN10:-output/visium_hd_exp1/final_deliverables/Jun10_CellTypeFirst_Qwen3_Skill}"

REQUESTS_CSV_LOCAL="${REQUESTS_CSV_LOCAL:-$LOCAL_JUN10/cell_type_first_requests.csv}"
STEP1_CSV_LOCAL="${STEP1_CSV_LOCAL:-$LOCAL_JUN10/step1_prompt_ensemble_baseline_plus_targeted_bronvessel/step1_hypotheses.csv}"
REQUESTS_CSV_REMOTE="$REMOTE_PROJECT/$REQUESTS_CSV_LOCAL"
STEP1_CSV_REMOTE="$REMOTE_PROJECT/$STEP1_CSV_LOCAL"

IMAGE_ROOT_REMOTE="${IMAGE_ROOT_REMOTE:-$REMOTE_PROJECT/$LOCAL_BASE/corrected_pool}"
OUT_BASE_REMOTE="${OUT_BASE_REMOTE:-/nfs/roberts/project/pi_xy48/hw646/Medical-SAM3_remote_runs/visium_hd_exp1/jun10_ficture_filter_ensemble_baseline_plus_targeted_bronvessel}"
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3-VL-32B-Instruct}"
LIMIT="${LIMIT:-0}"
PARTITION="${PARTITION:-gpu_b200,gpu_h200,gpu_rtx6000}"
PROMPT_STYLES="${PROMPT_STYLES:-strict_spatial soft_consistency tumor_stroma_narrow}"

echo "[Jun10 Step3 FICTURE ensemble] checking Bouchet SSH..."
HPC_HOST="$REMOTE_HOST" bash scripts/hpc_ssh_check.sh

echo "[Jun10 Step3 FICTURE ensemble] syncing scripts and CSV inputs..."
ssh -o BatchMode=yes "$REMOTE_HOST" "mkdir -p '$REMOTE_PROJECT/inference/visium_hd_exp1' '$REMOTE_PROJECT/scripts/hpc' '$REMOTE_PROJECT/$(dirname "$REQUESTS_CSV_LOCAL")' '$REMOTE_PROJECT/$(dirname "$STEP1_CSV_LOCAL")' '$REMOTE_PROJECT/$LOCAL_BASE/corrected_pool'"
scp inference/visium_hd_exp1/run_qwen3_ficture_image_hypothesis_verification.py \
  "$REMOTE_HOST:$REMOTE_PROJECT/inference/visium_hd_exp1/run_qwen3_ficture_image_hypothesis_verification.py"
scp inference/visium_hd_exp1/run_paired_vlm_hit_test.py \
  "$REMOTE_HOST:$REMOTE_PROJECT/inference/visium_hd_exp1/run_paired_vlm_hit_test.py"
scp scripts/hpc/jun10_ficture_image_filter_qwen3.sbatch \
  "$REMOTE_HOST:$REMOTE_PROJECT/scripts/hpc/jun10_ficture_image_filter_qwen3.sbatch"
scp "$REQUESTS_CSV_LOCAL" "$REMOTE_HOST:$REQUESTS_CSV_REMOTE"
scp "$STEP1_CSV_LOCAL" "$REMOTE_HOST:$STEP1_CSV_REMOTE"

echo "[Jun10 Step3 FICTURE ensemble] syncing FICTURE crop images..."
TMP_FILE_LIST="$(mktemp)"
python3 - "$REQUESTS_CSV_LOCAL" "$TMP_FILE_LIST" <<'PY'
import csv
import sys

requests_csv, file_list = sys.argv[1], sys.argv[2]
with open(requests_csv, newline="") as f, open(file_list, "w") as out:
    for row in csv.DictReader(f):
        out.write(f"corrected_pool/{row['ficture_crop_rel']}\n")
PY
tar -C "$LOCAL_BASE" -cf - -T "$TMP_FILE_LIST" | ssh -o BatchMode=yes "$REMOTE_HOST" \
  "mkdir -p '$REMOTE_PROJECT/$LOCAL_BASE' && tar -C '$REMOTE_PROJECT/$LOCAL_BASE' -xf -"
rm -f "$TMP_FILE_LIST"

echo "[Jun10 Step3 FICTURE ensemble] submitting prompt styles: $PROMPT_STYLES"
for PROMPT_STYLE in $PROMPT_STYLES; do
  ssh -o BatchMode=yes "$REMOTE_HOST" \
    "cd '$REMOTE_PROJECT' && LIMIT='$LIMIT' MODEL_NAME='$MODEL_NAME' PROMPT_STYLE='$PROMPT_STYLE' REQUESTS_CSV='$REQUESTS_CSV_REMOTE' STEP1_CSV='$STEP1_CSV_REMOTE' IMAGE_ROOT='$IMAGE_ROOT_REMOTE' OUT_BASE='$OUT_BASE_REMOTE' sbatch --partition='$PARTITION' scripts/hpc/jun10_ficture_image_filter_qwen3.sbatch"
done

echo "[Jun10 Step3 FICTURE ensemble] submitted."
