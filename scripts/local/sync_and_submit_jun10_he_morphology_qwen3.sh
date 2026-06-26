#!/bin/bash
set -euo pipefail

REMOTE_HOST="${REMOTE_HOST:-bouchet}"
REMOTE_PROJECT="${REMOTE_PROJECT:-/nfs/roberts/project/pi_xy48/hw646/Medical-SAM3}"
LOCAL_BASE="${LOCAL_BASE:-output/visium_hd_exp1/final_deliverables/Jun09_SkillRanker_ComponentAwareMaskSelection}"
LOCAL_JUN10="${LOCAL_JUN10:-output/visium_hd_exp1/final_deliverables/Jun10_CellTypeFirst_Qwen3_Skill}"
REQUESTS_CSV_LOCAL="${REQUESTS_CSV_LOCAL:-$LOCAL_JUN10/cell_type_first_requests.csv}"
REQUESTS_CSV_REMOTE="${REQUESTS_CSV_REMOTE:-$REMOTE_PROJECT/$LOCAL_JUN10/$(basename "$REQUESTS_CSV_LOCAL")}"
STEP1_CSV_LOCAL="${STEP1_CSV_LOCAL:-}"
STEP1_CSV_REMOTE="${STEP1_CSV_REMOTE:-}"
LIMIT="${LIMIT:-18}"
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3-VL-32B-Instruct}"
MODEL_SLUG="${MODEL_SLUG:-qwen3vl32b_he}"
IMAGE_MODE="${IMAGE_MODE:-he_crop_context}"
PROMPT_STYLE="${PROMPT_STYLE:-detailed}"
IMAGE_ROOT_REMOTE="${IMAGE_ROOT_REMOTE:-$REMOTE_PROJECT/$LOCAL_BASE/corrected_pool}"
CONTEXT_ROOT_REMOTE="${CONTEXT_ROOT_REMOTE:-$REMOTE_PROJECT/$LOCAL_BASE/region_aware_piece_context_locator_pack}"

echo "[Jun10 Step2] checking Bouchet SSH..."
HPC_HOST="$REMOTE_HOST" bash scripts/hpc_ssh_check.sh

echo "[Jun10 Step2] syncing scripts..."
ssh -o BatchMode=yes "$REMOTE_HOST" "mkdir -p '$REMOTE_PROJECT/inference/visium_hd_exp1' '$REMOTE_PROJECT/scripts/hpc' '$REMOTE_PROJECT/$LOCAL_JUN10' '$REMOTE_PROJECT/$LOCAL_BASE/corrected_pool' '$REMOTE_PROJECT/$LOCAL_BASE/region_aware_piece_context_locator_pack'"
scp inference/visium_hd_exp1/run_qwen3_he_morphology_verification.py \
  "$REMOTE_HOST:$REMOTE_PROJECT/inference/visium_hd_exp1/run_qwen3_he_morphology_verification.py"
scp inference/visium_hd_exp1/run_paired_vlm_hit_test.py \
  "$REMOTE_HOST:$REMOTE_PROJECT/inference/visium_hd_exp1/run_paired_vlm_hit_test.py"
scp scripts/hpc/jun10_he_morphology_qwen3_step2.sbatch \
  "$REMOTE_HOST:$REMOTE_PROJECT/scripts/hpc/jun10_he_morphology_qwen3_step2.sbatch"
scp "$REQUESTS_CSV_LOCAL" \
  "$REMOTE_HOST:$REQUESTS_CSV_REMOTE"
if [[ -n "$STEP1_CSV_LOCAL" ]]; then
  if [[ -z "$STEP1_CSV_REMOTE" ]]; then
    STEP1_CSV_REMOTE="$REMOTE_PROJECT/$LOCAL_JUN10/$(basename "$STEP1_CSV_LOCAL")"
  fi
  scp "$STEP1_CSV_LOCAL" \
    "$REMOTE_HOST:$STEP1_CSV_REMOTE"
fi

echo "[Jun10 Step2] syncing H&E crop/context images needed for 167-piece pool..."
TMP_FILE_LIST="$(mktemp)"
python3 - "$REQUESTS_CSV_LOCAL" "$TMP_FILE_LIST" "$IMAGE_MODE" <<'PY'
import csv
import sys

requests_csv, file_list, image_mode = sys.argv[1], sys.argv[2], sys.argv[3]
with open(requests_csv, newline="") as f, open(file_list, "w") as out:
    reader = csv.DictReader(f)
    for row in reader:
        uid = row["candidate_uid"]
        he_crop_rel = row["he_crop_rel"]
        out.write(f"corrected_pool/{he_crop_rel}\n")
        if image_mode != "he_crop_only":
            out.write(f"region_aware_piece_context_locator_pack/views/{uid}_he_local_context.jpg\n")
PY
tar -C "$LOCAL_BASE" -cf - -T "$TMP_FILE_LIST" | ssh -o BatchMode=yes "$REMOTE_HOST" \
  "mkdir -p '$REMOTE_PROJECT/$LOCAL_BASE' && tar -C '$REMOTE_PROJECT/$LOCAL_BASE' -xf -"
rm -f "$TMP_FILE_LIST"

echo "[Jun10 Step2] submitting Qwen3 H&E morphology verification..."
ssh -o BatchMode=yes "$REMOTE_HOST" \
  "cd '$REMOTE_PROJECT' && LIMIT='$LIMIT' MODEL_NAME='$MODEL_NAME' MODEL_SLUG='$MODEL_SLUG' IMAGE_MODE='$IMAGE_MODE' PROMPT_STYLE='$PROMPT_STYLE' REQUESTS_CSV='$REQUESTS_CSV_REMOTE' STEP1_CSV='${STEP1_CSV_REMOTE:-}' IMAGE_ROOT='$IMAGE_ROOT_REMOTE' CONTEXT_ROOT='$CONTEXT_ROOT_REMOTE' sbatch scripts/hpc/jun10_he_morphology_qwen3_step2.sbatch"

echo "[Jun10 Step2] submitted."
