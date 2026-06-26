#!/bin/bash
set -euo pipefail

REMOTE_HOST="${REMOTE_HOST:-bouchet}"
REMOTE_PROJECT="${REMOTE_PROJECT:-/nfs/roberts/project/pi_xy48/hw646/Medical-SAM3}"
LOCAL_BASE="${LOCAL_BASE:-output/visium_hd_exp1/final_deliverables/Jun09_SkillRanker_ComponentAwareMaskSelection}"
LOCAL_JUN10="${LOCAL_JUN10:-output/visium_hd_exp1/final_deliverables/Jun10_CellTypeFirst_Qwen3_Skill}"

MODE="${MODE:-weak58}"
if [[ "$MODE" == "weak58" ]]; then
  REQUESTS_CSV_LOCAL="${REQUESTS_CSV_LOCAL:-$LOCAL_JUN10/step2_next_prompt_ablation_weak58_input/weak58_requests.csv}"
  STEP1_CSV_LOCAL="${STEP1_CSV_LOCAL:-$LOCAL_JUN10/step2_next_prompt_ablation_weak58_input/step1_ensemble_weak58_hypotheses.csv}"
  OUT_BASE_REMOTE="${OUT_BASE_REMOTE:-/nfs/roberts/project/pi_xy48/hw646/Medical-SAM3_remote_runs/visium_hd_exp1/jun10_step2_next_prompt_ablation_weak58}"
else
  REQUESTS_CSV_LOCAL="${REQUESTS_CSV_LOCAL:-$LOCAL_JUN10/cell_type_first_requests.csv}"
  STEP1_CSV_LOCAL="${STEP1_CSV_LOCAL:-$LOCAL_JUN10/step1_prompt_ensemble_baseline_plus_targeted_bronvessel/step1_hypotheses.csv}"
  OUT_BASE_REMOTE="${OUT_BASE_REMOTE:-/nfs/roberts/project/pi_xy48/hw646/Medical-SAM3_remote_runs/visium_hd_exp1/jun10_step2_next_prompt_ablation_full167}"
fi

PROMPT_STYLES="${PROMPT_STYLES:-all_hypothesis_broad_class_veto all_hypothesis_direct_evidence_minrules comparative_hypothesis_veto comparative_soft_keep comparative_ranked_retention comparative_pattern_first comparative_primary_pattern_budget comparative_structural_priority comparative_false_positive_guard comparative_soft_broad_veto_retention comparative_fewshot_broad_veto comparative_structured_checklist comparative_minimal_retention comparative_structural_keep_broad_prune comparative_brief_class_definitions comparative_alveoli_septa_rescue comparative_stroma_immune_strict_veto}"
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3-VL-32B-Instruct}"
MODEL_SLUG_PREFIX="${MODEL_SLUG_PREFIX:-qwen3vl32b}"
IMAGE_MODE="${IMAGE_MODE:-he_crop_only}"
LIMIT="${LIMIT:-0}"
IMAGE_ROOT_REMOTE="${IMAGE_ROOT_REMOTE:-$REMOTE_PROJECT/$LOCAL_BASE/corrected_pool}"
CONTEXT_ROOT_REMOTE="${CONTEXT_ROOT_REMOTE:-$REMOTE_PROJECT/$LOCAL_BASE/region_aware_piece_context_locator_pack}"

REQUESTS_CSV_REMOTE="$REMOTE_PROJECT/$REQUESTS_CSV_LOCAL"
STEP1_CSV_REMOTE="$REMOTE_PROJECT/$STEP1_CSV_LOCAL"

echo "[Jun10 Step2 ablation] checking Bouchet SSH..."
HPC_HOST="$REMOTE_HOST" bash scripts/hpc_ssh_check.sh

echo "[Jun10 Step2 ablation] syncing scripts and CSV inputs..."
ssh -o BatchMode=yes "$REMOTE_HOST" "mkdir -p '$REMOTE_PROJECT/inference/visium_hd_exp1' '$REMOTE_PROJECT/scripts/hpc' '$REMOTE_PROJECT/$LOCAL_JUN10' '$REMOTE_PROJECT/$(dirname "$REQUESTS_CSV_LOCAL")' '$REMOTE_PROJECT/$(dirname "$STEP1_CSV_LOCAL")' '$REMOTE_PROJECT/$LOCAL_BASE/corrected_pool' '$REMOTE_PROJECT/$LOCAL_BASE/region_aware_piece_context_locator_pack'"
scp inference/visium_hd_exp1/run_qwen3_he_morphology_verification.py \
  "$REMOTE_HOST:$REMOTE_PROJECT/inference/visium_hd_exp1/run_qwen3_he_morphology_verification.py"
scp inference/visium_hd_exp1/run_paired_vlm_hit_test.py \
  "$REMOTE_HOST:$REMOTE_PROJECT/inference/visium_hd_exp1/run_paired_vlm_hit_test.py"
scp scripts/hpc/jun10_he_morphology_qwen3_step2.sbatch \
  "$REMOTE_HOST:$REMOTE_PROJECT/scripts/hpc/jun10_he_morphology_qwen3_step2.sbatch"
scp "$REQUESTS_CSV_LOCAL" "$REMOTE_HOST:$REQUESTS_CSV_REMOTE"
scp "$STEP1_CSV_LOCAL" "$REMOTE_HOST:$STEP1_CSV_REMOTE"

echo "[Jun10 Step2 ablation] syncing H&E crop images for $MODE..."
TMP_FILE_LIST="$(mktemp)"
python3 - "$REQUESTS_CSV_LOCAL" "$TMP_FILE_LIST" "$IMAGE_MODE" <<'PY'
import csv
import sys

requests_csv, file_list, image_mode = sys.argv[1], sys.argv[2], sys.argv[3]
with open(requests_csv, newline="") as f, open(file_list, "w") as out:
    for row in csv.DictReader(f):
        uid = row["candidate_uid"]
        out.write(f"corrected_pool/{row['he_crop_rel']}\n")
        if image_mode != "he_crop_only":
            out.write(f"region_aware_piece_context_locator_pack/views/{uid}_he_local_context.jpg\n")
PY
tar -C "$LOCAL_BASE" -cf - -T "$TMP_FILE_LIST" | ssh -o BatchMode=yes "$REMOTE_HOST" \
  "mkdir -p '$REMOTE_PROJECT/$LOCAL_BASE' && tar -C '$REMOTE_PROJECT/$LOCAL_BASE' -xf -"
rm -f "$TMP_FILE_LIST"

echo "[Jun10 Step2 ablation] submitting prompt styles: $PROMPT_STYLES"
for PROMPT_STYLE in $PROMPT_STYLES; do
  MODEL_SLUG="${MODEL_SLUG_PREFIX}_${MODE}_${PROMPT_STYLE}"
  ssh -o BatchMode=yes "$REMOTE_HOST" \
    "cd '$REMOTE_PROJECT' && LIMIT='$LIMIT' MODEL_NAME='$MODEL_NAME' MODEL_SLUG='$MODEL_SLUG' IMAGE_MODE='$IMAGE_MODE' PROMPT_STYLE='$PROMPT_STYLE' REQUESTS_CSV='$REQUESTS_CSV_REMOTE' STEP1_CSV='$STEP1_CSV_REMOTE' IMAGE_ROOT='$IMAGE_ROOT_REMOTE' CONTEXT_ROOT='$CONTEXT_ROOT_REMOTE' OUT_BASE='$OUT_BASE_REMOTE' sbatch scripts/hpc/jun10_he_morphology_qwen3_step2.sbatch"
done

echo "[Jun10 Step2 ablation] submitted."
