#!/bin/bash
set -euo pipefail

# Run this from the local Medical-SAM3 project root after Bouchet/Duo access works.
# It syncs the 17T runtime-policy second-SAM locator pack and submits a focused
# refinement run. By default it runs only bronchiola and vessels because those
# were the classes that passed the 17T refinement-ready gate.

REMOTE_HOST="${REMOTE_HOST:-bouchet}"
REMOTE_PROJECT="${REMOTE_PROJECT:-/nfs/roberts/project/pi_xy48/hw646/Medical-SAM3}"
BASE_REL="output/visium_hd_exp1/final_deliverables/Jun09_SkillRanker_ComponentAwareMaskSelection"
INPUT_REL="output/visium_hd_exp1/ficture_official_filtered_candidate_pool_input_roi"
LABELS="${LABELS:-bronchiola,vessels}"

echo "[Jun09 runtime 2SAM] checking Bouchet SSH..."
HPC_HOST="$REMOTE_HOST" bash scripts/hpc_ssh_check.sh

echo "[Jun09 runtime 2SAM] preparing remote directories..."
ssh -o BatchMode=yes "$REMOTE_HOST" \
  "mkdir -p '$REMOTE_PROJECT/inference/visium_hd_exp1' '$REMOTE_PROJECT/scripts/hpc' '$REMOTE_PROJECT/$BASE_REL/corrected_pool' '$REMOTE_PROJECT/$INPUT_REL'"

echo "[Jun09 runtime 2SAM] syncing runner and sbatch..."
scp inference/visium_hd_exp1/run_jun09_selected_piece_second_sam_refine.py \
  "$REMOTE_HOST:$REMOTE_PROJECT/inference/visium_hd_exp1/run_jun09_selected_piece_second_sam_refine.py"
scp scripts/hpc/jun09_runtime_policy_second_sam_refine.sbatch \
  "$REMOTE_HOST:$REMOTE_PROJECT/scripts/hpc/jun09_runtime_policy_second_sam_refine.sbatch"

echo "[Jun09 runtime 2SAM] syncing 17T locator pack and hidden truth..."
scp -r "$BASE_REL/runtime_policy_second_sam_locator_pack" \
  "$REMOTE_HOST:$REMOTE_PROJECT/$BASE_REL/runtime_policy_second_sam_locator_pack"
scp "$BASE_REL/corrected_pool/hidden_candidate_truth.csv" \
  "$REMOTE_HOST:$REMOTE_PROJECT/$BASE_REL/corrected_pool/hidden_candidate_truth.csv"

echo "[Jun09 runtime 2SAM] syncing official H&E/FICTURE ROI and annotations..."
scp "$INPUT_REL/he_roi_matching_official_ficture_coverage.png" \
  "$REMOTE_HOST:$REMOTE_PROJECT/$INPUT_REL/he_roi_matching_official_ficture_coverage.png"
scp "$INPUT_REL/ficture_official_filtered_roi_rgb.png" \
  "$REMOTE_HOST:$REMOTE_PROJECT/$INPUT_REL/ficture_official_filtered_roi_rgb.png"
scp -r "$INPUT_REL/cropped_annotation_masks" \
  "$REMOTE_HOST:$REMOTE_PROJECT/$INPUT_REL/cropped_annotation_masks"

echo "[Jun09 runtime 2SAM] submitting focused second-SAM refinement for LABELS=$LABELS..."
ssh -o BatchMode=yes "$REMOTE_HOST" \
  "cd '$REMOTE_PROJECT' && LABELS='$LABELS' sbatch scripts/hpc/jun09_runtime_policy_second_sam_refine.sbatch"

echo "[Jun09 runtime 2SAM] submitted. Check status with:"
echo "ssh -o BatchMode=yes $REMOTE_HOST 'squeue -u hw646 -o \"%.18i %.12P %.40j %.2t %.12M %.12l %.6C %.10m %.24R\"'"
