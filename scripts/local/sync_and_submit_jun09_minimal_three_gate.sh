#!/bin/bash
set -euo pipefail

# Run this from the local Medical-SAM3 project root after Bouchet/Duo access works.
# It syncs the strict-preflight oracle branch and submits the Jun09 minimal-three
# component oracle gate on Bouchet.

REMOTE_HOST="${REMOTE_HOST:-bouchet}"
REMOTE_PROJECT="${REMOTE_PROJECT:-/nfs/roberts/project/pi_xy48/hw646/Medical-SAM3}"

echo "[Jun09] checking Bouchet SSH..."
HPC_HOST="$REMOTE_HOST" bash scripts/hpc_ssh_check.sh

echo "[Jun09] syncing strict oracle and sbatch scripts..."
scp inference/visium_hd_exp1/componentwise_candidate_oracle.py \
  "$REMOTE_HOST:$REMOTE_PROJECT/inference/visium_hd_exp1/componentwise_candidate_oracle.py"
scp scripts/hpc/jun09_minimal_three_setting_component_oracle.sbatch \
  "$REMOTE_HOST:$REMOTE_PROJECT/scripts/hpc/jun09_minimal_three_setting_component_oracle.sbatch"
scp scripts/collect_jun09_minimal_three_setting_component_oracle.py \
  "$REMOTE_HOST:$REMOTE_PROJECT/scripts/collect_jun09_minimal_three_setting_component_oracle.py"

echo "[Jun09] submitting minimal-three component oracle..."
ssh -o BatchMode=yes "$REMOTE_HOST" \
  "cd '$REMOTE_PROJECT' && sbatch scripts/hpc/jun09_minimal_three_setting_component_oracle.sbatch"

echo "[Jun09] submitted. Check status with:"
echo "ssh -o BatchMode=yes $REMOTE_HOST 'squeue -u hw646 -o \"%.18i %.12P %.40j %.2t %.12M %.12l %.6C %.10m %.24R\"'"
