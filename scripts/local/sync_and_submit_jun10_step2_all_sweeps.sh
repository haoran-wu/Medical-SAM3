#!/bin/bash
set -euo pipefail

# Submit all Jun10 Step2 weak-class sweeps that are ready to run:
# 1) H&E-only prompt variants
# 2) cross-model H&E-only variants
# 3) H&E candidate crop + local-context variants
#
# This script intentionally delegates to the narrower submit wrappers so path
# syncing, official input roots, and sbatch behavior stay in one place.

REMOTE_HOST="${REMOTE_HOST:-bouchet}"
MODE="${MODE:-weak58}"

echo "[Jun10 Step2 all sweeps] checking Bouchet SSH..."
HPC_HOST="$REMOTE_HOST" bash scripts/hpc_ssh_check.sh

echo "[Jun10 Step2 all sweeps] submitting H&E-only prompt sweep..."
MODE="$MODE" scripts/local/sync_and_submit_jun10_step2_prompt_ablation.sh

echo "[Jun10 Step2 all sweeps] submitting model sweep..."
MODE="$MODE" scripts/local/sync_and_submit_jun10_step2_model_sweep.sh

echo "[Jun10 Step2 all sweeps] submitting context-aware sweep..."
MODE="$MODE" scripts/local/sync_and_submit_jun10_step2_context_sweep.sh

echo "[Jun10 Step2 all sweeps] submitted all requested sweeps."
