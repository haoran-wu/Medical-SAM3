#!/bin/bash
set -euo pipefail

# Fetch completed Jun10 Step2 weak-class sweeps from Bouchet and collect them
# into comparable CSV/Markdown summaries.

REMOTE_HOST="${REMOTE_HOST:-bouchet}"
REMOTE_RUN_BASE="${REMOTE_RUN_BASE:-/nfs/roberts/project/pi_xy48/hw646/Medical-SAM3_remote_runs/visium_hd_exp1}"
LOCAL_JUN10="${LOCAL_JUN10:-output/visium_hd_exp1/final_deliverables/Jun10_CellTypeFirst_Qwen3_Skill}"
LOCAL_FETCH_ROOT="${LOCAL_FETCH_ROOT:-$LOCAL_JUN10/remote_step2_sweep_outputs}"
LOCAL_COLLECT_ROOT="${LOCAL_COLLECT_ROOT:-$LOCAL_JUN10/step2_sweep_collected}"

declare -a SWEEPS=(
  "jun10_step2_next_prompt_ablation_weak58"
  "jun10_step2_model_sweep_weak58"
  "jun10_step2_context_sweep_weak58"
)

echo "[Jun10 Step2 fetch] checking Bouchet SSH..."
HPC_HOST="$REMOTE_HOST" bash scripts/hpc_ssh_check.sh

mkdir -p "$LOCAL_FETCH_ROOT" "$LOCAL_COLLECT_ROOT"

for SWEEP in "${SWEEPS[@]}"; do
  REMOTE_ROOT="$REMOTE_RUN_BASE/$SWEEP"
  LOCAL_SWEEP_ROOT="$LOCAL_FETCH_ROOT/$SWEEP"
  COLLECT_DIR="$LOCAL_COLLECT_ROOT/$SWEEP"

  echo "[Jun10 Step2 fetch] checking remote root: $REMOTE_ROOT"
  if ! ssh -o BatchMode=yes "$REMOTE_HOST" "test -d '$REMOTE_ROOT'"; then
    echo "[Jun10 Step2 fetch] remote root not found, skipping: $REMOTE_ROOT"
    continue
  fi

  mkdir -p "$LOCAL_SWEEP_ROOT"
  echo "[Jun10 Step2 fetch] copying completed outputs for $SWEEP"
  scp -r "$REMOTE_HOST:$REMOTE_ROOT/" "$LOCAL_SWEEP_ROOT/"

  echo "[Jun10 Step2 fetch] collecting $SWEEP"
  python3 scripts/collect_jun10_step2_prompt_ablation.py \
    --input-root "$LOCAL_SWEEP_ROOT" \
    --output-dir "$COLLECT_DIR"
done

echo "[Jun10 Step2 fetch] deciding next adaptive iteration"
python3 scripts/decide_jun10_step2_next_iteration.py \
  --collect-root "$LOCAL_COLLECT_ROOT" \
  --output-dir "$LOCAL_JUN10/step2_adaptive_iteration_decision"

echo "[Jun10 Step2 fetch] done. Local fetch root: $LOCAL_FETCH_ROOT"
echo "[Jun10 Step2 fetch] local collect root: $LOCAL_COLLECT_ROOT"
echo "[Jun10 Step2 fetch] adaptive decision: $LOCAL_JUN10/step2_adaptive_iteration_decision/adaptive_iteration_decision_report.md"
