#!/bin/bash
# Auto-submit the next SAM3/FICTURE experiment whenever Bouchet frees a job slot.

set -euo pipefail

PROJECT=/nfs/roberts/project/pi_xy48/hw646/Medical-SAM3
MAX_ACTIVE="${MAX_ACTIVE:-4}"
SLEEP_SEC="${SLEEP_SEC:-45}"
QUEUE_FILE="${QUEUE_FILE:-$PROJECT/results/visium_hd_exp1/logs/sam3_followup_queue.txt}"
STATE_FILE="${STATE_FILE:-$PROJECT/results/visium_hd_exp1/logs/sam3_followup_queue.state}"
LOG_FILE="${LOG_FILE:-$PROJECT/results/visium_hd_exp1/logs/sam3_followup_autosubmit.log}"

mkdir -p "$(dirname "$QUEUE_FILE")"
cd "$PROJECT"

if [[ ! -f "$QUEUE_FILE" ]]; then
  cat > "$QUEUE_FILE" <<'EOF'
sbatch scripts/hpc/visium_hd_exp1_sam3_extra_group.sbatch base_micro_text
sbatch scripts/hpc/visium_hd_exp1_sam3_extra_group.sbatch medical_micro_text
sbatch scripts/hpc/visium_hd_exp1_sam3_extra_group.sbatch base_micro_notext
sbatch scripts/hpc/visium_hd_exp1_sam3_extra_group.sbatch medical_micro_notext
sbatch scripts/hpc/visium_hd_exp1_sam3_extra_group.sbatch base_medium_multiscale
sbatch scripts/hpc/visium_hd_exp1_sam3_extra_group.sbatch medical_medium_multiscale
sbatch scripts/hpc/visium_hd_exp1_sam3_extra_group.sbatch base_dense_points
sbatch scripts/hpc/visium_hd_exp1_sam3_extra_group.sbatch medical_dense_points
sbatch scripts/hpc/visium_hd_exp1_sam3_extra_group.sbatch base_large_multiscale
sbatch scripts/hpc/visium_hd_exp1_sam3_extra_group.sbatch medical_large_multiscale
sbatch scripts/hpc/visium_hd_exp1_sam3_ficture_refine_group.sbatch base_top20_d48
sbatch scripts/hpc/visium_hd_exp1_sam3_ficture_refine_group.sbatch base_top40_d80
sbatch scripts/hpc/visium_hd_exp1_sam3_ficture_refine_group.sbatch base_top40_d160
sbatch scripts/hpc/visium_hd_exp1_sam3_ficture_refine_group.sbatch medical_top20_d80
sbatch scripts/hpc/visium_hd_exp1_sam3_ficture_refine_group.sbatch medical_top40_d120
sbatch scripts/hpc/visium_hd_exp1_sam3_ficture_refine_group.sbatch base_csv_top40_d80
sbatch scripts/hpc/visium_hd_exp1_sam3_ficture_refine_group.sbatch base_factor_focus
sbatch scripts/hpc/visium_hd_exp1_sam3_ficture_refine_group.sbatch medical_factor_focus
EOF
fi

next_index=1
if [[ -f "$STATE_FILE" ]]; then
  next_index="$(cat "$STATE_FILE")"
fi

total=$(grep -c '^sbatch ' "$QUEUE_FILE" || true)
echo "[$(date)] autosubmit starting at index $next_index / $total; max_active=$MAX_ACTIVE" >> "$LOG_FILE"

while [[ "$next_index" -le "$total" ]]; do
  # Count only GPU jobs for this GPU-followup queue. CPU/day summary jobs run
  # in parallel and should not prevent us from keeping GPU slots busy.
  active=$(squeue -u hw646 -h -o "%P" | awk '$1 ~ /^gpu/ {n++} END {print n+0}')
  if [[ "$active" -lt "$MAX_ACTIVE" ]]; then
    cmd=$(grep '^sbatch ' "$QUEUE_FILE" | sed -n "${next_index}p")
    echo "[$(date)] active=$active submitting #$next_index: $cmd" >> "$LOG_FILE"
    if output=$(eval "$cmd" 2>&1); then
      echo "[$(date)] submitted: $output" >> "$LOG_FILE"
      next_index=$((next_index + 1))
      echo "$next_index" > "$STATE_FILE"
      sleep 5
    else
      echo "[$(date)] submit failed: $output" >> "$LOG_FILE"
      sleep "$SLEEP_SEC"
    fi
  else
    echo "[$(date)] active=$active waiting" >> "$LOG_FILE"
    sleep "$SLEEP_SEC"
  fi
done

echo "[$(date)] autosubmit queue finished" >> "$LOG_FILE"
