#!/bin/bash
set -euo pipefail

URL="http://127.0.0.1:8765"
LOG_FILE="/tmp/medsam3_hpc_app.log"
PORT="${HPC_DASHBOARD_PORT:-8765}"
HISTORY_HOURS="${HPC_DASHBOARD_HISTORY_HOURS:-24}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DASHBOARD_SCRIPT="$SCRIPT_DIR/../Resources/hpc_dashboard.py"

is_up() {
  curl -fsS "$URL" >/dev/null 2>&1
}

if ! is_up; then
  nohup /usr/bin/python3 -u "$DASHBOARD_SCRIPT" \
    --port "$PORT" \
    --interval 20 \
    --history-hours "$HISTORY_HOURS" \
    >>"$LOG_FILE" 2>&1 &

  for _ in {1..20}; do
    if is_up; then
      break
    fi
    sleep 1
  done
fi

open "$URL"
