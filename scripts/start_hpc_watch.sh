#!/bin/bash
set -euo pipefail

ROOT="/Users/haoranwu/Desktop/Yan_Lab_Research/Medical-SAM3"
ENV_FILE="${HOME}/.config/hpc-notify.env"

if [ -f "$ENV_FILE" ]; then
  # shellcheck disable=SC1090
  source "$ENV_FILE"
fi

"$ROOT/scripts/hpc_ssh_check.sh"

PORT="${HPC_DASHBOARD_PORT:-8765}"
INTERVAL="${HPC_NOTIFY_INTERVAL:-120}"
HISTORY_HOURS="${HPC_DASHBOARD_HISTORY_HOURS:-24}"
DESKTOP_NOTIFY="${HPC_NOTIFY_LOCAL_ON:-change}"
EMAIL_TO="${HPC_NOTIFY_EMAIL:-}"
EMAIL_METHOD="${HPC_NOTIFY_EMAIL_METHOD:-smtp}"
EMAIL_ON="${HPC_NOTIFY_EMAIL_ON:-failure}"
PUSH_CHANNEL="${HPC_NOTIFY_PUSH_CHANNEL:-}"
PUSH_ON="${HPC_NOTIFY_PUSH_ON:-failure}"

echo "Starting dashboard on http://127.0.0.1:${PORT}"
echo "Reporter interval: ${INTERVAL}s"

/usr/bin/python3 -u "$ROOT/scripts/hpc_dashboard/hpc_dashboard.py" \
  --port "$PORT" \
  --interval 20 \
  --history-hours "$HISTORY_HOURS" \
  >/tmp/medsam3_hpc_dashboard.log 2>&1 &

DASHBOARD_PID=$!

cleanup() {
  if kill -0 "$DASHBOARD_PID" >/dev/null 2>&1; then
    kill "$DASHBOARD_PID" >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT

REPORTER_ARGS=(
  /usr/bin/python3 -u "$ROOT/scripts/hpc_dashboard/hpc_progress_reporter.py"
  --watch
  --interval "$INTERVAL"
  --state-file "${HOME}/.hpc_progress_reporter_state.json"
  --local-log "${HOME}/.hpc_progress_reporter.log"
  --notify-local-on "$DESKTOP_NOTIFY"
  --analyze-on-failure
  --analysis-dir "$ROOT/output/hpc_monitoring/analysis"
)

if [ -n "$EMAIL_TO" ]; then
  REPORTER_ARGS+=(--email-to "$EMAIL_TO" --email-method "$EMAIL_METHOD" --email-on "$EMAIL_ON")
fi

if [ -n "$PUSH_CHANNEL" ]; then
  REPORTER_ARGS+=(--push-channel "$PUSH_CHANNEL" --push-on "$PUSH_ON")
fi

echo "Dashboard log: /tmp/medsam3_hpc_dashboard.log"
echo "Reporter log: ${HOME}/.hpc_progress_reporter.log"
echo "Press Ctrl+C to stop both."

exec "${REPORTER_ARGS[@]}"
