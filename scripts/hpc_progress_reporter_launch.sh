#!/bin/bash
set -euo pipefail

ROOT="/Users/haoranwu/Desktop/Yan_Lab_Research/Medical-SAM3"
ENV_FILE="${HOME}/.config/hpc-notify.env"

if [ -f "$ENV_FILE" ]; then
  # shellcheck disable=SC1090
  source "$ENV_FILE"
fi

"$ROOT/scripts/hpc_ssh_check.sh"

exec /usr/bin/python3 -u "$ROOT/hpc_progress_reporter.py" \
  --watch \
  --interval "${HPC_NOTIFY_INTERVAL:-120}" \
  --email-to "${HPC_NOTIFY_EMAIL:-vettel.hwu@gmail.com}" \
  --email-method smtp \
  --email-on "${HPC_NOTIFY_EMAIL_ON:-change}" \
  --state-file "${HOME}/.hpc_progress_reporter_state.json" \
  --local-log "${HOME}/.hpc_progress_reporter.log"
