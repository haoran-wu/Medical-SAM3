#!/usr/bin/env bash
set -euo pipefail

HOST=${2:-${BOUCHET_SSH_HOST:-bouchet}}
INTERVAL=${BOUCHET_KEEPALIVE_INTERVAL:-3600}
LOG=${BOUCHET_KEEPALIVE_LOG:-"$HOME/.ssh/bouchet_keepalive.log"}
PIDFILE=${BOUCHET_KEEPALIVE_PIDFILE:-"$HOME/.ssh/bouchet_keepalive.pid"}

usage() {
  cat <<EOF
Usage: $0 {start|stop|status|loop} [host]

Keeps an already Duo-approved SSH ControlMaster alive. This script will not
answer Duo itself. First run an interactive SSH once:

  ssh -tt ${HOST} 'hostname'

Then start the keepalive:

  $0 start ${HOST}

Environment:
  BOUCHET_KEEPALIVE_INTERVAL  seconds between pings, default ${INTERVAL}
  BOUCHET_KEEPALIVE_LOG       log path, default ${LOG}
  BOUCHET_KEEPALIVE_PIDFILE   pid path, default ${PIDFILE}
EOF
}

master_status() {
  ssh -o BatchMode=yes -O check "$HOST"
}

require_master() {
  if ! master_status >/dev/null 2>&1; then
    echo "No active SSH ControlMaster for ${HOST}." >&2
    echo "Run: ssh -tt ${HOST} 'hostname'" >&2
    echo "Approve Duo, then rerun: $0 start ${HOST}" >&2
    exit 2
  fi
}

start_keepalive() {
  mkdir -p "$HOME/.ssh/controlmasters"
  chmod 700 "$HOME/.ssh" "$HOME/.ssh/controlmasters"
  require_master
  if [[ -f "$PIDFILE" ]]; then
    old_pid=$(cat "$PIDFILE" 2>/dev/null || true)
    if [[ -n "${old_pid:-}" ]] && kill -0 "$old_pid" 2>/dev/null; then
      echo "Keepalive already running for ${HOST}: pid ${old_pid}"
      exit 0
    fi
  fi
  nohup "$0" loop "$HOST" >>"$LOG" 2>&1 &
  pid=$!
  echo "$pid" >"$PIDFILE"
  echo "Started Bouchet SSH keepalive for ${HOST}: pid ${pid}, interval ${INTERVAL}s"
  echo "Log: ${LOG}"
}

stop_keepalive() {
  if [[ ! -f "$PIDFILE" ]]; then
    echo "No keepalive pidfile: ${PIDFILE}"
    exit 0
  fi
  pid=$(cat "$PIDFILE" 2>/dev/null || true)
  if [[ -n "${pid:-}" ]] && kill -0 "$pid" 2>/dev/null; then
    kill "$pid"
    echo "Stopped keepalive pid ${pid}"
  else
    echo "Keepalive process already stopped"
  fi
  rm -f "$PIDFILE"
}

status_keepalive() {
  if master_status; then
    true
  else
    echo "Master check failed"
  fi
  if [[ -f "$PIDFILE" ]]; then
    pid=$(cat "$PIDFILE" 2>/dev/null || true)
    if [[ -n "${pid:-}" ]] && kill -0 "$pid" 2>/dev/null; then
      echo "Keepalive running: pid ${pid}"
    else
      echo "Keepalive pidfile exists but process is not running: ${PIDFILE}"
    fi
  else
    echo "Keepalive not running"
  fi
}

loop_keepalive() {
  echo "[$(date)] keepalive loop started for ${HOST}, interval ${INTERVAL}s"
  while true; do
    if ! ssh -o BatchMode=yes -O check "$HOST"; then
      echo "[$(date)] ControlMaster check failed; exiting."
      exit 1
    fi
    if ! ssh -o BatchMode=yes "$HOST" 'true'; then
      echo "[$(date)] SSH ping failed; exiting."
      exit 1
    fi
    echo "[$(date)] SSH master alive for ${HOST}"
    sleep "$INTERVAL"
  done
}

cmd=${1:-}
case "$cmd" in
  start) start_keepalive ;;
  stop) stop_keepalive ;;
  status) status_keepalive ;;
  loop) loop_keepalive ;;
  *) usage; exit 2 ;;
esac
