#!/bin/bash
# hpc_ssh_check.sh
# Ensure the Bouchet SSH ControlMaster is alive.
# If it is dead, open Terminal for one interactive Duo login, then wait until
# the persistent master socket is ready for non-interactive Codex commands.

set -euo pipefail

HOST="${HPC_HOST:-bouchet}"
MAX_WAIT="${HPC_SSH_MAX_WAIT:-180}"
POLL="${HPC_SSH_POLL:-2}"

CONTROL_DIR="$HOME/.ssh/controlmasters"
mkdir -p "$CONTROL_DIR"
chmod 700 "$CONTROL_DIR"

control_path="$(ssh -G "$HOST" 2>/dev/null | awk '/^controlpath / {print $2; exit}')"
if [[ -z "${control_path:-}" ]]; then
    echo "ERROR: cannot resolve ssh ControlPath for $HOST" >&2
    exit 1
fi

check_alive() {
    ssh -O check "$HOST" >/dev/null 2>&1
}

if check_alive; then
    echo "ControlMaster already alive: $HOST"
    exit 0
fi

echo "ControlMaster is down for $HOST."
echo "Opening Terminal for Duo login. Leave it open until it prints CONNECTED."

osascript -e 'display notification "请在弹出的 Terminal 里完成 Duo 认证" with title "HPC SSH 需要重新连接" subtitle "ssh bouchet" sound name "Ping"' 2>/dev/null || true

terminal_cmd=$(
    cat <<EOF
echo 'HPC SSH reconnect for $HOST'
echo 'Please complete Duo/password if prompted.'
echo 'This window can be closed after it prints CONNECTED.'
ssh -M -S '$control_path' -o ControlMaster=yes -o ControlPersist=7d -o ServerAliveInterval=60 -o ServerAliveCountMax=10 '$HOST' 'echo CONNECTED: \$(hostname); sleep 5'
echo 'CONNECTED. ControlMaster should persist for 7 days.'
EOF
)

osascript - "$terminal_cmd" <<'EOF'
on run argv
    tell application "Terminal"
        activate
        do script (item 1 of argv)
    end tell
end run
EOF

elapsed=0
while ! check_alive; do
    sleep "$POLL"
    elapsed=$((elapsed + POLL))
    if [[ "$elapsed" -ge "$MAX_WAIT" ]]; then
        echo "ERROR: timed out waiting for Duo/ControlMaster (${MAX_WAIT}s)" >&2
        echo "Expected socket: $control_path" >&2
        exit 1
    fi
done

echo "ControlMaster established: $HOST"
exit 0
