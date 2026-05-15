#!/usr/bin/env bash
set -euo pipefail

host="${1:-bouchet}"

case "$host" in
  bouchet|grace|mccleary) ;;
  *)
    echo "Usage: $0 [bouchet|grace|mccleary]" >&2
    exit 2
    ;;
esac

if ssh -O check "$host" >/dev/null 2>&1; then
  echo "SSH master for '$host' is already running."
  exit 0
fi

echo "Starting persistent SSH master for '$host'."
echo "Complete Duo/MFA if prompted. After this succeeds, Codex can reuse it."

ssh -MNf "$host"

if ssh -O check "$host" >/dev/null 2>&1; then
  echo "SSH master for '$host' is ready."
else
  echo "SSH master for '$host' did not start cleanly." >&2
  exit 1
fi
