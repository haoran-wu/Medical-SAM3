#!/usr/bin/env bash
#SBATCH --job-name=all_tma_index
#SBATCH --partition=day,scavenge
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:20:00
#SBATCH --output=/nfs/roberts/project/pi_xy48/hw646/Medical-SAM3_all_tma_pre_vlm_tma39_exact_20260902/logs/index_%j.log
#SBATCH --mail-type=NONE

set -euo pipefail

ROOT=/nfs/roberts/project/pi_xy48/hw646/Medical-SAM3_all_tma_pre_vlm_tma39_exact_20260902
PY=/nfs/roberts/project/pi_xy48/hw646/ycrc_conda/envs/medsam3/bin/python
CODE=${ROOT}/shared/code

"${PY}" "${CODE}/build_all_tma_pre_vlm_index.py" \
  --root "${ROOT}" \
  --reports "${ROOT}/reports" \
  --output "${ROOT}/reports/index.html" \
  --summary-json "${ROOT}/reports/all_tma_summary.json"

test -s "${ROOT}/reports/index.html"
test -s "${ROOT}/reports/all_tma_summary.json"

"${PY}" "${CODE}/audit_all_tma_pre_vlm_reports.py" \
  --root "${ROOT}" \
  --reports "${ROOT}/reports" \
  --output "${ROOT}/reports/report_schema_audit.json"

test -s "${ROOT}/reports/report_schema_audit.json"
date --iso-8601=seconds
echo DONE
