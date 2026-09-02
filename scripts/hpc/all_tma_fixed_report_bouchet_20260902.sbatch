#!/usr/bin/env bash
#SBATCH --job-name=all_tma_report
#SBATCH --partition=day,scavenge
#SBATCH --array=0-9%3
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/nfs/roberts/project/pi_xy48/hw646/Medical-SAM3_all_tma_pre_vlm_20260902/logs/report_%A_%a.log
#SBATCH --mail-type=NONE

set -euo pipefail

TMAS=(TMA07 TMA24 TMA29 TMA30 TMA31 TMA34 TMA36 TMA39 TMA41 TMA42)
TMA=${TMAS[$SLURM_ARRAY_TASK_ID]}

ROOT=/nfs/roberts/project/pi_xy48/hw646/Medical-SAM3_all_tma_pre_vlm_20260902
RUN=${ROOT}/${TMA}
SHARED=${ROOT}/shared
CODE=${SHARED}/code
PY=/nfs/roberts/project/pi_xy48/hw646/ycrc_conda/envs/medsam3/bin/python
REPORT_DIR=${ROOT}/reports/${TMA}
REPORT=${REPORT_DIR}/${TMA}_Pre_VLM.html

export MPLCONFIGDIR=${RUN}/mplconfig/report_${SLURM_JOB_ID}
export FONTCONFIG_PATH=/etc/fonts
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONUNBUFFERED=1
mkdir -p "${MPLCONFIGDIR}" "${REPORT_DIR}"

for required in "${PY}" "${RUN}/pre_vlm_validation.json" \
  "${RUN}/frozen_pool/run_manifest.json" "${RUN}/frozen_pool/final_pool.csv" \
  "${CODE}/build_all_tma_pre_vlm_report.py" "${SHARED}/input_audit.json"; do
  test -s "${required}" || { echo "Missing required file: ${required}" >&2; exit 3; }
done

"${PY}" - "${RUN}/pre_vlm_validation.json" "${TMA}" <<'PY'
import json
import sys
from pathlib import Path

validation = json.loads(Path(sys.argv[1]).read_text())
run = validation["run"]
assert run["status"] == "valid"
assert run["validated_stage"] == "pre-vlm"
assert run["tma"] == sys.argv[2]
print("VALIDATED", json.dumps(run, sort_keys=True))
PY

"${PY}" "${CODE}/build_all_tma_pre_vlm_report.py" \
  --root "${RUN}" \
  --input-audit "${SHARED}/input_audit.json" \
  --output "${REPORT}" \
  > "${REPORT_DIR}/report_manifest.json"

test -s "${REPORT}"
grep -q "Interactive inventory" "${REPORT}"
grep -q "Registered K=12 strict-official raw FICTURE" "${REPORT}"
grep -q "Not run in this report" "${REPORT}"
cat "${REPORT_DIR}/report_manifest.json"
date --iso-8601=seconds
echo DONE
