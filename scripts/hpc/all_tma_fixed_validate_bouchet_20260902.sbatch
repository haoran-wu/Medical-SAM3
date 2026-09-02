#!/usr/bin/env bash
#SBATCH --job-name=all_tma_validate
#SBATCH --partition=day,scavenge
#SBATCH --array=0-9%4
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --time=00:45:00
#SBATCH --output=/nfs/roberts/project/pi_xy48/hw646/Medical-SAM3_all_tma_pre_vlm_tma39_exact_20260902/logs/validate_%A_%a.log
#SBATCH --mail-type=NONE

set -euo pipefail

TMAS=(TMA07 TMA24 TMA29 TMA30 TMA31 TMA34 TMA36 TMA39 TMA41 TMA42)
TMA=${TMAS[$SLURM_ARRAY_TASK_ID]}

ROOT=/nfs/roberts/project/pi_xy48/hw646/Medical-SAM3_all_tma_pre_vlm_tma39_exact_20260902
RUN=${ROOT}/${TMA}
SHARED=${ROOT}/shared
CODE=${SHARED}/code
PY=/nfs/roberts/project/pi_xy48/hw646/ycrc_conda/envs/medsam3/bin/python
VALIDATION=${RUN}/pre_vlm_validation.json
VALIDATION_TMP=${VALIDATION}.tmp.${SLURM_JOB_ID}
trap 'rm -f "${VALIDATION_TMP}"' EXIT

"${PY}" "${CODE}/validate_tma39_fixed_workflow.py" \
  --contract "${SHARED}/tma39_fixed_workflow_v1.json" \
  --run-root "${RUN}" \
  --stage pre-vlm \
  --input-audit "${SHARED}/input_audit.json" \
  > "${VALIDATION_TMP}"

mv "${VALIDATION_TMP}" "${VALIDATION}"

"${PY}" - "${VALIDATION}" "${TMA}" <<'PY'
import json
import sys
from pathlib import Path

run = json.loads(Path(sys.argv[1]).read_text())["run"]
assert run["status"] == "valid"
assert run["validated_stage"] == "pre-vlm"
assert run["tma"] == sys.argv[2]
if sys.argv[2] == "TMA39":
    assert run["tma39_reference_match"] is True
print(json.dumps(run, sort_keys=True))
PY

date --iso-8601=seconds
echo DONE
