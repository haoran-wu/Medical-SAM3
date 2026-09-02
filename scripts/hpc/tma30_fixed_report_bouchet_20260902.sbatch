#!/usr/bin/env bash
#SBATCH --job-name=t30_fixed_report
#SBATCH --partition=day
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=02:00:00
#SBATCH --output=/nfs/roberts/project/pi_xy48/hw646/Medical-SAM3_current_workflow_20260902/TMA30_fixed_tma39_pipeline/logs/report_%j.log
#SBATCH --mail-type=NONE

set -euo pipefail

RUN=/nfs/roberts/project/pi_xy48/hw646/Medical-SAM3_current_workflow_20260902/TMA30_fixed_tma39_pipeline
PYTHON=python
SCRIPT=${RUN}/code/build_tma30_fixed_tma39_complete_report.py
OUTPUT=${RUN}/TMA30_SAM3_Fixed_Workflow.html

module purge
module load SciPy-bundle/2024.05-gfbf-2024a
module load matplotlib/3.9.2-gfbf-2024a
module load Pillow/10.4.0-GCCcore-13.3.0
export PYTHONNOUSERSITE=1
export MPLCONFIGDIR=${RUN}/mplconfig
mkdir -p "${MPLCONFIGDIR}"

for required in "${SCRIPT}" \
  "${RUN}/frozen_pool/run_manifest.json" \
  "${RUN}/vlm_bundle/run_manifest.json" \
  "${RUN}/vlm_gemma4/run_manifest.json"; do
  test -s "${required}" || { echo "Missing required file: ${required}" >&2; exit 3; }
done

"${PYTHON}" "${SCRIPT}" \
  --root "${RUN}" \
  --bundle "${RUN}/vlm_bundle" \
  --results "${RUN}/vlm_gemma4" \
  --output "${OUTPUT}"

"${PYTHON}" - "${OUTPUT}" <<'PY'
import re
import sys
from pathlib import Path

path = Path(sys.argv[1])
content = path.read_text(encoding="utf-8")
assert path.stat().st_size > 10_000_000
assert content.count("data:image/") >= 350
assert "All 52 final candidates" in content
assert "All nine TMA30 GeneMap modules" in content
assert "G figure: all 45 prompts on H&amp;E" in content
assert "Fixed 12-factor RGB legend" in content
assert "Candidate-level visualization groups" in content
assert "https://" not in " ".join(re.findall(r'<img[^>]+src=[\"\']([^\"\']+)', content))
print("HTML_STATIC_QA_OK", path.stat().st_size, content.count("data:image/"))
PY

sha256sum "${OUTPUT}"
date --iso-8601=seconds
echo DONE
