#!/usr/bin/env bash
#SBATCH --job-name=t30_fixed_gemma
#SBATCH --partition=gpu_rtx6000
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=05:00:00
#SBATCH --output=/nfs/roberts/project/pi_xy48/hw646/Medical-SAM3_current_workflow_20260902/TMA30_fixed_tma39_pipeline/logs/gemma_%j.log
#SBATCH --mail-type=NONE

set -euo pipefail

RUN=/nfs/roberts/project/pi_xy48/hw646/Medical-SAM3_current_workflow_20260902/TMA30_fixed_tma39_pipeline
DEPLOYMENT=/nfs/roberts/project/pi_xy48/hw646/vlm_deployment
ENV_DIR=${DEPLOYMENT}/env
MODEL=/nfs/roberts/project/pi_xy48/hw646/vlm_models/Gemma-4-31B-Instruct
CODE=${RUN}/code
STAGING=${RUN}/staging
POOL=${RUN}/frozen_pool/final_pool.csv
BUNDLE=${RUN}/vlm_bundle
OUTPUT=${RUN}/vlm_gemma4

module purge
module load PyTorch/2.9.1-foss-2024a-CUDA-12.8.0
export PYTHONNOUSERSITE=1
export PYTHONPATH="${ENV_DIR}/lib/python3.12/site-packages${PYTHONPATH:+:${PYTHONPATH}}"
export HF_HOME=${DEPLOYMENT}/hf-home
export TRANSFORMERS_CACHE=${HF_HOME}/transformers
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

for required in "${ENV_DIR}/bin/python" "${MODEL}/config.json" "${POOL}" \
  "${STAGING}/TMA30_he.png" "${STAGING}/TMA30_ficture.png" "${STAGING}/factor_info.csv" \
  "${CODE}/prepare_tma30_fixed_pipeline_vlm_bundle.py" \
  "${CODE}/run_tma30_fixed_pipeline_gemma.py"; do
  test -s "${required}" || { echo "Missing required file: ${required}" >&2; exit 3; }
done
for label in airway arteriole venule alveoli; do
  test -s "${STAGING}/annotation/TMA30_authoritative_${label}_mask.png" || exit 3
done

if [[ ! -s "${BUNDLE}/run_manifest.json" ]]; then
  "${ENV_DIR}/bin/python" "${CODE}/prepare_tma30_fixed_pipeline_vlm_bundle.py" \
    --pool-csv "${POOL}" \
    --he "${STAGING}/TMA30_he.png" \
    --ficture "${STAGING}/TMA30_ficture.png" \
    --annotation-root "${STAGING}/annotation" \
    --factor-info "${STAGING}/factor_info.csv" \
    --output "${BUNDLE}"
fi

"${ENV_DIR}/bin/python" "${CODE}/run_tma30_fixed_pipeline_gemma.py" \
  --model-path "${MODEL}" \
  --bundle-root "${BUNDLE}" \
  --output-dir "${OUTPUT}" \
  --model-name google/gemma-4-31b-it \
  --max-attempts 3 \
  --max-new-tokens 512

"${ENV_DIR}/bin/python" - "${RUN}" <<'PY'
import json
import sys
from pathlib import Path

run = Path(sys.argv[1])
pool = json.loads((run / "frozen_pool/run_manifest.json").read_text())
bundle = json.loads((run / "vlm_bundle/run_manifest.json").read_text())
vlm = json.loads((run / "vlm_gemma4/run_manifest.json").read_text())
rows = json.loads((run / "vlm_gemma4/vlm_results.json").read_text())
expected = pool["final_candidate_count"]
assert bundle["candidate_count"] == expected
assert vlm["status"] == "complete"
assert vlm["candidate_count"] == expected
assert vlm["unique_candidate_count"] == expected
assert vlm["six_image_hashes_per_candidate"] is True
assert len(rows) == expected
assert len({row["candidate_uid"] for row in rows}) == expected
assert all(len(row["image_sha256"]) == 6 for row in rows)
print("VERIFIED", json.dumps({
    "candidate_count": expected,
    "evaluable": vlm["evaluable_candidate_count"],
    "correct": vlm["correct_evaluable_candidate_count"],
    "accuracy": vlm["accuracy_among_evaluable_candidates"],
}, sort_keys=True))
PY

date --iso-8601=seconds
echo DONE
