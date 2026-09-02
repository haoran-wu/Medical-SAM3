#!/usr/bin/env bash
#SBATCH --job-name=t30_fixed_sam3
#SBATCH --partition=gpu_devel
#SBATCH --gres=gpu:b200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=110G
#SBATCH --time=02:00:00
#SBATCH --output=/nfs/roberts/project/pi_xy48/hw646/Medical-SAM3_current_workflow_20260902/TMA30_fixed_tma39_pipeline/logs/sam3_%j.log
#SBATCH --mail-type=NONE

set -euo pipefail

PROJECT=/nfs/roberts/project/pi_xy48/hw646/Medical-SAM3
SAM3_ROOT=/nfs/roberts/project/pi_xy48/hw646/sam3
PY=/nfs/roberts/project/pi_xy48/hw646/ycrc_conda/envs/medsam3/bin/python
CHECKPOINT=${PROJECT}/checkpoints/SAM3-base/sam3.pt
RUN=/nfs/roberts/project/pi_xy48/hw646/Medical-SAM3_current_workflow_20260902/TMA30_fixed_tma39_pipeline
CODE=${RUN}/code
STAGING=${RUN}/staging
PROMPT_ROOT=${RUN}/prompts/final_sustained_edge3
PROMPTS=${PROMPT_ROOT}/adaptive_box_peak_sustained_edge3_context0.csv
LABELS=${PROMPT_ROOT}/adaptive_region_labels.npz
HE_IMAGE=${STAGING}/TMA30_he.png
FICTURE_IMAGE=${STAGING}/TMA30_ficture.png
SUPPORT=${STAGING}/TMA30_blind_he_tissue_support.png
GAP_ROOT=${RUN}/independent_he_gap_detection
SAM_ROOT=${RUN}/sam3_main
GAP_SAM_ROOT=${RUN}/sam3_independent_he_gap
POOL_ROOT=${RUN}/frozen_pool

export PYTHONPATH=${PROJECT}/inference:${PROJECT}/scripts:${PROJECT}:${SAM3_ROOT}:${PYTHONPATH:-}
export PROJECT_ROOT_OVERRIDE=${PROJECT}
export SAM3_ROOT_OVERRIDE=${SAM3_ROOT}
export PYTHONUNBUFFERED=1
export MPLCONFIGDIR=${RUN}/mplconfig/${SLURM_JOB_ID}
export HF_HOME=/nfs/roberts/project/pi_xy48/hw646/ycrc_conda/hf-home
export TRANSFORMERS_CACHE=${HF_HOME}/transformers
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "${MPLCONFIGDIR}" "${RUN}/logs"

for required in "${PY}" "${CHECKPOINT}" "${HE_IMAGE}" "${FICTURE_IMAGE}" \
  "${PROMPTS}" "${LABELS}" \
  "${CODE}/build_registered_he_tissue_support.py" \
  "${CODE}/prepare_tma39_raw_he_gap_prompts.py" \
  "${CODE}/run_tma39_auditable_internal_mask_selection_20260825.py" \
  "${CODE}/run_tma39_he_gap_uniform_box_point.py" \
  "${CODE}/run_fixed_he_gap_box_point_sam3.py" \
  "${CODE}/build_fixed_tma39_transfer_pool.py"; do
  test -s "${required}" || { echo "Missing required file: ${required}" >&2; exit 3; }
done

PROMPT_COUNT=$("${PY}" -c 'import csv,sys; print(sum(1 for _ in csv.DictReader(open(sys.argv[1]))))' "${PROMPTS}")
test "${PROMPT_COUNT}" -gt 0
echo "TMA30 fixed TMA39 transfer: ${PROMPT_COUNT} GeneMap regions"
echo "Main prompt: tight box (0% expansion) plus highest module-score point"
echo "Main selection: GeneMap support F1; FICTURE primary; H&E novelty threshold 40%"
echo "Independent H&E rule: fixed 64/35%/8%/6-cell detector plus 10% context box+point"
echo "Selection used annotation: False"
date --iso-8601=seconds

if [[ ! -s "${SUPPORT}" ]]; then
  "${PY}" "${CODE}/build_registered_he_tissue_support.py" \
    --he-image "${HE_IMAGE}" \
    --output "${SUPPORT}"
fi

if [[ ! -s "${GAP_ROOT}/prompt_manifest.json" ]]; then
  "${PY}" "${CODE}/prepare_tma39_raw_he_gap_prompts.py" \
    --tma TMA30 \
    --he "${HE_IMAGE}" \
    --ficture "${FICTURE_IMAGE}" \
    --output-dir "${GAP_ROOT}" \
    --cell-size 64 \
    --min-he-fraction 0.35 \
    --max-ficture-fraction 0.08 \
    --min-region-cells 6 \
    --box-expansion 0.10
fi

if [[ ! -s "${SAM_ROOT}/ficture/run_manifest.json" ]]; then
  "${PY}" "${CODE}/run_tma39_auditable_internal_mask_selection_20260825.py" \
    --image-path "${FICTURE_IMAGE}" \
    --foreground-mask-path "${SUPPORT}" \
    --prompts-csv "${PROMPTS}" \
    --region-labels-npz "${LABELS}" \
    --output-dir "${SAM_ROOT}/ficture" \
    --checkpoint "${CHECKPOINT}" \
    --source ficture \
    --device cuda \
    --min-area-frac 0.00002 \
    --max-area-frac 0.20 \
    --expected-prompts "${PROMPT_COUNT}"
fi

if [[ ! -s "${SAM_ROOT}/he/run_manifest.json" ]]; then
  "${PY}" "${CODE}/run_tma39_auditable_internal_mask_selection_20260825.py" \
    --image-path "${HE_IMAGE}" \
    --foreground-mask-path "${SUPPORT}" \
    --prompts-csv "${PROMPTS}" \
    --region-labels-npz "${LABELS}" \
    --output-dir "${SAM_ROOT}/he" \
    --checkpoint "${CHECKPOINT}" \
    --source he \
    --device cuda \
    --min-area-frac 0.00002 \
    --max-area-frac 0.20 \
    --expected-prompts "${PROMPT_COUNT}"
fi

if [[ ! -s "${GAP_SAM_ROOT}/run_manifest.json" ]]; then
  "${PY}" "${CODE}/run_fixed_he_gap_box_point_sam3.py" \
    --tma TMA30 \
    --input-root "${GAP_ROOT}" \
    --prompt-csv "${GAP_ROOT}/prompts_expanded10_box_point.csv" \
    --output-dir "${GAP_SAM_ROOT}" \
    --checkpoint "${CHECKPOINT}" \
    --he-image "${HE_IMAGE}" \
    --ficture-image "${FICTURE_IMAGE}" \
    --device cuda \
    --expansion-percent 10
fi

if [[ ! -s "${POOL_ROOT}/run_manifest.json" ]]; then
  "${PY}" "${CODE}/build_fixed_tma39_transfer_pool.py" \
    --tma TMA30 \
    --sam-root "${SAM_ROOT}" \
    --gap-root "${GAP_SAM_ROOT}" \
    --method genemap_support_f1 \
    --output-dir "${POOL_ROOT}"
fi

"${PY}" - "${RUN}" "${PROMPT_COUNT}" <<'PY'
import csv
import json
import sys
from pathlib import Path

run = Path(sys.argv[1])
expected = int(sys.argv[2])
for source in ("ficture", "he"):
    manifest = json.loads((run / f"sam3_main/{source}/run_manifest.json").read_text())
    selected = list(csv.DictReader((run / f"sam3_main/{source}/selected_candidates.csv").open()))
    assert manifest["prompt_count"] == expected
    assert manifest["image_encoding_count"] == 1
    assert manifest["inference_calls_per_prompt"] == 1
    assert manifest["total_inference_calls"] == expected
    assert manifest["selection_used_annotation"] is False
    assert manifest["positive_point_required"] is True
    support = [row for row in selected if row["selection_method"] == "genemap_support_f1"]
    assert len(support) == expected

gap = json.loads((run / "sam3_independent_he_gap/run_manifest.json").read_text())
pool = json.loads((run / "frozen_pool/run_manifest.json").read_text())
rows = list(csv.DictReader((run / "frozen_pool/final_pool.csv").open()))
assert gap["inference_calls_per_prompt"] == 1
assert gap["selected_candidate_count"] == gap["prompt_count"]
assert pool["main_prompt_count"] == expected
assert pool["final_candidate_count"] == len(rows)
assert pool["selection_used_annotation"] is False
assert len({row["candidate_id"] for row in rows}) == len(rows)
print("VERIFIED", json.dumps({
    "main_prompts": expected,
    "independent_he_prompts": gap["prompt_count"],
    "final_candidates": len(rows),
    "source_counts": pool["final_source_counts"],
    "selection_used_annotation": False,
}, sort_keys=True))
PY

date --iso-8601=seconds
echo DONE
