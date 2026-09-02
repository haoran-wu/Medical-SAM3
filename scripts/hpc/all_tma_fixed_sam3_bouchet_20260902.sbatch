#!/usr/bin/env bash
#SBATCH --job-name=all_tma_sam3
#SBATCH --partition=gpu_devel
#SBATCH --gres=gpu:b200:1
#SBATCH --array=0-9%2
#SBATCH --cpus-per-task=8
#SBATCH --mem=110G
#SBATCH --time=02:00:00
#SBATCH --output=/nfs/roberts/project/pi_xy48/hw646/Medical-SAM3_all_tma_pre_vlm_20260902/logs/sam3_%A_%a.log
#SBATCH --mail-type=NONE

set -euo pipefail

TMAS=(TMA07 TMA24 TMA29 TMA30 TMA31 TMA34 TMA36 TMA39 TMA41 TMA42)
TMA=${TMAS[$SLURM_ARRAY_TASK_ID]}

PROJECT=/nfs/roberts/project/pi_xy48/hw646/Medical-SAM3
SAM3_ROOT=/nfs/roberts/project/pi_xy48/hw646/sam3
PY=/nfs/roberts/project/pi_xy48/hw646/ycrc_conda/envs/medsam3/bin/python
CHECKPOINT=${PROJECT}/checkpoints/SAM3-base/sam3.pt
ROOT=/nfs/roberts/project/pi_xy48/hw646/Medical-SAM3_all_tma_pre_vlm_20260902
RUN=${ROOT}/${TMA}
SHARED=${ROOT}/shared
CODE=${SHARED}/code
STAGING=${RUN}/staging
PROMPT_ROOT=${RUN}/prompts/final_sustained_edge3
PROMPTS=${PROMPT_ROOT}/adaptive_box_peak_sustained_edge3_context0.csv
LABELS=${PROMPT_ROOT}/adaptive_region_labels.npz
HE_IMAGE=${STAGING}/${TMA}_he.png
FICTURE_IMAGE=${STAGING}/${TMA}_ficture.png
SUPPORT=${STAGING}/${TMA}_blind_he_tissue_support.png
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
  "${STAGING}/factor_info.csv" "${PROMPTS}" "${LABELS}" \
  "${CODE}/build_registered_he_tissue_support.py" \
  "${CODE}/prepare_tma39_raw_he_gap_prompts.py" \
  "${CODE}/run_tma39_auditable_internal_mask_selection_20260825.py" \
  "${CODE}/run_tma39_he_gap_uniform_box_point.py" \
  "${CODE}/run_fixed_he_gap_box_point_sam3.py" \
  "${CODE}/build_fixed_tma39_transfer_pool.py" \
  "${CODE}/validate_tma39_fixed_workflow.py"; do
  test -s "${required}" || { echo "Missing required file: ${required}" >&2; exit 3; }
done

PROMPT_COUNT=$("${PY}" -c 'import csv,sys; print(sum(1 for _ in csv.DictReader(open(sys.argv[1]))))' "${PROMPTS}")
test "${PROMPT_COUNT}" -gt 0
echo "${TMA}: ${PROMPT_COUNT} fixed GeneMap box-plus-point prompts"
echo "Main selection: GeneMap support F1; FICTURE primary; H&E novelty threshold 40%"
echo "Independent H&E: 64 px, 35% H&E, 8% FICTURE, 6 connected cells, 10% context box plus point"
date --iso-8601=seconds

if [[ ! -s "${SUPPORT}" ]]; then
  "${PY}" "${CODE}/build_registered_he_tissue_support.py" \
    --he-image "${HE_IMAGE}" \
    --output "${SUPPORT}"
fi

if [[ ! -s "${GAP_ROOT}/prompt_manifest.json" ]]; then
  "${PY}" "${CODE}/prepare_tma39_raw_he_gap_prompts.py" \
    --tma "${TMA}" \
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
    --tma "${TMA}" \
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
    --tma "${TMA}" \
    --sam-root "${SAM_ROOT}" \
    --gap-root "${GAP_SAM_ROOT}" \
    --method genemap_support_f1 \
    --output-dir "${POOL_ROOT}"
fi

"${PY}" "${CODE}/validate_tma39_fixed_workflow.py" \
  --contract "${SHARED}/tma39_fixed_workflow_v1.json" \
  --run-root "${RUN}" \
  --stage pre-vlm \
  --input-audit "${SHARED}/input_audit.json" \
  > "${RUN}/pre_vlm_validation.json"
cat "${RUN}/pre_vlm_validation.json"

date --iso-8601=seconds
echo DONE
