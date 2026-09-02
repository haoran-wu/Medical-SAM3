#!/usr/bin/env bash
#SBATCH --job-name=all_tma_gene
#SBATCH --partition=day,scavenge
#SBATCH --array=0-9%4
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=/vast/palmer/pi/xiting_yan/hw646/Medical-SAM3_all_tma_pre_vlm_tma39_exact_20260902/logs/genemap_%A_%a.log
#SBATCH --mail-type=NONE

set -euo pipefail

source /etc/profile.d/modules.sh 2>/dev/null || true
module purge
module load Python/3.12.3-GCCcore-13.3.0
module load SciPy-bundle/2024.05-gfbf-2024a
module load matplotlib/3.9.2-gfbf-2024a
module load Pillow/10.4.0-GCCcore-13.3.0

TMAS=(TMA07 TMA24 TMA29 TMA30 TMA31 TMA34 TMA36 TMA39 TMA41 TMA42)
SLIDES=(TMA07_770-1 TMA24_770-1 TMA29_771-1 TMA30_771-1 TMA31_771-1 TMA34_771-1 TMA36_771-1 TMA39_771-1 TMA41_771-1 TMA42_771-1)
TMA=${TMAS[$SLURM_ARRAY_TASK_ID]}
SLIDE=${SLIDES[$SLURM_ARRAY_TASK_ID]}

PROJECT=/vast/palmer/pi/xiting_yan/hw646/Medical-SAM3
ROOT=/vast/palmer/pi/xiting_yan/hw646/Medical-SAM3_all_tma_pre_vlm_tma39_exact_20260902
RUN=${ROOT}/${TMA}
SHARED=${ROOT}/shared
CODE=${SHARED}/code
STAGING=${RUN}/staging
TRANSCRIPTS=${PROJECT}/output/xenium_silica_true_punkst_input_20260626/${SLIDE}/transcripts.sorted.tsv.gz

export MPLCONFIGDIR=${RUN}/mplconfig/${SLURM_JOB_ID}
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONUNBUFFERED=1
mkdir -p "${MPLCONFIGDIR}" "${RUN}/logs"

for required in "${TRANSCRIPTS}" "${SHARED}/highres_he_crop_manifest.json" \
  "${SHARED}/TMA39_fixed_gene_modules.csv" "${STAGING}/${TMA}_he.png" \
  "${CODE}/build_fixed_genemap_scores.py" \
  "${CODE}/build_tma39_genemap_adaptive_prompts.py" \
  "${CODE}/build_tma39_genemap_sustained_edge_prompts.py"; do
  test -s "${required}" || { echo "Missing required file: ${required}" >&2; exit 3; }
done

echo "${TMA}: frozen TMA39 120-gene and 9-module membership"
echo "Prompt rule: top 1% center, top 2.5% connection, top 5% extent, 0% box expansion, peak point"
echo "Region merge: same module plus at least 3 shared edge-adjacent GeneMap bins"
date --iso-8601=seconds

if [[ ! -s "${RUN}/genemap_scores/summary.json" ]]; then
  python "${CODE}/build_fixed_genemap_scores.py" \
    --tma "${TMA}" \
    --transcripts "${TRANSCRIPTS}" \
    --overview-manifest "${SHARED}/highres_he_crop_manifest.json" \
    --fixed-gene-modules "${SHARED}/TMA39_fixed_gene_modules.csv" \
    --output "${RUN}/genemap_scores"
fi

if [[ ! -s "${RUN}/prompts/adaptive_base/adaptive_prompt_manifest.json" ]]; then
  python "${CODE}/build_tma39_genemap_adaptive_prompts.py" \
    --tma "${TMA}" \
    --module-npz "${RUN}/genemap_scores/gene_module_scores_unclipped.npz" \
    --reference-image "${STAGING}/${TMA}_he.png" \
    --context-bins 0 \
    --variant-name context0 \
    --output "${RUN}/prompts/adaptive_base"
fi

if [[ ! -s "${RUN}/prompts/final_sustained_edge3/adaptive_prompt_manifest.json" ]]; then
  python "${CODE}/build_tma39_genemap_sustained_edge_prompts.py" \
    --tma "${TMA}" \
    --old-labels-npz "${RUN}/prompts/adaptive_base/adaptive_region_labels.npz" \
    --module-npz "${RUN}/genemap_scores/gene_module_scores_unclipped.npz" \
    --reference-image "${STAGING}/${TMA}_he.png" \
    --old-regions-csv "${RUN}/prompts/adaptive_base/adaptive_regions.csv" \
    --minimum-shared-edge-bins 3 \
    --expected-count 0 \
    --output "${RUN}/prompts/final_sustained_edge3"
fi

python - "${RUN}" "${TMA}" <<'PY'
import csv
import json
import sys
from pathlib import Path

run, tma = Path(sys.argv[1]), sys.argv[2]
score = json.loads((run / "genemap_scores/summary.json").read_text())
base = json.loads((run / "prompts/adaptive_base/adaptive_prompt_manifest.json").read_text())
final = json.loads((run / "prompts/final_sustained_edge3/adaptive_prompt_manifest.json").read_text())
rows = list(csv.DictReader((run / "prompts/final_sustained_edge3/adaptive_box_peak_sustained_edge3_context0.csv").open()))
assert score["tma"] == base["tma"] == final["tma"] == tma
assert score["fixed_gene_count"] == score["active_gene_count"] == 120
assert score["gene_module_count"] == base["module_count"] == 9
assert score["fixed_module_source_sha256"] == "23246f95bd1461529c2022c08b50b0b34e5fca4006deae800c06c3f67787d9a2"
assert base["context_pixels"] == [0, 0]
assert final["minimum_shared_edge_bins"] == 3
assert len(rows) == final["prompt_count"] > 0
assert all(row["selection_used_annotation"].lower() == "false" for row in rows)
if tma == "TMA39":
    assert len(rows) == 44
if tma == "TMA30":
    assert len(rows) == 45
print("VERIFIED", json.dumps({
    "tma": tma,
    "active_genes": score["active_gene_count"],
    "base_regions": base["prompt_count"],
    "final_regions": final["prompt_count"],
    "module_prompt_counts": final["module_prompt_counts"],
    "selection_used_annotation": False,
}, sort_keys=True))
PY

date --iso-8601=seconds
echo DONE
