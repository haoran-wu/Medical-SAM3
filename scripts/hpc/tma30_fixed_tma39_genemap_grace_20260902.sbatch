#!/usr/bin/env bash
#SBATCH --job-name=t30_fixed_gene
#SBATCH --partition=day,scavenge
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=/vast/palmer/pi/xiting_yan/hw646/Medical-SAM3_current_workflow_20260902/TMA30_fixed_tma39_pipeline/logs/genemap_%j.log
#SBATCH --mail-type=NONE

set -euo pipefail

source /etc/profile.d/modules.sh 2>/dev/null || true
module purge
module load Python/3.12.3-GCCcore-13.3.0
module load SciPy-bundle/2024.05-gfbf-2024a
module load matplotlib/3.9.2-gfbf-2024a
module load Pillow/10.4.0-GCCcore-13.3.0

PROJECT=/vast/palmer/pi/xiting_yan/hw646/Medical-SAM3
RUN=/vast/palmer/pi/xiting_yan/hw646/Medical-SAM3_current_workflow_20260902/TMA30_fixed_tma39_pipeline
STAGING=${RUN}/staging
CODE=${RUN}/code
TRANSCRIPTS=${PROJECT}/output/xenium_silica_true_punkst_input_20260626/TMA30_771-1/transcripts.sorted.tsv.gz

export MPLCONFIGDIR=${RUN}/mplconfig/${SLURM_JOB_ID}
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONUNBUFFERED=1
mkdir -p "${MPLCONFIGDIR}" "${RUN}/logs"

for required in "${TRANSCRIPTS}" "${STAGING}/highres_he_crop_manifest.json" \
  "${STAGING}/TMA39_fixed_gene_modules.csv" "${STAGING}/TMA30_he.png" \
  "${CODE}/build_fixed_genemap_scores.py" \
  "${CODE}/build_tma39_genemap_adaptive_prompts.py" \
  "${CODE}/build_tma39_genemap_sustained_edge_prompts.py"; do
  test -s "${required}" || { echo "Missing required file: ${required}" >&2; exit 3; }
done

echo "TMA30 transfer uses the frozen TMA39 120-gene, 9-module membership."
echo "Prompt rule: top-1% core, top-2.5% confirmation, top-5% extent, 0% box expansion, peak point."
echo "Region merge rule: same module and at least 3 shared edge-adjacent GeneMap bins."
echo "Selection used annotation: False"
date --iso-8601=seconds

if [[ ! -s "${RUN}/genemap_scores/summary.json" ]]; then
  python "${CODE}/build_fixed_genemap_scores.py" \
    --tma TMA30 \
    --transcripts "${TRANSCRIPTS}" \
    --overview-manifest "${STAGING}/highres_he_crop_manifest.json" \
    --fixed-gene-modules "${STAGING}/TMA39_fixed_gene_modules.csv" \
    --output "${RUN}/genemap_scores"
fi

if [[ ! -s "${RUN}/prompts/adaptive_base/adaptive_prompt_manifest.json" ]]; then
  python "${CODE}/build_tma39_genemap_adaptive_prompts.py" \
    --tma TMA30 \
    --module-npz "${RUN}/genemap_scores/gene_module_scores_unclipped.npz" \
    --reference-image "${STAGING}/TMA30_he.png" \
    --context-bins 0 \
    --variant-name context0 \
    --output "${RUN}/prompts/adaptive_base"
fi

if [[ ! -s "${RUN}/prompts/final_sustained_edge3/adaptive_prompt_manifest.json" ]]; then
  python "${CODE}/build_tma39_genemap_sustained_edge_prompts.py" \
    --tma TMA30 \
    --old-labels-npz "${RUN}/prompts/adaptive_base/adaptive_region_labels.npz" \
    --module-npz "${RUN}/genemap_scores/gene_module_scores_unclipped.npz" \
    --reference-image "${STAGING}/TMA30_he.png" \
    --old-regions-csv "${RUN}/prompts/adaptive_base/adaptive_regions.csv" \
    --minimum-shared-edge-bins 3 \
    --expected-count 0 \
    --output "${RUN}/prompts/final_sustained_edge3"
fi

python - "${RUN}" <<'PY'
import csv
import json
import sys
from pathlib import Path

run = Path(sys.argv[1])
score = json.loads((run / "genemap_scores/summary.json").read_text())
base = json.loads((run / "prompts/adaptive_base/adaptive_prompt_manifest.json").read_text())
final = json.loads((run / "prompts/final_sustained_edge3/adaptive_prompt_manifest.json").read_text())
prompts = list(csv.DictReader((run / "prompts/final_sustained_edge3/adaptive_box_peak_sustained_edge3_context0.csv").open()))
assert score["tma"] == base["tma"] == final["tma"] == "TMA30"
assert score["gene_module_count"] == base["module_count"] == 9
assert score["selection_used_annotation"] is False
assert base["selection_used_annotation"] is False
assert final["selection_used_annotation"] is False
assert len(prompts) == final["prompt_count"] > 0
assert base["context_pixels"] == [0, 0]
assert all(row["selection_used_annotation"].lower() == "false" for row in prompts)
print("VERIFIED", json.dumps({
    "active_genes": score["active_gene_count"],
    "base_regions": base["prompt_count"],
    "final_regions": final["prompt_count"],
    "module_prompt_counts": final["module_prompt_counts"],
    "selection_used_annotation": False,
}, sort_keys=True))
PY

date --iso-8601=seconds
echo DONE
