#!/bin/bash
#SBATCH --job-name=xen_hr_score
#SBATCH --partition=day
#SBATCH --array=0-259%96
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --output=/nfs/roberts/project/pi_xy48/hw646/logs/xen_hr_score_%A_%a.log
#SBATCH --mail-type=NONE

set -euo pipefail

# Hidden-GT scoring array for the current high-resolution, FICTURE-driven
# Xenium Silica SAM candidate bundle.  This scores paired runtime inputs:
#   he + ficture_molecule_k12

PROJECT=${PROJECT:-/nfs/roberts/project/pi_xy48/hw646/Medical-SAM3}
PY=${PY:-/nfs/roberts/project/pi_xy48/hw646/ycrc_conda/envs/medsam3/bin/python}
PREPARED_ROOT=${PREPARED_ROOT:-/nfs/roberts/project/pi_xy48/hw646/aaai_xenium_silica_20260623/xenium_hdstyle_sam_inputs_highres_ficturedriven_20260626}
CANDIDATE_ROOT=${CANDIDATE_ROOT:-/nfs/roberts/project/pi_xy48/hw646/aaai_xenium_silica_20260623/xenium_hdstyle_sam_candidates_highres_ficturedriven_20260626}
SCORE_ROOT=${SCORE_ROOT:-/nfs/roberts/project/pi_xy48/hw646/aaai_xenium_silica_20260623/xenium_hdstyle_sam_component_scores_highres_ficturedriven_20260626}
SETTINGS_MANIFEST=${SETTINGS_MANIFEST:-settings_manifest.csv}
ANNOTATION_POLICY=${ANNOTATION_POLICY:-dense}
CLUSTER_THRESHOLD=${CLUSTER_THRESHOLD:-0.90}

mkdir -p /nfs/roberts/project/pi_xy48/hw646/logs "$SCORE_ROOT"

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

cd "$PROJECT"

eval "$("$PY" - "$PREPARED_ROOT" "$SETTINGS_MANIFEST" "$SLURM_ARRAY_TASK_ID" <<'PY'
import csv
import shlex
import sys
from pathlib import Path

prepared = Path(sys.argv[1])
settings_manifest = sys.argv[2]
task = int(sys.argv[3])
tmas = sorted({int(row["tma"]) for row in csv.DictReader((prepared / "source_manifest.csv").open(newline=""))})
settings = list(csv.DictReader((prepared / settings_manifest).open(newline="")))
n_settings = len(settings)
tma_index = task // n_settings
setting_index = task % n_settings
if tma_index >= len(tmas):
    raise SystemExit(f"Task {task} exceeds {len(tmas) * n_settings} score tasks")
row = settings[setting_index]
values = {
    "TMA": tmas[tma_index],
    "SETTING_NAME": row["setting_name"],
    "SETTING_ID": row["setting_id"],
}
for key, value in values.items():
    print(f"{key}={shlex.quote(str(value))}")
PY
)"

OUT_DIR="$SCORE_ROOT/TMA$(printf '%02d' "$TMA")/he_plus_ficture_molecule_k12/${SETTING_NAME}"

echo "Start high-res FICTURE-driven component score task: $(date)"
echo "Host=$(hostname)"
echo "Array=${SLURM_ARRAY_TASK_ID} TMA=$TMA Setting=$SETTING_NAME"
echo "Prepared=$PREPARED_ROOT"
echo "SettingsManifest=$SETTINGS_MANIFEST"
echo "Candidates=$CANDIDATE_ROOT"
echo "Out=$OUT_DIR"
echo "AnnotationPolicy=$ANNOTATION_POLICY"

"$PY" scripts/xenium_score_hdstyle_sam_candidates.py \
  --prepared-root "$PREPARED_ROOT" \
  --candidate-root "$CANDIDATE_ROOT/sam_candidate_masks" \
  --tma "$TMA" \
  --setting-name "$SETTING_NAME" \
  --source-keys he ficture_molecule_k12 \
  --annotation-policy "$ANNOTATION_POLICY" \
  --cluster-threshold "$CLUSTER_THRESHOLD" \
  --out-dir "$OUT_DIR"

echo "Done high-res FICTURE-driven component score task: $(date)"
