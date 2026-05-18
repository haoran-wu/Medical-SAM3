#!/bin/bash
#SBATCH --job-name=mm_rank_fic_cpu
#SBATCH --partition=devel
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=/nfs/roberts/project/pi_xy48/hw646/Medical-SAM3/results/visium_hd_exp1/logs/mm_rank_fic_cpu_%j.log
#SBATCH --mail-type=NONE

set -euo pipefail

# CPU official filtered FICTURE semantic/ranking pass. Use this for non-CLIP
# rankers so GPU queues are not occupied by NumPy/IO-only work.

PROJECT=/nfs/roberts/project/pi_xy48/hw646/Medical-SAM3
PY=/nfs/roberts/project/pi_xy48/hw646/ycrc_conda/envs/medsam3/bin/python
INPUT_DIR=$PROJECT/output/visium_hd_exp1/ficture_official_filtered_candidate_pool_input_roi

RUN_NAME=${RUN_NAME:-official_ficture_semantic_cpu}
CAND_LIMIT=${CAND_LIMIT:-120}
TOP_KS_LIST=${TOP_KS_LIST:-"1:2:3:5:8:12:20"}
MAX_OVERLAP=${MAX_OVERLAP:-0.86}
DISABLE_CONSENSUS=${DISABLE_CONSENSUS:-0}
CONSENSUS_SAMPLE_LIMIT=${CONSENSUS_SAMPLE_LIMIT:-80}
MASK_CACHE_SIZE=${MASK_CACHE_SIZE:-32}
MIN_FICTURE_REPORTS=${MIN_FICTURE_REPORTS:-48}
MIN_UNIQUE_SETTINGS=${MIN_UNIQUE_SETTINGS:-48}
FACTOR_ANNOTATION=${FACTOR_ANNOTATION:-$PROJECT/output/visium_hd_exp1/molecular_prior_experiments/dx-60_dy80/factor_annotation.csv}
FACTOR_SEMANTIC_LEGEND=${FACTOR_SEMANTIC_LEGEND:-$PROJECT/output/visium_hd_exp1/ficture_official_filtered_he_aligned/factor_semantic_legend.json}

OUT=$PROJECT/output/visium_hd_exp1/multimodal_candidate_ranking/${RUN_NAME}_${SLURM_JOB_ID}
FICTURE_ROOT=$PROJECT/results/visium_hd_exp1/ficture_official_filtered_candidate_pool

cd "$PROJECT"
mkdir -p "$OUT" "$PROJECT/results/visium_hd_exp1/logs"
export MPLCONFIGDIR="$PROJECT/output/.mplconfig"
export PYTHONUNBUFFERED=1

"$PY" - <<PY
from pathlib import Path
import json
import numpy as np
from PIL import Image

project = Path("$PROJECT")
input_dir = Path("$INPUT_DIR")
official_summary = project / "output/visium_hd_exp1/ficture_official_filtered_he_aligned/summary_official.json"
required = [
    official_summary,
    input_dir / "he_roi_matching_official_ficture_coverage.png",
    input_dir / "ficture_official_filtered_roi_rgb.png",
    input_dir / "ficture_official_filtered_roi_factor_index.npy",
    input_dir / "region_summary_official_filtered_roi_hpc.json",
    Path("$FICTURE_ROOT"),
    Path("$FACTOR_SEMANTIC_LEGEND"),
]
for path in required:
    if not path.exists():
        raise SystemExit(f"Missing official FICTURE prerequisite: {path}")
if json.loads(official_summary.read_text()).get("status") != "PASS_OFFICIAL":
    raise SystemExit(f"Official FICTURE summary is not PASS_OFFICIAL: {official_summary}")
he_size = Image.open(required[1]).size
fic_size = Image.open(required[2]).size
factor_shape = np.load(required[3], mmap_mode="r").shape
if he_size != fic_size or factor_shape != (fic_size[1], fic_size[0]):
    raise SystemExit(f"Official input size mismatch: he={he_size} ficture={fic_size} factor={factor_shape}")
reports = sorted(Path("$FICTURE_ROOT").glob("**/candidate_report.json"))
if len(reports) < int("$MIN_FICTURE_REPORTS"):
    raise SystemExit(f"Official FICTURE candidate reports incomplete: {len(reports)} < $MIN_FICTURE_REPORTS")
unique_settings = {p.parent.name for p in reports}
if len(unique_settings) < int("$MIN_UNIQUE_SETTINGS"):
    raise SystemExit(f"Official FICTURE unique settings incomplete: {len(unique_settings)} < $MIN_UNIQUE_SETTINGS")
print(f"Official CPU preflight OK: image_size={fic_size}, factor_shape={factor_shape}, reports={len(reports)}, unique_settings={len(unique_settings)}")
PY

consensus_args=()
if [[ "$DISABLE_CONSENSUS" == "1" ]]; then
  consensus_args+=(--disable-consensus)
else
  consensus_args+=(--consensus-sample-limit "$CONSENSUS_SAMPLE_LIMIT")
fi

IFS=':' read -r -a top_ks <<< "$TOP_KS_LIST"

date
echo "RUN_NAME=$RUN_NAME"
echo "CPU semantic official ranking: CAND_LIMIT=$CAND_LIMIT TOP_KS=$TOP_KS_LIST MAX_OVERLAP=$MAX_OVERLAP"
echo "FACTOR_SEMANTIC_LEGEND=$FACTOR_SEMANTIC_LEGEND"
"$PY" inference/visium_hd_exp1/rank_multimodal_sam_candidates.py \
  --candidate-root "ficture=$FICTURE_ROOT" \
  --he-image "$INPUT_DIR/he_roi_matching_official_ficture_coverage.png" \
  --ficture-image "$INPUT_DIR/ficture_official_filtered_roi_rgb.png" \
  --factor-label-npy "$INPUT_DIR/ficture_official_filtered_roi_factor_index.npy" \
  --summary-path "$INPUT_DIR/region_summary_official_filtered_roi_hpc.json" \
  --factor-annotation "$FACTOR_ANNOTATION" \
  --factor-semantic-legend "$FACTOR_SEMANTIC_LEGEND" \
  --output-dir "$OUT" \
  --top-ks "${top_ks[@]}" \
  --max-overlap "$MAX_OVERLAP" \
  --max-candidates-per-source-setting "$CAND_LIMIT" \
  --mask-cache-size "$MASK_CACHE_SIZE" \
  "${consensus_args[@]}"
date
echo "Output: $OUT"
