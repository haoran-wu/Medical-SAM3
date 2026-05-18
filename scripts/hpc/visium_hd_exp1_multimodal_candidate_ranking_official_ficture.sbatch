#!/bin/bash
#SBATCH --job-name=mm_rank_fic_off
#SBATCH --partition=gpu_devel
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=/nfs/roberts/project/pi_xy48/hw646/Medical-SAM3/results/visium_hd_exp1/logs/mm_rank_fic_off_%j.log
#SBATCH --mail-type=NONE

set -euo pipefail

# Official filtered FICTURE only. Do not point this script to any pre-official
# ficture_coord_scaled / ficture_corrected inputs or roots.

PROJECT=/nfs/roberts/project/pi_xy48/hw646/Medical-SAM3
PY=/nfs/roberts/project/pi_xy48/hw646/ycrc_conda/envs/medsam3/bin/python
INPUT_DIR=$PROJECT/output/visium_hd_exp1/ficture_official_filtered_candidate_pool_input_roi

RUN_NAME=${RUN_NAME:-official_ficture_rank_lite}
CANDIDATE_MODE=${CANDIDATE_MODE:-ficture}
CAND_LIMIT=${CAND_LIMIT:-40}
TOP_KS_LIST=${TOP_KS_LIST:-"1:2:3:5"}
MAX_OVERLAP=${MAX_OVERLAP:-0.72}
DISABLE_CONSENSUS=${DISABLE_CONSENSUS:-1}
CONSENSUS_SAMPLE_LIMIT=${CONSENSUS_SAMPLE_LIMIT:-80}
MASK_CACHE_SIZE=${MASK_CACHE_SIZE:-24}
MIN_FICTURE_REPORTS=${MIN_FICTURE_REPORTS:-48}
MIN_UNIQUE_SETTINGS=${MIN_UNIQUE_SETTINGS:-48}
FACTOR_ANNOTATION=${FACTOR_ANNOTATION:-$PROJECT/output/visium_hd_exp1/molecular_prior_experiments/dx-60_dy80/factor_annotation.csv}
CLIP_MODEL=${CLIP_MODEL:-}
CLIP_BATCH_SIZE=${CLIP_BATCH_SIZE:-32}

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
he = input_dir / "he_roi_matching_official_ficture_coverage.png"
fic = input_dir / "ficture_official_filtered_roi_rgb.png"
factor = input_dir / "ficture_official_filtered_roi_factor_index.npy"
summary = input_dir / "region_summary_official_filtered_roi_hpc.json"
candidate_root = Path("$FICTURE_ROOT")
for path in [official_summary, he, fic, factor, summary, candidate_root]:
    if not path.exists():
        raise SystemExit(f"Missing official FICTURE prerequisite: {path}")
status = json.loads(official_summary.read_text()).get("status")
if status != "PASS_OFFICIAL":
    raise SystemExit(f"Official FICTURE summary is not PASS_OFFICIAL: {official_summary}")
he_size = Image.open(he).size
fic_size = Image.open(fic).size
factor_shape = np.load(factor, mmap_mode="r").shape
if he_size != fic_size:
    raise SystemExit(f"Official H&E/FICTURE size mismatch: {he_size} vs {fic_size}")
if factor_shape != (fic_size[1], fic_size[0]):
    raise SystemExit(f"Official factor-index shape mismatch: {factor_shape} vs {(fic_size[1], fic_size[0])}")
reports = sorted(candidate_root.glob("**/candidate_report.json"))
if len(reports) < int("$MIN_FICTURE_REPORTS"):
    raise SystemExit(f"Official FICTURE candidate reports incomplete: {len(reports)} < $MIN_FICTURE_REPORTS")
unique_settings = {p.parent.name for p in reports}
if len(unique_settings) < int("$MIN_UNIQUE_SETTINGS"):
    raise SystemExit(
        f"Official FICTURE unique settings incomplete: {len(unique_settings)} < $MIN_UNIQUE_SETTINGS"
    )
if "deprecated_ficture_pre_official" in str(candidate_root) or "ficture_corrected_candidate_pool" in str(candidate_root):
    raise SystemExit(f"Refusing pre-official FICTURE candidate root: {candidate_root}")
print(
    f"Official preflight OK: image_size={fic_size}, factor_shape={factor_shape}, "
    f"reports={len(reports)}, unique_settings={len(unique_settings)}"
)
PY

candidate_args=()
case "$CANDIDATE_MODE" in
  ficture)
    candidate_args+=(--candidate-root "ficture=$FICTURE_ROOT")
    ;;
  *)
    echo "Official script currently permits only CANDIDATE_MODE=ficture; got $CANDIDATE_MODE" >&2
    exit 2
    ;;
esac

consensus_args=()
if [[ "$DISABLE_CONSENSUS" == "1" ]]; then
  consensus_args+=(--disable-consensus)
else
  consensus_args+=(--consensus-sample-limit "$CONSENSUS_SAMPLE_LIMIT")
fi

clip_args=()
if [[ -n "$CLIP_MODEL" ]]; then
  clip_args+=(--clip-model "$CLIP_MODEL" --clip-batch-size "$CLIP_BATCH_SIZE" --clip-device cuda)
fi

IFS=':' read -r -a top_ks <<< "$TOP_KS_LIST"

date
echo "RUN_NAME=$RUN_NAME"
echo "CANDIDATE_MODE=$CANDIDATE_MODE CAND_LIMIT=$CAND_LIMIT TOP_KS=$TOP_KS_LIST CONSENSUS_DISABLED=$DISABLE_CONSENSUS"
echo "Official input: $INPUT_DIR"
echo "FICTURE_ROOT=$FICTURE_ROOT"
echo "CLIP_MODEL=${CLIP_MODEL:-none}"
"$PY" inference/visium_hd_exp1/rank_multimodal_sam_candidates.py \
  "${candidate_args[@]}" \
  --he-image "$INPUT_DIR/he_roi_matching_official_ficture_coverage.png" \
  --factor-label-npy "$INPUT_DIR/ficture_official_filtered_roi_factor_index.npy" \
  --summary-path "$INPUT_DIR/region_summary_official_filtered_roi_hpc.json" \
  --factor-annotation "$FACTOR_ANNOTATION" \
  --output-dir "$OUT" \
  --top-ks "${top_ks[@]}" \
  --max-overlap "$MAX_OVERLAP" \
  --max-candidates-per-source-setting "$CAND_LIMIT" \
  --mask-cache-size "$MASK_CACHE_SIZE" \
  "${consensus_args[@]}" \
  "${clip_args[@]}"
date
echo "Output: $OUT"
