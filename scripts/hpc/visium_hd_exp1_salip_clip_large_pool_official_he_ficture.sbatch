#!/bin/bash
#SBATCH --job-name=salip_he_fic
#SBATCH --partition=gpu_devel
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=72G
#SBATCH --time=03:00:00
#SBATCH --output=/nfs/roberts/project/pi_xy48/hw646/Medical-SAM3/results/visium_hd_exp1/logs/salip_he_fic_%j.log
#SBATCH --mail-type=NONE

set -euo pipefail

# Pure CLIP image-text retrieval after the missing same-ROI H&E candidate pool
# has been generated. The two roots are safe to fuse only when both masks live
# on the same 3144 x 3327 H&E/FICTURE crop.

PROJECT=/nfs/roberts/project/pi_xy48/hw646/Medical-SAM3
PY=/nfs/roberts/project/pi_xy48/hw646/ycrc_conda/envs/medsam3/bin/python
INPUT_DIR=$PROJECT/output/visium_hd_exp1/ficture_official_filtered_candidate_pool_input_roi
OFFICIAL_SUMMARY=$PROJECT/output/visium_hd_exp1/ficture_official_filtered_he_aligned/summary_official.json

HE_ROOT=${HE_ROOT:?Set HE_ROOT to the same-ROI H&E candidate root, e.g. /home/hw646/codex_he_official_roi_candidate_pool/he_official_<arrayjobid>}
FICTURE_ROOT=${FICTURE_ROOT:-$PROJECT/results/visium_hd_exp1/ficture_official_filtered_candidate_pool}
RUN_NAME=${RUN_NAME:-official_same_roi_he_plus_ficture_salip_clip_large_pool}
CLIP_MODEL=${CLIP_MODEL:-openai/clip-vit-base-patch32}
CLIP_BATCH_SIZE=${CLIP_BATCH_SIZE:-64}
TOP_KS_LIST=${TOP_KS_LIST:-"1:2:3:5:8:12:20"}
MAX_OVERLAP=${MAX_OVERLAP:-0.86}
MASK_CACHE_SIZE=${MASK_CACHE_SIZE:-64}
OUT_BASE=${OUT_BASE:-$PROJECT/output/visium_hd_exp1/salip_clip_large_pool}
LOG_DIR=${LOG_DIR:-$PROJECT/results/visium_hd_exp1/logs}
RANK_SCRIPT=${RANK_SCRIPT:-$PROJECT/inference/visium_hd_exp1/rank_salip_clip_large_pool.py}

OUT=$OUT_BASE/${RUN_NAME}_${SLURM_JOB_ID}

cd "$PROJECT"
mkdir -p "$OUT" "$LOG_DIR"
export MPLCONFIGDIR="${MPLCONFIGDIR_OVERRIDE:-$PROJECT/output/.mplconfig}"
export HF_HOME="${HF_HOME_OVERRIDE:-/nfs/roberts/project/pi_xy48/hw646/ycrc_conda/hf-home}"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
export PYTHONUNBUFFERED=1
export PYTHONPATH="$PROJECT/inference/visium_hd_exp1:$PROJECT/inference:$PROJECT:${PYTHONPATH:-}"
mkdir -p "$MPLCONFIGDIR" "$HF_HOME"

"$PY" - <<PY
import json
from pathlib import Path
from PIL import Image

official = Path("$OFFICIAL_SUMMARY")
payload = json.loads(official.read_text())
if payload.get("status") != "PASS_OFFICIAL":
    raise SystemExit(f"Official FICTURE summary is not PASS_OFFICIAL: {official}")
failed = [k for k, v in payload.get("status_checks", {}).items() if not v]
if failed:
    raise SystemExit(f"Official FICTURE checks failed: {failed}")
he_size = Image.open("$INPUT_DIR/he_roi_matching_official_ficture_coverage.png").size
fic_size = Image.open("$INPUT_DIR/ficture_official_filtered_roi_rgb.png").size
if he_size != fic_size:
    raise SystemExit(f"H&E/FICTURE input images differ: he={he_size}, ficture={fic_size}")
if he_size != (3144, 3327):
    raise SystemExit(f"Unexpected same-ROI size: {he_size}")
for root_name, root in [("HE_ROOT", "$HE_ROOT"), ("FICTURE_ROOT", "$FICTURE_ROOT")]:
    root_path = Path(root)
    if not root_path.exists():
        raise SystemExit(f"{root_name} does not exist: {root_path}")
print(f"Same-ROI fused preflight OK: image_size={he_size}")
PY

IFS=':' read -r -a top_ks <<< "$TOP_KS_LIST"

date
echo "RUN_NAME=$RUN_NAME"
echo "Ranker=salip_clip only"
echo "Plain meaning: CLIP compares each candidate-masked H&E crop to tissue-label text prompts."
echo "CLIP_MODEL=$CLIP_MODEL"
echo "HE_ROOT=$HE_ROOT"
echo "FICTURE_ROOT=$FICTURE_ROOT"
echo "Output=$OUT"
"$PY" "$RANK_SCRIPT" \
  --candidate-root "he=$HE_ROOT" \
  --candidate-root "ficture=$FICTURE_ROOT" \
  --he-image "$INPUT_DIR/he_roi_matching_official_ficture_coverage.png" \
  --summary-path "$INPUT_DIR/region_summary_official_filtered_roi_hpc.json" \
  --official-summary "$OFFICIAL_SUMMARY" \
  --output-dir "$OUT" \
  --clip-model "$CLIP_MODEL" \
  --clip-device cuda \
  --clip-batch-size "$CLIP_BATCH_SIZE" \
  --top-ks "${top_ks[@]}" \
  --max-overlap "$MAX_OVERLAP" \
  --mask-cache-size "$MASK_CACHE_SIZE"
date
echo "Output: $OUT"
