#!/bin/bash
#SBATCH --job-name=he_medpt24_chunk
#SBATCH --partition=gpu_b200,gpu_h200,gpu_rtx6000
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --array=0-7%4
#SBATCH --output=/home/hw646/codex_he_official_roi_candidate_pool_b200/logs/he_medpt24_chunk_%A_%a.log
#SBATCH --mail-type=NONE

set -euo pipefail

PROJECT=/nfs/roberts/project/pi_xy48/hw646/Medical-SAM3
PY=/nfs/roberts/project/pi_xy48/hw646/ycrc_conda/envs/medsam3/bin/python
INPUT_DIR=$PROJECT/output/visium_hd_exp1/ficture_official_filtered_candidate_pool_input_roi
IMAGE=$INPUT_DIR/he_roi_matching_official_ficture_coverage.png
FICTURE_IMAGE=$INPUT_DIR/ficture_official_filtered_roi_rgb.png
SUMMARY=$INPUT_DIR/region_summary_official_filtered_roi_hpc.json
OFFICIAL_SUMMARY=$PROJECT/output/visium_hd_exp1/ficture_official_filtered_he_aligned/summary_official.json
CHUNK_ROOT=${CHUNK_ROOT:-/home/hw646/codex_he_official_roi_candidate_pool_chunks/medical_official_points_step24_${SLURM_ARRAY_JOB_ID}}
CHUNK_SIZE=${CHUNK_SIZE:-450}
TASK=${SLURM_ARRAY_TASK_ID}
START=$((TASK * CHUNK_SIZE))
OUT=$CHUNK_ROOT/medical_official_points_step24_chunk_${TASK}
MED=$PROJECT/checkpoints/Medical-SAM3/checkpoint.pt

mkdir -p "$OUT" "$CHUNK_ROOT" /home/hw646/codex_he_official_roi_candidate_pool_b200/logs
cd "$PROJECT"

"$PY" - <<PY
import json
from pathlib import Path
from PIL import Image

payload = json.loads(Path("$OFFICIAL_SUMMARY").read_text())
if payload.get("status") != "PASS_OFFICIAL":
    raise SystemExit(f"Official FICTURE summary is not PASS_OFFICIAL: $OFFICIAL_SUMMARY")
failed = [k for k, v in payload.get("status_checks", {}).items() if not v]
if failed:
    raise SystemExit(f"Official FICTURE checks failed: {failed}")
he = Image.open("$IMAGE").size
fic = Image.open("$FICTURE_IMAGE").size
if he != fic or he != (3144, 3327):
    raise SystemExit(f"Unexpected same-ROI image sizes: he={he}, ficture={fic}")
print(f"Chunk preflight OK: size={he}", flush=True)
PY

export MPLCONFIGDIR="${MPLCONFIGDIR_OVERRIDE:-$PROJECT/output/.mplconfig}"
export HF_HOME="${HF_HOME_OVERRIDE:-/nfs/roberts/project/pi_xy48/hw646/ycrc_conda/hf-home}"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
export PROJECT_ROOT_OVERRIDE="$PROJECT"
export PYTHONUNBUFFERED=1
mkdir -p "$MPLCONFIGDIR" "$HF_HOME"

if [[ -s "$OUT/candidate_report.json" ]]; then
  echo "Existing chunk report found; skipping: $OUT"
  exit 0
fi

date
echo "Run: medical_official_points_step24 chunk $TASK"
echo "Prompt start=$START count=$CHUNK_SIZE"
echo "Output=$OUT"
"$PY" /home/hw646/codex_salip_scripts/run_sam3_candidate_proposals_chunkable.py \
  --image-path "$IMAGE" \
  --summary-path "$SUMMARY" \
  --checkpoint "$MED" \
  --device cuda \
  --output-dir "$OUT" \
  --proposal-mode point \
  --max-side 2048 \
  --min-area-frac 0.00002 \
  --max-area-frac 0.95 \
  --nms-iou 0 \
  --save-mask-limit 1000 \
  --contact-sheet-limit 0 \
  --grid-step 24 \
  --border 12 \
  --max-points 3600 \
  --prompt-start "$START" \
  --prompt-count "$CHUNK_SIZE" \
  --skip-label-eval
date
echo "Output: $OUT"
