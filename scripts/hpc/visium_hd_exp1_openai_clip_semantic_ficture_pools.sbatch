#!/bin/bash
#SBATCH --job-name=clip_sem_fic
#SBATCH --partition=gpu_rtx6000
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=72G
#SBATCH --time=03:00:00
#SBATCH --output=/home/hw646/codex_salip_outputs/logs/openai_clip_semantic_ficture_%j.log
#SBATCH --mail-type=NONE

set -euo pipefail

echo "Batch shell started on $(hostname) at $(date)"

# Direct feature-aware CLIP retrieval:
# candidate mask -> crop official FICTURE factor-color image -> OpenAI CLIP
# image/text similarity with factor/cell-type/gene semantic prompts -> top-k
# masks -> Dice/Precision/Recall.

PROJECT=/nfs/roberts/project/pi_xy48/hw646/Medical-SAM3
PY=/nfs/roberts/project/pi_xy48/hw646/ycrc_conda/envs/medsam3/bin/python
INPUT_DIR=$PROJECT/output/visium_hd_exp1/ficture_official_filtered_candidate_pool_input_roi
OFFICIAL_SUMMARY=$PROJECT/output/visium_hd_exp1/ficture_official_filtered_he_aligned/summary_official.json
SEMANTIC_LEGEND=$PROJECT/output/visium_hd_exp1/ficture_official_filtered_he_aligned/factor_semantic_legend.json

HE_ROOT=${HE_ROOT:?Set HE_ROOT to the same-ROI H&E candidate root}
FICTURE_ROOT=${FICTURE_ROOT:-$PROJECT/results/visium_hd_exp1/ficture_official_filtered_candidate_pool}
RUN_NAME=${RUN_NAME:-official_same_roi_he_plus_ficture_openai_clip_semantic}
CLIP_MODEL=${CLIP_MODEL:-openai/clip-vit-base-patch32}
CLIP_DEVICE=${CLIP_DEVICE:-cuda}
CLIP_BATCH_SIZE=${CLIP_BATCH_SIZE:-64}
TOP_KS_LIST=${TOP_KS_LIST:-"1:2:3:5:8:12:20"}
MAX_OVERLAP=${MAX_OVERLAP:-0.86}
MASK_CACHE_SIZE=${MASK_CACHE_SIZE:-64}
OUT_BASE=${OUT_BASE:-/home/hw646/codex_salip_outputs/openai_clip_semantic_merged_pool}
RANK_SCRIPT=${RANK_SCRIPT:-/home/hw646/codex_salip_scripts/rank_openai_clip_semantic_ficture_pools.py}

OUT=$OUT_BASE/${RUN_NAME}_${SLURM_JOB_ID}

cd "$PROJECT"
mkdir -p "$OUT" "$OUT_BASE" /home/hw646/codex_salip_outputs/logs
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
fic_size = Image.open("$INPUT_DIR/ficture_official_filtered_roi_rgb.png").size
if fic_size != (3144, 3327):
    raise SystemExit(f"Unexpected official FICTURE ROI size: {fic_size}")
for name, root in [("HE_ROOT", "$HE_ROOT"), ("FICTURE_ROOT", "$FICTURE_ROOT")]:
    root_path = Path(root)
    if not root_path.exists():
        raise SystemExit(f"{name} does not exist: {root_path}")
legend = Path("$SEMANTIC_LEGEND")
if not legend.exists():
    raise SystemExit(f"Missing semantic legend: {legend}")
print(f"OpenAI CLIP semantic preflight OK: ficture_size={fic_size}")
PY

IFS=':' read -r -a top_ks <<< "$TOP_KS_LIST"

date
echo "RUN_NAME=$RUN_NAME"
echo "Ranker=openai_clip_ficture_semantic only"
echo "Plain meaning: CLIP compares each candidate-masked FICTURE crop to feature-aware tissue-label text prompts."
echo "CLIP_MODEL=$CLIP_MODEL"
echo "CLIP_DEVICE=$CLIP_DEVICE"
echo "HE_ROOT=$HE_ROOT"
echo "FICTURE_ROOT=$FICTURE_ROOT"
echo "Output=$OUT"
"$PY" "$RANK_SCRIPT" \
  --candidate-root "he=$HE_ROOT" \
  --candidate-root "ficture=$FICTURE_ROOT" \
  --ficture-image "$INPUT_DIR/ficture_official_filtered_roi_rgb.png" \
  --summary-path "$INPUT_DIR/region_summary_official_filtered_roi_hpc.json" \
  --official-summary "$OFFICIAL_SUMMARY" \
  --factor-semantic-legend "$SEMANTIC_LEGEND" \
  --output-dir "$OUT" \
  --clip-model "$CLIP_MODEL" \
  --clip-device "$CLIP_DEVICE" \
  --clip-batch-size "$CLIP_BATCH_SIZE" \
  --top-ks "${top_ks[@]}" \
  --max-overlap "$MAX_OVERLAP" \
  --mask-cache-size "$MASK_CACHE_SIZE"
date
echo "Output: $OUT"
