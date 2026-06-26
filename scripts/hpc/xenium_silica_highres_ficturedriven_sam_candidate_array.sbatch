#!/bin/bash
#SBATCH --job-name=xen_hr_sam
#SBATCH --partition=gpu
#SBATCH --array=0-519%48
#SBATCH --time=03:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --output=/nfs/roberts/project/pi_xy48/hw646/logs/xen_hr_sam_%A_%a.log
#SBATCH --mail-type=NONE

set -euo pipefail

# Candidate-generation array for the current high-resolution, FICTURE-driven
# Xenium Silica bundle.  The input bundle should be synced from local:
#   output/aaai_xenium_silica_20260623/xenium_hdstyle_sam_inputs_highres_ficturedriven_20260626

PROJECT=${PROJECT:-/nfs/roberts/project/pi_xy48/hw646/Medical-SAM3}
PY=${PY:-/nfs/roberts/project/pi_xy48/hw646/ycrc_conda/envs/medsam3/bin/python}
INPUT_ROOT=${INPUT_ROOT:-/nfs/roberts/project/pi_xy48/hw646/aaai_xenium_silica_20260623/xenium_hdstyle_sam_inputs_highres_ficturedriven_20260626}
TASKS_CSV=${TASKS_CSV:-$INPUT_ROOT/sam_array_tasks.csv}
OUTPUT_ROOT=${OUTPUT_ROOT:-/nfs/roberts/project/pi_xy48/hw646/aaai_xenium_silica_20260623/xenium_hdstyle_sam_candidates_highres_ficturedriven_20260626}
SAVE_MASK_LIMIT=${SAVE_MASK_LIMIT:-100000}
CONTACT_SHEET_LIMIT=${CONTACT_SHEET_LIMIT:-80}
NMS_IOU=${NMS_IOU:-0.97}
AUTO_SKIP_NMS_MIN_CANDIDATES=${AUTO_SKIP_NMS_MIN_CANDIDATES:-5000}
SKIP_EXISTING_REPORT=${SKIP_EXISTING_REPORT:-1}

BASE=${BASE:-$PROJECT/checkpoints/SAM3-base/sam3.pt}
MED=${MED:-$PROJECT/checkpoints/Medical-SAM3/checkpoint.pt}

mkdir -p /nfs/roberts/project/pi_xy48/hw646/logs "$OUTPUT_ROOT"

export PYTHONUNBUFFERED=1
export MPLCONFIGDIR=${MPLCONFIGDIR:-/tmp/mpl_${USER}_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}}
export HF_HOME=${HF_HOME:-/nfs/roberts/project/pi_xy48/hw646/.cache/huggingface}
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

cd "$PROJECT"

eval "$("$PY" - "$INPUT_ROOT" "$TASKS_CSV" "$SLURM_ARRAY_TASK_ID" <<'PY'
import csv
import shlex
import sys
from pathlib import Path

input_root = Path(sys.argv[1])
tasks_csv = Path(sys.argv[2])
task_id = int(sys.argv[3])
rows = list(csv.DictReader(tasks_csv.open(newline="")))
if task_id < 0 or task_id >= len(rows):
    raise SystemExit(f"Task id {task_id} outside task table of {len(rows)} rows")
row = rows[task_id]
values = {
    "TASK_ID": row["task_id"],
    "TMA": row["tma"],
    "SLIDE": row["slide"],
    "SOURCE_KEY": row["source_key"],
    "IMAGE_PATH": str(input_root / row["image_rel"]),
    "SETTING_ID": row["setting_id"],
    "SETTING_NAME": row["setting_name"],
    "CHECKPOINT_KIND": row["checkpoint_kind"],
    "PROPOSAL_MODE": row["proposal_mode"],
    "MAX_SIDE": row["max_side"],
    "EXTRA_ARGS_STR": row["extra_args"],
    "OUTPUT_REL": row["output_rel"],
}
for key, value in values.items():
    print(f"{key}={shlex.quote(str(value))}")
PY
)"

if [[ "$CHECKPOINT_KIND" == "base" ]]; then
  CKPT="$BASE"
elif [[ "$CHECKPOINT_KIND" == "medical" ]]; then
  CKPT="$MED"
else
  echo "Unknown checkpoint kind: $CHECKPOINT_KIND" >&2
  exit 2
fi

OUT_DIR="$OUTPUT_ROOT/$OUTPUT_REL"
mkdir -p "$OUT_DIR"

read -r -a EXTRA_ARGS <<< "$EXTRA_ARGS_STR"

if [[ "$SKIP_EXISTING_REPORT" == "1" && -s "$OUT_DIR/candidate_report.json" ]]; then
  echo "Skip existing completed candidate report: $OUT_DIR/candidate_report.json"
  exit 0
fi

echo "Start high-res FICTURE-driven Xenium SAM task: $(date)"
echo "Host=$(hostname)"
echo "Task=$TASK_ID Array=${SLURM_ARRAY_TASK_ID}"
echo "TMA=$TMA Slide=$SLIDE Source=$SOURCE_KEY"
echo "Setting=$SETTING_NAME id=$SETTING_ID mode=$PROPOSAL_MODE checkpoint=$CHECKPOINT_KIND"
echo "Image=$IMAGE_PATH"
echo "Output=$OUT_DIR"

"$PY" inference/visium_hd_exp1/run_sam3_candidate_proposals.py \
  --image-path "$IMAGE_PATH" \
  --summary-path "$INPUT_ROOT/no_label_eval_summary_missing.json" \
  --checkpoint "$CKPT" \
  --device cuda \
  --output-dir "$OUT_DIR" \
  --proposal-mode "$PROPOSAL_MODE" \
  --max-side "$MAX_SIDE" \
  --min-area-frac 0.00002 \
  --max-area-frac 0.95 \
  --nms-iou "$NMS_IOU" \
  --auto-skip-nms-min-candidates "$AUTO_SKIP_NMS_MIN_CANDIDATES" \
  --save-mask-limit "$SAVE_MASK_LIMIT" \
  --contact-sheet-limit "$CONTACT_SHEET_LIMIT" \
  "${EXTRA_ARGS[@]}"

echo "Done high-res FICTURE-driven Xenium SAM task: $(date)"
