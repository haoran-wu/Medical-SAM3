#!/bin/bash
#SBATCH --job-name=vlm_judge_off
#SBATCH --partition=gpu_h200
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --output=/nfs/roberts/project/pi_xy48/hw646/Medical-SAM3/results/visium_hd_exp1/logs/vlm_judge_off_%j.log
#SBATCH --mail-type=NONE

set -euo pipefail

PROJECT=/nfs/roberts/project/pi_xy48/hw646/Medical-SAM3
PY=/nfs/roberts/project/pi_xy48/hw646/ycrc_conda/envs/medsam3/bin/python
INPUT_DIR=$PROJECT/output/visium_hd_exp1/ficture_official_filtered_candidate_pool_input_roi
RANK_BASE=$PROJECT/output/visium_hd_exp1/multimodal_candidate_ranking
OUTBASE=$PROJECT/output/visium_hd_exp1/vlm_candidate_judge

RUN_NAME=${RUN_NAME:-official_qwen25vl7b_ficture}
MODEL_NAME=${MODEL_NAME:-Qwen/Qwen2.5-VL-7B-Instruct}
FACTOR_SEMANTIC_LEGEND=${FACTOR_SEMANTIC_LEGEND:-$PROJECT/output/visium_hd_exp1/ficture_official_filtered_he_aligned/factor_semantic_legend.json}
PER_LABEL=${PER_LABEL:-24}
TOP_KS_LIST=${TOP_KS_LIST:-"1:2:3:5:8:12"}
MAX_OVERLAP=${MAX_OVERLAP:-0.82}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-96}
SCORE_GLOB=${SCORE_GLOB:-official_ficture_*}

OUT=$OUTBASE/${RUN_NAME}_${SLURM_JOB_ID}

cd "$PROJECT"
mkdir -p "$OUT" "$PROJECT/results/visium_hd_exp1/logs" "$PROJECT/output/.hf_cache"
export MPLCONFIGDIR="$PROJECT/output/.mplconfig"
export HF_HOME=/nfs/roberts/project/pi_xy48/hw646/ycrc_conda/hf-home
export TRANSFORMERS_CACHE=$HF_HOME/hub
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
    input_dir / "he_roi_matching_official_ficture_coverage.png",
    input_dir / "ficture_official_filtered_roi_rgb.png",
    input_dir / "ficture_official_filtered_roi_factor_index.npy",
    input_dir / "region_summary_official_filtered_roi_hpc.json",
    official_summary,
    Path("$FACTOR_SEMANTIC_LEGEND"),
]
for path in required:
    if not path.exists():
        raise SystemExit(f"Missing official VLM prerequisite: {path}")
if json.loads(official_summary.read_text()).get("status") != "PASS_OFFICIAL":
    raise SystemExit(f"Official FICTURE summary is not PASS_OFFICIAL: {official_summary}")
he_size = Image.open(required[0]).size
fic_size = Image.open(required[1]).size
factor_shape = np.load(required[2], mmap_mode="r").shape
if he_size != fic_size or factor_shape != (fic_size[1], fic_size[0]):
    raise SystemExit(f"Official input size mismatch: he={he_size} ficture={fic_size} factor={factor_shape}")
PY

score_args=()
while IFS= read -r score_csv; do
  score_args+=(--score-csv "$score_csv")
done < <(find "$RANK_BASE" -maxdepth 2 -path "$RANK_BASE/$SCORE_GLOB/candidate_multimodal_scores.csv" -type f | sort)

if [[ ${#score_args[@]} -eq 0 ]]; then
  echo "No official score CSVs found with SCORE_GLOB=$SCORE_GLOB under $RANK_BASE" >&2
  exit 2
fi

IFS=':' read -r -a top_ks <<< "$TOP_KS_LIST"

date
echo "RUN_NAME=$RUN_NAME"
echo "MODEL_NAME=$MODEL_NAME"
echo "FACTOR_SEMANTIC_LEGEND=$FACTOR_SEMANTIC_LEGEND"
echo "PER_LABEL=$PER_LABEL TOP_KS=$TOP_KS_LIST MAX_OVERLAP=$MAX_OVERLAP"
printf 'Official score args: %s\n' "${score_args[@]}"
nvidia-smi || true
"$PY" inference/visium_hd_exp1/vlm_candidate_judge.py \
  "${score_args[@]}" \
  --he-image "$INPUT_DIR/he_roi_matching_official_ficture_coverage.png" \
  --ficture-image "$INPUT_DIR/ficture_official_filtered_roi_rgb.png" \
  --factor-label-npy "$INPUT_DIR/ficture_official_filtered_roi_factor_index.npy" \
  --factor-annotation "$PROJECT/output/visium_hd_exp1/molecular_prior_experiments/dx-60_dy80/factor_annotation.csv" \
  --factor-semantic-legend "$FACTOR_SEMANTIC_LEGEND" \
  --summary-path "$INPUT_DIR/region_summary_official_filtered_roi_hpc.json" \
  --output-dir "$OUT" \
  --model "$MODEL_NAME" \
  --device cuda \
  --per-label "$PER_LABEL" \
  --top-ks "${top_ks[@]}" \
  --max-overlap "$MAX_OVERLAP" \
  --max-new-tokens "$MAX_NEW_TOKENS"
date
echo "Output: $OUT"
