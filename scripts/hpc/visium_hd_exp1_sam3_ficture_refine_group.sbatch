#!/bin/bash
#SBATCH --job-name=sam3_fictx
#SBATCH --partition=gpu_devel
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=03:00:00
#SBATCH --output=/nfs/roberts/project/pi_xy48/hw646/Medical-SAM3/results/visium_hd_exp1/logs/sam3_fictx_%x_%j.log
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=vettel.hwu@gmail.com

set -euo pipefail

GROUP="${1:?Usage: sbatch visium_hd_exp1_sam3_ficture_refine_group.sbatch GROUP}"

PROJECT=/nfs/roberts/project/pi_xy48/hw646/Medical-SAM3
PY=/nfs/roberts/project/pi_xy48/hw646/ycrc_conda/envs/medsam3/bin/python
ROOT_OUT=$PROJECT/results/visium_hd_exp1/sam3_reference_candidate_refine

cd "$PROJECT"
export MPLCONFIGDIR="$PROJECT/output/.mplconfig"
export HF_HOME=/nfs/roberts/project/pi_xy48/hw646/ycrc_conda/hf-home
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
mkdir -p "$MPLCONFIGDIR" "$ROOT_OUT" "$PROJECT/results/visium_hd_exp1/logs"

echo "Job: ${SLURM_JOB_ID}"
echo "Group: $GROUP"
echo "Host: $(hostname)"
date

BASE=$PROJECT/checkpoints/SAM3-base/sam3.pt
MED=$PROJECT/checkpoints/Medical-SAM3/checkpoint.pt
CAND=output/visium_hd_exp1/reference_map_candidates/hex12_k12_original_smooth
REF=output/visium_hd_exp1/assets/hex_12.k12.pixel.png
HE=output/visium_hd_exp1/assets/tissue_hires_image.png
TFORM=output/visium_hd_exp1/registration_debug/ficture_coordinate_ref_to_he.json

run_refine () {
  local name="$1"
  local ckpt="$2"
  local limit="$3"
  local max_side="$4"
  local margin="$5"
  local dilate="$6"
  local sort_by="$7"
  local factors="${8:-}"
  local out="$ROOT_OUT/${name}_${SLURM_JOB_ID}"
  local factor_args=()
  if [[ -n "$factors" ]]; then
    factor_args=(--factors "$factors")
  fi

  echo "======================================================================"
  echo "Refine: $name limit=$limit max_side=$max_side margin=$margin dilate=$dilate sort=$sort_by factors=${factors:-all}"

  "$PY" inference/visium_hd_exp1/run_sam3_refine_reference_candidates.py \
    --candidate-root "$CAND" \
    --reference-map "$REF" \
    --target-image "$HE" \
    --output-dir "$out" \
    --checkpoint "$ckpt" \
    --device cuda \
    --max-side "$max_side" \
    --limit "$limit" \
    --candidate-sort "$sort_by" \
    --box-margin "$margin" \
    --transform-json "$TFORM" \
    --use-affine-prior-mask \
    --clip-to-tissue \
    --prior-dilate-radius "$dilate" \
    "${factor_args[@]}"
}

case "$GROUP" in
  base_top20_d48)
    run_refine "ficture_base_top20_d48_area" "$BASE" 20 4096 0.04 48 area ""
    ;;
  base_top40_d80)
    run_refine "ficture_base_top40_d80_area" "$BASE" 40 4096 0.05 80 area ""
    ;;
  base_top40_d160)
    run_refine "ficture_base_top40_d160_area" "$BASE" 40 4096 0.05 160 area ""
    ;;
  medical_top20_d80)
    run_refine "ficture_medical_top20_d80_area" "$MED" 20 4096 0.05 80 area ""
    ;;
  medical_top40_d120)
    run_refine "ficture_medical_top40_d120_area" "$MED" 40 4096 0.06 120 area ""
    ;;
  base_csv_top40_d80)
    run_refine "ficture_base_top40_d80_csv" "$BASE" 40 4096 0.05 80 csv ""
    ;;
  base_factor_focus)
    run_refine "ficture_base_factor_focus_d96" "$BASE" 48 4096 0.05 96 csv "0,1,2,3,4,5,6,7,8,9,10,11"
    ;;
  medical_factor_focus)
    run_refine "ficture_medical_factor_focus_d96" "$MED" 48 4096 0.05 96 csv "0,1,2,3,4,5,6,7,8,9,10,11"
    ;;
  *)
    echo "Unknown group: $GROUP" >&2
    exit 2
    ;;
esac

date
