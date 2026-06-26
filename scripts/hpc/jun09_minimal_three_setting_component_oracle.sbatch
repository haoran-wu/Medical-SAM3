#!/bin/bash
#SBATCH --job-name=jun09_min3_component_oracle
#SBATCH --partition=day_amd
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --array=0-5
#SBATCH --output=/nfs/roberts/project/pi_xy48/hw646/codex_salip_outputs/logs/jun09_min3_component_oracle_%A_%a.log
#SBATCH --error=/nfs/roberts/project/pi_xy48/hw646/codex_salip_outputs/logs/jun09_min3_component_oracle_%A_%a.log

set -euo pipefail

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

PROJECT_ROOT="${PROJECT_ROOT:-/nfs/roberts/project/pi_xy48/hw646/Medical-SAM3}"
RUN_ROOT="${RUN_ROOT:-/nfs/roberts/project/pi_xy48/hw646/Medical-SAM3_remote_runs/visium_hd_exp1/jun09_minimal_three_setting_component_oracle_${SLURM_ARRAY_JOB_ID}}"
mkdir -p "$RUN_ROOT" /nfs/roberts/project/pi_xy48/hw646/codex_salip_outputs/logs

LABELS=(
  lung_bronchiola
  lung_alveoli
  lung_vessels
  tumor
  stroma
  immune_infiltration
)
LABEL="${LABELS[$SLURM_ARRAY_TASK_ID]}"
OUT_DIR="$RUN_ROOT/$LABEL"

echo "job=$SLURM_JOB_ID task=$SLURM_ARRAY_TASK_ID label=$LABEL"
echo "project=$PROJECT_ROOT"
echo "out=$OUT_DIR"

cd "$PROJECT_ROOT"

python inference/visium_hd_exp1/componentwise_candidate_oracle.py \
  --labels "$LABEL" \
  --out-dir "$OUT_DIR" \
  --combined-source-name minimal_three_settings \
  --top-k-per-component 5 \
  --min-component-area 256 \
  --strict-root-integrity \
  --candidate-root "he_medpt24=/home/hw646/Medical-SAM3_remote_runs/visium_hd_exp1/codex_home_20260530/codex_he_official_roi_candidate_pool_b200/he_official_12445451/medical_official_points_step24" \
  --candidate-root "ficture_medpt24=/home/hw646/Medical-SAM3_remote_runs/visium_hd_exp1/ficture_official_filtered_candidate_pool_full_masks/ficture_official_medpt24_full_13707092/medical_official_points_step24" \
  --candidate-root "he_base_official_points_step24=/home/hw646/Medical-SAM3_remote_runs/visium_hd_exp1/fullmask_repair_candidate_pools/he_fullmask_repair_13723426/base_official_points_step24" \
  --candidate-root "ficture_base_official_points_step24=/home/hw646/Medical-SAM3_remote_runs/visium_hd_exp1/fullmask_repair_candidate_pools/ficture_fullmask_repair_13723427/base_official_points_step24" \
  --candidate-root "he_base_box384_s128_m1536=/home/hw646/Medical-SAM3_remote_runs/visium_hd_exp1/codex_home_20260530/codex_he_official_roi_candidate_pool_b200/he_official_12445451/base_box384_s128_m1536" \
  --candidate-root "ficture_base_box384_s128_m1536=/nfs/roberts/project/pi_xy48/hw646/Medical-SAM3/results/visium_hd_exp1/ficture_official_filtered_candidate_pool/ficture_official_11938227/base_box384_s128_m1536"

echo "done label=$LABEL out=$OUT_DIR"
