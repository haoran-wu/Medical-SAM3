#!/bin/bash
#SBATCH --job-name=vlm_fic_multi
#SBATCH --partition=gpu_h200
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=120G
#SBATCH --time=04:00:00
#SBATCH --output=/nfs/roberts/project/pi_xy48/hw646/Medical-SAM3/results/visium_hd_exp1/logs/vlm_fic_multi_%j.log
#SBATCH --mail-type=NONE

set -u

PROJECT=/nfs/roberts/project/pi_xy48/hw646/Medical-SAM3
JUDGE_SCRIPT=$PROJECT/scripts/hpc/visium_hd_exp1_vlm_candidate_judge_official_ficture.sbatch
LOG_DIR=$PROJECT/results/visium_hd_exp1/logs

mkdir -p "$LOG_DIR"
cd "$PROJECT" || exit 1

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# Non-Qwen VLM coverage test. Keep each pass small enough to learn quickly,
# then expand the best-compatible model(s) in a follow-up run.
MODELS=(
  "llava-hf/llava-1.5-7b-hf"
  "llava-hf/llava-v1.6-vicuna-7b-hf"
  "microsoft/Phi-3.5-vision-instruct"
  "OpenGVLab/InternVL2_5-8B"
  "openbmb/MiniCPM-V-2_6"
)

echo "Started multi-model official FICTURE VLM smoke: $(date)"
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-unknown}"
echo "Models:"
printf '  %s\n' "${MODELS[@]}"
echo

status=0
for model in "${MODELS[@]}"; do
  slug=$(echo "$model" | tr '[:upper:]' '[:lower:]' | sed 's#[^a-z0-9]#_#g' | sed 's#_#_#g' | sed 's#^_*##; s#_*$##')
  run_name="official_openai_clip_${slug}_judge_smoke"
  child_log="$LOG_DIR/vlm_fic_multi_${SLURM_JOB_ID}_${slug}.log"

  echo "============================================================"
  echo "MODEL=$model"
  echo "RUN_NAME=$run_name"
  echo "CHILD_LOG=$child_log"
  echo "Start: $(date)"

  (
    set -euo pipefail
    RUN_NAME="$run_name" \
    MODEL_NAME="$model" \
    SCORE_GLOB="official_ficture_openai_clip_semantic_*" \
    PER_LABEL=8 \
    TOP_KS_LIST="1:2:3:5" \
    MAX_OVERLAP=0.82 \
    MAX_NEW_TOKENS=64 \
    MAX_FACTOR_CONTEXT_CHARS=1800 \
    FACTOR_SEMANTIC_LEGEND="$PROJECT/output/visium_hd_exp1/ficture_official_filtered_he_aligned/factor_semantic_legend.json" \
      bash "$JUDGE_SCRIPT"
  ) >"$child_log" 2>&1
  rc=$?

  if [[ "$rc" -ne 0 ]]; then
    echo "FAILED model=$model rc=$rc"
    echo "Last log lines:"
    tail -80 "$child_log" || true
    status=1
  else
    echo "SUCCESS model=$model"
    tail -20 "$child_log" || true
  fi
  echo "End: $(date)"
done

echo "Finished multi-model official FICTURE VLM smoke: $(date)"
exit "$status"
