#!/bin/bash
#SBATCH --job-name=alv_pairwise
#SBATCH --partition=gpu_h200,gpu_b200,gpu_rtx6000
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=120G
#SBATCH --time=03:00:00
#SBATCH --output=/nfs/roberts/project/pi_xy48/hw646/codex_salip_outputs/logs/jun07_alveoli_pairwise_%j.log
#SBATCH --mail-type=NONE

set -euo pipefail

PROJECT=${PROJECT:-/nfs/roberts/project/pi_xy48/hw646/Medical-SAM3}
PY=${PY:-/nfs/roberts/project/pi_xy48/hw646/ycrc_conda/envs/medsam3/bin/python}
RUN_ROOT=${RUN_ROOT:-/nfs/roberts/project/pi_xy48/hw646/Medical-SAM3_remote_runs/visium_hd_exp1/jun07_alveoli_box384_setting_rank_20260607}
SCRIPT_DIR=${SCRIPT_DIR:-/nfs/roberts/project/pi_xy48/hw646/codex_salip_scripts_jun07}
MODEL_NAME=${MODEL_NAME:-Qwen/Qwen3-VL-32B-Instruct}
MODEL_SLUG=${MODEL_SLUG:-qwen3vl32b}
IMAGE_MODE=${IMAGE_MODE:-he_context}
CANDIDATE_IDS=${CANDIDATE_IDS:-90,82,11,12,13,24}
DECISION_PROFILE=${DECISION_PROFILE:-standard}
SUMMARY_CSV=${SUMMARY_CSV:-$RUN_ROOT/alveoli_box384_ficture_text_summary.csv}

CONTEXT_POOL=$RUN_ROOT/alveoli_box384_context_pool
PAIRWISE_OUT=$RUN_ROOT/pairwise_choice_outputs/${MODEL_SLUG}_${IMAGE_MODE}_${SLURM_JOB_ID}

mkdir -p "$PAIRWISE_OUT" /nfs/roberts/project/pi_xy48/hw646/codex_salip_outputs/logs
cd "$PROJECT"

export PYTHONUNBUFFERED=1
export PYTHONPATH="$SCRIPT_DIR:$PROJECT/inference/visium_hd_exp1:$PROJECT/inference:$PROJECT:${PYTHONPATH:-}"
export MPLCONFIGDIR="${MPLCONFIGDIR_OVERRIDE:-$PROJECT/output/.mplconfig}"
export HF_HOME="${HF_HOME_OVERRIDE:-/nfs/roberts/project/pi_xy48/hw646/hf-home}"
export HF_HUB_CACHE="$HF_HOME/hub"
export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
export VLM_DEVICE_MAP="${VLM_DEVICE_MAP:-auto}"
export VLM_OFFLOAD_FOLDER="${VLM_OFFLOAD_FOLDER:-$RUN_ROOT/offload_pairwise_${SLURM_JOB_ID}}"
export VLM_ALLOW_AUTOMODEL_FALLBACK="${VLM_ALLOW_AUTOMODEL_FALLBACK:-1}"
export VLM_MAX_MEMORY_CUDA="${VLM_MAX_MEMORY_CUDA:-70GiB}"
export VLM_MAX_MEMORY_CPU="${VLM_MAX_MEMORY_CPU:-140GiB}"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
mkdir -p "$MPLCONFIGDIR" "$HF_HOME" "$VLM_OFFLOAD_FOLDER"

DEFAULT_HF_TOKEN_PATH=${DEFAULT_HF_TOKEN_PATH:-$HOME/.cache/huggingface/token}
if [[ -z "${HF_TOKEN:-}" && -s "$DEFAULT_HF_TOKEN_PATH" ]]; then
  export HF_TOKEN
  HF_TOKEN="$(cat "$DEFAULT_HF_TOKEN_PATH")"
fi

date
echo "Jun07 alveoli pairwise candidate choice"
echo "MODEL_NAME=$MODEL_NAME"
echo "IMAGE_MODE=$IMAGE_MODE"
echo "CANDIDATE_IDS=$CANDIDATE_IDS"
echo "DECISION_PROFILE=$DECISION_PROFILE"
echo "SUMMARY_CSV=$SUMMARY_CSV"
echo "PAIRWISE_OUT=$PAIRWISE_OUT"

SUMMARY_ARGS=()
if [[ -s "$SUMMARY_CSV" ]]; then
  SUMMARY_ARGS=(--summary-csv "$SUMMARY_CSV")
fi

"$PY" "$SCRIPT_DIR/run_local_vlm_pairwise_candidate_choice.py" \
  --pool-csv "$CONTEXT_POOL/public_vlm_requests.csv" \
  --output-dir "$PAIRWISE_OUT" \
  --model "$MODEL_NAME" \
  --device cuda \
  --target-display alveoli \
  --target-description "alveolar lung parenchyma, open air spaces, thin septa" \
  --candidate-ids "$CANDIDATE_IDS" \
  --image-mode "$IMAGE_MODE" \
  --decision-profile "$DECISION_PROFILE" \
  "${SUMMARY_ARGS[@]}" \
  --max-new-tokens 96

date
echo "Output: $PAIRWISE_OUT"
