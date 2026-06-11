#!/bin/bash
set -euo pipefail

# Submit a weak58 Step2 sweep across local/open-source VLMs.
#
# This is intentionally a wrapper around sync_and_submit_jun10_step2_prompt_ablation.sh
# so all path syncing, official-pool inputs, and collector compatibility stay in one place.

MODE="${MODE:-weak58}"
IMAGE_MODE="${IMAGE_MODE:-he_crop_only}"
PROMPT_STYLES="${PROMPT_STYLES:-comparative_soft_broad_veto_retention comparative_minimal_retention comparative_structural_keep_broad_prune comparative_alveoli_septa_rescue comparative_stroma_immune_strict_veto comparative_soft_keep}"
OUT_BASE_REMOTE="${OUT_BASE_REMOTE:-/nfs/roberts/project/pi_xy48/hw646/Medical-SAM3_remote_runs/visium_hd_exp1/jun10_step2_model_sweep_weak58}"

MODEL_LIST="${MODEL_LIST:-Qwen/Qwen3-VL-8B-Instruct Qwen/Qwen3-VL-32B-Instruct OpenGVLab/InternVL3_5-8B-Instruct google/medgemma-4b-it}"

slug_for_model() {
  case "$1" in
    Qwen/Qwen3-VL-8B-Instruct) echo "qwen3vl8b" ;;
    Qwen/Qwen3-VL-32B-Instruct) echo "qwen3vl32b" ;;
    Qwen/Qwen2.5-VL-7B-Instruct) echo "qwen25vl7b" ;;
    OpenGVLab/InternVL3_5-8B-Instruct) echo "internvl35_8b" ;;
    OpenGVLab/InternVL3_5-38B-Instruct) echo "internvl35_38b" ;;
    google/medgemma-4b-it) echo "medgemma4b" ;;
    mistralai/Mistral-Small-3.1-24B-Instruct-2503) echo "mistral_small31_24b" ;;
    *) echo "$1" | tr '/:.' '___' | tr -cd '[:alnum:]_' | tr '[:upper:]' '[:lower:]' ;;
  esac
}

echo "[Jun10 Step2 model sweep] mode=$MODE image_mode=$IMAGE_MODE"
echo "[Jun10 Step2 model sweep] prompt_styles=$PROMPT_STYLES"
echo "[Jun10 Step2 model sweep] models=$MODEL_LIST"

for MODEL_NAME in $MODEL_LIST; do
  MODEL_SLUG_PREFIX="$(slug_for_model "$MODEL_NAME")"
  echo "[Jun10 Step2 model sweep] submitting model=$MODEL_NAME slug=$MODEL_SLUG_PREFIX"
  MODE="$MODE" \
    IMAGE_MODE="$IMAGE_MODE" \
    PROMPT_STYLES="$PROMPT_STYLES" \
    MODEL_NAME="$MODEL_NAME" \
    MODEL_SLUG_PREFIX="$MODEL_SLUG_PREFIX" \
    OUT_BASE_REMOTE="$OUT_BASE_REMOTE" \
    scripts/local/sync_and_submit_jun10_step2_prompt_ablation.sh
done

echo "[Jun10 Step2 model sweep] submitted all requested models."
