#!/bin/bash
set -euo pipefail

# Submit a weak58 Step2 input-ablation using H&E candidate crop + H&E local context.
# The prompt styles here explicitly mention the second context image.

MODE="${MODE:-weak58}"
IMAGE_MODE="${IMAGE_MODE:-he_crop_context}"
PROMPT_STYLES="${PROMPT_STYLES:-comparative_context_minimal_retention comparative_context_soft_broad_veto comparative_context_structured_checklist all_hypothesis_context_rescue}"
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3-VL-32B-Instruct}"
MODEL_SLUG_PREFIX="${MODEL_SLUG_PREFIX:-qwen3vl32b_context}"
OUT_BASE_REMOTE="${OUT_BASE_REMOTE:-/nfs/roberts/project/pi_xy48/hw646/Medical-SAM3_remote_runs/visium_hd_exp1/jun10_step2_context_sweep_weak58}"

echo "[Jun10 Step2 context sweep] mode=$MODE image_mode=$IMAGE_MODE model=$MODEL_NAME"
echo "[Jun10 Step2 context sweep] prompt_styles=$PROMPT_STYLES"

MODE="$MODE" \
  IMAGE_MODE="$IMAGE_MODE" \
  PROMPT_STYLES="$PROMPT_STYLES" \
  MODEL_NAME="$MODEL_NAME" \
  MODEL_SLUG_PREFIX="$MODEL_SLUG_PREFIX" \
  OUT_BASE_REMOTE="$OUT_BASE_REMOTE" \
  scripts/local/sync_and_submit_jun10_step2_prompt_ablation.sh

echo "[Jun10 Step2 context sweep] submitted."
