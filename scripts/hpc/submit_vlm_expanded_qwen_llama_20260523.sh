#!/bin/bash
set -euo pipefail

# Submit the expanded VLM-only direct retrieval test:
# - build a 6-class pool with 10 GOOD + 10 MID + 10 BAD per class
# - render light-gray masked crops and gray reverse-blur crops
# - run Qwen/Llama plus two comparison baselines on both crop modes

REMOTE=${REMOTE:-bouchet}
REMOTE_SCRIPT_DIR=${REMOTE_SCRIPT_DIR:-/home/hw646/codex_salip_scripts}
RUN_NAME=${RUN_NAME:-paired_he_ficture_good_mid_bad_6class_perbucket10_20260523}
PER_BUCKET=${PER_BUCKET:-10}

PLAIN_POOL=/home/hw646/codex_vlm_hit_tests_expanded_plain/${RUN_NAME}_lightgray
REVERSE_POOL=/home/hw646/codex_vlm_hit_tests_expanded_reverseblur/${RUN_NAME}_reverseblur
PLAIN_OUT_BASE=/home/hw646/codex_vlm_hit_tests_expanded_results_plain
REVERSE_OUT_BASE=/home/hw646/codex_vlm_hit_tests_expanded_results_reverseblur

scp \
  inference/visium_hd_exp1/build_paired_vlm_hit_test_pool.py \
  inference/visium_hd_exp1/render_paired_vlm_pool_plain_crops.py \
  inference/visium_hd_exp1/run_paired_vlm_hit_test.py \
  scripts/reports/render_vlm_direct_retrieval_simple_report.py \
  scripts/hpc/visium_hd_exp1_build_paired_vlm_hit_test_pool.sbatch \
  scripts/hpc/visium_hd_exp1_build_vlm_expanded_pool_and_crops.sbatch \
  scripts/hpc/visium_hd_exp1_vlm_direct_retrieval_model_pair.sbatch \
  "$REMOTE:$REMOTE_SCRIPT_DIR/"

build_job=$(
  ssh -o BatchMode=yes "$REMOTE" \
    "sbatch --parsable --export=ALL,RUN_NAME=$RUN_NAME,PER_BUCKET=$PER_BUCKET,OVERWRITE=1 $REMOTE_SCRIPT_DIR/visium_hd_exp1_build_vlm_expanded_pool_and_crops.sbatch"
)
echo "Submitted expanded pool build/render job: $build_job"

submit_model() {
  local model_name=$1
  local model_slug=$2
  local extra_export=${3:-}
  local export_vars="ALL,MODEL_NAME=$model_name,MODEL_SLUG=$model_slug,PLAIN_POOL=$PLAIN_POOL,REVERSE_POOL=$REVERSE_POOL,PLAIN_OUT_BASE=$PLAIN_OUT_BASE,REVERSE_OUT_BASE=$REVERSE_OUT_BASE,PROMPT_MODE=description_only,MAX_NEW_TOKENS=256"
  if [[ -n "$extra_export" ]]; then
    export_vars="$export_vars,$extra_export"
  fi
  ssh -o BatchMode=yes "$REMOTE" \
    "sbatch --parsable --dependency=afterok:$build_job --export=$export_vars $REMOTE_SCRIPT_DIR/visium_hd_exp1_vlm_direct_retrieval_model_pair.sbatch"
}

qwen_job=$(submit_model "Qwen/Qwen2.5-VL-7B-Instruct" "qwen25_vl_7b")
llama_job=$(submit_model "meta-llama/Llama-3.2-11B-Vision-Instruct" "llama32_11b_vision")
medgemma_job=$(submit_model "google/medgemma-4b-it" "medgemma_4b")
mistral_job=$(submit_model "mistralai/Mistral-Small-3.1-24B-Instruct-2503" "mistral_small31_24b" "VLM_MAX_MEMORY_CUDA=78GiB,VLM_MAX_MEMORY_CPU=160GiB")

echo "Submitted VLM jobs:"
echo "  Qwen2.5-VL-7B: $qwen_job"
echo "  Llama-3.2-11B-Vision: $llama_job"
echo "  MedGemma-4B baseline: $medgemma_job"
echo "  Mistral-Small-3.1-24B baseline: $mistral_job"
echo
echo "Pools:"
echo "  $PLAIN_POOL"
echo "  $REVERSE_POOL"
echo
echo "Monitor:"
echo "  ssh -o BatchMode=yes $REMOTE 'squeue -j $build_job,$qwen_job,$llama_job,$medgemma_job,$mistral_job -o \"%.18i %.12P %.40j %.2t %.12M %.12l %.6C %.10m %.24R\"'"
