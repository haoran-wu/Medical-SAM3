#!/bin/bash
#SBATCH --job-name=openrouter_xlabel
#SBATCH --partition=day_amd
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=04:00:00
#SBATCH --output=/home/hw646/Medical-SAM3_remote_runs/visium_hd_exp1/logs/openrouter_cross_label_%j.log
#SBATCH --mail-type=NONE

set -euo pipefail

PROJECT=${PROJECT:-/nfs/roberts/project/pi_xy48/hw646/Medical-SAM3}
PY=${PY:-/nfs/roberts/project/pi_xy48/hw646/ycrc_conda/envs/medsam3/bin/python}
RUN_SCRIPT=${RUN_SCRIPT:-/home/hw646/codex_salip_scripts/run_openrouter_cross_label_top1_accuracy.py}
POOL_CSV=${POOL_CSV:?Set POOL_CSV}
FACTOR_LEGEND_CSV=${FACTOR_LEGEND_CSV:-/home/hw646/Medical-SAM3_remote_runs/visium_hd_exp1/current_project_inputs/ficture_factor_prompt_legend_from_html.csv}
OUTPUT_DIR=${OUTPUT_DIR:?Set OUTPUT_DIR}
MODEL=${MODEL:-openai/gpt-5.5}
VLM_PROMPT_STYLE=${VLM_PROMPT_STYLE:-compact_short}
MAX_TOKENS=${MAX_TOKENS:-256}
MAX_SIDE=${MAX_SIDE:-768}
JPEG_QUALITY=${JPEG_QUALITY:-85}
TEMPERATURE=${TEMPERATURE:-0}
TIMEOUT=${TIMEOUT:-180}
RETRIES=${RETRIES:-3}
API_KEY_FILE=${API_KEY_FILE:-/home/hw646/.openrouter_api_key}
DISABLE_RESPONSE_FORMAT=${DISABLE_RESPONSE_FORMAT:-0}

mkdir -p "$OUTPUT_DIR" /home/hw646/Medical-SAM3_remote_runs/visium_hd_exp1/logs
cd "$PROJECT"

export PYTHONUNBUFFERED=1
export PYTHONPATH="/home/hw646/codex_salip_scripts:$PROJECT/inference/visium_hd_exp1:$PROJECT/inference:$PROJECT:${PYTHONPATH:-}"
export VLM_PROMPT_STYLE

date
echo "OpenRouter cross-label API run"
echo "MODEL=$MODEL"
echo "VLM_PROMPT_STYLE=$VLM_PROMPT_STYLE"
echo "POOL_CSV=$POOL_CSV"
echo "OUTPUT_DIR=$OUTPUT_DIR"
echo "MAX_TOKENS=$MAX_TOKENS"
echo "MAX_SIDE=$MAX_SIDE"
if [[ -s "$API_KEY_FILE" ]]; then
  echo "API key file: present"
else
  echo "API key file: missing"
fi

EXTRA_ARGS=()
if [[ "$DISABLE_RESPONSE_FORMAT" == "1" ]]; then
  EXTRA_ARGS+=(--disable-response-format)
fi

"$PY" "$RUN_SCRIPT" \
  --pool-csv "$POOL_CSV" \
  --output-dir "$OUTPUT_DIR" \
  --factor-legend-csv "$FACTOR_LEGEND_CSV" \
  --model "$MODEL" \
  --api-key-file "$API_KEY_FILE" \
  --temperature "$TEMPERATURE" \
  --max-tokens "$MAX_TOKENS" \
  --max-side "$MAX_SIDE" \
  --jpeg-quality "$JPEG_QUALITY" \
  --timeout "$TIMEOUT" \
  --retries "$RETRIES" \
  --resume \
  "${EXTRA_ARGS[@]}"

date
echo "Done: $OUTPUT_DIR"
