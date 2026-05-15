#!/bin/bash
#SBATCH --job-name=visiumhd_punkst_k12
#SBATCH --partition=week
#SBATCH --time=3-00:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --output=/nfs/roberts/project/pi_xy48/hw646/Medical-SAM3/logs/visiumhd_punkst_k12_%j.out
#SBATCH --error=/nfs/roberts/project/pi_xy48/hw646/Medical-SAM3/logs/visiumhd_punkst_k12_%j.err

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/nfs/roberts/project/pi_xy48/hw646/Medical-SAM3}"
TOOLS_ROOT="${TOOLS_ROOT:-/nfs/roberts/project/pi_xy48/hw646/ficture_tools}"
INPUT_DIR="${INPUT_DIR:-/nfs/roberts/project/pi_xy48/hw646/FICTURE/spatula_output/VisiumHD_Exp1}"
RUN_DIR="${RUN_DIR:-/nfs/roberts/project/pi_xy48/hw646/FICTURE/punkst_runs/VisiumHD_Exp1_hex12_k12_v2}"
TMP_ROOT="${TMP_ROOT:-/nfs/roberts/project/pi_xy48/hw646/FICTURE/tmp}"
THREADS="${THREADS:-${SLURM_CPUS_PER_TASK:-16}}"

mkdir -p "$PROJECT_ROOT/logs" "$RUN_DIR" "$TMP_ROOT"

module purge
module load CMake/3.31.8-GCCcore-13.3.0
module load GCC/13.3.0
module load tbb/2021.13.0-GCCcore-13.3.0
module load zlib/1.3.1-GCCcore-13.3.0
module load bzip2/1.0.8-GCCcore-13.3.0
module load XZ/5.4.5-GCCcore-13.3.0
module load cURL/8.7.1-GCCcore-13.3.0
module load Python/3.12.3-GCCcore-13.3.0
module load Python-bundle-PyPI/2024.06-GCCcore-13.3.0

export PATH="$TOOLS_ROOT/src/punkst/bin:$PATH"
export LD_LIBRARY_PATH="$TOOLS_ROOT/local_opencv/lib64:$TOOLS_ROOT/local_libpng/lib:$TOOLS_ROOT/src/spatula/submodules/htslib:${LD_LIBRARY_PATH:-}"

TRANSCRIPTS="$INPUT_DIR/transcripts.sorted.tsv.gz"
TMPDIR="$TMP_ROOT/VisiumHD_Exp1_hex12_k12_${SLURM_JOB_ID:-manual}"
CONFIG_JSON="$RUN_DIR/config.hex12.k12.json"
MAKEFILE="$RUN_DIR/Makefile"
RUN_SH="$RUN_DIR/run.sh"

if [ ! -s "$TRANSCRIPTS" ]; then
  echo "Missing sorted transcript input: $TRANSCRIPTS" >&2
  echo "Run scripts/hpc/run_visiumhd_exp1_ficture_prep_bouchet.sbatch first." >&2
  exit 1
fi

if ! command -v punkst >/dev/null; then
  echo "punkst is not on PATH. Check TOOLS_ROOT=$TOOLS_ROOT" >&2
  exit 1
fi

python - <<PY
from pathlib import Path
import json

config = {
    "workflow": {
        "threads": int("${THREADS}"),
        "seed": 1,
        "gitpath": "${TOOLS_ROOT}/src/punkst",
        "datadir": "${RUN_DIR}",
        "tmpdir": "${TMPDIR}",
        "transcripts": "${TRANSCRIPTS}",
        "exclude_feature_regex": "'^(?:Gm[0-9]+|Blank|BLANK|NegCon|NegPrb|mt-|MT-)'",
        "icol_x": 0,
        "icol_y": 1,
        "icol_feature": 2,
        "icol_count": 3,
        "skip": 1,
        "tilesize": 500,
        "hexgrids": [12],
        "topics": [12],
        "use_fixed_color_table": True,
        "nepochs": 2,
        "pixhex": 12,
        "nmove": 2,
        "res": 2,
        "scale": 2,
    },
    "job": {
        "time": "3-00:00:00",
        "cpus-per-task": int("${THREADS}"),
        "mem": "128G",
        "job-name": "visiumhd_punkst_k12",
        "output": "${PROJECT_ROOT}/logs/visiumhd_punkst_k12_generated_%j.out",
        "error": "${PROJECT_ROOT}/logs/visiumhd_punkst_k12_generated_%j.err",
        "extra_lines": "module purge\\nmodule load CMake/3.31.8-GCCcore-13.3.0 GCC/13.3.0 tbb/2021.13.0-GCCcore-13.3.0 zlib/1.3.1-GCCcore-13.3.0 bzip2/1.0.8-GCCcore-13.3.0 XZ/5.4.5-GCCcore-13.3.0 cURL/8.7.1-GCCcore-13.3.0 Python/3.12.3-GCCcore-13.3.0 Python-bundle-PyPI/2024.06-GCCcore-13.3.0\\nexport PATH=${TOOLS_ROOT}/src/punkst/bin:\\$PATH\\nexport LD_LIBRARY_PATH=${TOOLS_ROOT}/local_opencv/lib64:${TOOLS_ROOT}/local_libpng/lib:${TOOLS_ROOT}/src/spatula/submodules/htslib:\\${LD_LIBRARY_PATH:-}",
    },
}

Path("${RUN_DIR}").mkdir(parents=True, exist_ok=True)
with open("${CONFIG_JSON}", "w") as fh:
    json.dump(config, fh, indent=2)
    fh.write("\\n")
print("${CONFIG_JSON}")
PY

python "$TOOLS_ROOT/src/punkst/ext/py/generate_workflow.py" \
  -c "$CONFIG_JSON" \
  -o "$RUN_SH" \
  -m "$MAKEFILE" \
  -t "$TOOLS_ROOT/src/punkst/examples/basic/Makefile"

echo "Generated workflow:"
ls -lh "$CONFIG_JSON" "$MAKEFILE" "$RUN_SH"

echo
echo "Dry run:"
make -C "$RUN_DIR" -f "$MAKEFILE" --dry-run

echo
echo "Running punkst pixel-level workflow:"
make -C "$RUN_DIR" -f "$MAKEFILE" -j 1

echo
echo "Final outputs:"
ls -lh "$RUN_DIR"/hex_12.k12.pixel.png "$RUN_DIR"/hex_12.k12.pixel.info.html "$RUN_DIR"/hex_12.k12.pixel.info.tsv

python - <<PY
from pathlib import Path
from PIL import Image

png = Path("${RUN_DIR}") / "hex_12.k12.pixel.png"
html = Path("${RUN_DIR}") / "hex_12.k12.pixel.info.html"
if not png.is_file() or png.stat().st_size < 1024:
    raise SystemExit(f"Bad/missing PNG: {png}")
if not html.is_file() or html.stat().st_size < 1024:
    raise SystemExit(f"Bad/missing HTML report: {html}")

with Image.open(png) as im:
    print(f"Verified image: {png} size={im.size} mode={im.mode}")
print(f"Verified report: {html} bytes={html.stat().st_size}")
PY
