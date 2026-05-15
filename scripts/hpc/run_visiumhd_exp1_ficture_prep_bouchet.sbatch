#!/bin/bash
#SBATCH --job-name=visiumhd_ficture_prep
#SBATCH --partition=day
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=/nfs/roberts/project/pi_xy48/hw646/Medical-SAM3/logs/visiumhd_ficture_prep_%j.out
#SBATCH --error=/nfs/roberts/project/pi_xy48/hw646/Medical-SAM3/logs/visiumhd_ficture_prep_%j.err

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/nfs/roberts/project/pi_xy48/hw646/Medical-SAM3}"
TOOLS_ROOT="${TOOLS_ROOT:-/nfs/roberts/project/pi_xy48/hw646/ficture_tools}"
DATADIR="${DATADIR:-/nfs/roberts/project/pi_xy48/hw646/Exp1/binned_outputs}"
OUTDIR="${OUTDIR:-/nfs/roberts/project/pi_xy48/hw646/FICTURE/spatula_output/VisiumHD_Exp1}"
PARQUET_PYTHON="${PARQUET_PYTHON:-$TOOLS_ROOT/envs/parquet/bin/python}"

mkdir -p "$PROJECT_ROOT/logs" "$OUTDIR"

module purge
module load CMake/3.31.8-GCCcore-13.3.0
module load GCC/13.3.0
module load tbb/2021.13.0-GCCcore-13.3.0
module load zlib/1.3.1-GCCcore-13.3.0
module load bzip2/1.0.8-GCCcore-13.3.0
module load XZ/5.4.5-GCCcore-13.3.0
module load cURL/8.7.1-GCCcore-13.3.0

export PATH="$TOOLS_ROOT/src/spatula/bin:$TOOLS_ROOT/src/punkst/bin:$PATH"
export LD_LIBRARY_PATH="$TOOLS_ROOT/local_opencv/lib64:$TOOLS_ROOT/local_libpng/lib:$TOOLS_ROOT/src/spatula/submodules/htslib:${LD_LIBRARY_PATH:-}"
export CFLAGS="-pthread"
export CXXFLAGS="-pthread"
export LDFLAGS="-pthread"
export DATADIR

echo "DATADIR=$DATADIR"
echo "OUTDIR=$OUTDIR"
echo "spatula=$(command -v spatula || true)"
echo "punkst=$(command -v punkst || true)"

if ! command -v spatula >/dev/null; then
  echo "spatula is missing. Submit scripts/hpc/install_ficture_stack_bouchet.sbatch first." >&2
  exit 1
fi

if [ ! -f "$DATADIR/square_002um/spatial/tissue_positions.csv.gz" ]; then
  if [ ! -x "$PARQUET_PYTHON" ]; then
    echo "Missing parquet helper Python: $PARQUET_PYTHON" >&2
    echo "Submit scripts/hpc/install_ficture_stack_bouchet.sbatch first." >&2
    exit 1
  fi
  "$PARQUET_PYTHON" - <<'PY'
from pathlib import Path
import gzip
import pandas as pd

import os

datadir = Path(os.environ["DATADIR"])
src = datadir / "square_002um/spatial/tissue_positions.parquet"
dst = datadir / "square_002um/spatial/tissue_positions.csv.gz"
df = pd.read_parquet(src)
with gzip.open(dst, "wt") as fh:
    df.to_csv(fh, index=False)
print(dst)
PY
fi

spatula convert-sge \
  --in-sge "$DATADIR/square_002um/filtered_feature_bc_matrix" \
  --pos "$DATADIR/square_002um/spatial/tissue_positions.csv.gz" \
  --units-per-um 3.652 \
  --colnames-count Count \
  --out-sge "$OUTDIR" \
  --out-tsv "$OUTDIR" \
  --icols-mtx 1 \
  --exclude-feature-regex '^(BLANK|NegCon|NegPrb|mt-|MT-|Gm[0-9]+$)'

if [ -f "$OUTDIR/transcripts.sorted.tsv.gz" ] && [ "$(stat -c%s "$OUTDIR/transcripts.sorted.tsv.gz")" -lt 1024 ]; then
  mv "$OUTDIR/transcripts.sorted.tsv.gz" "$OUTDIR/transcripts.sorted.tsv.gz.bad_$(date +%Y%m%d_%H%M%S)"
fi

if [ ! -f "$OUTDIR/transcripts.sorted.tsv.gz" ]; then
  tmp_sort="${TMPDIR:-/tmp}/visiumhd_ficture_sort_${SLURM_JOB_ID:-$$}"
  mkdir -p "$tmp_sort"
  (
    set +o pipefail
    gzip -cd "$OUTDIR/transcripts.unsorted.tsv.gz" | head -1
    set -o pipefail
    gzip -cd "$OUTDIR/transcripts.unsorted.tsv.gz" | tail -n +2 | sort -T "$tmp_sort" -S 48G -gk1
  ) | gzip -c > "$OUTDIR/transcripts.sorted.tsv.gz"
fi

echo "Prepared FICTURE/punkst input:"
ls -lh "$OUTDIR"/transcripts.*.tsv.gz

echo
echo "Next step:"
echo "  Copy/edit punkst examples/basic/config.json, point it at:"
echo "  $OUTDIR/transcripts.sorted.tsv.gz"
echo "Then generate/run workflow with:"
echo "  python $TOOLS_ROOT/src/punkst/ext/py/generate_workflow.py -c CONFIG.json -o run.sh -m Makefile -t $TOOLS_ROOT/src/punkst/examples/basic/Makefile"
echo "  make -f Makefile"
