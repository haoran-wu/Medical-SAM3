#!/bin/bash
#SBATCH --job-name=install_ficture_stack
#SBATCH --partition=day
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=/nfs/roberts/project/pi_xy48/hw646/Medical-SAM3/logs/install_ficture_stack_%j.out
#SBATCH --error=/nfs/roberts/project/pi_xy48/hw646/Medical-SAM3/logs/install_ficture_stack_%j.err

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/nfs/roberts/project/pi_xy48/hw646/Medical-SAM3}"
TOOLS_ROOT="${TOOLS_ROOT:-/nfs/roberts/project/pi_xy48/hw646/ficture_tools}"
OPENCV_PREFIX="${OPENCV_PREFIX:-$TOOLS_ROOT/local_opencv}"
LIBPNG_PREFIX="${LIBPNG_PREFIX:-$TOOLS_ROOT/local_libpng}"
PARQUET_ENV="${PARQUET_ENV:-$TOOLS_ROOT/envs/parquet}"

mkdir -p "$PROJECT_ROOT/logs" "$TOOLS_ROOT/src"
cd "$TOOLS_ROOT/src"

module purge
module load CMake/3.18
module load GCC/12.2.0
module load tbb/2021.10.0-GCCcore-12.2.0
module load zlib/1.2.12-GCCcore-12.2.0
module load bzip2/1.0.8-GCCcore-12.2.0
module load XZ/5.2.7-GCCcore-12.2.0
module load cURL/7.86.0-GCCcore-12.2.0
module load Autoconf
module load Automake
module load binutils

echo "Installing tools under $TOOLS_ROOT"

if [ ! -x "$PARQUET_ENV/bin/python" ]; then
  if [ -d "$PARQUET_ENV" ]; then
    mv "$PARQUET_ENV" "${PARQUET_ENV}.incomplete_$(date +%Y%m%d_%H%M%S)"
  fi
  module load miniconda/24.11.3
  conda create -y -p "$PARQUET_ENV" python=3.10 pandas pyarrow
else
  echo "Parquet helper env already exists: $PARQUET_ENV"
fi

module purge
module load CMake/3.18
module load GCC/12.2.0
module load tbb/2021.10.0-GCCcore-12.2.0
module load zlib/1.2.12-GCCcore-12.2.0
module load bzip2/1.0.8-GCCcore-12.2.0
module load XZ/5.2.7-GCCcore-12.2.0
module load cURL/7.86.0-GCCcore-12.2.0
module load Autoconf
module load Automake
module load binutils

if [ ! -x "$OPENCV_PREFIX/bin/opencv_version" ]; then
  if [ ! -d opencv ]; then
    git clone https://github.com/opencv/opencv.git
  fi
  git -C opencv checkout 4.x
  cmake -S opencv -B opencv/build \
    -D CMAKE_BUILD_TYPE=Release \
    -D CMAKE_INSTALL_PREFIX="$OPENCV_PREFIX"
  cmake --build opencv/build --parallel "${SLURM_CPUS_PER_TASK:-4}"
  cmake --install opencv/build
else
  echo "OpenCV already installed: $OPENCV_PREFIX"
fi

if [ ! -x "$TOOLS_ROOT/src/punkst/bin/punkst" ]; then
  if [ ! -d punkst ]; then
    git clone --recursive https://github.com/Yichen-Si/punkst.git
  fi
  git -C punkst submodule update --init --recursive
  cmake -S punkst -B punkst/build \
    -DOpenCV_DIR="$OPENCV_PREFIX/lib64/cmake/opencv4" \
    -DENABLE_REMOTE_IO=OFF
  cmake --build punkst/build --parallel "${SLURM_CPUS_PER_TASK:-4}"
else
  echo "punkst already built: $TOOLS_ROOT/src/punkst"
fi

if [ ! -f "$LIBPNG_PREFIX/lib/libpng.so" ]; then
  if [ ! -f libpng-1.6.43.tar.gz ]; then
    wget -O libpng-1.6.43.tar.gz https://download.sourceforge.net/libpng/libpng-1.6.43.tar.gz
  fi
  rm -rf libpng-1.6.43
  tar -xf libpng-1.6.43.tar.gz
  cd libpng-1.6.43
  ./configure --prefix="$LIBPNG_PREFIX"
  make -j "${SLURM_CPUS_PER_TASK:-4}"
  make install
  cd "$TOOLS_ROOT/src"
else
  echo "libpng already installed: $LIBPNG_PREFIX"
fi

if [ ! -x "$TOOLS_ROOT/src/spatula/bin/spatula" ]; then
  if [ ! -d spatula ]; then
    git clone --recursive https://github.com/seqscope/spatula.git
  fi
  cd spatula/submodules
  export CFLAGS="-pthread"
  export CXXFLAGS="-pthread"
  export LDFLAGS="-pthread"
  if [ -d ./qgenlib/build ]; then
    mv ./qgenlib/build "./qgenlib/build.failed_$(date +%Y%m%d_%H%M%S)"
  fi
  sed -i '1{/^#include <cstdint>/!s/^/#include <cstdint>\n/}' ./qgenlib/qgenlib/qgen_utils.h
  sed -i '1{/^#include <cstdint>/!s/^/#include <cstdint>\n/}' ./qgenlib/qgenlib/qgen_error.h
  sed -i '1{/^#include <cstdint>/!s/^/#include <cstdint>\n/}' ./qgenlib/qgenlib/dataframe.h
  sh -x build.sh
  cd ..
  if [ -d ./build ]; then
    mv ./build "./build.failed_$(date +%Y%m%d_%H%M%S)"
  fi
  cmake -S . -B build \
    -DPNG_LIBRARY="$LIBPNG_PREFIX/lib/libpng.so" \
    -DPNG_PNG_INCLUDE_DIR="$LIBPNG_PREFIX/include" \
    -DCURL_LIBRARY="$EBROOTCURL/lib/libcurl.so" \
    -DCURL_INCLUDE_DIR="$EBROOTCURL/include"
  cmake --build build --parallel "${SLURM_CPUS_PER_TASK:-4}"
  cd "$TOOLS_ROOT/src"
else
  echo "spatula already built: $TOOLS_ROOT/src/spatula"
fi

echo "Done."
echo "punkst: $TOOLS_ROOT/src/punkst/bin/punkst"
echo "spatula: $TOOLS_ROOT/src/spatula/bin/spatula"
