# intro
This is related to get a FICTURE like cell type information.

# Install required software


## install OpenCV first
git clone the repo to project dir. 
then 
install OpenCV to $HOME/local_opencv

```bash
module load CMake/3.18
module load GCC/12.2.0
module load tbb/2021.10.0-GCCcore-12.2.0
module load zlib/1.2.12-GCCcore-12.2.0
module load bzip2/1.0.8-GCCcore-12.2.0
module load XZ/5.2.7-GCCcore-12.2.0

git clone https://github.com/opencv/opencv.git
git -C opencv checkout 4.x
cd opencv
mkdir -p build && cd build

cmake -D CMAKE_BUILD_TYPE=Release \
      -D CMAKE_INSTALL_PREFIX=~/local_opencv \
      ..

cmake --build . --parallel 4
cmake --install .
```

## install punkst
git clone it to your project directory. not home directory.

```bash
module load CMake/3.18
module load GCC/12.2.0
module load tbb/2021.10.0-GCCcore-12.2.0
module load zlib/1.2.12-GCCcore-12.2.0
module load bzip2/1.0.8-GCCcore-12.2.0
module load XZ/5.2.7-GCCcore-12.2.0

export OpenCV_DIR=$HOME/local_opencv/lib64/cmake/opencv4
# 1) Clone the repository
git clone --recursive https://github.com/Yichen-Si/punkst.git
cd punkst
git submodule update --init
# 2) Create and enter a build directory
mkdir build && cd build
# 3) Configure
cmake .. \
  -DOpenCV_DIR=${OpenCV_DIR} \
  -DENABLE_REMOTE_IO=OFF


# 4) Build
cmake --build . --parallel # or make

# 5) check
cd ..
./bin/punkst --help

# 6) add punkst/bin to PATH. 
export
```

## install libpng
git clone the repo to project dir. 
then 
install libpng to $HOME/local_libpng

```bash
ml CMake
ml Autoconf
ml XZ
module load Automake
module load binutils

# Download the libpng source code
wget https://download.sourceforge.net/libpng/libpng-1.6.43.tar.gz
tar -xf libpng-1.6.43.tar.gz
cd libpng-1.6.43

# Configure it to install into a folder called "local_libpng" in your home directory
./configure --prefix=$HOME/local_libpng
make
make install
```

## install spatula

```bash
ml CMake
ml Autoconf
ml XZ
module load Automake
module load binutils

## STEP 1 : CLONE THE REPOSITORY
## clone the current snapshot of this repository
git clone --recursive https://github.com/seqscope/spatula.git

## move to the spatula directory
cd spatula

## STEP 2 : BUILD THE SUBMODULES
## move to the submodules directory
cd submodules

export CFLAGS="-pthread"
export CXXFLAGS="-pthread"
export LDFLAGS="-pthread"
# because the GCC is too new in HPC, we need to do this:
sed -i '1s/^/#include <cstdint>\n/' ./qgenlib/qgenlib/qgen_utils.h
sed -i '1s/^/#include <cstdint>\n/' ./qgenlib/qgenlib/qgen_error.h
sed -i '1s/^/#include <cstdint>\n/' ./qgenlib/qgenlib/dataframe.h 

## build the submodules using build.sh script
sh -x build.sh

## move to the spatula directory
cd ..

## STEP 3 : BUILD SPATULA
## create a build directory
mkdir build
cd build

cmake -DPNG_LIBRARY=$HOME/local_libpng/lib/libpng.so -DPNG_PNG_INCLUDE_DIR=$HOME/local_libpng/include ..
make

# test
# put spatula/bin to PATH

spatula --help
```

# data preparation

require format is introduced in [ficture](https://seqscope.github.io/ficture/format_input/visiumHD/)

use [spatula](https://seqscope.github.io/spatula/tools/convert_sge/#converting-10x-genomics-visium-hd-feature-barcode-matrix) to convert the data from visiumHD or Seq-Scope/NovaScope to the format that FICTURE can use.



### prepare the data for FICTURE
install parquet-tools in a conda env with python 3.10
```bash
conda create -n parquet python=3.10
conda activate parquet
pip install parquet-tools

DATADIR=    #/home/hw568/scratch_pi_xy48/hw568/VisiumHD/binned_outputs
grep -w microns_per_pixel ${DATADIR}/square_002um/spatial/scalefactors_json.json 
parquet-tools csv ${DATADIR}/square_002um/spatial/tissue_positions.parquet \
    | gzip -c > ${DATADIR}/square_002um/spatial/tissue_positions.csv.gz

# check data
gzip -cd ${DATADIR}/square_002um/spatial/tissue_positions.csv.gz | head


output_dir=    #/home/hw568/project_pi_xy48/hw568/FICTURE/spatula_output/VisiumHD_Exp1
mkdir -p $output_dir


ml CMake
ml Autoconf
ml XZ
module load Automake
module load binutils

export CFLAGS="-pthread"
export CXXFLAGS="-pthread"
export LDFLAGS="-pthread"

spatula convert-sge \
    --in-sge ${DATADIR}/square_002um/filtered_feature_bc_matrix \
    --pos ${DATADIR}/square_002um/spatial/tissue_positions.csv.gz \
    --units-per-um 3.652 \
    --colnames-count Count \
    --out-sge $output_dir \
    --out-tsv $output_dir \
    --icols-mtx 1 \
    --exclude-feature-regex '^(BLANK|NegCon|NegPrb|mt-|MT-|Gm\d+$$)'


## Uncomment the following line if you want to exclude commonly ignored features
#   --exclude-feature-regex '^(BLANK|NegCon|NegPrb|mt-|MT-|Gm\d+$$)'
## The following parameters follow the default values (unnecessary to specify)
#   --sge-bcd barcode.tsv.gz \
#   --sge-ftr features.tsv.gz \
#   --sge-mtx matrix.mtx.gz \
#   --icols-mtx 1 \
#   --icol-bcd-barcode 1 \
#   --icol-ftr-id 1 \
#   --icol-ftr-name 2 \
#   --pos-colname-barcode barcode \
#   --pos-colname-x pxl_row_in_fullres \
#   --pos-colname-y pxl_col_in_fullres \
#   --pos-delim , \
#   --colname-x X \
#   --colname-y Y \
#   --colname-feature-name gene 


(gzip -cd $output_dir/transcripts.unsorted.tsv.gz \
    | head -1; gzip -cd $output_dir/transcripts.unsorted.tsv.gz \
    | tail -n +2 | sort -S 1G -gk1) \
    | gzip -c > $output_dir/transcripts.sorted.tsv.gz


gzip -cd $output_dir/transcripts.sorted.tsv.gz | head
```

### run punkst
https://yichen-si.github.io/punkst/workflows/#generic-input-format-and-example-data

steps:
1. copy punkst/examples/basic/config.json and edit it.
2. generate a Makefile
3. exectutes the workflow

```bash
module load CMake/3.18
module load GCC/12.2.0
module load tbb/2021.10.0-GCCcore-12.2.0
module load zlib/1.2.12-GCCcore-12.2.0
module load bzip2/1.0.8-GCCcore-12.2.0
module load XZ/5.2.7-GCCcore-12.2.0

ml miniconda
conda activate ficture

####################
# Mannually edit config.json to point to the right paths
#
CONFIG_PATH=
####################

repopath=     #/home/hw568/project_pi_xy48/hw568/punkst
python ${repopath}/ext/py/generate_workflow.py \
  -c ${CONFIG_PATH} -o run.sh -m Makefile \
  -t ${repopath}/examples/basic/Makefile

make -f Makefile --dry-run
make -f Makefile

``` 

### celltype annotation
*.info.html provide these for each factor(cluster):
- TopGene_pval
- TopGene_specific
- TopGene_fc
- TopGene_weight 
Use AI/cell type atlases to annotate the cell types.
