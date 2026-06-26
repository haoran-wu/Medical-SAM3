# Medical-SAM3

Medical SAM3: A Foundation Model for Universal Prompt-Driven Medical Image Segmentation

## Setup

```bash
# Create conda environment
conda create -n medsam3 python=3.10
conda activate medsam3

# Install dependencies
pip install -r requirements.txt

# Install SAM3 from source
git clone https://github.com/facebookresearch/sam3.git
pip install -e sam3/
```

## Evaluation

### Base SAM3

```bash
python run_evaluation.py
```

### MedSAM3

```bash
python run_medsam3_evaluation.py \
    --checkpoint /path/to/checkpoint.pt \
    --model-name medsam3
```

**Options:** `--max-samples N`, `--datasets "Dataset1,Dataset2"`

## TMA24 Prompt Experiments

The TMA24 experiments are indexed in `TMA24_EXPERIMENTS.md`.

Whole-image text/box prompt test:

```bash
python run_tma24_whole_image_prompts.py \
    --checkpoint ../checkpoints/Medical-SAM3/checkpoint.pt
```

Left-to-right transfer test:

```bash
python run_tma24_left_to_right_transfer.py \
    --checkpoint ../checkpoints/Medical-SAM3/checkpoint.pt
```

## Single-image spatialLIBD inference

This repo also includes a single-image inference entrypoint for the current
`spatialLIBD` workflow.

Default input image:

- `../data/spatialLIBD/151673/tissue_hires_image.png`

Full-resolution source image:

- `../data/spatialLIBD/151673/151673_full_image.tif`

Prompt list:

- `prompts/spatiallibd_dlpfc_layers.txt`

Run:

```bash
python run_spatiallibd_prompts.py
```

Optional custom checkpoint:

```bash
python run_spatiallibd_prompts.py \
    --checkpoint /path/to/checkpoint.pt
```

Notes:

- The default script input is the `2000 x 2000` PNG.
- The full-resolution TIFF is still available when you want the raw source image.
- The default inference setting resizes the longest side to `2048` to keep memory manageable.
- Outputs are written to `../output/spatialLIBD_151673/`.

## Visualization

```bash
cd visualization
python visualize_all_datasets.py
```

## Output

- **Results:** `results/*.csv`, `*.json`, `*.md`
- **Visualizations:** `visualization/{dataset}/comparison_*.png`

## Datasets

**Evaluation data path:** `../medsam_data/`

`../examples/legacy_examples/kvasir_seg/Kvasir-SEG/` is only bundled as example
data for local smoke tests.
