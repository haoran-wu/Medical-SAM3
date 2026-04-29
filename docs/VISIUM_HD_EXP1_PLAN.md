# Visium HD Exp1 Plan

This is Line 2: try a separate Visium HD dataset with GeoJSON annotations and
an H&E image.

## Data

HPC directory:

```text
/nfs/roberts/project/pi_xy48/hw646/Exp1
```

Known key files:

```text
/nfs/roberts/project/pi_xy48/hw646/Exp1/VisiumHD_Exp1.geojson
/nfs/roberts/project/pi_xy48/hw646/Exp1/spatial/tissue_hires_image.png
```

Additional files confirmed in `Exp1`:

```text
binned_outputs/square_002um/filtered_feature_bc_matrix.h5
binned_outputs/square_008um/filtered_feature_bc_matrix.h5
binned_outputs/square_016um/filtered_feature_bc_matrix.h5
binned_outputs/square_002um/spatial/scalefactors_json.json
binned_outputs/square_008um/spatial/scalefactors_json.json
binned_outputs/square_016um/spatial/scalefactors_json.json
Visium_HD_Human_Lung_Cancer_HD_Only_Experiment1_feature_slice.h5
Visium_HD_Human_Lung_Cancer_HD_Only_Experiment1_molecule_info.h5
```

Initial inspection:

```text
tissue_hires_image.png: RGB, 3524 x 6000
GeoJSON size: 51 MB
GeoJSON type: FeatureCollection
GeoJSON features: 3821
Geometry types: Polygon, MultiPolygon
GeoJSON label field: properties.region
GeoJSON full-res coordinate bounds: x 59..25370, y 200..32142
tissue_hires_scalef: 0.13752006
```

The GeoJSON coordinates are full-resolution Space Ranger coordinates. To render
on `tissue_hires_image.png`, multiply GeoJSON coordinates by
`tissue_hires_scalef`.

Region label counts:

| Region | Count |
|---|---:|
| stroma | 1281 |
| erythorocytes | 1265 |
| tumor | 693 |
| immune infiltration | 494 |
| pigment | 74 |
| lung vessels | 9 |
| Lung Bronchiola | 4 |
| lung alveoli (normal adjacent) | 1 |

## Working Plan

### Stage 1: Annotation To Masks

Goal: make region masks and overlays from `VisiumHD_Exp1.geojson`.

Inputs:

```text
VisiumHD_Exp1.geojson
spatial/tissue_hires_image.png
binned_outputs/square_008um/spatial/scalefactors_json.json
```

Outputs:

```text
output/visium_hd_exp1/geojson_overlays/
output/visium_hd_exp1/region_masks/
output/visium_hd_exp1/region_summary.json
```

Notes:

- use `properties.region` as the label key
- transform coordinates by `tissue_hires_scalef`
- group polygons by region label
- save one binary mask per region and one combined colored overlay

### Stage 2: SAM3 / MedicalSAM3 Oracle Prompt Baseline On New Sample

Goal: test whether SAM3 or MedicalSAM3 can recover expert tissue-region masks
under favorable prompt conditions.

Current result summary:

- The main completed baseline is tile-based multipoint prompting.
- Positive points are sampled from expert GeoJSON-derived masks.
- Negative points are sampled near the region bounding box but outside the mask.
- The same expert mask is used as the evaluation target.
- This is an oracle-style baseline, not an automatic recognition experiment.

Outputs:

```text
output/visium_hd_exp1/final_sam3_medicalsam3_overlays/
output/visium_hd_exp1/sam3_runs/base_sam3_multipoint/
output/visium_hd_exp1/sam3_runs/base_sam3_multipoint_remaining_dev/
output/visium_hd_exp1/sam3_runs/medical_sam3_multipoint/
output/visium_hd_exp1/sam3_runs/medical_sam3_multipoint_remaining_dev/
```

Key finding:

Even with oracle multipoint prompts sampled from expert annotations, SAM3 and
MedicalSAM3 only partially recover pathology-defined tissue regions. SAM3 tends
to oversegment with higher recall and lower precision, while MedicalSAM3 tends
to be more conservative with lower recall. This suggests the bottleneck is the
segmentation backbone / task formulation rather than simply the prompt modality.

Detailed results and interpretation:

```text
docs/EXP1_SAM3_ORACLE_BASELINE_AND_DIRECTION_UPDATE.md
```

### Stage 3: Expression Matrix And Spatial Labels

Goal: connect Visium HD bins to region labels.

Start with 8um bins:

```text
binned_outputs/square_008um/filtered_feature_bc_matrix.h5
binned_outputs/square_008um/spatial/tissue_positions.parquet
binned_outputs/square_008um/spatial/scalefactors_json.json
```

Why 8um first:

- lighter than 2um
- enough spatial resolution for H&E patches
- easier to debug expression/image alignment

Needed exports:

```text
output/visium_hd_exp1/exported_expression/bin_metadata_008um.csv
output/visium_hd_exp1/exported_expression/expression_008um_top_genes.npz
output/visium_hd_exp1/exported_expression/bin_region_labels_008um.csv
```

Each bin should eventually have:

```text
barcode/bin_id
full-res x/y
hires x/y
region label
gene expression vector
```

### Stage 4: Gene Expression Methods On Exp1

Once Stage 1-3 are stable, run the more novel methods. Based on the SAM3
oracle-prompt baseline, these methods should emphasize retrieval, localization,
and proposal ranking rather than assuming expression prompts can directly make
SAM3 produce accurate pixel-level semantic masks.

1. Gene expression to prompt space
   - high-risk as a direct `expression -> SAM prompt -> mask` route
   - better framed as `expression -> region query / proposal score`

2. H&E patch and expression alignment
   - crop H&E patches around 8um bins or aggregated local neighborhoods
   - train image encoder and expression encoder with contrastive loss
   - retrieve or localize image regions from gene-expression queries

3. Expression exemplar
   - compute mean expression profile for labels such as tumor/stroma/immune
   - use expression profile as query to find matching image patches or proposals

## Immediate Checks

- inspect `Exp1` file tree
- inspect GeoJSON geometry types, labels, and coordinate bounds
- inspect image size
- check for expression matrix files such as:
  - `filtered_feature_bc_matrix.h5`
  - `filtered_feature_bc_matrix/`
  - `matrix.mtx.gz`
  - `features.tsv.gz`
  - `barcodes.tsv.gz`
  - `spatial/tissue_positions*`
- render GeoJSON labels over `tissue_hires_image.png`

Status:

- file tree inspected
- image size inspected
- GeoJSON label field and region counts inspected
- expression H5 files confirmed
- tissue positions parquet exists, but the current HPC Python environment lacks
  `pyarrow`; use an environment with `pyarrow`, R/arrow, or install a small local
  dependency before reading parquet.

## Local Code Area

```text
inference/visium_hd_exp1/
```

Planned files:

```text
inspect_exp1_files.py
inspect_geojson.py
render_geojson_overlay.py
export_exp1_expression.py
run_exp1_sam3_prompts.py
```

Existing related script:

```text
inference/render_visiumhd_geojson_masks.py
```

This can either stay where it is for now or later be moved into
`inference/visium_hd_exp1/` once the Exp1 path and coordinate transform are
confirmed.

Recommended near-term file additions:

```text
inference/visium_hd_exp1/inspect_exp1_files.py
inference/visium_hd_exp1/render_geojson_masks.py
inference/visium_hd_exp1/export_008um_expression.py
inference/visium_hd_exp1/run_exp1_region_prompts.py
inference/visium_hd_exp1/summarize_exp1_results.py
```

## Local Output Area

```text
output/visium_hd_exp1/
```

Planned outputs:

```text
file_inspection/
geojson_overlays/
exported_expression/
sam3_runs/
```

## How This Differs From Line 1

Line 1 uses existing TMA/Xenium work and adds gene-expression prompts to the
current SAM3 workflow. Line 2 starts with a separate Visium HD dataset and tries
to build a cleaner image/annotation/expression pipeline from scratch.
