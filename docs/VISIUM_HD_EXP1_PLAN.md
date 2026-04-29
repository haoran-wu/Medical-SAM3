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

