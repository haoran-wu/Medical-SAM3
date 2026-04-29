# Project Directions

This project is now organized around two active lines.

## Line 1: TMA/Xenium + SAM3 + Gene Expression Prompts

Goal: continue from the existing TMA24/SAM3 workflow, but move beyond text and
box prompts by adding molecular information.

Primary gene-expression object on HPC:

```text
/nfs/roberts/project/pi_xy48/hw646/nucXensFiltered.Sept5.Seurat.Robj
```

Known contents from inspection:

- Seurat object: `nucXensFiltered`
- cells: `831,791`
- genes/features: `479`
- assays: `Xenium`, `BlankCodeword`, `ControlCodeword`, `ControlProbe`, `SCT`
- expression matrix: `Xenium` assay `counts` / `data`
- normalized matrix: `SCT` assay
- spatial/sample metadata: `xCoordinate`, `coordinateX`, `coordinateY`, `TMApunch`, `SampleID`, `PatientID`, `Disease`, `Stage`

Novel method directions:

- A: gene expression to SAM/SAM3 prompt space
- B: cross-modal alignment between H&E patches and gene expression vectors
- C: gene expression as an exemplar query

Local code area:

```text
inference/gene_expression_prompt/
```

Local output area:

```text
output/gene_expression_prompt/
```

## Line 2: Visium HD Exp1

Goal: try a separate Visium HD dataset with explicit GeoJSON annotations and an
H&E image, then decide whether it gives a cleaner setup for region-level
segmentation and expression/image alignment.

Primary data directory on HPC:

```text
/nfs/roberts/project/pi_xy48/hw646/Exp1
```

Known key files:

```text
/nfs/roberts/project/pi_xy48/hw646/Exp1/VisiumHD_Exp1.geojson
/nfs/roberts/project/pi_xy48/hw646/Exp1/spatial/tissue_hires_image.png
```

Next checks:

- inspect GeoJSON labels and coordinate system
- confirm whether expression matrix exists under `Exp1`
- render GeoJSON masks on `tissue_hires_image.png`
- map expression bins/spots to image patches

Local code area:

```text
inference/visium_hd_exp1/
```

Local output area:

```text
output/visium_hd_exp1/
```

## Existing TMA24/SAM3 Baseline

The current SAM3/Medical-SAM3 prompt experiments remain in `inference/` for now
to avoid breaking working command paths. The main index is:

```text
inference/TMA24_EXPERIMENTS.md
```

Current local summary tables:

```text
output/tma24_sam3_results_summary.csv
output/tma24_sam3_results_summary.md
```
