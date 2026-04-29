# Gene Expression Prompt Plan

This is Line 1: keep the previous TMA/SAM3 workflow, but add Xenium gene
expression as a new prompt or query modality.

## Data

HPC Seurat object:

```text
/nfs/roberts/project/pi_xy48/hw646/nucXensFiltered.Sept5.Seurat.Robj
```

Inspected equivalent object contents:

- object name: `nucXensFiltered`
- class: `Seurat`
- cells: `831,791`
- features: `479`
- default assay: `SCT`
- raw/normalized Xenium assay: `Xenium`
- metadata includes spatial coordinates and sample fields

Important matrices:

```r
GetAssayData(nucXensFiltered, assay = "Xenium", slot = "counts")
GetAssayData(nucXensFiltered, assay = "Xenium", slot = "data")
GetAssayData(nucXensFiltered, assay = "SCT", slot = "data")
```

## Methods

### A. Gene Expression To Prompt Space

Train a small MLP/adapter:

```text
gene expression vector -> SAM/SAM3 prompt embedding
```

First deliverable:

- export cell metadata and expression matrix
- build a toy adapter over selected genes
- test whether expression-conditioned prompts can recover known pseudo-mask regions

### B. Cross-Modal Alignment

Train two encoders into one embedding space:

```text
H&E patch encoder <-> gene expression encoder
```

First deliverable:

- crop H&E patches around cells/spots
- pair each patch with its expression vector
- train contrastive alignment
- retrieve image regions from gene-expression queries

### C. Gene Expression As Exemplar

Use a cell type or cluster average expression profile as an exemplar query:

```text
cell type expression profile -> image regions matching that molecular phenotype
```

First deliverable:

- confirm usable labels from `seurat_clusters`, `renamed1`, `renamed2`, or other metadata columns
- compute average expression profile by label
- test retrieval/segmentation against image regions

## Local Code Area

```text
inference/gene_expression_prompt/
```

Planned files:

```text
inspect_xenium_seurat.R
export_xenium_expression.R
train_expression_prompt_mlp.py
train_cross_modal_alignment.py
run_expression_exemplar.py
```

## Local Output Area

```text
output/gene_expression_prompt/
```

Planned outputs:

```text
xenium_schema/
exported_expression/
expression_prompt_mlp/
cross_modal_alignment/
expression_exemplar/
```

