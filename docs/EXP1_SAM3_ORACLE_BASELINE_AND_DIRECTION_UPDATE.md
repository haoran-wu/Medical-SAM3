# Exp1 SAM3 Oracle Baseline And Direction Update

Date: 2026-04-29

This note records the current interpretation of the Exp1 SAM3/MedicalSAM3
results and how they change the next research direction.

## Context

Dataset:

```text
/nfs/roberts/project/pi_xy48/hw646/Exp1
```

Key inputs:

```text
VisiumHD_Exp1.geojson
spatial/tissue_hires_image.png
binned_outputs/square_008um/filtered_feature_bc_matrix.h5
```

The GeoJSON annotations were rasterized into 8 per-region binary masks on the
`tissue_hires_image.png` coordinate system. The correct coordinate transform is:

```text
x_hires = x_geojson * tissue_hires_scalef
y_hires = y_geojson * tissue_hires_scalef
```

No y-axis flip is used.

## Experiment: Oracle Multipoint Prompt Baseline

This experiment evaluates SAM3 and MedicalSAM3 under favorable prompt
conditions.

For each region label and each image tile:

1. Use the expert GeoJSON-derived mask as annotation `A`.
2. Sample positive points from inside `A`.
3. Sample negative points near the bounding box but outside `A`.
4. Run SAM3 or MedicalSAM3 with these points.
5. Stitch tile predictions into a whole-image mask `M`.
6. Compare `M` against `A` using Dice, IoU, precision, and recall.

This is therefore an oracle-style baseline:

```text
known annotation A -> points sampled from A -> predicted mask M -> compare M to A
```

It is not an automatic recognition experiment and should not be presented as
the model independently discovering tissue regions. It asks a more basic
question:

> Even with favorable points sampled from expert annotations, can SAM3 or
> MedicalSAM3 recover the annotated pathology-defined tissue region?

Tile inference settings:

| Setting | Value |
|---|---:|
| H&E image size | 3524 x 6000 px |
| tile size | 1024 x 1024 px |
| tile overlap | 128 px |
| total tiles | 28 |
| positive points per prompted tile | 10 |
| negative points per prompted tile | 5 |

## Results

Final local outputs:

```text
output/visium_hd_exp1/final_sam3_medicalsam3_overlays/
```

Key files:

```text
overview/exp1_sam3_vs_medicalsam3_all_8_panels.png
overview/exp1_sam3_vs_medicalsam3_all_8_prediction_overlays.png
tables/exp1_sam3_medicalsam3_multipoint_metrics.csv
tables/exp1_sam3_medicalsam3_multipoint_metrics.md
```

Related note:

```text
docs/EXP1_TILE_SIZE_ABLATION.md
```

| Label | SAM3 Dice | MedicalSAM3 Dice | SAM3 IoU | MedicalSAM3 IoU | SAM3 Precision | MedicalSAM3 Precision | SAM3 Recall | MedicalSAM3 Recall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Lung Bronchiola | 0.080 | 0.470 | 0.041 | 0.307 | 0.050 | 0.947 | 0.203 | 0.313 |
| erythorocytes | 0.074 | 0.089 | 0.038 | 0.046 | 0.039 | 0.062 | 0.666 | 0.152 |
| immune infiltration | 0.071 | 0.162 | 0.037 | 0.088 | 0.038 | 0.126 | 0.722 | 0.229 |
| lung alveoli (normal adjacent) | 0.229 | 0.053 | 0.129 | 0.027 | 0.177 | 0.612 | 0.324 | 0.028 |
| lung vessels | 0.273 | 0.350 | 0.158 | 0.212 | 0.167 | 0.358 | 0.752 | 0.342 |
| pigment | 0.002 | 0.042 | 0.001 | 0.022 | 0.001 | 0.023 | 0.117 | 0.210 |
| stroma | 0.382 | 0.225 | 0.236 | 0.127 | 0.259 | 0.307 | 0.726 | 0.178 |
| tumor | 0.351 | 0.146 | 0.213 | 0.079 | 0.239 | 0.267 | 0.658 | 0.101 |

## Interpretation

### What The Metrics Mean

Dice and IoU both measure overlap between the predicted mask and the expert
mask. Dice is the common segmentation score and is usually numerically higher
than IoU. IoU is stricter because it measures intersection over union.

Precision asks:

```text
Of the pixels predicted as this region, how many are actually in the expert mask?
```

Recall asks:

```text
Of the expert-mask pixels, how many did the model recover?
```

Thus:

- high recall and low precision means oversegmentation
- high precision and low recall means undersegmentation
- high Dice/IoU requires a better precision-recall balance

### Main Finding

SAM3 and MedicalSAM3 behave differently:

- SAM3 is aggressive: recall is often high, but precision is low.
- MedicalSAM3 is conservative: precision is often higher, but recall drops.

Examples:

- SAM3 has higher Dice on `stroma`, `tumor`, and `lung alveoli`.
- MedicalSAM3 has higher Dice on `Lung Bronchiola`, `lung vessels`,
  `immune infiltration`, `erythorocytes`, and `pigment`.
- MedicalSAM3 achieves very high precision for `Lung Bronchiola` but low recall,
  meaning it finds a small high-confidence part rather than the full region.

The important conclusion is that even under oracle multipoint prompts, neither
model reliably recovers the expert semantic tissue regions.

## Why The Oracle Baseline Is Still Weak

The weak upper-bound result suggests the main bottleneck is the segmentation
backbone and task formulation, not just the prompt type.

Likely reasons:

1. Pathology region labels are semantic tissue concepts, not clean object masks.
   SAM-style models are better at object-like boundaries than at labels such as
   `stroma`, `tumor`, or `immune infiltration`.

2. GeoJSON annotations are often fragmented, interleaved, and defined by expert
   pathology criteria. SAM-style region expansion tends to produce more
   continuous visual masks.

3. Tile-based inference introduces square artifacts. Each 1024 px tile is
   predicted independently and stitched back, so oversegmentation can appear as
   block-like regions.

4. Multipoint prompts are still sparse for large irregular tissue regions.
   Ten positive points and five negative points per tile are favorable but not
   enough to fully specify complex semantic boundaries.

5. MedicalSAM3 may be medical-domain tuned, but its training masks may not match
   these VisiumHD lung cancer semantic region annotations.

6. Increasing tile size from `1024` to `1536` does not fix the problem. Dice
   decreases while recall tends to rise, which suggests that larger context is
   mostly encouraging broader expansion rather than more accurate semantic
   segmentation.

## Consequence For The Project

This baseline should be described as a sanity check / oracle upper-bound-ish
experiment, not as a final recognition method.

Recommended wording:

> We performed an oracle-style multipoint prompt baseline by sampling positive
> and negative points from expert GeoJSON annotations. Even under these
> favorable prompt conditions, SAM3 and MedicalSAM3 only partially recovered
> pathology-defined tissue regions, suggesting a mismatch between SAM-style
> object segmentation and semantic histopathology region segmentation.

This result motivates replacing or supplementing SAM-style segmentation rather
than only improving prompts.

## Updated ABC Directions

### Direction A: Gene Expression To Prompt Space

Original idea:

```text
gene expression vector -> SAM prompt embedding -> mask
```

Updated view:

This remains exploratory but high risk as the main path. If oracle point prompts
already fail, gene-expression prompts alone are unlikely to make SAM3 produce
accurate pixel-level region masks.

Better use:

```text
gene expression -> region query / proposal score / coarse localization
```

Use gene expression to decide what biological state to search for, then combine
it with a stronger segmentation or proposal model.

### Direction B: Cross-Modal Alignment

This direction remains useful and is now more central.

Goal:

```text
H&E patch embedding <-> gene expression embedding
```

Expected outputs:

- gene pattern to H&E region retrieval
- H&E patch to expression profile retrieval
- region heatmaps from molecular queries
- morphology-expression correspondence analysis

This does not require SAM to draw perfect masks. It tests whether histology and
spatial transcriptomics share a usable embedding space.

### Direction C: Gene Expression As Exemplar

Original idea:

```text
cell type or region expression profile -> exemplar prompt -> SAM mask
```

Updated view:

Use expression exemplars primarily for retrieval and ranking:

```text
reference expression profile -> retrieve similar image patches/proposals
```

Then optionally use a segmentation model only to visualize or refine the
selected regions.

## Updated Recommended Pipeline

The strongest next direction is:

1. Keep SAM3/MedicalSAM3 oracle prompt results as a baseline showing the
   limitation of promptable object segmentation.
2. Test pathology-specific segmentation or panoptic models such as CellViT /
   HistoLytics / HoVerNet-style models.
3. Train lightweight models with frozen pathology foundation encoders rather
   than training large segmentation models from scratch.
4. Use gene expression for retrieval, localization, and proposal re-ranking
   rather than expecting it to directly solve pixel-level SAM prompting.

Practical target:

```text
segmentation/proposal model -> candidate regions
gene expression alignment model -> score or retrieve biologically matched regions
```

This is more realistic than:

```text
gene expression -> SAM prompt -> accurate semantic tissue mask
```

## Sample Size Consideration

With around 25 TMAs, training a large segmentation model from scratch is risky.

More realistic options:

- frozen pathology foundation encoder + small classifier head
- frozen encoder + lightweight segmentation decoder
- H&E patch classification / localization
- cross-modal retrieval
- proposal ranking using expression and image embeddings

The correct validation split should be held-out TMA / held-out patient, not
random patch split, because patches from the same section are highly correlated.
