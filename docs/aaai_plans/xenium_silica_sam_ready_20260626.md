# Xenium Silica SAM-Ready H&E/FICTURE/GT Inputs

Date: 2026-06-26

## Current Local Artifact

The current SAM-ready visual QC report is saved locally at:

`output/aaai_xenium_silica_20260623/tma_sample_overview_ficturedriven_highres_20260626/index.html`

The current high-resolution H&E crop root is:

`output/aaai_xenium_silica_20260623/highres_he_crops_ficturedriven_20260626`

The current molecule-level FICTURE root is:

`output/aaai_xenium_silica_20260623/xenium_true_molecule_ficture_sync_20260626`

The current SAM input bundle is:

`output/aaai_xenium_silica_20260623/xenium_hdstyle_sam_inputs_highres_ficturedriven_20260626`

These local output folders are intentionally not committed because they contain
large image artifacts.

## Status

This version is ready to use as the input basis for the next SAM candidate
generation step.

## Inputs

- H&E: full-resolution H&E TIFF crops from the original slide images.
- FICTURE: molecule-level Xenium transcript FICTURE, K=12, using the official
  cmap48 color table.
- GT annotation: evaluation-only ground truth from original
  `Xenium_Silica/with_annotation/TMA*.csv` `new_annotation` rows, matched to
  Xenium `cells.parquet` and `cell_boundaries.parquet`.
- SAM runtime bundle: `source_manifest.csv` contains only `he` and
  `ficture_molecule_k12`; old low-resolution small-core H&E and deprecated
  cell-proxy FICTURE variants are excluded.

## Crop Policy

Each TMA crop is driven by:

`union(old manual H&E crop, K=12 molecule-level FICTURE non-black footprint) + guard margin`

The FICTURE image is coordinate-warped into the expanded H&E canvas. It is not
blindly resized.

## GT Display Policy

The HTML report has two collapsible GT sections per TMA:

- original annotation: raw cell-level annotation points on H&E and blank mask;
- continuous region annotation: same original annotation after nearest-cell
  matching, cell-boundary fill, small closing radius 8 px, and hole filling.

The continuous GT does not use a final dilation step.

## SAM Input Policy

The high-resolution SAM input bundle contains:

- `source_manifest.csv`: 20 runtime source images, two per TMA
  (`he`, `ficture_molecule_k12`).
- `label_manifest.csv`: 36 hidden annotation masks. `dense_mask_rel` is the
  current continuous-region GT, while `point_mask_rel` is the original sparse
  annotation-point GT.
- `sam_array_tasks.csv`: 520 candidate-generation tasks using the current
  small-core prompt settings.

The expected scoring command should pass:

`--source-keys he ficture_molecule_k12 --annotation-policy dense`

Dedicated Bouchet launch scripts:

- `scripts/hpc/xenium_silica_highres_ficturedriven_sam_candidate_array.sbatch`
  writes candidate masks to
  `xenium_hdstyle_sam_candidates_highres_ficturedriven_20260626`.
- `scripts/hpc/xenium_silica_highres_ficturedriven_component_score_array.sbatch`
  scores `he + ficture_molecule_k12` against hidden continuous GT.

## QC Checks Passed

- 10 annotated TMAs are present.
- H&E, FICTURE, original GT, and continuous GT dimensions match per TMA.
- New SAM input bundle has 20 runtime source images, 36 hidden GT masks, and
  520 SAM array tasks.
- HTML image/link references resolve.
- GT provenance points to original annotation CSVs plus `cells.parquet` and
  `cell_boundaries.parquet`.
- Deprecated low-resolution/proxy and over-connected GT outputs are archived
  under `output/aaai_xenium_silica_20260623/_archive_*`.

## Rebuild Script

Main report script:

`scripts/tools/build_xenium_silica_ficture_driven_highres_overview.py`

SAM input bundle script:

`scripts/xenium_prepare_ficturedriven_highres_sam_inputs.py`

Bouchet SAM/scoring scripts:

- `scripts/hpc/xenium_silica_highres_ficturedriven_sam_candidate_array.sbatch`
- `scripts/hpc/xenium_silica_highres_ficturedriven_component_score_array.sbatch`
