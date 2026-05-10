# Weak Transfer Pipeline Checkpoint

Date: 2026-05-10

This checkpoint is for the real data situation: 25 total samples, but only one sample currently has manual annotation. The pipeline therefore has two separate deliverables:

1. A final high-Dice segmentation for the annotated Exp1 sample.
2. A weak-transfer pseudo-segmentation workflow for future unlabeled samples.

It is not valid to claim that a supervised classifier has been trained and validated across 25 samples yet.

## Current Annotated-Sample Result

Final deliverable:

`output/visium_hd_exp1/final_deliverables/current_assets_dice08_masks/`

Original-resolution non-overlap Dice:

| label | Dice |
|---|---:|
| lung_bronchiola | 0.982 |
| erythorocytes | 0.803 |
| immune_infiltration | 0.890 |
| lung_alveoli_normal_adjacent | 0.987 |
| lung_vessels | 0.978 |
| pigment | 0.820 |
| stroma | 0.895 |
| tumor | 0.895 |

Lowest Dice is 0.803. This is the best current mask set for the one annotated slide.

## Weak-Transfer Pipeline

Script:

`inference/visium_hd_exp1/run_weak_transfer_pipeline.py`

Inputs per sample:

- H&E image registered to analysis coordinates.
- FICTURE factor label image already projected into H&E coordinates.
- Annotation is optional. If annotation is absent, the pipeline exports pseudo masks, confidence maps, and review tiles, but cannot report Dice.

Features:

- H&E RGB/HSV/LAB color.
- Gray and Sobel texture.
- FICTURE factor one-hot channels.
- Multi-scale smoothed H&E and factor-context channels.
- No x/y coordinate features by default, so the model is less tied to the single annotated slide layout.

Current best weak-transfer command:

```bash
python3 inference/visium_hd_exp1/run_weak_transfer_pipeline.py \
  --max-side 1536 \
  --n-estimators 260 \
  --max-per-class 75000 \
  --calibrate-thresholds \
  --calibrate-competition \
  --output-dir output/visium_hd_exp1/weak_transfer_pseudoseg_competition_calibrated
```

Current weak-transfer output:

`output/visium_hd_exp1/weak_transfer_pseudoseg_competition_calibrated/`

Annotated-sample metrics for this transfer-oriented mode:

| label | Dice | precision | recall |
|---|---:|---:|---:|
| lung_bronchiola | 0.990 | 0.995 | 0.985 |
| erythorocytes | 0.726 | 0.836 | 0.642 |
| immune_infiltration | 0.778 | 0.959 | 0.655 |
| lung_alveoli_normal_adjacent | 0.926 | 0.912 | 0.940 |
| lung_vessels | 0.987 | 0.985 | 0.988 |
| pigment | 0.759 | 0.820 | 0.706 |
| stroma | 0.855 | 0.937 | 0.785 |
| tumor | 0.868 | 0.842 | 0.896 |

Mean Dice is 0.861. The limiting classes are erythorocytes, pigment, and immune infiltration; these are also the most annotation-sensitive small/patchy classes.

## How To Add The Other 24 Samples

Edit or replace:

`output/visium_hd_exp1/weak_transfer_pseudoseg/sample_manifest.csv`

Expected columns:

```csv
sample_id,he_path,factor_label_path,has_annotation
sample_02,/path/to/sample_02/tissue_hires_image.png,/path/to/sample_02/ficture_factor_label_image_he.npy,False
```

For each unannotated sample, the pipeline will write:

- `pseudo_multiclass_label_index.png`
- `pseudo_overlay.png`
- `confidence_map.png`
- `masks/<label>_pseudo_mask.png`
- `uncertain_review_tiles/`
- `sample_summary.json`

## Interpretation

Use `current_assets_dice08_masks` for the current annotated Exp1 slide.

Use `weak_transfer_pseudoseg_competition_calibrated` as the first pass for unannotated slides. Those outputs should be treated as pseudo-labels plus uncertainty, not as validated ground truth.

The scientifically defensible next step is to run the weak-transfer script on all 24 unlabeled samples, manually correct the uncertain review tiles, then retrain once 3 to 5 additional samples have partial or full labels.
