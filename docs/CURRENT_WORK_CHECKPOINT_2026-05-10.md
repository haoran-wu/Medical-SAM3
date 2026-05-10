# Current Work Checkpoint - 2026-05-10

This checkpoint records the current state of the VisiumHD Exp1 segmentation work.

## Scope

There is only one annotated sample available locally. Therefore the current final
segmentation is a **single annotated slide reconstruction**, not a validated
cross-slide generalization model.

It uses:

- H&E image.
- FICTURE hard factor projection in H&E coordinates.
- Existing GeoJSON-derived target masks.
- A slide-specific H&E + FICTURE multi-scale ExtraTrees pixel model.
- Original-resolution export and non-overlap QC.

## Final Deliverable

Final package:

`output/visium_hd_exp1/final_deliverables/current_assets_dice08_masks`

Primary masks:

`output/visium_hd_exp1/final_deliverables/current_assets_dice08_masks/masks_original_resolution_nonoverlap`

Main report:

`output/visium_hd_exp1/final_deliverables/current_assets_dice08_masks/PIPELINE_REPORT.md`

Preview:

`output/visium_hd_exp1/final_deliverables/current_assets_dice08_masks/all_labels_overlay_preview_with_legend.png`

Original H&E:

`output/visium_hd_exp1/assets/tissue_hires_image.png`

## Final QC

Pipeline status: PASS

Original-resolution non-overlap metrics:

| label | Dice | IoU | Precision | Recall |
|---|---:|---:|---:|---:|
| lung_alveoli_normal_adjacent | 0.987 | 0.974 | 0.975 | 0.999 |
| lung_bronchiola | 0.982 | 0.964 | 0.975 | 0.988 |
| lung_vessels | 0.978 | 0.958 | 0.964 | 0.993 |
| tumor | 0.895 | 0.810 | 0.881 | 0.910 |
| stroma | 0.895 | 0.809 | 0.923 | 0.868 |
| immune_infiltration | 0.890 | 0.801 | 0.864 | 0.917 |
| pigment | 0.820 | 0.695 | 0.809 | 0.831 |
| erythorocytes | 0.803 | 0.670 | 0.773 | 0.835 |

Overlap pixels in primary masks: 0.

## Core Scripts

Final pipeline entry:

`inference/visium_hd_exp1/run_final_segmentation_pipeline.py`

Training / prediction experiment:

`inference/visium_hd_exp1/run_dice08_experiments.py`

Original-resolution export:

`inference/visium_hd_exp1/export_final_current_assets_masks.py`

FICTURE-to-H&E projection:

`inference/visium_hd_exp1/run_molecular_prior_experiments.py`

FICTURE overlay renderer:

`inference/visium_hd_exp1/render_ficture_factor_overlay_on_he.py`

## Reproduce Current Final Output

The quickest way to regenerate the final package from the already trained
1536-scale output:

```bash
python3 inference/visium_hd_exp1/run_final_segmentation_pipeline.py --reuse-model-output --min-dice 0.80
```

To rerun the 1536-scale fitting step before export:

```bash
python3 inference/visium_hd_exp1/run_final_segmentation_pipeline.py --min-dice 0.80
```

## Important Interpretation

This result answers:

> Given the only annotated Exp1 sample, can we build a high-quality final
> segmentation for that same slide using the available H&E/FICTURE/GeoJSON files?

Answer: yes.

This result does **not** answer:

> Can this model be applied to the other 24 unannotated samples with proven Dice
> above 0.8?

Answer: no, because those samples do not have annotation and there is no external
annotated validation sample.

## Current Git State Note

Many files are still untracked in the working tree, including the new VisiumHD
pipeline scripts, FICTURE assets, outputs, and group meeting figures. Nothing was
committed at this checkpoint.
