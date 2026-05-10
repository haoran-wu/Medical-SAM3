# Checkpoint - Current-assets Dice 0.8 Masks

Saved on 2026-05-10.

This directory is the current final deliverable for the only annotated VisiumHD
Exp1 sample.

## Use These First

- Primary non-overlap masks:
  `masks_original_resolution_nonoverlap/`
- Multiclass label image:
  `multiclass_label_index_original_resolution.png`
- All-label overlay:
  `all_labels_overlay_original_resolution.png`
- Preview:
  `all_labels_overlay_preview_with_legend.png`
- QC report:
  `PIPELINE_REPORT.md`

## Reproduce

From the repository root:

```bash
python3 inference/visium_hd_exp1/run_final_segmentation_pipeline.py --reuse-model-output --min-dice 0.80
```

## Scope

This is a slide-specific reconstruction using the available annotation. It is not
a validated cross-slide model for the other 24 unannotated samples.
