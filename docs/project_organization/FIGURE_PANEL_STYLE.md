# Figure Panel Style Guide

Use this default panel style for VisiumHD Exp1 segmentation, candidate-pool,
component-union, and before/after visualizations.

## Default Format

When Haoran asks for a figure, visualization, union comparison, or "补上和没补
对比", use the official one-row six-panel layout unless he explicitly asks for a
different layout.

Each row must contain these six panels in this order:

1. `Annotation on H&E`
2. `Final test mask on H&E`
3. `Final test mask only`
4. `Annotation mask only`
5. `H&E ROI`
6. `FICTURE ROI`

Keep all six panels the same visual size and aligned in one horizontal row.
Avoid ad hoc multi-box or local zoom-only layouts as the primary figure.

## Before/After Comparisons

For "补上 vs 没补", use two stacked rows with the same six-panel layout:

- Row 1: current / not boosted / before.
- Row 2: boosted / targeted / after.

The row header should include:

- class name.
- method/rule.
- source.
- Dice, Precision, Recall.
- one short plain-language note explaining what changed.

## Color Convention

Use stable colors:

- annotation overlay: green.
- candidate/final mask overlay: blue.
- H&E ROI and FICTURE ROI: unmodified source images.

Do not use blue-highlight-only crops as the default summary figure, because
FICTURE already contains strong blue/cyan colors.

## Current Example

The current canonical example is:

```text
output/visium_hd_exp1/final_deliverables/Jun01_final_union_test_inputs_preview/assets/immune_infiltration_final_test_input_panel.png
```

The current before/after example is:

```text
output/visium_hd_exp1/final_deliverables/Jun01_immune_recall_fullpool_stream_top100/targeted_left_lower_top8_rank15/immune_current_vs_targeted_two_row_official_panel.png
```
