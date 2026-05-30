# Current Main Example: VisiumHD H&E + FICTURE

This is the main example for the current project story.

## What this example shows

```text
Same ROI H&E + official FICTURE
  -> candidate mask pool
  -> paired H&E/FICTURE candidate crops
  -> component-aware union
  -> Test1: Cross-Label Tissue Classification
  -> Test2: Same-Class Candidate Mask Retrieval
```

## Example crop

The candidate region is sharp and in color. The outside context is grayscale and blurred.

| Image | File |
|---|---|
| H&E crop | `crops/bronchiola_good_he_reverse_blur.png` |
| FICTURE crop | `crops/bronchiola_good_ficture_reverse_blur.png` |

## Union examples

These figures show why one mask can be insufficient for disconnected tissue structures.

| Tissue class | File |
|---|---|
| bronchiola merged H&E + FICTURE union | `union/bronchiola_merged_he_ficture_union.png` |
| vessels merged H&E + FICTURE union | `union/vessels_merged_he_ficture_union.png` |
| bronchiola component-aware assembly sheet | `union/bronchiola_component_aware_assembly_sheet.png` |
| vessels component-aware assembly sheet | `union/vessels_component_aware_assembly_sheet.png` |

## Full report

Use this HTML for the complete result:

```text
output/visium_hd_exp1/final_deliverables/May30_detailed_union_test1_test2_report/index.html
```

## Source assets

The files in this example folder are copied from:

```text
output/visium_hd_exp1/final_deliverables/May30_detailed_union_test1_test2_report/assets/
```

