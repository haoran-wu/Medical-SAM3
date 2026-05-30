# Example Manifest

This manifest explains which examples are current and which are historical.

## Current main example

The current main example is:

```text
examples/current_visium_hd_exp1/
```

Use this as the main storyline:

```text
H&E + official FICTURE paired candidate crop
-> candidate mask pool
-> component-aware union
-> Test1 and Test2 VLM evaluation
```

This example is copied from the May30 detailed report:

```text
output/visium_hd_exp1/final_deliverables/May30_detailed_union_test1_test2_report/
```

## What the current example contains

| Type | Files |
|---|---|
| Paired candidate crop | `examples/current_visium_hd_exp1/crops/bronchiola_good_he_reverse_blur.png`, `examples/current_visium_hd_exp1/crops/bronchiola_good_ficture_reverse_blur.png` |
| Component-aware union | `examples/current_visium_hd_exp1/union/bronchiola_merged_he_ficture_union.png`, `examples/current_visium_hd_exp1/union/vessels_merged_he_ficture_union.png` |
| Full assembly sheets | `examples/current_visium_hd_exp1/union/bronchiola_component_aware_assembly_sheet.png`, `examples/current_visium_hd_exp1/union/vessels_component_aware_assembly_sheet.png` |

## Legacy examples

The following root-level files are old examples and are kept only because old scripts
still use them as defaults:

| File | Status | Notes |
|---|---|---|
| `example1.jpg` | legacy | Older TMA24 / silicosis H&E example |
| `example2.png` | legacy | Older spatialLIBD / prompt-inference example |

Do not use `example1.jpg` or `example2.png` as the main VisiumHD project story.

## Rule going forward

When adding a new example:

1. Put it under `examples/<short_name>/`.
2. Add a README explaining what question the example answers.
3. Update this manifest.
4. Keep old examples marked as `legacy` instead of mixing them with the main result.

