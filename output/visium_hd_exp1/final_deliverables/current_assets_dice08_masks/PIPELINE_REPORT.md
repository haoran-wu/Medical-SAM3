# Final Segmentation Pipeline Report

Status: **PASS**

Model output reused: `True`
Dice threshold: `0.800`
Overlap pixels in primary non-overlap masks: `0`

## Original-resolution non-overlap metrics

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

## Outputs

- Primary masks: `/Users/haoranwu/Desktop/Yan_Lab_Research/Medical-SAM3/output/visium_hd_exp1/final_deliverables/current_assets_dice08_masks/masks_original_resolution_nonoverlap`
- Multiclass label image: `/Users/haoranwu/Desktop/Yan_Lab_Research/Medical-SAM3/output/visium_hd_exp1/final_deliverables/current_assets_dice08_masks/multiclass_label_index_original_resolution.png`
- Original-resolution overlay: `/Users/haoranwu/Desktop/Yan_Lab_Research/Medical-SAM3/output/visium_hd_exp1/final_deliverables/current_assets_dice08_masks/all_labels_overlay_original_resolution.png`
- Preview with legend: `/Users/haoranwu/Desktop/Yan_Lab_Research/Medical-SAM3/output/visium_hd_exp1/final_deliverables/current_assets_dice08_masks/all_labels_overlay_preview_with_legend.png`

## Scope

This is the final Exp1 segmentation package built from the currently available local assets. It is a slide-specific reconstruction, not a validated cross-slide generalization model.
