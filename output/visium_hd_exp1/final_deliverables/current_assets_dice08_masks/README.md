# Current-assets Dice 0.8 Mask Deliverable

Original H&E shape: `6000 x 3524` pixels.

## Original-resolution metrics

Primary masks are non-overlapping masks exported from `multiclass_label_index_original_resolution.png`.

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

## Important Scope

This is the best current deliverable using only files already present locally: H&E, FICTURE hard factor image/metadata, and existing GeoJSON-derived target masks.
It is suitable as an Exp1 slide reconstruction/mask package. It is not claimed as a validated cross-slide model.
