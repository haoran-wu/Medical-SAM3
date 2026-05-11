# VisiumHD SAM3 Current Result Snapshot

Saved on 2026-05-11.

## Training-assisted diagnostic upper bound

These files use annotation for model/run selection and are not the deployable no-training pipeline:

- `training_assisted_best_labelmap_preview.png`
- `training_assisted_best_overlay_preview.png`
- `training_assisted_best_records.csv`

## Annotation-free proxy output

These files do not use annotation training or Dice-based selection. They map FICTURE/expression factors to the same 8-class output format as the training-assisted visualization:

- `no_training_8class_labelmap_preview.png`
- `no_training_8class_overlay_preview.png`
- `no_training_8class_records.csv`

Current generation script:

- `inference/visium_hd_exp1/build_annotation_free_ficture_sam_8class.py`

Remote source run used for the no-training proxy:

- `/nfs/roberts/project/pi_xy48/hw646/Medical-SAM3/results/visium_hd_exp1/sam3_reference_candidate_refine/he_ficturecoord_quality_top12_hi4096_11240236`

Important caveat: the no-training output is a marker-gene/FICTURE factor proxy, not yet a reliable final 8-class segmentation.
