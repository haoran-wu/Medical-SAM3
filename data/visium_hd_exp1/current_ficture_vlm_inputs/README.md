# Current FICTURE VLM Input Bundle

This folder is the clean input bundle for the current VisiumHD Exp1 VLM tests.
It keeps the FICTURE legend, 90-row crop pool, crop image entrypoint, and prompt
snapshot in one place so Test1 and Test2 do not depend on scattered files.

## What This Bundle Is For

- Test1: Cross-label tissue classification.
- Test2: Same-class candidate mask retrieval, computed from the same six-class
  score outputs.
- Image style: gray reverse-blur paired H&E and FICTURE crops.
- ROI: same official 3144 x 3327 FICTURE/H&E ROI.

## Files

- `source_matched_factor_info_with_llm_inferred_celltypes.html`  
  Human-readable source-matched FICTURE legend. This is the visual reference for
  the prompt factor/cell-type descriptions.
- `ficture_factor_prompt_legend_from_html.csv`  
  Minimal machine-readable prompt legend extracted directly from
  `source_matched_factor_info_with_llm_inferred_celltypes.html`. It contains only
  Factor, RGB, Major Compartment, and cell type, which are the fields shown to
  the VLM.
- `ficture_factor_legend_for_prompt.csv`  
  Full machine-readable legend copied from the same source-matched FICTURE map.
  This is kept for traceability, but current prompt text is generated from the
  smaller HTML-extracted file above.
- `prompt_user_from_source_matched_ficture_legend.txt`  
  Current user prompt snapshot generated from
  `ficture_factor_prompt_legend_from_html.csv`.
- `public_vlm_requests.csv`  
  Clean 90-row candidate input table. It keeps crop paths and evaluation grouping,
  but removes legacy prompt columns so old factor descriptions cannot leak into
  new runs.
- `hidden_candidate_truth.csv`  
  Evaluation-only labels and quality buckets. These are not shown to the model.
- `candidate_pair_crops`  
  Symlink to the paired H&E/FICTURE gray reverse-blur crop images.
- `final_ficture_roi_3144x3327.png`  
  Symlink to the source-matched official FICTURE ROI image.
- `ficture_official_filtered_roi_factor_index.npy`  
  Symlink to the official factor-index array for the same ROI.
- `he_roi_matching_official_ficture_coverage.png`  
  Symlink to the H&E ROI aligned to the official FICTURE coverage.
- `reverse_blur_pool_index.html`  
  Symlink to the visual index for the reverse-blur 90-candidate crop pool.
- `final_union_test_inputs/`  
  Six-class final-mask input bundle for the next Test1/Test2 runs. These are
  the masks that should be sent to tests after applying the current union policy:
  bronchiola uses the merged H&E+FICTURE component union, alveoli uses one
  single-best mask, vessels uses the merged H&E+FICTURE component union,
  tumor/stroma use single-best plus precision-aware component-union masks, and
  immune infiltration uses the Bouchet full-pool HE+FICTURE top40/rank25
  recall-push union.

## Rules

- New final-mask tests should use `final_union_test_inputs/final_test_requests.csv`,
  not the original 90-row raw candidate pool, unless the goal is explicitly to
  re-run the raw candidate-pool evaluation.
- Keep RGB, Major Compartment, and cell type aligned to
  `source_matched_factor_info_with_llm_inferred_celltypes.html`.
- Current runner defaults use `ficture_factor_prompt_legend_from_html.csv`, which
  is extracted from that HTML table.
- Do not use the older `target_factor_hints`, `candidate_factor_composition`, or
  row-level `prompt_text` fields from earlier pool CSVs for new tests.
- `sample_bucket` is allowed in `public_vlm_requests.csv` only for evaluation
  after scoring. It is not included in the prompt.
- Full-pool mask rescans and reported metric recomputation must run on Bouchet
  Slurm compute nodes. Local files in this folder are packaged inputs/previews
  copied back from those remote runs.
- Future GPT/OpenRouter and local VLM runs should use this folder as the default
  input bundle.

## How The 90-Row GOOD/MID/BAD Pool Was Sampled

The 90-row pool is a controlled VLM sanity-test pool, not the final six-class
union input. It has 6 tissue classes x 15 candidates per class:

- 5 `GOOD`: same-class candidates with the highest hidden Dice.
- 5 `MID`: same-class candidates closest to the same-class median hidden Dice.
- 5 `BAD`: candidates that were good enough for another class but have the
  lowest hidden Dice for the current target class.

The bucket labels and Dice/Precision/Recall values are hidden from the model and
used only after scoring. The exact sampling code is:

```text
inference/visium_hd_exp1/build_paired_vlm_hit_test_pool.py
```

The current hidden Dice ranges are:

| Class | GOOD Dice range | MID Dice range | BAD Dice range |
|---|---:|---:|---:|
| bronchiola | 0.753-0.758 | 0.364-0.378 | 0.000 |
| alveoli | 0.604-0.677 | 0.214-0.227 | 0.000 |
| vessels | 0.608-0.627 | 0.337-0.421 | 0.000 |
| tumor | 0.513-0.527 | 0.192-0.203 | 0.000 |
| stroma | 0.522-0.528 | 0.115-0.124 | 0.000 |
| immune infiltration | 0.270-0.297 | 0.149-0.151 | 0.000 |
