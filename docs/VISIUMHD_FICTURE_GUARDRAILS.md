# VisiumHD FICTURE Guardrails

These checks are mandatory before submitting or reporting any FICTURE-dependent
candidate pool, ranking, VLM judge, or direct-grounding result for VisiumHD Exp1.

## Correct FICTURE Source

- Use the direction-corrected, filtered FICTURE ROI image:
  `output/visium_hd_exp1/ficture_from_folder_he_direction_corrected/ficture_from_folder_he_direction_corrected_roi.png`
- The corrected candidate-pool input directory is:
  `output/visium_hd_exp1/ficture_corrected_candidate_pool_input_roi/`
- Do not use the continuous redraw from `hex_12.k12.results.tsv` as the final
  FICTURE ranking image unless it has been explicitly revalidated against the
  corrected source. In particular, treat these older inputs/results as suspect:
  `output/visium_hd_exp1/ficture_coord_scaled_hires_continuous_candidate_pool_input_roi/`
  and
  `results/visium_hd_exp1/ficture_coord_scaled_hires_candidate_pool/`.

## Pre-submit Checks

Before submitting any dependent job:

1. Verify source path points to `ficture_corrected_candidate_pool_input_roi`.
2. Verify FICTURE ROI and matching H&E ROI have identical dimensions.
3. Visually verify orientation against the H&E ROI before reporting figures.
4. Verify background/no-tissue speckles are removed by comparing against the
   corrected folder-derived FICTURE map.
5. If any check fails, do not submit ranking/VLM jobs; regenerate the corrected
   candidate input first.

## Reporting Rules

- Old FICTURE-dependent rankings from the continuous redraw are reference-only,
  not final results.
- New bests must state the corrected FICTURE source and job id.
- Figures shown to the user must be visually inspected first for orientation,
  size match, and absence of the old unfiltered background artifacts.
