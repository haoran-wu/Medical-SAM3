# Current Main Result: H&E + FICTURE Candidate Pool, Union, and VLM Tests

This is the current main storyline of the project.

## One-line story

Use a paired H&E + official FICTURE example, build a candidate mask pool, show that
component-aware union can recover disconnected tissue pieces, then evaluate whether
VLMs can understand and rank the candidate masks.

## Main pipeline

```text
Official FICTURE aligned to H&E same ROI
  -> candidate mask pool
  -> paired H&E/FICTURE candidate crop examples
  -> component-aware / precision-aware union
  -> Test1: Cross-Label Tissue Classification
  -> Test2: Same-Class Candidate Mask Retrieval
```

## Compute rule

All formal computation for this storyline should run on Bouchet Slurm compute
nodes unless Haoran explicitly asks for local execution. Do not run formal
Python mask scans, metric recomputation, SAM/CLIP/VLM inference, or large report
jobs on the Bouchet login node. The login node is only for `sbatch`, job/status
checks, log inspection, and small file staging.

Current candidate-pool rescans should use the Bouchet HE and official FICTURE
candidate roots, not a small local exported subset, unless the user explicitly
asks for a subset debug run. Final reported metrics should come from compute-node
runs.

See:

```text
docs/project_organization/COMPUTE_POLICY.md
```

## Step 0: Official FICTURE to H&E alignment

All current FICTURE-dependent results must come from the official filtered
FICTURE map and the same H&E ROI. The current aligned map is not a manual visual
shift. It uses:

- source: filtered FICTURE pixel image `hex_12.k12.pixel.png`.
- orientation: `fliplr(raw filtered PNG)`.
- coordinate mapping: convert FICTURE 2 micron pixel coordinates into H&E hires
  pixels using `microns_per_pixel = 0.2737554241192739` and
  `tissue_hires_scalef = 0.13752006`.
- no manual dx/dy shift.
- official crop bbox: `[75, 40, 3219, 3367]`.
- ROI size: `3144 x 3327`.
- status check: `output/visium_hd_exp1/ficture_official_filtered_he_aligned/summary_official.json`
  must say `PASS_OFFICIAL`.

The source-matched FICTURE legend and prompt inputs are collected here:

```text
data/visium_hd_exp1/current_ficture_vlm_inputs/
```

For VLM prompts, the FICTURE color legend must be generated from:

```text
data/visium_hd_exp1/current_ficture_vlm_inputs/ficture_factor_prompt_legend_from_html.csv
```

That file is extracted directly from:

```text
data/visium_hd_exp1/current_ficture_vlm_inputs/source_matched_factor_info_with_llm_inferred_celltypes.html
```

So the prompt uses the same `RGB`, `Major Compartment`, and `cell type` shown in
the source-matched HTML.

## Step 1: H&E + FICTURE paired example

The current example uses both:

- H&E pathology crop.
- Official PASS_OFFICIAL FICTURE crop from the same ROI.

For VLM testing, the candidate is shown as a paired crop:

- Image 1: H&E gray reverse-blur crop.
- Image 2: official FICTURE gray reverse-blur crop.

The candidate region stays sharp and in color; outside context is grayscale and blurred.

The clean input bundle for current VLM reruns is:

```text
data/visium_hd_exp1/current_ficture_vlm_inputs/
```

It includes the source-matched FICTURE HTML legend, the HTML-extracted CSV factor
legend used in the prompt, the 90-row candidate crop table, and symlinks to the
crop images. The prompt legend should stay aligned to
`source_matched_factor_info_with_llm_inferred_celltypes.html`.

The current example bundle is:

```text
examples/current_visium_hd_exp1/
```

Older TMA24 and spatialLIBD examples now live under `examples/legacy_examples/`
and are not the current main storyline.

## Step 2: Candidate pool

The candidate pool contains SAM/Medical-SAM3 candidate masks generated from H&E and
official FICTURE inputs. The purpose is to provide many possible masks for each tissue
class, then test whether a downstream method can select or combine the useful ones.

## Step 3: Component-aware union

Some annotations, especially bronchiola and vessels, are not one connected object.
They have multiple separated tissue pieces. A single best mask often captures only
the largest piece, which limits recall.

Component-aware union does this:

1. Split the manual annotation into connected components.
2. Find the best candidate mask for each important component.
3. Union those selected masks.
4. Compare the union mask with the annotation using Dice, Precision, and Recall.

Current headline results:

- bronchiola merged H&E + FICTURE union: Dice 0.887, Precision 0.881, Recall 0.892.
- vessels merged H&E + FICTURE union: Dice 0.891, Precision 0.855, Recall 0.930.

For tumor, stroma, and immune infiltration, the policy is different. These
annotations can have many connected components, but that does not mean every
piece should be unioned. The goal is not "recover every small component"; the
goal is to build a final mask with better Precision while keeping useful Recall.

Use precision-aware subset selection:

1. Start from important components only, such as the largest components that
   cover most of the annotation.
2. For each component, test candidate masks from the H&E and official FICTURE
   pools.
3. Add a candidate only if its new pixels have high enough Precision and the
   final union does not lose too much Precision.
4. Stop when adding more components no longer improves the precision-weighted
   score.

The reusable policy is documented in:

```text
docs/project_organization/PRECISION_AWARE_COMPONENT_UNION.md
```

For the next Test1/Test2 runs, the model should not receive raw single masks
from the original pool when a final union mask exists. The current six-class
test-input rule is:

| Class | Final mask sent to tests |
|---|---|
| bronchiola | merged H&E + FICTURE component-aware union |
| alveoli | one single-best mask |
| vessels | merged H&E + FICTURE component-aware union |
| tumor | FICTURE single best + FICTURE precision-aware component union |
| stroma | FICTURE single best + HE precision-aware component union |
| immune infiltration | HE single best + HE precision-aware component union + one FICTURE C2 recall-boost component |

The current preview bundle is:

```text
data/visium_hd_exp1/current_ficture_vlm_inputs/final_union_test_inputs/
```

The readable preview report is:

```text
output/visium_hd_exp1/final_deliverables/Jun01_final_union_test_inputs_preview/index.html
```

## Step 4: Test1

Test1 is called:

```text
Cross-Label Tissue Classification
```

Question:

```text
Does the VLM recognize what tissue type the candidate is?
```

For each candidate, the VLM returns six scores:

- bronchiola
- alveoli
- vessels
- tumor
- stroma
- immune infiltration

The predicted tissue class is the class with the highest score.

## Step 5: Test2

Test2 is called:

```text
Same-Class Candidate Mask Retrieval
```

Question:

```text
Within one tissue class, can the VLM rank better masks higher?
```

The same six-class scores are reused. For example:

- bronchiola candidates are ranked by the bronchiola score.
- vessels candidates are ranked by the vessels score.

Main metrics:

- `Top1 GOOD`: whether the highest-ranked candidate is a hidden GOOD mask.
- `GOOD in Top5`: how many hidden GOOD masks appear among the top five.
- `Unique scores`: whether the model used enough distinct scores or just tied many candidates.

## Current primary report

Use this HTML as the main entrypoint:

```text
/Users/haoranwu/Desktop/Yan_Lab_Research/Medical-SAM3/output/visium_hd_exp1/final_deliverables/May30_detailed_union_test1_test2_report/index.html
```
