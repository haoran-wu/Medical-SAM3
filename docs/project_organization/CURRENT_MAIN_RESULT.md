# Current Main Result: H&E + FICTURE Candidate Pool, Union, and VLM Tests

This is the current main storyline of the project.

## One-line story

Use a paired H&E + official FICTURE example, build a candidate mask pool, show that
component-aware union can recover disconnected tissue pieces, then evaluate whether
VLMs can understand and rank the candidate masks.

## Main pipeline

```text
Same ROI H&E + official FICTURE
  -> candidate mask pool
  -> paired H&E/FICTURE candidate crop examples
  -> component-aware union
  -> Test1: Cross-Label Tissue Classification
  -> Test2: Same-Class Candidate Mask Retrieval
```

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

It includes the source-matched FICTURE HTML legend, the CSV factor legend used in
the prompt, the 90-row candidate crop table, and symlinks to the crop images. The
prompt legend should stay aligned to
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
