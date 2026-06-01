# Precision-Aware Component Union Policy

This document records the current rule for component-aware assembly in VisiumHD
Exp1 so the project does not drift back to "union every component".

## Why This Exists

Some tissue annotations are disconnected. A single best SAM candidate can miss
small but real pieces. This is clearly useful for bronchiola and vessels, where
the annotation has a few meaningful pieces and one-mask oracle bests are recall
limited.

Tumor, stroma, and immune infiltration are different. They may have many
connected components, but many are tiny, noisy, or difficult to isolate. For
these classes, blindly taking one candidate per component can increase false
positive area and lower Precision. The target is therefore a precision-aware
union, not a full component recovery.

## Definitions

- Candidate mask: one SAM or Medical-SAM3 proposed mask from H&E or official
  FICTURE.
- Annotation: manual ground-truth mask used only for evaluation.
- Precision: among pixels predicted by the candidate, how many are actually in
  the annotation.
- Recall: among annotation pixels, how many are covered by the prediction.
- Dice: overlap score balancing Precision and Recall.

## Class Policy

| Tissue class | Current union policy | Reason |
|---|---|---|
| bronchiola | Component-aware recovery | Few meaningful pieces; single-best mask misses smaller real pieces. |
| vessels | Component-aware recovery with precision gates | Several meaningful pieces; single-best mask is recall-limited. |
| tumor | Precision-aware subset selection | Many components; best single already has high recall but low Precision. |
| stroma | Precision-aware subset selection | Broad fragmented tissue; adding every piece risks many false positives. |
| immune infiltration | Recall-push subset selection with Precision floor | Many small clusters; use the full candidate pool to recover more real clusters, but keep a Precision floor so noisy pieces do not dominate. |

## Selection Rule

For tumor, stroma, and immune infiltration:

1. Split the annotation into connected components.
2. Keep only important components for the search, for example the largest
   components or components covering at least a meaningful area fraction.
3. For each important component, scan candidate masks from both H&E and official
   FICTURE pools.
4. Compute component-level Precision/Recall/Dice, plus incremental Precision:
   how clean the newly added candidate pixels are after considering the masks
   already selected.
5. Add a candidate only when:
   - component Precision passes a threshold,
   - incremental Precision is high enough,
   - overlap with already selected masks is not excessive,
   - final union Precision stays above the chosen floor or does not drop beyond
     a small tolerance,
   - and a precision-weighted score, such as F0.5 or Dice with a Precision
     guardrail, improves.
6. Stop when no remaining component candidate improves the precision-aware
   objective.

## Practical Defaults

These are starting thresholds for the next full-pool rerun, not fixed biological
truth:

| Class group | Min component Precision | Min incremental Precision | Max Precision drop | Objective |
|---|---:|---:|---:|---|
| bronchiola/vessels | 0.70 | 0.70 | 0.03 | recover real pieces while keeping Precision |
| tumor/stroma | 0.55 | 0.60 | 0.02 | improve or preserve Precision first |
| immune infiltration | 0.45 | 0.55 | 0.02 | keep only cleaner immune clusters |

For immune infiltration, the current reported mask uses a full-pool scan on
Bouchet rather than the older local top-k subset:

```text
HE pool: 15,899 masks
FICTURE pool: 15,571 masks
Selection: top40 annotation components, rank<=25 candidate rows, recall-push rule
Result: Dice 0.509, Precision 0.438, Recall 0.607
```

The reusable implementations are:

```text
inference/visium_hd_exp1/componentwise_candidate_assembly.py
inference/visium_hd_exp1/stream_component_candidate_oracle.py
inference/visium_hd_exp1/search_recall_boost_from_component_candidates.py
```

Current policy presets:

- `precision_focused`: stricter policy for bronchiola/vessels-like components.
- `balanced`: looser component-recovery policy.
- `precision_broad_tissue`: starting policy for tumor and stroma.
- `precision_immune`: starting policy for immune infiltration.
- `recall_oracle_all`: diagnostic only; this keeps nearly everything and should
  not be interpreted as the final method for broad fragmented classes.

The script writes `incremental_precision` and `union_precision_after` for each
accepted component candidate. These are the key fields for deciding whether
adding that component helped or hurt the final union.

## What To Report

For every class, report:

- selected candidate count, not just connected-component count.
- final Dice, Precision, and Recall.
- whether the selected union improved Precision over the best single candidate.
- which candidates were skipped by the precision gate.

## Important Guardrail

Do not interpret "many connected components" as automatic evidence that component
assembly should use all of them. Component assembly is only useful if the unioned
mask improves the final segmentation objective. For tumor, stroma, and immune
infiltration, the main objective is higher Precision with acceptable Recall.
