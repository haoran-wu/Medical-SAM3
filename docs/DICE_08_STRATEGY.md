# Strategy to Push VisiumHD Exp1 Region Dice Toward 0.8

Current best local results:

| label | best current signal | Dice |
|---|---|---:|
| tumor | FICTURE factor 0 | 0.512 |
| stroma | FICTURE factor 1 | 0.459 |
| immune_infiltration | FICTURE factor 4 | 0.395 |

Base SAM3 box refinement helps some component boundaries, but it does not solve
semantic region discovery by itself. Medical-SAM3 is more conservative and was
worse on this slide. Therefore the path to 0.8 Dice should make molecular/spatial
domain modeling the main model and use SAM only as a boundary proposal/refiner.

## Literature signals

- FICTURE supports using high-resolution molecular factors directly as the main
  tissue-state evidence. It is designed for segmentation-free submicron spatial
  transcriptomics and reports strength in difficult fibrotic, vascular, muscular
  and complex tissue regions.
- SpaGCN and related spatial-domain methods combine gene expression, spatial
  location and histology through graph convolution or spatial regularization.
  This is closer to the current task than pure SAM prompting.
- BANKSY frames the task as spatially aware cell typing/domain segmentation. It
  suggests smoothing molecular states with neighborhood context rather than
  treating each pixel/component independently.
- iIMPACT is especially relevant because it combines image and molecular profiles
  and uses spatial/statistical domain modeling. The key idea to borrow is not
  the exact implementation but the multi-stage structure: image profile +
  molecular profile + spatial MRF/domain model.
- MedSAM shows that promptable medical segmentation works best when a target is
  already specified by a strong prompt; it is not a substitute for semantic
  pathology-region discovery.

## Main proposal

Build a supervised/weakly-supervised region model with these inputs:

1. H&E RGB patch.
2. FICTURE factor channels, preferably soft factor proportions if available.
3. Molecular marker channels for each target label.
4. Spatial coordinates and local neighborhood statistics.
5. Optional SAM/base-SAM masks as boundary candidates, not as labels.

The model should output label probability maps for tumor, stroma,
immune_infiltration and background/other. Post-process with graph cut or CRF
using H&E edge strength and FICTURE factor smoothness.

## Experiment ladder

### E1. Strong non-neural baseline

Goal: determine how far we can get without training a deep model.

- Make per-label probability maps from FICTURE factors:
  - tumor: F0 plus tumor marker score.
  - stroma: F1 plus smooth-muscle/stromal markers.
  - immune: F4, with explicit penalty for F6 pigment/macrophage regions unless
    immune markers dominate.
- Add spatial smoothing with MRF/graph cut:
  - unary = molecular label score.
  - pairwise = penalize label changes unless H&E edge is strong.
  - enforce minimum component size and hole filling.
- Tune thresholds on small validation regions.

Expected: may lift factor-level Dice from 0.4-0.5 to roughly 0.55-0.7 if the
main error is ragged boundaries and fragmented factors.

### E2. Patch-level segmentation model

Goal: learn the mapping from H&E + molecular maps to GeoJSON labels.

- Train a lightweight U-Net / DeepLabV3+ / SegFormer on tiles.
- Input channels: RGB + one-hot/soft FICTURE factors + marker maps.
- Loss: Dice + CE + boundary loss.
- Sampling: oversample positive regions and hard negatives near confusing
  boundaries.
- Validation: split by spatial blocks, not random pixels, to avoid leakage.

Expected: this is the most realistic path to 0.8 on the current labeled slide,
especially for tumor/stroma. It needs careful validation because same-slide
tile splits can overestimate generalization.

### E3. SAM as proposal generator only

Goal: avoid relying on SAM semantics.

- For each molecular component, generate several prompts:
  - tight bbox.
  - expanded bbox.
  - centroid positive points.
  - positive points plus negative points outside molecular support.
- Run base SAM3.
- Rank masks by:
  - molecular support inside mask.
  - penalty for covering conflicting factors.
  - H&E boundary agreement.
  - learned classifier score from E2 if available.

Expected: improves boundaries for components already selected correctly; unlikely
to reach 0.8 alone.

### E4. Image-expression retrieval/ranker

Goal: replace marker heuristics with learned retrieval.

- Train a patch encoder using H&E patch -> expression/factor vector.
- Start from pathology encoders such as UNI/CONCH/GigaPath if available.
- Use contrastive loss between image patches and nearby expression profiles.
- Use retrieval/ranker output to select label-consistent molecular components
  before SAM refinement.

Expected: helps component selection, especially immune vs pigment/macrophage
confusions. Ranking alone still needs a segmentation backend.

## Most likely route to Dice 0.8

The practical path is E1 -> E2 -> E3:

1. Use FICTURE to generate cleaner pseudo-labels and probability maps.
2. Train a small segmentation model on H&E + molecular channels.
3. Use SAM/base-SAM only to provide boundary alternatives.
4. Select final masks with molecular + image + boundary scoring.

If we allow supervised learning on the existing GeoJSON masks, Dice 0.8 is
plausible for tumor and stroma. Immune_infiltration will need either better
annotation granularity or stronger marker/factor disambiguation, because current
F4 only reaches 0.395 at full-factor level and components cover only parts of
the target.

## Immediate next run

Implement E1 first:

- Export dense per-label molecular probability maps.
- Add graph-cut/CRF refinement using H&E gradients.
- Sweep thresholds and pairwise smoothness on the three labels with existing GT.
- Compare:
  - raw factor mask.
  - cleaned morphology mask.
  - graph-cut mask.
  - graph-cut + SAM boundary candidate.

This will tell us whether 0.8 is reachable mostly through post-processing or
whether we need E2 supervised training.
