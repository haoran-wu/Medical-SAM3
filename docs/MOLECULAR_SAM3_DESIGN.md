# Molecularly Guided SAM3 for VisiumHD Histopathology

Date: 2026-05-10

## Current checkpoint

Current accepted FICTURE-to-H&E registration:

```bash
python3 inference/visium_hd_exp1/render_ficture_factor_overlay_on_he.py \
  --output-dir output/visium_hd_exp1/ficture_molecular_he_overlay_shifted_dx-60_dy80 \
  --shift-x-px -60 \
  --shift-y-px 80
```

The FICTURE PNG is treated as a factor-assignment grid, not pasted as an image.
Factor pixels are decoded with `hex_12.k12.pixel.info.tsv`, mapped to H&E hires
space, shifted by `dx=-60, dy=80`, clipped to tissue, then redrawn on H&E.

## Literature map

### 1. SAM + CLIP mask selection

SaLIP uses SAM segment-everything mode to produce many part masks, crops the
image by those masks, uses CLIP with visually descriptive text prompts to select
the target ROI, and feeds the retrieved ROI box back to SAM for final
segmentation.

Related papers:

- SaLIP: Test-Time Adaptation with SaLIP, CVPRW 2024.
  https://openaccess.thecvf.com/content/CVPR2024W/DEF-AI-MIA/html/Aleem_Test-Time_Adaptation_with_SaLIP_A_Cascade_of_SAM_and_CLIP_CVPRW_2024_paper.html
- MedCLIP-SAM: text/image bridging for universal medical segmentation.
  https://papers.miccai.org/miccai-2024/498-Paper2311.html
- SAM for digital pathology zero-shot evaluation: useful for large connected
  structures, weaker for dense instance tasks and multi-scale WSI settings.
  https://2023.midl.io/papers/s080
- WSI-SAM: adapts SAM for whole-slide multi-resolution pathology.
  https://proceedings.mlr.press/v254/liu24a.html

Takeaway for this project:

SaLIP is image-language driven: it asks CLIP to choose a semantic ROI from
SAM-generated visual crops. Our setting has stronger information than CLIP text:
spatial transcriptomics and FICTURE factors can provide molecularly grounded
region priors before SAM3 refinement.

### 2. Spatial transcriptomics and molecular factor maps

FICTURE is directly relevant because it produces pixel-level spatial factors from
high-resolution spatial transcriptomics without requiring cell segmentation. This
is a strong prior for complex regions such as tumor, stroma, vasculature,
fibrosis, immune niches, and mixed epithelial states.

Related papers:

- FICTURE, Nature Methods 2024.
  https://www.nature.com/articles/s41592-024-02415-2
- IAMSAM, Genome Biology 2024: uses SAM for morphology-based ROI selection in
  spatial transcriptomics downstream analysis.
  https://genomebiology.biomedcentral.com/articles/10.1186/s13059-024-03380-x
- Vispro, Genome Biology 2025: Visium image processing and segmentation support.
  https://genomebiology.biomedcentral.com/articles/10.1186/s13059-025-03648-w

Takeaway for this project:

Most existing ST+SAM work uses SAM to help select morphology ROIs and then does
omics analysis. Our direction flips this: use omics/FICTURE to propose or rank
histology regions, then use SAM3 to refine boundaries.

### 3. H&E to spatial transcriptomics representation learning

Several works predict or retrieve spatial gene expression from H&E. This supports
our plan to learn an image-expression embedding or ranker.

Related papers:

- ST-Net, Nature Biomedical Engineering 2020: predicts local gene expression
  from H&E patches in breast cancer.
  https://www.nature.com/articles/s41551-020-0578-x
- Hist2ST, Briefings in Bioinformatics 2022: combines CNN, Transformer, and GNN
  modules for spatial expression prediction.
  https://academic.oup.com/bib/article/23/5/bbac297/6645485
- Benchmarking spatial gene expression prediction from histology, Nature
  Communications 2025: compares eleven methods and emphasizes generalizability,
  translational utility, and computational efficiency.
  https://www.nature.com/articles/s41467-025-56618-y
- OmiCLIP/Loki, Nature Methods 2025: visual-omics foundation model aligning H&E
  patches with transcriptomics "sentences"; supports annotation, retrieval, cell
  type decomposition, and expression prediction.
  https://www.nature.com/articles/s41592-025-02707-1
- FineST, Nature Communications 2026: contrastive histology-ST integration for
  nuclei-resolved analysis.
  https://www.nature.com/articles/s41467-026-70528-7

Takeaway for this project:

The field already supports the premise that H&E morphology and spatial gene
expression can be aligned. Our novelty should not be "predict genes from H&E";
it should be "use molecular alignment as a prompt/proposal/ranking signal for
SAM3 segmentation."

### 4. Pathology foundation models

For image encoders and text/image retrieval baselines, pathology-specific models
are better motivated than generic CLIP.

Related papers:

- CONCH, Nature Medicine 2024: pathology vision-language foundation model for
  classification, segmentation, captioning, and image-text retrieval.
  https://www.nature.com/articles/s41591-024-02856-4
- UNI, Nature Medicine 2024: general-purpose self-supervised pathology model.
  https://www.nature.com/articles/s41591-024-02857-3
- Prov-GigaPath, Nature 2024: whole-slide foundation model pretrained on
  1.3B pathology tiles.
  https://www.nature.com/articles/s41586-024-07441-w
- Virchow, Nature Medicine 2024: large computational pathology foundation model.
  https://www.nature.com/articles/s41591-024-03141-0

Takeaway for this project:

Use generic CLIP/SaLIP as an image-only baseline, but use GigaPath/UNI/CONCH as
the serious pathology image encoders for retrieval/ranking.

## Proposed system structure

The central idea:

```text
Molecular prior from VisiumHD/FICTURE/expression
        +
Histology morphology from H&E/pathology encoder
        +
SAM3/Medical-SAM3 boundary refinement
        =
Molecularly grounded segmentation proposal ranking/refinement
```

This should be framed as molecularly guided test-time prompting/ranking for
pathology segmentation, not as another SAM+CLIP cascade.

## Method family A: FICTURE-prior SAM3, no training

This is the fastest and most defensible next experiment.

Pipeline:

```text
FICTURE factor grid
  -> H&E-space factor masks using dx=-60, dy=80
  -> connected components per factor
  -> boxes/points/mask prompts for SAM3
  -> SAM3 refined masks
  -> rank by factor weight, component size, tissue overlap, and marker genes
```

Outputs:

```text
factor_masks_he/
factor_components/
factor_annotation.csv
candidate_components.csv
sam3_refined_components/
ranked_sam3_molecular_candidates.csv
```

Why it matters:

This directly tests whether molecular regions provide better SAM3 prompts than
blind dense grid proposals.

Primary evaluation:

- Component recall against expert GeoJSON labels.
- Dice/IoU after SAM3 refinement.
- Boundary F1 before versus after SAM3.
- Per-factor biological consistency with expected region labels.

Baselines:

- SAM3 dense point/grid proposals.
- FICTURE mask only, without SAM3 refinement.
- GeoJSON bbox oracle prompts.
- SaLIP-like image-only mask selection if feasible.

## Method family B: Molecular ranker over SAM3 proposals

This separates proposal generation from molecular selection.

Pipeline:

```text
SAM3 dense proposals on H&E
  -> crop each proposal region
  -> pathology encoder embedding
  -> expression/FICTURE query embedding
  -> rank proposals by image-molecular similarity
```

Possible query embeddings:

- Factor-level top genes from `hex_12.k12.pixel.info.tsv`.
- Region-level 8um expression bins pooled inside a candidate.
- Expert label marker sets, e.g. tumor/stroma/vessel/immune/bronchiolar.

Model choices:

- Zero-shot text/image: CONCH or BiomedCLIP with factor top-gene descriptions.
- Trainable contrastive: image crop encoder versus expression encoder.
- Lightweight ranker: concatenate image embedding, factor overlap statistics,
  component geometry, and expression summary.

Why it matters:

This is closest to SaLIP structurally, but replaces generic organ text prompts
with molecular queries and pathology/ST-specific embeddings.

## Method family C: Patch-expression retrieval model

This uses the existing 8um expression export direction.

Training units:

- H&E patch centered at each 8um bin.
- Expression vector or reduced gene program vector.
- Region label from GeoJSON masks, when available.

Losses:

- Supervised label classification for region labels.
- Image-expression contrastive loss.
- Optional gene-program prediction, not full 18k-gene regression at first.

Use at inference:

```text
Candidate mask -> covered bins / sampled patches
  -> pooled image embedding and/or expression-query score
  -> candidate ranking for SAM3 proposals
```

Why it matters:

This builds a reusable model that can score whether an H&E region matches a
molecular phenotype, instead of only using raw FICTURE overlap.

## Method family D: SaLIP-plus-molecular hybrid

This is the clearest comparison to the SaLIP paper.

Pipeline:

```text
SAM3 segment-everything/dense proposals
  -> image crops
  -> CLIP/CONCH scores from text prompts
  -> molecular scores from FICTURE/expression overlap
  -> fused score
  -> top candidates prompt SAM3 refinement
```

Fusion score:

```text
score = w1 * vision_language_score
      + w2 * ficture_factor_overlap
      + w3 * expression_marker_score
      + w4 * morphology_quality_score
      - w5 * background_penalty
```

Why it matters:

This lets us say exactly how we differ from and improve on SaLIP: SaLIP uses
CLIP only; we add molecular evidence as a second, grounded selection channel.

## Recommended immediate implementation

Do these in order:

1. Export H&E-space FICTURE factor masks using the accepted `dx=-60, dy=80`.
2. Split each factor mask into connected components.
3. Add factor annotation from top genes and a provisional biological label.
4. Generate SAM3 prompts from components:
   - bbox prompt,
   - center positive points,
   - optional negative points just outside component,
   - optional prior mask if SAM3 supports it.
5. Run SAM3/Medical-SAM3 refinement for top components.
6. Evaluate against existing expert GeoJSON labels.
7. Only after this works, add trainable image-expression retrieval.

## Working hypothesis

SAM3 is good at drawing boundaries when given a plausible region, but weak at
knowing which pathology/molecular region to segment. FICTURE/expression is good
at locating molecular tissue states, but its boundaries are noisy and tied to
capture/resolution. The useful system is therefore not "SAM3 versus FICTURE";
it is:

```text
FICTURE/expression decides where and what.
SAM3 decides the final visual boundary.
```

## Novelty claim to test

Compared with SaLIP:

- SaLIP is image-only and language-selected.
- This project is image-plus-molecular and expression-selected.
- SaLIP retrieves generic organ masks.
- This project retrieves/ranks molecular pathology niches and refines them with
  SAM3.

Compared with H&E-to-ST prediction:

- Those methods predict expression or retrieve transcriptomics from images.
- This project uses ST/FICTURE to improve segmentation proposal selection and
  mask refinement.

Compared with IAMSAM:

- IAMSAM uses SAM to select morphology ROIs for downstream molecular analysis.
- This project uses molecular maps to select/refine morphology masks.

That direction is specific, testable, and probably the cleanest contribution.
