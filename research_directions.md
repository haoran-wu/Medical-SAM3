# Research Directions: ST + H&E Segmentation

## Direction 1 — CONCH/PLIP as Text Encoder
**Gene expression needed:** No

Replace Medical-SAM3's text encoder with a pathology-specific vision-language model (CONCH or PLIP), which are pretrained on histology image-text pairs and understand terms like "alveoli", "granuloma", "stroma". Current text prompts fail because the default text encoder has no histology prior. This is zero-shot, no training required.

**Data needed:** H&E image + text labels (GeoJSON annotation labels are sufficient)

---

## Direction 2 — Gene Expression → Prompt Space
**Gene expression needed:** Yes

Train a lightweight MLP/adapter to map per-spot gene expression vectors into SAM's prompt embedding space. Instead of deriving a text or box prompt, the model receives a molecular signal directly as its "instruction" for what to segment. Bypasses text as an intermediate representation entirely.

**Data needed:** H&E image + per-spot gene expression matrix + spatial coordinates

---

## Direction 3 — Cross-Modal Alignment (H&E ↔ Gene Expression)
**Gene expression needed:** Yes

Learn a joint embedding space between H&E image patches and gene expression vectors from corresponding spatial locations. Once aligned, a gene expression query can retrieve matching image regions, and vice versa. This is a representation learning problem that enables downstream conditioning.

**Data needed:** H&E image + per-spot gene expression matrix + spatial coordinates

---

## Direction 4 — Gene Expression as Exemplar
**Gene expression needed:** Yes

SAM3 supports exemplar-based prompting (PerSAM-style): give a reference image patch, find all similar regions. Replace the image patch exemplar with a gene expression vector — given a cell type's typical gene expression profile, the model searches the image for morphologically matching regions. Reuses SAM3's existing exemplar pathway, only changes the input modality.

**Data needed:** H&E image + per-spot gene expression matrix + cell type labels

---

## Summary

| Direction | Gene Expression | Training Required | Novelty |
|---|---|---|---|
| 1. CONCH/PLIP text encoder | No | No (zero-shot) | Low |
| 2. Gene expression → prompt space | Yes | Yes (small MLP) | High |
| 3. Cross-modal alignment | Yes | Yes (contrastive) | High |
| 4. Gene expression as exemplar | Yes | Yes (small adapter) | High |

## Datasets

- **TMA24 (Xenium):** Single-cell ST, ~15k cells, 3 annotated labels (granuloma border, mixed alveoli, hyalinized granuloma). Raw gene expression sparse matrix availability TBD.
- **VisiumHD Human Lung (Exp1):** Per-bin ST at high spatial resolution, 8 annotated region types (tumor, stroma, immune infiltration, etc.). Gene expression matrix likely available under `filtered_feature_bc_matrix/`.
