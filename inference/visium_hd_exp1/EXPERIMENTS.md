# VisiumHD Exp1 Experiment Log

Direction 3: Cross-Modal H&E Patch ↔ Gene Expression Contrastive Alignment.

Full plan: `docs/VISIUM_HD_EXP1_PLAN.md` and `.claude/plans/zippy-swinging-coral.md`.

---

## Dataset

| Item | Value |
|---|---|
| HPC path | `/nfs/roberts/project/pi_xy48/hw646/Exp1` |
| H&E image | `spatial/tissue_hires_image.png` — RGB 3524×6000 |
| GeoJSON | `VisiumHD_Exp1.geojson` — 51 MB, 3821 features |
| Bin resolution | 8μm (`binned_outputs/square_008um/`) |
| Expression file | `binned_outputs/square_008um/filtered_feature_bc_matrix.h5` |
| Spatial positions | `binned_outputs/square_008um/spatial/tissue_positions.parquet` |
| Scale factor | `tissue_hires_scalef = 0.13752006` |
| Gene count | **18,085** (Visium HD full transcriptome, not Xenium 479-gene panel) |
| In-tissue bins | **448,109** (at 8μm resolution) |

Region label counts (from Stage 2 output):

| Region | Bins |
|---|---:|
| tumor | 152,941 |
| stroma | 152,765 |
| lung alveoli (normal adjacent) | 29,093 |
| lung vessels | 29,982 |
| immune infiltration | 22,184 |
| Lung Bronchiola | 18,744 |
| erythorocytes | 14,250 |
| pigment | 686 |
| background | 27,464 |

Priority labels for modeling: **tumor**, **stroma**, **immune infiltration**.

---

## Coordinate System Notes

GeoJSON coordinates are full-resolution Space Ranger coordinates.
Transform to `tissue_hires_image.png` pixel space:

```
x_hires = x_geojson * tissue_hires_scalef
y_hires = y_geojson * tissue_hires_scalef
```

**No y-axis flip needed.** Space Ranger uses image coordinates (top-left origin,
y increases downward). Confirmed by visual comparison against
`spatial/geojson.png` (Space Ranger's own annotation overlay).

---

## Scripts

| Script | Purpose |
|---|---|
| `run_exp1_multipoint.py` | SAM3 baseline A: tile-based multi-point prompts |
| `run_exp1_bbox.py` | SAM3 baseline B: whole-region bbox *(not yet written)* |
| `run_whole_region_prompts.py` | (deprioritized) whole-region prompt baseline |
| `run_dense_box_proposals.py` | (deprioritized) dense box proposal baseline |
| `render_geojson_masks.py` | Stage 1: GeoJSON → binary PNG per region label |
| `export_008um_expression.py` | Stage 2: bin metadata CSV + sparse expression NPZ |
| `build_patch_expression_dataset.py` | Stage 3: normalized expression + spatial split + `PatchExpressionDataset` |
| `train_contrastive.py` | Stage 4a: ResNet50 + MLP InfoNCE contrastive training |
| `train_classification.py` | Stage 4b: dual-tower region classification + embedding alignment |
| `evaluate_retrieval.py` | Stage 5: recall@k retrieval + linear probe |

---

## SAM3 Inference Plan

### Why this dataset is different from TMA24

TMA24 had small, compact cell nuclei — bounding boxes are tight and informative. VisiumHD Exp1
has large, irregular tissue regions (tumor/stroma each span hundreds of µm) on a 3524×6000 px
image. This changes the inference strategy significantly:

- **No left-right transfer**: Exp1 is a single continuous tissue section (not a TMA with separate
  cores). The spatial train/val/test split (right 10% = test) already tests spatial generalization.
- **Text prompts probably useless**: SAM3 text-only gave Dice=0.000 on TMA24 H&E. Skip.
- **Single bbox per region is weak**: A region's bounding box contains large amounts of adjacent
  tissue. Multi-point prompts from the polygon interior are much more informative.
- **Image too large for single-pass SAM3**: 3524×6000 downscaled to max-side 2048 loses detail.
  Tile-based inference (1024×1024 tiles with overlap) is the right approach.

### Experiment priority

| Priority | Experiment | Prompt | Models | Script |
|---|---|---|---|---|
| 1 | Multi-point from polygon | K points sampled inside each region polygon + negative points outside | SAM3, Medical-SAM3 | `run_exp1_multipoint.py` |
| 2 | Whole-region bbox | Bounding box of each GeoJSON polygon | SAM3, Medical-SAM3 | `run_exp1_bbox.py` |
| 3 | Expression-guided | Classification embedding → projected point/box prompt | Medical-SAM3 | `run_exp1_expr_prompted.py` |

Left-right transfer and dense sliding-box proposals are deprioritized (low signal-to-effort ratio
for this dataset).

### Tile strategy

All scripts use tile-based inference to avoid quality loss from aggressive downscaling:
- Tile size: 1024×1024 px (hires coordinates)
- Overlap: 128 px (to avoid boundary artifacts)
- Each tile is fed to SAM3 independently; predictions are stitched back with overlap averaging
- Region mask (GT) is also cropped per tile for Dice computation

### Metric

Per-region Dice against GeoJSON ground truth masks (same masks used in Stage 1).
Report mean Dice over tumor, stroma, immune infiltration (the three priority labels).

---

## SAM3 Baseline A: Multi-Point Prompts

**Status: complete, oracle baseline result available** (`run_exp1_multipoint.py`)

For each region label:
1. Load the binary mask (from Stage 1 output)
2. Sample K=10 positive points uniformly from foreground pixels
3. Sample K=5 negative points from the region's bounding box but outside the mask
4. Run SAM3 / Medical-SAM3 tile-by-tile, stitch predictions, compute Dice

Important interpretation: this is an oracle-style baseline. The expert mask is
used both to sample the prompt points and as the evaluation target. It measures
whether SAM3/MedicalSAM3 can recover expert tissue-region annotations under
favorable prompt conditions; it does not measure automatic region discovery.

Final local outputs:

```text
output/visium_hd_exp1/final_sam3_medicalsam3_overlays/
```

Summary:

| Label | SAM3 Dice | MedicalSAM3 Dice | SAM3 Recall | MedicalSAM3 Recall |
|---|---:|---:|---:|---:|
| Lung Bronchiola | 0.080 | 0.470 | 0.203 | 0.313 |
| erythorocytes | 0.074 | 0.089 | 0.666 | 0.152 |
| immune infiltration | 0.071 | 0.162 | 0.722 | 0.229 |
| lung alveoli (normal adjacent) | 0.229 | 0.053 | 0.324 | 0.028 |
| lung vessels | 0.273 | 0.350 | 0.752 | 0.342 |
| pigment | 0.002 | 0.042 | 0.117 | 0.210 |
| stroma | 0.382 | 0.225 | 0.726 | 0.178 |
| tumor | 0.351 | 0.146 | 0.658 | 0.101 |

Interpretation: SAM3 is generally more aggressive with higher recall and lower
precision, while MedicalSAM3 is more conservative. The weak oracle baseline
suggests that the main limitation is the segmentation backbone/task mismatch:
SAM-style object segmentation does not map cleanly onto pathology-defined
semantic tissue regions.

Detailed interpretation:

```text
docs/EXP1_SAM3_ORACLE_BASELINE_AND_DIRECTION_UPDATE.md
```

```bash
# SAM3-base
/home/hw646/.conda/envs/medsam3/bin/python \
  ~/Medical-SAM3/inference/visium_hd_exp1/run_exp1_multipoint.py \
  --image-path /nfs/roberts/project/pi_xy48/hw646/Exp1/spatial/tissue_hires_image.png \
  --masks-dir  ~/Medical-SAM3/output/visium_hd_exp1/region_masks \
  --output-dir ~/Medical-SAM3/output/visium_hd_exp1/sam3_runs/base_sam3_multipoint \
  --checkpoint ~/Medical-SAM3/checkpoints/SAM3-base/sam3.pt \
  --n-pos 10 --n-neg 5 --tile-size 1024 --tile-overlap 128

# Medical-SAM3
/home/hw646/.conda/envs/medsam3/bin/python \
  ~/Medical-SAM3/inference/visium_hd_exp1/run_exp1_multipoint.py \
  --image-path /nfs/roberts/project/pi_xy48/hw646/Exp1/spatial/tissue_hires_image.png \
  --masks-dir  ~/Medical-SAM3/output/visium_hd_exp1/region_masks \
  --output-dir ~/Medical-SAM3/output/visium_hd_exp1/sam3_runs/medsam3_multipoint \
  --checkpoint ~/Medical-SAM3/checkpoints/Medical-SAM3/checkpoint.pt \
  --n-pos 10 --n-neg 5 --tile-size 1024 --tile-overlap 128
```

---

## SAM3 Baseline B: Whole-Region Bbox

**Status: 🔲 Script not yet written** (`run_exp1_bbox.py`)

For each region label, use the bounding box of the full GeoJSON polygon mask as a single
box prompt. Expected to perform worse than multi-point for large irregular regions (tumor/stroma),
but useful as a lower-bound reference.

```bash
# SAM3-base
/home/hw646/.conda/envs/medsam3/bin/python \
  ~/Medical-SAM3/inference/visium_hd_exp1/run_exp1_bbox.py \
  --image-path /nfs/roberts/project/pi_xy48/hw646/Exp1/spatial/tissue_hires_image.png \
  --masks-dir  ~/Medical-SAM3/output/visium_hd_exp1/region_masks \
  --output-dir ~/Medical-SAM3/output/visium_hd_exp1/sam3_runs/base_sam3_bbox \
  --checkpoint ~/Medical-SAM3/checkpoints/SAM3-base/sam3.pt \
  --tile-size 1024 --tile-overlap 128

# Medical-SAM3
/home/hw646/.conda/envs/medsam3/bin/python \
  ~/Medical-SAM3/inference/visium_hd_exp1/run_exp1_bbox.py \
  --image-path /nfs/roberts/project/pi_xy48/hw646/Exp1/spatial/tissue_hires_image.png \
  --masks-dir  ~/Medical-SAM3/output/visium_hd_exp1/region_masks \
  --output-dir ~/Medical-SAM3/output/visium_hd_exp1/sam3_runs/medsam3_bbox \
  --checkpoint ~/Medical-SAM3/checkpoints/Medical-SAM3/checkpoint.pt \
  --tile-size 1024 --tile-overlap 128
```

---

## Direction 3 Payoff: Expression-Guided Prompts

**Status: reframed after oracle SAM3 baseline** (`run_exp1_expr_prompted.py`)

Once the classification model is trained, the expression encoder maps each 8µm bin's expression
vector → 128-dim embedding. This embedding is projected (small MLP adapter) into SAM3's prompt
space to generate a point or box prompt without any spatial annotation. This is the core
contribution: segmentation driven purely by gene expression.

Updated view: direct `expression -> SAM prompt -> accurate semantic tissue mask`
is high-risk because even oracle multipoint prompts only partially recover the
expert masks. Expression-guided methods should first be framed as retrieval,
localization, and proposal re-ranking:

```text
segmentation/proposal model -> candidate regions
gene expression model -> retrieve, score, or rank biologically matched regions
```

Design (to be finalized after seeing Stage 4b results):
- For each tile, identify which expression bins fall inside it (use `items.csv` hires_x/y)
- Run expression encoder on those bins → embeddings
- Project to SAM3 prompt token space via lightweight adapter (trained on Stage 4b embeddings)
- Run SAM3 with projected prompts, compute Dice

---

## Stage 1: GeoJSON → Region Masks

**Status: ✅ Complete**

```bash
/home/hw646/.conda/envs/medsam3/bin/python \
  ~/Medical-SAM3/inference/visium_hd_exp1/render_geojson_masks.py \
  --image-path /nfs/roberts/project/pi_xy48/hw646/Exp1/spatial/tissue_hires_image.png \
  --geojson-path /nfs/roberts/project/pi_xy48/hw646/Exp1/VisiumHD_Exp1.geojson \
  --output-dir ~/Medical-SAM3/output/visium_hd_exp1/region_masks \
  --label-key region \
  --scalef 0.13752006
```

Outputs on HPC:
```
~/Medical-SAM3/output/visium_hd_exp1/region_masks/
  tumor.png
  stroma.png
  immune_infiltration.png
  lung_alveoli_normal_adjacent.png
  lung_vessels.png
  lung_bronchiola.png
  erythorocytes.png
  pigment.png
  overlays/
  region_summary.json
```

---

## Stage 2: Export 8μm Bin Metadata + Expression

**Status: ✅ Complete**  (SLURM job 10123016, ran 2026-04-29)

```bash
sbatch --job-name=exp1_export --mem=32G --cpus-per-task=4 --time=1:00:00 \
  --output=~/Medical-SAM3/output/visium_hd_exp1/export_stage2_%j.log \
  --wrap="/home/hw646/.conda/envs/medsam3/bin/python \
    ~/Medical-SAM3/inference/visium_hd_exp1/export_008um_expression.py \
    --data-dir /nfs/roberts/project/pi_xy48/hw646/Exp1 \
    --masks-dir ~/Medical-SAM3/output/visium_hd_exp1/region_masks \
    --output-dir ~/Medical-SAM3/output/visium_hd_exp1/exported_expression"
```

Outputs on HPC:
```
~/Medical-SAM3/output/visium_hd_exp1/exported_expression/
  bin_metadata_008um.csv   — 47 MB, 448109 rows
                             columns: barcode, in_tissue, fullres_x, fullres_y,
                                      hires_x, hires_y, region_label
  expression_008um.npz     — 619 MB (sparse CSR)
                             keys: data, indices, indptr, shape, gene_names, barcodes
```

---

## Stage 3: Build Patch+Expression Dataset

**Status: ✅ Complete** (SLURM job 10123373, ran 2026-04-29)

**Bug found and fixed (2026-04-29):** `PatchExpressionDataset.__getitem__` was using positional
index `self.X[idx]` to look up expression rows, but after filtering by split the positional index
no longer matched the original matrix row. Fixed by preserving `item_idx` as the DataFrame index
and using `self.X[self.df.index[idx]]` instead.

```bash
sbatch --job-name=exp1_dataset --mem=64G --cpus-per-task=4 --time=1:00:00 \
  --output=~/Medical-SAM3/output/visium_hd_exp1/stage3_%j.log \
  --wrap="/home/hw646/.conda/envs/medsam3/bin/python \
    ~/Medical-SAM3/inference/visium_hd_exp1/build_patch_expression_dataset.py \
    --metadata ~/Medical-SAM3/output/visium_hd_exp1/exported_expression/bin_metadata_008um.csv \
    --expression ~/Medical-SAM3/output/visium_hd_exp1/exported_expression/expression_008um.npz \
    --image-path /nfs/roberts/project/pi_xy48/hw646/Exp1/spatial/tissue_hires_image.png \
    --patch-size 64 \
    --output-dir ~/Medical-SAM3/output/visium_hd_exp1/patch_dataset \
    --sanity-check"
```

Outputs:
```
~/Medical-SAM3/output/visium_hd_exp1/patch_dataset/
  items.csv           — 420,645 bins with split assignment (background + small labels filtered)
  expr_log1p.npz      — 156 MB sparse log1p expression (scipy CSR, NOT dense)
  scaler.npz          — per-gene mean/std for z-scoring at runtime
  sanity_patches.png  — 16 sample patches
```

First run (job 10123016) OOM'd at 48GB because of dense conversion. Fixed by rewriting
`build_patch_expression_dataset.py` to stay fully sparse — log1p on `.data` only,
mean/std computed with sparse ops, expression matrix never densified.

---

## Stage 4a: InfoNCE Contrastive Training

**Status: ❌ Complete, negative result** (SLURM job 10124612, finished 2026-04-29)

Bottom line: InfoNCE did not learn meaningful cross-modal alignment on Exp1.
Best validation performance occurred at epoch 1 and stayed near random afterward,
so this direction is treated as a failed baseline rather than an active run.

Architecture:
- Image encoder: ResNet50 (ImageNet pretrained) + projection head → 128-dim L2-normalized
- Expression encoder: MLP (18085 → 256 → 256 → 128, LayerNorm + dropout=0.3) + L2-normalize
- Loss: symmetric InfoNCE, fixed temperature τ=0.07
- Phase 1 (epoch 0-29): frozen backbone
- Phase 2 (epoch 30-79): full fine-tune

**Bug fixed:** Originally used learnable log_τ; τ collapsed to 0.046 by epoch 10 causing
val loss to explode (train=3.50, val=7.74). Fixed to fixed τ=0.07. Also added dropout=0.3.
After fix: training was more numerically stable, but the model still failed to
produce useful validation performance.

```bash
sbatch -p gpu_devel --job-name=exp1_train --gres=gpu:1 --mem=48G --cpus-per-task=8 \
  --time=12:00:00 \
  --output=~/Medical-SAM3/output/visium_hd_exp1/train_%j.log \
  --wrap="/home/hw646/.conda/envs/medsam3/bin/python \
    ~/Medical-SAM3/inference/visium_hd_exp1/train_contrastive.py \
    --items-csv ~/Medical-SAM3/output/visium_hd_exp1/patch_dataset/items.csv \
    --expr-npz  ~/Medical-SAM3/output/visium_hd_exp1/patch_dataset/expr_log1p.npz \
    --scaler-npz ~/Medical-SAM3/output/visium_hd_exp1/patch_dataset/scaler.npz \
    --image-path /nfs/roberts/project/pi_xy48/hw646/Exp1/spatial/tissue_hires_image.png \
    --output-dir ~/Medical-SAM3/output/visium_hd_exp1/contrastive_checkpoints \
    --patch-size 64 --batch-size 256 --total-epochs 80"
```

**Limitation:** InfoNCE requires instance-level matching (same bin's patch and expression must
be the positive pair in a batch of 256). Spatial autocorrelation means nearby bins look very
similar, making the task artificially hard at train time but trivially easy via neighborhood
lookup — not the semantic alignment we want.

---

## Stage 4b: Classification-Based Alignment

**Status: 🔄 Running** (current SLURM jobs 10140513 and 10140514 on `gpu_devel` + `gpu`)

Two variants submitted in parallel:
- `exp1_cls` (job 10140513): `align_weight=0.5` → output: `cls_checkpoints/`
- `exp1_cls_a1` (job 10140514): `align_weight=1.0` → output: `cls_a1_checkpoints/`

**Why this is better than InfoNCE for spatial omics:**
- Trains on semantic region labels (tumor vs stroma vs immune) instead of instance-level matching
- Val accuracy is directly interpretable (random baseline = 1/8 = 12.5%)
- Robust to spatial distribution shift between train/val regions
- Weighted cross-entropy handles class imbalance (pigment=686 vs tumor=152k)

Architecture:
- Same image/expression encoders as Stage 4a
- Shared `ClassificationHead(128 → n_classes)` applied to both embeddings
- Loss = `CE(img_pred, label) + CE(expr_pred, label) + align_weight * MSE(norm(z_img), norm(z_expr))`
- MSE term on L2-normalized embeddings pulls both towers into shared space
- Phase 1 (epoch 0-19): frozen backbone
- Phase 2 (epoch 20-59): full fine-tune

```bash
# align_weight=0.5
sbatch -p gpu --job-name=exp1_cls --gres=gpu:1 --mem=48G --cpus-per-task=8 \
  --time=12:00:00 \
  --output=~/Medical-SAM3/output/visium_hd_exp1/cls_%j.log \
  --wrap="... train_classification.py ... --align-weight 0.5"

# align_weight=1.0
sbatch -p gpu --job-name=exp1_cls_a1 --gres=gpu:1 --mem=48G --cpus-per-task=8 \
  --time=12:00:00 \
  --output=~/Medical-SAM3/output/visium_hd_exp1/cls_a1_%j.log \
  --wrap="... train_classification.py ... --align-weight 1.0"
```

---

## Stage 5: Evaluation

**Status: pending** — run after Stage 4a/4b complete

```bash
# Evaluate InfoNCE checkpoint
/home/hw646/.conda/envs/medsam3/bin/python \
  ~/Medical-SAM3/inference/visium_hd_exp1/evaluate_retrieval.py \
  --model-type contrastive \
  --checkpoint ~/Medical-SAM3/output/visium_hd_exp1/contrastive_checkpoints/best.pt \
  --items-csv  ~/Medical-SAM3/output/visium_hd_exp1/patch_dataset/items.csv \
  --expr-npz   ~/Medical-SAM3/output/visium_hd_exp1/patch_dataset/expr_log1p.npz \
  --scaler-npz ~/Medical-SAM3/output/visium_hd_exp1/patch_dataset/scaler.npz \
  --image-path /nfs/roberts/project/pi_xy48/hw646/Exp1/spatial/tissue_hires_image.png \
  --output-dir ~/Medical-SAM3/output/visium_hd_exp1/retrieval_eval_contrastive \
  --save-examples

# Evaluate classification checkpoint (pick best of align_weight=0.5 vs 1.0)
/home/hw646/.conda/envs/medsam3/bin/python \
  ~/Medical-SAM3/inference/visium_hd_exp1/evaluate_retrieval.py \
  --model-type classification \
  --checkpoint ~/Medical-SAM3/output/visium_hd_exp1/cls_checkpoints/best.pt \
  --items-csv  ~/Medical-SAM3/output/visium_hd_exp1/patch_dataset/items.csv \
  --expr-npz   ~/Medical-SAM3/output/visium_hd_exp1/patch_dataset/expr_log1p.npz \
  --scaler-npz ~/Medical-SAM3/output/visium_hd_exp1/patch_dataset/scaler.npz \
  --image-path /nfs/roberts/project/pi_xy48/hw646/Exp1/spatial/tissue_hires_image.png \
  --output-dir ~/Medical-SAM3/output/visium_hd_exp1/retrieval_eval_cls \
  --save-examples
```

Target metrics:
- recall@10 overall > 50% (random baseline = 12.5%)
- recall@10 for tumor and stroma > 60% (largest and most distinct classes)

---

## Key Findings So Far

1. Gene count is 18,085 (full transcriptome), not 479 — richer signal for contrastive learning.
2. No y-flip needed: Space Ranger uses top-left origin image coordinates.
3. Tumor and stroma each have ~150k bins — balanced and sufficient for contrastive training.
4. Background (unannotated) is only 6% of in-tissue bins — masks cover almost all tissue.
