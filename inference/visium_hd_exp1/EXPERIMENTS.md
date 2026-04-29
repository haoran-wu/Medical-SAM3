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
| `render_geojson_masks.py` | Stage 1: GeoJSON → binary PNG per region label |
| `export_008um_expression.py` | Stage 2: bin metadata CSV + sparse expression NPZ |
| `build_patch_expression_dataset.py` | Stage 3: normalized expression + spatial split + `PatchExpressionDataset` |
| `train_contrastive.py` | Stage 4: ResNet50 + MLP InfoNCE contrastive training |
| `evaluate_retrieval.py` | Stage 5: recall@k retrieval + linear probe |

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

**Status: 🔄 Running** (SLURM job 10124612, gpu_devel, started 2026-04-29 ~05:06)

Architecture:
- Image encoder: ResNet50 (ImageNet pretrained) + projection head → 128-dim L2-normalized
- Expression encoder: MLP (18085 → 256 → 256 → 128, LayerNorm + dropout=0.3) + L2-normalize
- Loss: symmetric InfoNCE, fixed temperature τ=0.07
- Phase 1 (epoch 0-29): frozen backbone
- Phase 2 (epoch 30-79): full fine-tune

**Bug fixed:** Originally used learnable log_τ; τ collapsed to 0.046 by epoch 10 causing
val loss to explode (train=3.50, val=7.74). Fixed to fixed τ=0.07. Also added dropout=0.3.
After fix: train/val gap dropped from 4.2 to 1.29 by epoch 11.

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

**Status: 🔄 Pending** (SLURM jobs 10125666 and 10125669, gpu partition, submitted 2026-04-29)

Two variants submitted in parallel:
- `exp1_cls` (job 10125666): `align_weight=0.5` → output: `cls_checkpoints/`
- `exp1_cls_a1` (job 10125669): `align_weight=1.0` → output: `cls_a1_checkpoints/`

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
