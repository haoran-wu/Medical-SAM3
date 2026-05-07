# ResNet and GigaPath Experiment Details

Last updated: 2026-05-07

This note freezes the current baseline comparison for Visium HD Exp1. We are not
adding more pathology foundation-model backbones here. The goal is to make the
existing ResNet and GigaPath experiments reproducible, interpretable, and easy to
present.

## Project Paths

| Item | Path |
|---|---|
| Project results root | `/nfs/roberts/project/pi_xy48/hw646/Medical-SAM3/results/visium_hd_exp1` |
| Training runs | `/nfs/roberts/project/pi_xy48/hw646/Medical-SAM3/results/visium_hd_exp1/training_runs` |
| Retrieval evals | `/nfs/roberts/project/pi_xy48/hw646/Medical-SAM3/results/visium_hd_exp1/retrieval_eval` |
| Logs | `/nfs/roberts/project/pi_xy48/hw646/Medical-SAM3/results/visium_hd_exp1/logs` |
| Patch/expression dataset | `/nfs/roberts/project/pi_xy48/hw646/Medical-SAM3/output_visium_hd_exp1_legacy/visium_hd_exp1/patch_dataset` |
| H&E image | `/nfs/roberts/project/pi_xy48/hw646/Exp1/spatial/tissue_hires_image.png` |
| Presentation figures | `/nfs/roberts/project/pi_xy48/hw646/Medical-SAM3/results/visium_hd_exp1/figures/group_meeting` |

## What Was Compared

| Experiment | Job | Backbone | Objective | Classes | Crop/Input | Main question |
|---|---:|---|---|---|---|---|
| ResNet contrastive | `11016975` | ResNet50 | InfoNCE image-expression contrastive | 8 | `64/64` | Does a simple CNN contrastive baseline learn useful cross-modal retrieval? |
| GigaPath crop64 | `11016957` | GigaPath tile encoder | Dual-tower classification + alignment | 8 | `64/224` | Does pathology-pretrained image encoding improve classification/retrieval? |
| GigaPath crop128 | `11016973` | GigaPath tile encoder | Dual-tower classification + alignment | 8 | `128/224` | Does larger local tissue context help the 8-class task? |
| GigaPath crop128 3-class | `11016974` | GigaPath tile encoder | Dual-tower classification + alignment | 3 | `128/224` | Does coarse tissue grouping give a more stable biologically useful target? |

## Data and Splits

All four experiments use the same canonical Visium HD Exp1 patch-expression
dataset:

```text
items.csv
expr_log1p.npz
scaler.npz
```

The training scripts use the split stored in `items.csv`. The full retrieval
evaluation uses the full available train/test split from the dataset, while the
training jobs used sample caps for speed and queue practicality.

Label set for 8-class experiments:

```text
Lung Bronchiola
erythorocytes
immune infiltration
lung alveoli (normal adjacent)
lung vessels
pigment
stroma
tumor
```

Label set for 3-class experiments:

```text
immune infiltration
stroma
tumor
```

## Model Definitions

### ResNet Contrastive Baseline

Script:

```text
inference/visium_hd_exp1/train_contrastive.py
```

HPC script:

```text
scripts/hpc/visium_hd_exp1_contrastive_resnet_project_run.sbatch
```

Core parameters:

| Parameter | Value |
|---|---|
| `--image-backbone` | `resnet50` |
| `--crop-size` | `64` |
| `--input-size` | `64` |
| `--batch-size` | `256` |
| `--total-epochs` | `20` |
| `--freeze-epochs` | `20` |
| `--max-train-samples` | `50000` |
| `--max-val-samples` | `10000` |
| Temperature `tau` | `0.07` |

Loss:

```text
InfoNCE(image_embedding, expression_embedding)
```

Interpretation:

This is the cleanest cross-modal baseline. It does not use tissue class labels
as direct supervision. It asks whether paired H&E crops and expression profiles
can be embedded close together while unpaired examples are pushed apart.

Observed loss:

| Epoch | Train InfoNCE | Val InfoNCE |
|---:|---:|---:|
| 1 | 5.4607 | 5.2028 |
| 4 | lower than epoch 1 | **best val around 4.7586** |
| 20 | 2.9692 | 5.8492 |

The training loss keeps decreasing while validation loss worsens after the best
epoch. This suggests overfitting or insufficient semantic structure in the plain
ResNet features for this dataset.

### GigaPath Dual-Tower Classification + Alignment

Script:

```text
inference/visium_hd_exp1/train_classification.py
```

HPC scripts:

```text
scripts/hpc/visium_hd_exp1_g64_project_run_devel.sbatch
scripts/hpc/visium_hd_exp1_g128_project_run.sbatch
scripts/hpc/visium_hd_exp1_g128_3cls_project_run.sbatch
```

Core shared parameters:

| Parameter | Value |
|---|---|
| `--image-backbone` | `gigapath` |
| `--input-size` | `224` |
| `--batch-size` | `32` |
| `--total-epochs` | `8` |
| `--freeze-epochs` | `8` |
| `--max-train-samples` | `10000` |
| `--max-val-samples` | `2500` |
| `--class-weight-mode` | `sqrt_inv` |
| `--sampler` | `balanced` |
| `--monitor` | `val_macro_f1_expr` |
| `--early-stop-patience` | `2` |
| `--align-weight` | `0.5` |
| Seed | `42` |

Loss:

```text
total_loss = CE(image_logits, label)
           + CE(expression_logits, label)
           + 0.5 * MSE(normalized_image_embedding, normalized_expression_embedding)
```

Loss components:

| Component | Meaning |
|---|---|
| `CE(image_logits, label)` | Trains the H&E image tower to predict the region/tissue class. |
| `CE(expression_logits, label)` | Trains the expression tower to predict the same region/tissue class. |
| `MSE(normalized embeddings)` | Pulls the image and expression embeddings into a shared latent space. |

This setup is not a pure retrieval model. It is a supervised dual-tower model
with an explicit but lightweight alignment term. That makes it easier to use for
class-level tissue interpretation, but it is less directly CLIP-like than the
ResNet InfoNCE baseline.

## Training Results

| Run | Best epoch | Best val expr acc | Best val image acc | Best val expr macro-F1 | Training interpretation |
|---|---:|---:|---:|---:|---|
| GigaPath crop64 8-class | 3 | 0.5948 | 0.6344 | 0.4368 | Learns signal, but fine-grained labels remain hard. |
| GigaPath crop128 8-class | 4 | 0.6100 | 0.5328 | 0.4366 | Slightly higher expr acc than crop64, but worse image acc and similar macro-F1. |
| GigaPath crop128 3-class | 2 | 0.7544 | 0.6800 | 0.6724 | Much cleaner coarse tissue target; best current supervised model. |

Key interpretation:

The 3-class model is substantially more stable than the 8-class models. The
8-class models can fit the training data, but validation expression CE remains
high and rare/fine labels such as `pigment` and `erythorocytes` are not reliable.
This supports treating 8-class as a diagnostic/fine-grained analysis, not as the
main model for SAM3 integration yet.

## Retrieval and Linear-Probe Results

Retrieval is expression-to-image: for a query expression embedding, retrieve
nearest image embeddings and check whether the retrieved image has the same
label.

| Run | Classes | Recall@1 | Recall@5 | Recall@10 | Image probe | Expr probe | Concat probe |
|---|---:|---:|---:|---:|---:|---:|---:|
| ResNet contrastive crop64 | 8 | 0.5571 | 0.8219 | 0.8851 | 0.6688 | 0.6675 | 0.7266 |
| GigaPath crop64 8-class | 8 | 0.5999 | 0.7215 | 0.7622 | 0.8138 | 0.6555 | 0.8237 |
| GigaPath crop128 8-class | 8 | 0.5879 | 0.7229 | 0.7603 | 0.7939 | 0.6510 | 0.8179 |
| GigaPath crop128 3-class | 3 | 0.7406 | 0.8301 | 0.8634 | 0.8067 | 0.7751 | 0.8325 |

Interpretation:

- ResNet contrastive has strong Recall@5/10, but weaker linear-probe accuracy.
  It retrieves broad neighborhoods well, but its embedding is less class
  discriminative.
- GigaPath 8-class is better for image/combined class separability, especially
  in linear probe, but its retrieval Recall@5/10 is lower than ResNet.
- GigaPath 3-class is the best overall current result because it combines high
  Recall@1 with strong image/expression/concat probe scores.

## Per-Label Retrieval Notes

### GigaPath crop128 3-class

| Label | Recall@1 | Recall@5 | Recall@10 |
|---|---:|---:|---:|
| immune infiltration | 0.3517 | 0.4576 | 0.4932 |
| stroma | 0.8059 | 0.8930 | 0.9323 |
| tumor | 0.7317 | 0.8212 | 0.8483 |

The model is strongest on `stroma` and `tumor`, weaker on `immune infiltration`.
This is biologically plausible: immune infiltration can be spatially mixed and
patch-level morphology may be less visually homogeneous.

### GigaPath crop64 vs crop128 on 8-class

| Label | Crop64 R@1 | Crop128 R@1 | Better |
|---|---:|---:|---|
| Lung Bronchiola | 0.7401 | 0.7145 | crop64 |
| erythorocytes | 0.0077 | 0.0042 | both poor |
| immune infiltration | 0.3170 | 0.4152 | crop128 |
| lung alveoli | 0.4307 | 0.4404 | crop128 slight |
| lung vessels | 0.5547 | 0.4320 | crop64 |
| pigment | 0.0000 | 0.0000 | both fail |
| stroma | 0.6334 | 0.6243 | crop64 slight |
| tumor | 0.6890 | 0.6767 | crop64 slight |

Crop128 helps `immune infiltration`, likely because it adds tissue context.
Crop64 is slightly better overall and better for several visually local labels.
Neither crop solves rare labels.

### ResNet contrastive 8-class

| Label | Recall@1 | Recall@5 | Recall@10 |
|---|---:|---:|---:|
| Lung Bronchiola | 0.6926 | 0.7940 | 0.8303 |
| erythorocytes | 0.0393 | 0.1944 | 0.3460 |
| immune infiltration | 0.5176 | 0.7439 | 0.8025 |
| lung alveoli | 0.3802 | 0.7013 | 0.7934 |
| lung vessels | 0.4843 | 0.7965 | 0.8509 |
| pigment | 0.0147 | 0.0147 | 0.0147 |
| stroma | 0.5575 | 0.8641 | 0.9322 |
| tumor | 0.6443 | 0.8844 | 0.9351 |

ResNet contrastive is surprisingly competitive for Recall@5/10. That makes it a
useful retrieval baseline, but the weaker linear-probe scores mean it should not
replace GigaPath as the main semantic tissue model.

## Current Recommendation

For presentation and next SAM3 integration:

1. Use **GigaPath crop128 3-class** as the main supervised result.
2. Use **ResNet contrastive crop64** as the simple cross-modal baseline.
3. Use **GigaPath crop64/crop128 8-class** as a fine-grained diagnostic showing
   that the current 8-class labels are harder and rare classes are unstable.
4. Do not test more backbones until these results are clearly presented and the
   loss/backbone ablations are framed from literature.

## Active Loss Ablation

Status on 2026-05-07: submitted to Bouchet as SLURM array `11028639_[0-11%2]`.

Script:

```text
scripts/hpc/visium_hd_exp1_gigapath_loss_ablation.sbatch
```

Output root:

```text
/nfs/roberts/project/pi_xy48/hw646/Medical-SAM3/results/visium_hd_exp1/training_runs/loss_ablation
```

Log pattern:

```text
/nfs/roberts/project/pi_xy48/hw646/Medical-SAM3/results/visium_hd_exp1/logs/loss_ablation_11028639_*.log
```

Design:

| Task IDs | Class target | Runs |
|---|---|---|
| `0-5` | 3-class: tumor, stroma, immune infiltration | CE-only; CE+InfoNCE lambda 0.05/0.10/0.20/0.50; InfoNCE-only |
| `6-11` | 8-class: all region labels | CE-only; CE+InfoNCE lambda 0.05/0.10/0.20/0.50; InfoNCE-only |

Reference baseline already completed:

```text
CE_img + CE_expr + 0.5 * MSE alignment
```

New ablation questions:

- `CE-only`: is retrieval good because both towers learn the same labels, even
  without explicit alignment?
- `CE+InfoNCE`: does CLIP-style paired alignment improve shared-space behavior
  over pairwise MSE?
- `InfoNCE-only`: can paired image-expression alignment work without tissue
  labels?
- `3-class vs 8-class`: is the 8-class issue mainly fine-label noise/imbalance,
  or a failure of cross-modal alignment?

The first two subtasks started successfully and reached epoch 1 without script
or environment failure, so the new loss code path is valid.

## Files to Cite in Slides

Training histories:

```text
/nfs/roberts/project/pi_xy48/hw646/Medical-SAM3/results/visium_hd_exp1/training_runs/contrastive_resnet50_crop64_rtx6000_project_v2_11016975/history.json
/nfs/roberts/project/pi_xy48/hw646/Medical-SAM3/results/visium_hd_exp1/training_runs/cls_gigapath_crop64_metric_gpu_devel_project_v2_devel_11016957/history.json
/nfs/roberts/project/pi_xy48/hw646/Medical-SAM3/results/visium_hd_exp1/training_runs/cls_gigapath_crop128_metric_gpu_rtx6000_project_v2_11016973/history.json
/nfs/roberts/project/pi_xy48/hw646/Medical-SAM3/results/visium_hd_exp1/training_runs/cls_gigapath_crop128_3class_gpu_rtx6000_project_v2_11016974/history.json
```

Retrieval metrics:

```text
/nfs/roberts/project/pi_xy48/hw646/Medical-SAM3/results/visium_hd_exp1/retrieval_eval/resnet_contrastive_crop64_11025621/metrics.json
/nfs/roberts/project/pi_xy48/hw646/Medical-SAM3/results/visium_hd_exp1/retrieval_eval/g64_8class_11020754_0/metrics.json
/nfs/roberts/project/pi_xy48/hw646/Medical-SAM3/results/visium_hd_exp1/retrieval_eval/g128_8class_11020754_1/metrics.json
/nfs/roberts/project/pi_xy48/hw646/Medical-SAM3/results/visium_hd_exp1/retrieval_eval/g128_3class_devel_11017417_2/metrics.json
```

Slide figures:

```text
/nfs/roberts/project/pi_xy48/hw646/Medical-SAM3/results/visium_hd_exp1/figures/group_meeting/presentation_3class_vs_8class_metrics.png
/nfs/roberts/project/pi_xy48/hw646/Medical-SAM3/results/visium_hd_exp1/figures/group_meeting/presentation_3class_total_loss_decomposition.png
/nfs/roberts/project/pi_xy48/hw646/Medical-SAM3/results/visium_hd_exp1/figures/group_meeting/presentation_expression_ce_generalization_gap.png
/nfs/roberts/project/pi_xy48/hw646/Medical-SAM3/results/visium_hd_exp1/figures/group_meeting/resnet_contrastive_loss.png
/nfs/roberts/project/pi_xy48/hw646/Medical-SAM3/results/visium_hd_exp1/figures/group_meeting/retrieval_overall_recall.png
```
