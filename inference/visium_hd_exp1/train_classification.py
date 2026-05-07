#!/usr/bin/env python3
"""
Dual-tower classification-based alignment for H&E patches <-> gene expression.

Each encoder independently predicts the region label (cross-entropy).
An optional alignment term pulls the two 128-dim embeddings together.

Loss = img_ce_weight * CE(img_pred, label)
     + expr_ce_weight * CE(expr_pred, label)
     + align_weight * alignment_loss(img_emb, expr_emb)

Why this is better than InfoNCE for spatial omics:
- Trains on region semantics (tumor vs stroma vs immune) instead of
  exact spatial instance matching (which expression bin goes with which patch)
- Val accuracy is directly interpretable
- Robust to spatial distribution shift between train/val regions

After training, both encoders map to the same 128-dim space (implicitly aligned
via shared label supervision + explicit MSE term). Use the embeddings for retrieval
the same way as in evaluate_retrieval.py.

Usage:
  python train_classification.py \
    --items-csv  output/visium_hd_exp1/patch_dataset/items.csv \
    --expr-npz   output/visium_hd_exp1/patch_dataset/expr_log1p.npz \
    --scaler-npz output/visium_hd_exp1/patch_dataset/scaler.npz \
    --image-path /nfs/.../Exp1/spatial/tissue_hires_image.png \
    --output-dir output/visium_hd_exp1/cls_checkpoints/
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = Path("/nfs/roberts/project/pi_xy48/hw646/Exp1")
DEFAULT_IMAGE_PATH = DATA_DIR / "spatial" / "tissue_hires_image.png"
PATCH_DATASET_DIR = PROJECT_ROOT / "output" / "visium_hd_exp1" / "patch_dataset"


# ─── Encoders ────────────────────────────────────────────────────────────────

class ExpressionEncoder(nn.Module):
    def __init__(self, n_genes: int, hidden_dim: int = 256, embed_dim: int = 128,
                 dropout: float = 0.3) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(n_genes, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
            nn.LayerNorm(embed_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)


class ImageEncoder(nn.Module):
    def __init__(
        self,
        embed_dim: int = 128,
        freeze_backbone: bool = True,
        image_backbone: str = "resnet50",
    ) -> None:
        super().__init__()
        from image_backbones import create_image_backbone

        self.backbone_name = image_backbone
        self.backbone, self.feature_dim = create_image_backbone(image_backbone)
        self.proj = nn.Sequential(
            nn.Linear(self.feature_dim, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, embed_dim),
            nn.LayerNorm(embed_dim),
        )
        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(self.backbone(x).flatten(1))

    def unfreeze_backbone(self, trainable_blocks: int | None = None) -> None:
        """Unfreeze all backbone params or only the last N major blocks."""
        for p in self.backbone.parameters():
            p.requires_grad_(False)
        if trainable_blocks == 0:
            return
        if trainable_blocks is None or trainable_blocks < 0:
            for p in self.backbone.parameters():
                p.requires_grad_(True)
            return

        target = getattr(self.backbone, "model", self.backbone)
        block_seq = None
        for attr in ("blocks", "layers", "stages"):
            if hasattr(target, attr):
                block_seq = list(getattr(target, attr))
                break
        if block_seq is None:
            block_seq = list(target.children())
        if not block_seq:
            for p in self.backbone.parameters():
                p.requires_grad_(True)
            return

        for block in block_seq[-trainable_blocks:]:
            for p in block.parameters():
                p.requires_grad_(True)
        for name, module in target.named_modules():
            if name.endswith("norm") or ".norm" in name:
                for p in module.parameters(recurse=False):
                    p.requires_grad_(True)


def count_trainable_parameters(module: nn.Module) -> tuple[int, int]:
    trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
    total = sum(p.numel() for p in module.parameters())
    return trainable, total


class ClassificationHead(nn.Module):
    def __init__(self, embed_dim: int, n_classes: int) -> None:
        super().__init__()
        self.fc = nn.Linear(embed_dim, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


# ─── Training ────────────────────────────────────────────────────────────────

def make_optimizer(img_enc, expr_enc, cls_head, lr_proj, lr_backbone, weight_decay):
    return torch.optim.AdamW([
        {"params": img_enc.proj.parameters(),      "lr": lr_proj},
        {"params": img_enc.backbone.parameters(),  "lr": lr_backbone},
        {"params": expr_enc.parameters(),          "lr": lr_proj},
        {"params": cls_head.parameters(),          "lr": lr_proj},
    ], weight_decay=weight_decay)


def make_lr_scheduler(optimizer, schedule: str, total_steps: int, warmup_steps: int):
    if schedule == "constant":
        return None
    if schedule != "warmup_cosine":
        raise ValueError(f"Unknown LR schedule: {schedule}")
    total_steps = max(1, total_steps)
    warmup_steps = min(max(0, warmup_steps), total_steps - 1)

    def lr_lambda(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def get_subset_labels(dataset) -> np.ndarray:
    base_ds = dataset.dataset if isinstance(dataset, Subset) else dataset
    labels = np.asarray(base_ds.labels_idx)
    if isinstance(dataset, Subset):
        labels = labels[np.asarray(dataset.indices)]
    return labels


def compute_class_weights(label_counts: np.ndarray, mode: str, device: torch.device) -> torch.Tensor | None:
    if mode == "none":
        return None
    counts = np.maximum(label_counts.astype(np.float32), 1.0)
    if mode == "sqrt_inv":
        weights = 1.0 / np.sqrt(counts)
    elif mode == "inverse":
        weights = 1.0 / counts
    else:
        raise ValueError(f"Unknown class weight mode: {mode}")
    weights = weights / weights.sum() * len(weights)
    return torch.tensor(weights, dtype=torch.float32, device=device)


def make_balanced_sampler(labels: np.ndarray) -> WeightedRandomSampler:
    counts = np.bincount(labels).astype(np.float64)
    sample_weights = 1.0 / np.maximum(counts[labels], 1.0)
    return WeightedRandomSampler(sample_weights, num_samples=len(labels), replacement=True)


def classification_metrics(
    y_true: list[int],
    y_img: list[int],
    y_expr: list[int],
    label_list: list[str],
) -> dict:
    def one_side(y_pred: list[int]) -> dict:
        true = np.asarray(y_true, dtype=np.int64)
        pred = np.asarray(y_pred, dtype=np.int64)
        n_classes = len(label_list)
        cm = np.zeros((n_classes, n_classes), dtype=np.int64)
        for t, p in zip(true, pred):
            cm[t, p] += 1

        per_class = {}
        recalls = []
        precisions = []
        f1s = []
        for idx, label in enumerate(label_list):
            tp = float(cm[idx, idx])
            fp = float(cm[:, idx].sum() - cm[idx, idx])
            fn = float(cm[idx, :].sum() - cm[idx, idx])
            support = int(cm[idx, :].sum())
            precision = tp / (tp + fp) if tp + fp > 0 else 0.0
            recall = tp / (tp + fn) if tp + fn > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
            per_class[label] = {
                "support": support,
                "precision": precision,
                "recall": recall,
                "f1": f1,
            }
            if support > 0:
                precisions.append(precision)
                recalls.append(recall)
                f1s.append(f1)

        return {
            "macro_precision": float(np.mean(precisions)) if precisions else 0.0,
            "macro_recall": float(np.mean(recalls)) if recalls else 0.0,
            "macro_f1": float(np.mean(f1s)) if f1s else 0.0,
            "confusion_matrix": cm.tolist(),
            "per_class": per_class,
        }

    return {"img": one_side(y_img), "expr": one_side(y_expr)}


def write_confusion_csv(path: Path, matrix: list[list[int]], label_list: list[str]) -> None:
    import csv

    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["true_label", *label_list])
        for label, row in zip(label_list, matrix):
            writer.writerow([label, *row])


def json_ready(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(v) for v in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def symmetric_infonce(z_img: torch.Tensor, z_expr: torch.Tensor, temperature: float) -> torch.Tensor:
    z_img = F.normalize(z_img, dim=-1)
    z_expr = F.normalize(z_expr, dim=-1)
    logits = z_img @ z_expr.T / temperature
    targets = torch.arange(z_img.shape[0], device=z_img.device)
    return 0.5 * (
        F.cross_entropy(logits, targets)
        + F.cross_entropy(logits.T, targets)
    )


def alignment_loss(
    z_img: torch.Tensor,
    z_expr: torch.Tensor,
    loss_type: str,
    temperature: float,
) -> torch.Tensor:
    if loss_type == "mse":
        return F.mse_loss(F.normalize(z_img, dim=-1), F.normalize(z_expr, dim=-1))
    if loss_type == "infonce":
        return symmetric_infonce(z_img, z_expr, temperature)
    if loss_type == "none":
        return z_img.new_zeros(())
    raise ValueError(f"Unknown alignment loss: {loss_type}")


def run_epoch(loader, img_enc, expr_enc, cls_head, optimizer,
              device, align_weight, train, class_weights=None, label_list=None,
              align_loss_type="mse", contrastive_temperature=0.07,
              img_ce_weight=1.0, expr_ce_weight=1.0, scheduler=None):
    img_enc.train(train)
    expr_enc.train(train)
    cls_head.train(train)

    total_loss = total_ce_img = total_ce_expr = total_align = 0.0
    total_correct_img = total_correct_expr = total_n = 0
    y_true: list[int] = []
    y_pred_img: list[int] = []
    y_pred_expr: list[int] = []

    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for patches, exprs, labels in loader:
            patches = patches.to(device)
            exprs   = exprs.to(device)
            labels  = labels.to(device)

            z_img  = img_enc(patches)   # [B, embed_dim]
            z_expr = expr_enc(exprs)    # [B, embed_dim]

            logits_img  = cls_head(z_img)
            logits_expr = cls_head(z_expr)

            ce_img  = F.cross_entropy(logits_img,  labels, weight=class_weights)
            ce_expr = F.cross_entropy(logits_expr, labels, weight=class_weights)
            align = alignment_loss(z_img, z_expr, align_loss_type, contrastive_temperature)
            loss = img_ce_weight * ce_img + expr_ce_weight * ce_expr + align_weight * align

            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()

            n = len(labels)
            total_loss       += loss.item()    * n
            total_ce_img     += ce_img.item()  * n
            total_ce_expr    += ce_expr.item() * n
            total_align      += align.item()   * n
            pred_img = logits_img.argmax(1)
            pred_expr = logits_expr.argmax(1)
            total_correct_img  += (pred_img  == labels).sum().item()
            total_correct_expr += (pred_expr == labels).sum().item()
            total_n += n
            if not train:
                y_true.extend(labels.cpu().tolist())
                y_pred_img.extend(pred_img.cpu().tolist())
                y_pred_expr.extend(pred_expr.cpu().tolist())

    N = total_n
    result = {
        "loss":       total_loss / N,
        "ce_img":     total_ce_img / N,
        "ce_expr":    total_ce_expr / N,
        "align":      total_align / N,
        "acc_img":    total_correct_img / N,
        "acc_expr":   total_correct_expr / N,
    }
    if not train and label_list is not None:
        result["metrics"] = classification_metrics(y_true, y_pred_img, y_pred_expr, label_list)
    return result


# ─── Main ────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Train dual-tower classification alignment.")
    parser.add_argument("--items-csv",  type=Path, default=PATCH_DATASET_DIR / "items.csv")
    parser.add_argument("--expr-npz",   type=Path, default=PATCH_DATASET_DIR / "expr_log1p.npz")
    parser.add_argument("--scaler-npz", type=Path, default=PATCH_DATASET_DIR / "scaler.npz")
    parser.add_argument("--image-path", type=Path, default=DEFAULT_IMAGE_PATH)
    parser.add_argument("--output-dir", type=Path,
                        default=PROJECT_ROOT / "output" / "visium_hd_exp1" / "cls_checkpoints")
    parser.add_argument("--embed-dim",    type=int,   default=128)
    parser.add_argument("--patch-size",   type=int,   default=64,
                        help="Legacy alias for crop/input size when crop-size/input-size are unset")
    parser.add_argument("--crop-size",    type=int,   default=None,
                        help="Hires H&E pixels to crop around each bin")
    parser.add_argument("--input-size",   type=int,   default=None,
                        help="Pixel size after resizing the crop for the image backbone")
    parser.add_argument("--image-backbone", type=str, default="resnet50",
                        choices=["resnet50", "gigapath", "uni", "conch", "virchow", "virchow2", "musk"])
    parser.add_argument("--batch-size",   type=int,   default=256)
    parser.add_argument("--freeze-epochs",type=int,   default=20)
    parser.add_argument("--total-epochs", type=int,   default=60)
    parser.add_argument("--lr",           type=float, default=1e-4)
    parser.add_argument("--lr-backbone",  type=float, default=1e-5)
    parser.add_argument("--trainable-backbone-blocks", type=int, default=-1,
                        help="-1 unfreezes the full backbone after freeze-epochs; 0 keeps it frozen; N unfreezes last N major blocks")
    parser.add_argument("--lr-schedule", type=str, default="constant",
                        choices=["constant", "warmup_cosine"],
                        help="Learning-rate schedule")
    parser.add_argument("--warmup-steps", type=int, default=0,
                        help="Absolute warmup steps for warmup_cosine; overrides warmup-ratio when >0")
    parser.add_argument("--warmup-ratio", type=float, default=0.0,
                        help="Fraction of training steps used for warmup when warmup-steps is 0")
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout",      type=float, default=0.3)
    parser.add_argument("--align-weight", type=float, default=0.5,
                        help="Weight for embedding alignment term")
    parser.add_argument("--align-loss", type=str, default="mse",
                        choices=["mse", "infonce", "none"],
                        help="Embedding alignment objective")
    parser.add_argument("--contrastive-temperature", type=float, default=0.07,
                        help="Temperature for symmetric InfoNCE alignment")
    parser.add_argument("--img-ce-weight", type=float, default=1.0,
                        help="Weight for image branch cross-entropy")
    parser.add_argument("--expr-ce-weight", type=float, default=1.0,
                        help="Weight for expression branch cross-entropy")
    parser.add_argument("--class-weight-mode", type=str, default="inverse",
                        choices=["inverse", "sqrt_inv", "none"],
                        help="Class weighting for cross-entropy")
    parser.add_argument("--sampler", type=str, default="shuffle",
                        choices=["shuffle", "balanced"],
                        help="Use class-balanced sampling for the train loader")
    parser.add_argument("--include-labels", type=str, default=None,
                        help="Comma-separated region labels to keep, e.g. 'tumor,stroma,immune infiltration'")
    parser.add_argument("--early-stop-patience", type=int, default=0,
                        help="Stop after this many epochs without monitor improvement; 0 disables")
    parser.add_argument("--early-stop-min-delta", type=float, default=0.0,
                        help="Minimum monitor improvement required to reset patience")
    parser.add_argument("--monitor", type=str, default="val_acc_expr",
                        choices=["val_acc_expr", "val_macro_f1_expr", "val_loss"],
                        help="Validation metric used for best checkpoint and early stopping")
    parser.add_argument("--num-workers",  type=int,   default=8)
    parser.add_argument("--max-train-samples", type=int, default=None,
                        help="Optional cap for a fast train subset")
    parser.add_argument("--max-val-samples", type=int, default=None,
                        help="Optional cap for a fast validation subset")
    parser.add_argument("--seed",         type=int,   default=42)
    args = parser.parse_args()
    args.crop_size = args.crop_size if args.crop_size is not None else args.patch_size
    args.input_size = args.input_size if args.input_size is not None else args.crop_size
    include_labels = None
    if args.include_labels:
        include_labels = [x.strip() for x in args.include_labels.split(",") if x.strip()]

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from build_patch_expression_dataset import PatchExpressionDataset

    print("Loading datasets...")
    train_ds = PatchExpressionDataset(
        args.items_csv, args.expr_npz, args.scaler_npz, args.image_path,
        patch_size=args.patch_size, crop_size=args.crop_size, input_size=args.input_size,
        split="train", include_labels=include_labels, augment=True,
    )
    val_ds = PatchExpressionDataset(
        args.items_csv, args.expr_npz, args.scaler_npz, args.image_path,
        patch_size=args.patch_size, crop_size=args.crop_size, input_size=args.input_size,
        split="val", include_labels=include_labels, augment=False,
    )
    base_train_ds = train_ds.dataset if isinstance(train_ds, Subset) else train_ds
    n_classes = base_train_ds.n_classes

    if args.max_train_samples is not None and args.max_train_samples < len(train_ds):
        subset_idx = np.random.permutation(len(train_ds))[:args.max_train_samples]
        train_ds = Subset(train_ds, subset_idx.tolist())
    if args.max_val_samples is not None and args.max_val_samples < len(val_ds):
        subset_idx = np.random.permutation(len(val_ds))[:args.max_val_samples]
        val_ds = Subset(val_ds, subset_idx.tolist())

    label_list = base_train_ds.label_list
    print(f"  Train: {len(train_ds)}, Val: {len(val_ds)}, Classes: {n_classes}")
    print(f"  Labels: {label_list}")

    # Inverse-frequency class weights to handle severe imbalance (pigment=686 vs tumor=152k)
    train_labels = get_subset_labels(train_ds)
    label_counts = np.bincount(train_labels, minlength=n_classes).astype(np.float32)
    class_weights = compute_class_weights(label_counts, args.class_weight_mode, device)
    if class_weights is None:
        print("  Class weights: disabled")
    else:
        print(f"  Class weights ({args.class_weight_mode}): { {label_list[i]: f'{class_weights[i].item():.3f}' for i in range(n_classes)} }")

    train_sampler = make_balanced_sampler(train_labels) if args.sampler == "balanced" else None
    print(f"  Train sampler: {args.sampler}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=(train_sampler is None),
                               sampler=train_sampler,
                               num_workers=args.num_workers, pin_memory=True, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False,
                               num_workers=args.num_workers, pin_memory=True)

    img_enc  = ImageEncoder(
        embed_dim=args.embed_dim,
        freeze_backbone=True,
        image_backbone=args.image_backbone,
    ).to(device)
    expr_enc = ExpressionEncoder(n_genes=base_train_ds.n_genes, embed_dim=args.embed_dim,
                                 dropout=args.dropout).to(device)
    cls_head = ClassificationHead(args.embed_dim, n_classes).to(device)

    optimizer = make_optimizer(img_enc, expr_enc, cls_head,
                               lr_proj=args.lr, lr_backbone=0.0,
                               weight_decay=args.weight_decay)
    total_steps = max(1, args.total_epochs * len(train_loader))
    warmup_steps = args.warmup_steps if args.warmup_steps > 0 else int(total_steps * args.warmup_ratio)
    scheduler = make_lr_scheduler(optimizer, args.lr_schedule, total_steps, warmup_steps)
    tr_img, total_img = count_trainable_parameters(img_enc.backbone)
    print(
        f"  LR schedule: {args.lr_schedule}, warmup_steps={warmup_steps}, "
        f"total_steps={total_steps}"
    )
    print(f"  Image backbone trainable params: {tr_img:,}/{total_img:,}")

    best_monitor = -float("inf")
    checkpoint = None
    history = []
    epochs_without_improvement = 0

    for epoch in range(args.total_epochs):
        if epoch == args.freeze_epochs:
            print(
                f"\nEpoch {epoch}: unfreezing backbone blocks={args.trainable_backbone_blocks} "
                f"(lr={args.lr_backbone})"
            )
            img_enc.unfreeze_backbone(args.trainable_backbone_blocks)
            optimizer = make_optimizer(img_enc, expr_enc, cls_head,
                                       lr_proj=args.lr, lr_backbone=args.lr_backbone,
                                       weight_decay=args.weight_decay)
            remaining_steps = max(1, (args.total_epochs - epoch) * len(train_loader))
            remaining_warmup = args.warmup_steps if args.warmup_steps > 0 else int(remaining_steps * args.warmup_ratio)
            scheduler = make_lr_scheduler(optimizer, args.lr_schedule, remaining_steps, remaining_warmup)
            tr_img, total_img = count_trainable_parameters(img_enc.backbone)
            print(f"  Image backbone trainable params: {tr_img:,}/{total_img:,}")

        tr = run_epoch(train_loader, img_enc, expr_enc, cls_head, optimizer,
                       device, args.align_weight, train=True, class_weights=class_weights,
                       align_loss_type=args.align_loss,
                       contrastive_temperature=args.contrastive_temperature,
                       img_ce_weight=args.img_ce_weight,
                       expr_ce_weight=args.expr_ce_weight,
                       scheduler=scheduler)
        va = run_epoch(val_loader,   img_enc, expr_enc, cls_head, None,
                       device, args.align_weight, train=False, class_weights=class_weights,
                       label_list=label_list,
                       align_loss_type=args.align_loss,
                       contrastive_temperature=args.contrastive_temperature,
                       img_ce_weight=args.img_ce_weight,
                       expr_ce_weight=args.expr_ce_weight)

        val_macro_f1_expr = va["metrics"]["expr"]["macro_f1"]

        print(
            f"Epoch {epoch+1:03d}/{args.total_epochs} | "
            f"train_acc img={tr['acc_img']:.3f} expr={tr['acc_expr']:.3f} | "
            f"val_acc img={va['acc_img']:.3f} expr={va['acc_expr']:.3f} | "
            f"macro_f1 expr={val_macro_f1_expr:.3f} | align={va['align']:.4f}"
        )
        history.append({"epoch": epoch + 1, "train": tr, "val": va})

        if args.monitor == "val_acc_expr":
            monitor_value = va["acc_expr"]
        elif args.monitor == "val_macro_f1_expr":
            monitor_value = val_macro_f1_expr
        elif args.monitor == "val_loss":
            monitor_value = -va["loss"]
        else:
            raise ValueError(f"Unknown monitor: {args.monitor}")
        if monitor_value > best_monitor + args.early_stop_min_delta:
            best_monitor = monitor_value
            epochs_without_improvement = 0
            checkpoint = {
                "epoch": epoch + 1,
                "monitor": args.monitor,
                "monitor_value": monitor_value,
                "val_acc_expr": va["acc_expr"],
                "val_acc_img":  va["acc_img"],
                "val_macro_f1_expr": val_macro_f1_expr,
                "val_metrics": va["metrics"],
                "image_backbone": args.image_backbone,
                "feature_dim": img_enc.feature_dim,
                "img_enc":  img_enc.state_dict(),
                "expr_enc": expr_enc.state_dict(),
                "cls_head": cls_head.state_dict(),
                "args":      json_ready(vars(args)),
                "label_list": label_list,
                "n_genes":    base_train_ds.n_genes,
            }
            torch.save(checkpoint, args.output_dir / "best.pt")
            write_confusion_csv(
                args.output_dir / "best_confusion_expr.csv",
                va["metrics"]["expr"]["confusion_matrix"],
                label_list,
            )
            write_confusion_csv(
                args.output_dir / "best_confusion_img.csv",
                va["metrics"]["img"]["confusion_matrix"],
                label_list,
            )
            (args.output_dir / "best_val_metrics.json").write_text(json.dumps(va["metrics"], indent=2))
        else:
            epochs_without_improvement += 1

        if args.early_stop_patience > 0 and epochs_without_improvement >= args.early_stop_patience:
            print(
                f"\nEarly stopping at epoch {epoch + 1}: "
                f"no {args.monitor} improvement for {args.early_stop_patience} epochs."
            )
            break

    if checkpoint is None:
        last_val = history[-1]["val"] if history else {}
        last_metrics = last_val.get("metrics", {})
        checkpoint = {
            "epoch": args.total_epochs,
            "monitor": args.monitor,
            "monitor_value": best_monitor if best_monitor > -float("inf") else None,
            "val_acc_expr": last_val.get("acc_expr"),
            "val_acc_img": last_val.get("acc_img"),
            "val_macro_f1_expr": last_metrics.get("expr", {}).get("macro_f1"),
            "val_metrics": last_metrics,
            "image_backbone": args.image_backbone,
            "feature_dim": img_enc.feature_dim,
            "img_enc": img_enc.state_dict(),
            "expr_enc": expr_enc.state_dict(),
            "cls_head": cls_head.state_dict(),
            "args": json_ready(vars(args)),
            "label_list": label_list,
            "n_genes": base_train_ds.n_genes,
        }
        torch.save(checkpoint, args.output_dir / "best.pt")

    torch.save(checkpoint, args.output_dir / "final.pt")
    (args.output_dir / "history.json").write_text(json.dumps(history, indent=2))
    summary = {
        "monitor": args.monitor,
        "best_monitor": best_monitor if best_monitor > -float("inf") else None,
        "best_epoch": checkpoint.get("epoch"),
        "best_val_acc_expr": checkpoint.get("val_acc_expr"),
        "best_val_acc_img": checkpoint.get("val_acc_img"),
        "best_val_macro_f1_expr": checkpoint.get("val_macro_f1_expr"),
        "label_list": label_list,
        "args": json_ready(vars(args)),
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nDone. Best {args.monitor}: {best_monitor:.4f}")
    print(f"Best val expr acc: {checkpoint.get('val_acc_expr'):.4f}")
    print(f"Best val expr macro-F1: {checkpoint.get('val_macro_f1_expr'):.4f}")
    print(f"Checkpoints saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
