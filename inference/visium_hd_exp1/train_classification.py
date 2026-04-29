#!/usr/bin/env python3
"""
Dual-tower classification-based alignment for H&E patches <-> gene expression.

Each encoder independently predicts the region label (cross-entropy).
An optional L2 alignment term pulls the two 128-dim embeddings together.

Loss = CE(img_pred, label) + CE(expr_pred, label) + align_weight * MSE(img_emb, expr_emb)

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
from torch.utils.data import DataLoader

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
    def __init__(self, embed_dim: int = 128, freeze_backbone: bool = True) -> None:
        super().__init__()
        from torchvision.models import resnet50, ResNet50_Weights
        backbone = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        self.backbone = nn.Sequential(*list(backbone.children())[:-1])
        self.proj = nn.Sequential(
            nn.Linear(2048, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, embed_dim),
            nn.LayerNorm(embed_dim),
        )
        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(self.backbone(x).flatten(1))

    def unfreeze_backbone(self) -> None:
        for p in self.backbone.parameters():
            p.requires_grad_(True)


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


def run_epoch(loader, img_enc, expr_enc, cls_head, optimizer,
              device, align_weight, train, class_weights=None):
    img_enc.train(train)
    expr_enc.train(train)
    cls_head.train(train)

    total_loss = total_ce_img = total_ce_expr = total_align = 0.0
    total_correct_img = total_correct_expr = total_n = 0

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
            align   = F.mse_loss(F.normalize(z_img, dim=-1),
                                 F.normalize(z_expr, dim=-1))
            loss    = ce_img + ce_expr + align_weight * align

            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            n = len(labels)
            total_loss       += loss.item()    * n
            total_ce_img     += ce_img.item()  * n
            total_ce_expr    += ce_expr.item() * n
            total_align      += align.item()   * n
            total_correct_img  += (logits_img.argmax(1)  == labels).sum().item()
            total_correct_expr += (logits_expr.argmax(1) == labels).sum().item()
            total_n += n

    N = total_n
    return {
        "loss":       total_loss / N,
        "ce_img":     total_ce_img / N,
        "ce_expr":    total_ce_expr / N,
        "align":      total_align / N,
        "acc_img":    total_correct_img / N,
        "acc_expr":   total_correct_expr / N,
    }


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
    parser.add_argument("--patch-size",   type=int,   default=64)
    parser.add_argument("--batch-size",   type=int,   default=256)
    parser.add_argument("--freeze-epochs",type=int,   default=20)
    parser.add_argument("--total-epochs", type=int,   default=60)
    parser.add_argument("--lr",           type=float, default=1e-4)
    parser.add_argument("--lr-backbone",  type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout",      type=float, default=0.3)
    parser.add_argument("--align-weight", type=float, default=0.5,
                        help="Weight for MSE embedding alignment term")
    parser.add_argument("--num-workers",  type=int,   default=8)
    parser.add_argument("--seed",         type=int,   default=42)
    args = parser.parse_args()

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
        patch_size=args.patch_size, split="train", augment=True,
    )
    val_ds = PatchExpressionDataset(
        args.items_csv, args.expr_npz, args.scaler_npz, args.image_path,
        patch_size=args.patch_size, split="val", augment=False,
    )
    n_classes = train_ds.n_classes
    print(f"  Train: {len(train_ds)}, Val: {len(val_ds)}, Classes: {n_classes}")
    print(f"  Labels: {train_ds.label_list}")

    # Inverse-frequency class weights to handle severe imbalance (pigment=686 vs tumor=152k)
    label_counts = np.bincount(train_ds.labels_idx, minlength=n_classes).astype(np.float32)
    class_weights = torch.tensor(1.0 / np.maximum(label_counts, 1)).to(device)
    class_weights = class_weights / class_weights.sum() * n_classes  # normalize to n_classes mean=1
    print(f"  Class weights: { {train_ds.label_list[i]: f'{class_weights[i].item():.3f}' for i in range(n_classes)} }")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                               num_workers=args.num_workers, pin_memory=True, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False,
                               num_workers=args.num_workers, pin_memory=True)

    img_enc  = ImageEncoder(embed_dim=args.embed_dim, freeze_backbone=True).to(device)
    expr_enc = ExpressionEncoder(n_genes=train_ds.n_genes, embed_dim=args.embed_dim,
                                  dropout=args.dropout).to(device)
    cls_head = ClassificationHead(args.embed_dim, n_classes).to(device)

    optimizer = make_optimizer(img_enc, expr_enc, cls_head,
                               lr_proj=args.lr, lr_backbone=0.0,
                               weight_decay=args.weight_decay)

    best_val_acc = 0.0
    history = []

    for epoch in range(args.total_epochs):
        if epoch == args.freeze_epochs:
            print(f"\nEpoch {epoch}: unfreezing backbone (lr={args.lr_backbone})")
            img_enc.unfreeze_backbone()
            optimizer = make_optimizer(img_enc, expr_enc, cls_head,
                                       lr_proj=args.lr, lr_backbone=args.lr_backbone,
                                       weight_decay=args.weight_decay)

        tr = run_epoch(train_loader, img_enc, expr_enc, cls_head, optimizer,
                       device, args.align_weight, train=True, class_weights=class_weights)
        va = run_epoch(val_loader,   img_enc, expr_enc, cls_head, None,
                       device, args.align_weight, train=False, class_weights=class_weights)

        print(
            f"Epoch {epoch+1:03d}/{args.total_epochs} | "
            f"train_acc img={tr['acc_img']:.3f} expr={tr['acc_expr']:.3f} | "
            f"val_acc img={va['acc_img']:.3f} expr={va['acc_expr']:.3f} | "
            f"align={va['align']:.4f}"
        )
        history.append({"epoch": epoch + 1, "train": tr, "val": va})

        # Save best by val expression accuracy (the harder task)
        val_acc = va["acc_expr"]
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            checkpoint = {
                "epoch": epoch + 1,
                "val_acc_expr": val_acc,
                "val_acc_img":  va["acc_img"],
                "img_enc":  img_enc.state_dict(),
                "expr_enc": expr_enc.state_dict(),
                "cls_head": cls_head.state_dict(),
                "args":      vars(args),
                "label_list": train_ds.label_list,
                "n_genes":    train_ds.n_genes,
            }
            torch.save(checkpoint, args.output_dir / "best.pt")

    torch.save(checkpoint, args.output_dir / "final.pt")
    (args.output_dir / "history.json").write_text(json.dumps(history, indent=2))
    print(f"\nDone. Best val expr acc: {best_val_acc:.4f}")
    print(f"Checkpoints saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
