#!/usr/bin/env python3
"""
Contrastive alignment training: H&E image patches <-> gene expression vectors.

Architecture:
  Image encoder:      ResNet50 (ImageNet pretrained) + projection head -> 128-dim L2-normalized
  Expression encoder: MLP (n_genes -> 256 -> 256 -> 128) + L2-normalize

Loss: symmetric InfoNCE (NT-Xent) with learnable temperature.

Training schedule:
  Phase 1 (epochs 0..freeze_epochs): backbone frozen, only projection head + expr encoder trained
  Phase 2 (freeze_epochs..total_epochs): full network fine-tuned at lower lr

Usage (submit as SLURM GPU job):
  python train_contrastive.py \
    --items-csv output/visium_hd_exp1/patch_dataset/items.csv \
    --expr-npz  output/visium_hd_exp1/patch_dataset/expr_norm.npz \
    --image-path /vast/.../Exp1/spatial/tissue_hires_image.png \
    --output-dir output/visium_hd_exp1/contrastive_checkpoints/
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
DATA_DIR = Path("/vast/palmer/pi/xiting_yan/hw568/collections_spatial_datasets/VisiumHD_Human_Lung/Exp1")
DEFAULT_IMAGE_PATH = DATA_DIR / "spatial" / "tissue_hires_image.png"
PATCH_DATASET_DIR = PROJECT_ROOT / "output" / "visium_hd_exp1" / "patch_dataset"


# ─── Model components ────────────────────────────────────────────────────────

class ProjectionHead(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int = 256, out_dim: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.net(x), dim=-1)


class ExpressionEncoder(nn.Module):
    def __init__(self, n_genes: int, hidden_dim: int = 256, out_dim: int = 128, dropout: float = 0.3) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_genes, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.net(x), dim=-1)


class ImageEncoder(nn.Module):
    def __init__(self, out_dim: int = 128, freeze_backbone: bool = True) -> None:
        super().__init__()
        from torchvision.models import resnet50, ResNet50_Weights
        backbone = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        self.backbone = nn.Sequential(*list(backbone.children())[:-1])  # drop final FC
        self.proj = ProjectionHead(2048, 256, out_dim)
        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.backbone(x).flatten(1)
        return self.proj(feat)

    def unfreeze_backbone(self) -> None:
        for p in self.backbone.parameters():
            p.requires_grad_(True)


# ─── Loss ────────────────────────────────────────────────────────────────────

class InfoNCELoss(nn.Module):
    def __init__(self, tau: float = 0.07) -> None:
        super().__init__()
        self.tau = tau

    def forward(self, z_img: torch.Tensor, z_expr: torch.Tensor) -> torch.Tensor:
        sim = z_img @ z_expr.T / self.tau  # [B, B]
        labels = torch.arange(len(z_img), device=z_img.device)
        loss_i2e = F.cross_entropy(sim, labels)
        loss_e2i = F.cross_entropy(sim.T, labels)
        return (loss_i2e + loss_e2i) / 2


# ─── Training helpers ─────────────────────────────────────────────────────────

def make_optimizer(img_enc: ImageEncoder, expr_enc: ExpressionEncoder,
                   loss_fn: InfoNCELoss, lr_proj: float, lr_backbone: float,
                   weight_decay: float) -> torch.optim.Optimizer:
    return torch.optim.AdamW([
        {"params": img_enc.proj.parameters(), "lr": lr_proj},
        {"params": img_enc.backbone.parameters(), "lr": lr_backbone},
        {"params": expr_enc.parameters(), "lr": lr_proj},
    ], weight_decay=weight_decay)


def run_epoch(loader: DataLoader, img_enc: ImageEncoder, expr_enc: ExpressionEncoder,
              loss_fn: InfoNCELoss, optimizer: torch.optim.Optimizer | None,
              device: torch.device, train: bool) -> float:
    img_enc.train(train)
    expr_enc.train(train)
    total_loss = 0.0
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for patches, exprs, _ in loader:
            patches = patches.to(device)
            exprs = exprs.to(device)
            z_img = img_enc(patches)
            z_expr = expr_enc(exprs)
            loss = loss_fn(z_img, z_expr)
            if train and optimizer is not None:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * len(patches)
    return total_loss / len(loader.dataset)


# ─── Main ────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Train contrastive H&E <-> expression model.")
    parser.add_argument("--items-csv", type=Path, default=PATCH_DATASET_DIR / "items.csv")
    parser.add_argument("--expr-npz", type=Path, default=PATCH_DATASET_DIR / "expr_log1p.npz")
    parser.add_argument("--scaler-npz", type=Path, default=PATCH_DATASET_DIR / "scaler.npz")
    parser.add_argument("--image-path", type=Path, default=DEFAULT_IMAGE_PATH)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "output" / "visium_hd_exp1" / "contrastive_checkpoints")
    parser.add_argument("--patch-size", type=int, default=64)
    parser.add_argument("--embed-dim", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--freeze-epochs", type=int, default=30,
                        help="Epochs to train with frozen backbone")
    parser.add_argument("--total-epochs", type=int, default=80)
    parser.add_argument("--lr", type=float, default=1e-4, help="LR for projection head and expression encoder")
    parser.add_argument("--lr-backbone", type=float, default=1e-5, help="LR for image backbone after unfreezing")
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--tau", type=float, default=0.07, help="InfoNCE temperature (fixed)")
    parser.add_argument("--dropout", type=float, default=0.3, help="Dropout rate in expression encoder")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Import dataset from build_patch_expression_dataset.py
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
    print(f"  Train: {len(train_ds)}, Val: {len(val_ds)}")
    print(f"  n_genes: {train_ds.n_genes}, n_classes: {train_ds.n_classes}")
    print(f"  Labels: {train_ds.label_list}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                               num_workers=args.num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers, pin_memory=True)

    img_enc = ImageEncoder(out_dim=args.embed_dim, freeze_backbone=True).to(device)
    expr_enc = ExpressionEncoder(n_genes=train_ds.n_genes, out_dim=args.embed_dim, dropout=args.dropout).to(device)
    loss_fn = InfoNCELoss(tau=args.tau)

    optimizer = make_optimizer(img_enc, expr_enc, loss_fn,
                               lr_proj=args.lr, lr_backbone=0.0,
                               weight_decay=args.weight_decay)

    best_val_loss = float("inf")
    history = []

    for epoch in range(args.total_epochs):
        # Phase transition: unfreeze backbone and rebuild optimizer
        if epoch == args.freeze_epochs:
            print(f"\nEpoch {epoch}: unfreezing backbone, lr_backbone={args.lr_backbone}")
            img_enc.unfreeze_backbone()
            optimizer = make_optimizer(img_enc, expr_enc, loss_fn,
                                       lr_proj=args.lr, lr_backbone=args.lr_backbone,
                                       weight_decay=args.weight_decay)

        train_loss = run_epoch(train_loader, img_enc, expr_enc, loss_fn, optimizer, device, train=True)
        val_loss = run_epoch(val_loader, img_enc, expr_enc, loss_fn, None, device, train=False)

        print(f"Epoch {epoch + 1:03d}/{args.total_epochs} | train={train_loss:.4f} val={val_loss:.4f} tau={loss_fn.tau:.4f}")
        history.append({"epoch": epoch + 1, "train_loss": train_loss, "val_loss": val_loss, "tau": loss_fn.tau})

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint = {
                "epoch": epoch + 1,
                "val_loss": val_loss,
                "img_enc": img_enc.state_dict(),
                "expr_enc": expr_enc.state_dict(),
                "args": vars(args),
                "label_list": train_ds.label_list,
                "n_genes": train_ds.n_genes,
            }
            torch.save(checkpoint, args.output_dir / "best.pt")

    torch.save(checkpoint, args.output_dir / "final.pt")

    (args.output_dir / "history.json").write_text(json.dumps(history, indent=2))
    print(f"\nDone. Best val loss: {best_val_loss:.4f}")
    print(f"Checkpoints saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
