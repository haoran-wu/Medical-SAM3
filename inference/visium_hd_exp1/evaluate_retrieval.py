#!/usr/bin/env python3
"""
Evaluate the trained contrastive model.

Metrics:
  1. Cross-modal retrieval recall@k (expression -> image patch)
     For each test expression vector, retrieve top-k image patches by cosine similarity.
     Report recall@{1,5,10} per region label and overall.

  2. Linear probe accuracy
     Freeze both encoders, train logistic regression on 128-dim embeddings.
     Compare: image-only, expression-only, concatenated.

Outputs (in --output-dir):
  metrics.json        — recall@k table and linear probe results
  embeddings.npz      — all test embeddings for further analysis
  retrieval_examples/ — saved image grids showing top-5 retrieved patches

Usage:
  python evaluate_retrieval.py \
    --checkpoint output/visium_hd_exp1/contrastive_checkpoints/best.pt \
    --items-csv  output/visium_hd_exp1/patch_dataset/items.csv \
    --expr-npz   output/visium_hd_exp1/patch_dataset/expr_norm.npz \
    --image-path /vast/.../Exp1/spatial/tissue_hires_image.png \
    --output-dir output/visium_hd_exp1/retrieval_eval/
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = Path("/vast/palmer/pi/xiting_yan/hw568/collections_spatial_datasets/VisiumHD_Human_Lung/Exp1")
DEFAULT_IMAGE_PATH = DATA_DIR / "spatial" / "tissue_hires_image.png"
PATCH_DATASET_DIR = PROJECT_ROOT / "output" / "visium_hd_exp1" / "patch_dataset"
CHECKPOINT_DIR = PROJECT_ROOT / "output" / "visium_hd_exp1" / "contrastive_checkpoints"


def load_model(checkpoint_path: Path, device: torch.device, model_type: str = "contrastive"):
    sys.path.insert(0, str(Path(__file__).parent))

    ckpt = torch.load(checkpoint_path, map_location=device)
    n_genes   = ckpt["n_genes"]
    a         = ckpt["args"]
    embed_dim = a.get("embed_dim", 128)
    dropout   = a.get("dropout", 0.3)

    if model_type == "classification":
        from train_classification import ImageEncoder, ExpressionEncoder
        img_enc  = ImageEncoder(embed_dim=embed_dim, freeze_backbone=False).to(device)
        expr_enc = ExpressionEncoder(n_genes=n_genes, embed_dim=embed_dim, dropout=dropout).to(device)
    else:
        from train_contrastive import ImageEncoder, ExpressionEncoder
        img_enc  = ImageEncoder(out_dim=embed_dim, freeze_backbone=False).to(device)
        expr_enc = ExpressionEncoder(n_genes=n_genes, out_dim=embed_dim, dropout=dropout).to(device)

    img_enc.load_state_dict(ckpt["img_enc"])
    expr_enc.load_state_dict(ckpt["expr_enc"])
    img_enc.eval()
    expr_enc.eval()
    return img_enc, expr_enc, ckpt["label_list"]


@torch.no_grad()
def encode_all(loader: DataLoader, img_enc, expr_enc, device: torch.device):
    all_z_img, all_z_expr, all_labels = [], [], []
    for patches, exprs, labels in loader:
        patches = patches.to(device)
        exprs = exprs.to(device)
        all_z_img.append(img_enc(patches).cpu())
        all_z_expr.append(expr_enc(exprs).cpu())
        all_labels.append(labels)
    return (
        torch.cat(all_z_img),
        torch.cat(all_z_expr),
        torch.cat(all_labels),
    )


def recall_at_k(z_query: torch.Tensor, z_gallery: torch.Tensor,
                query_labels: torch.Tensor, gallery_labels: torch.Tensor,
                ks=(1, 5, 10)) -> dict:
    """
    For each query (expression vector), retrieve top-k from gallery (image embeddings)
    by cosine similarity. A hit = at least one retrieved item shares the query's label.
    """
    sim = z_query @ z_gallery.T  # [N_query, N_gallery]
    results = {}
    for k in ks:
        topk_indices = sim.topk(k, dim=1).indices  # [N_query, k]
        topk_labels = gallery_labels[topk_indices]  # [N_query, k]
        hits = (topk_labels == query_labels.unsqueeze(1)).any(dim=1).float()
        results[f"recall@{k}"] = float(hits.mean())
    return results


def recall_by_label(z_query, z_gallery, query_labels, gallery_labels, label_list, ks=(1, 5, 10)):
    sim = z_query @ z_gallery.T
    per_label = {}
    for lb_idx, lb_name in enumerate(label_list):
        mask = (query_labels == lb_idx)
        if mask.sum() == 0:
            continue
        sim_lb = sim[mask]
        q_lb = query_labels[mask]
        per_label[lb_name] = {}
        for k in ks:
            topk_indices = sim_lb.topk(k, dim=1).indices
            topk_labels = gallery_labels[topk_indices]
            hits = (topk_labels == q_lb.unsqueeze(1)).any(dim=1).float()
            per_label[lb_name][f"recall@{k}"] = float(hits.mean())
    return per_label


def linear_probe(z_train: np.ndarray, y_train: np.ndarray,
                 z_test: np.ndarray, y_test: np.ndarray) -> float:
    from sklearn.linear_model import LogisticRegression  # type: ignore
    clf = LogisticRegression(max_iter=500, C=1.0)
    clf.fit(z_train, y_train)
    return float(clf.score(z_test, y_test))


def save_retrieval_examples(
    z_query: torch.Tensor,
    z_gallery: torch.Tensor,
    query_labels: torch.Tensor,
    gallery_labels: torch.Tensor,
    label_list: list,
    gallery_df,
    image,
    patch_size: int,
    output_dir: Path,
    n_examples: int = 3,
    top_k: int = 5,
) -> None:
    from PIL import Image, ImageDraw

    sim = z_query @ z_gallery.T
    w, h = image.size
    half = patch_size // 2

    for lb_idx, lb_name in enumerate(label_list):
        mask = (query_labels == lb_idx)
        if mask.sum() == 0:
            continue
        indices = mask.nonzero(as_tuple=True)[0]
        chosen = indices[:n_examples]
        topk_per_query = sim[chosen].topk(top_k, dim=1).indices

        rows = []
        for qi, row_indices in zip(chosen.tolist(), topk_per_query.tolist()):
            row_patches = []
            # query patch
            row_df = gallery_df.iloc[qi]
            cx, cy = int(round(row_df["hires_x"])), int(round(row_df["hires_y"]))
            patch = image.crop((max(0, cx - half), max(0, cy - half),
                                min(w, cx + half), min(h, cy + half)))
            if patch.size != (patch_size, patch_size):
                patch = patch.resize((patch_size, patch_size))
            row_patches.append(patch)
            for gi in row_indices:
                row_df_g = gallery_df.iloc[gi]
                cx2, cy2 = int(round(row_df_g["hires_x"])), int(round(row_df_g["hires_y"]))
                g_patch = image.crop((max(0, cx2 - half), max(0, cy2 - half),
                                      min(w, cx2 + half), min(h, cy2 + half)))
                if g_patch.size != (patch_size, patch_size):
                    g_patch = g_patch.resize((patch_size, patch_size))
                row_patches.append(g_patch)
            rows.append(row_patches)

        n_cols = 1 + top_k
        grid = Image.new("RGB", (n_cols * patch_size, n_examples * patch_size), (200, 200, 200))
        for ri, row_patches in enumerate(rows):
            for ci, p in enumerate(row_patches):
                grid.paste(p, (ci * patch_size, ri * patch_size))
        # Red border around query column
        draw = ImageDraw.Draw(grid)
        for ri in range(len(rows)):
            draw.rectangle(
                [0, ri * patch_size, patch_size - 1, (ri + 1) * patch_size - 1],
                outline=(220, 30, 30), width=3,
            )
        out_path = output_dir / f"retrieval_{lb_name.replace(' ', '_')}.png"
        grid.save(out_path)
        print(f"  Saved retrieval examples for '{lb_name}' → {out_path.name}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate contrastive retrieval.")
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT_DIR / "best.pt")
    parser.add_argument("--model-type", type=str, default="contrastive",
                        choices=["contrastive", "classification"],
                        help="Which training script produced the checkpoint")
    parser.add_argument("--items-csv", type=Path, default=PATCH_DATASET_DIR / "items.csv")
    parser.add_argument("--expr-npz", type=Path, default=PATCH_DATASET_DIR / "expr_log1p.npz")
    parser.add_argument("--scaler-npz", type=Path, default=PATCH_DATASET_DIR / "scaler.npz")
    parser.add_argument("--image-path", type=Path, default=DEFAULT_IMAGE_PATH)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "output" / "visium_hd_exp1" / "retrieval_eval")
    parser.add_argument("--patch-size", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--save-examples", action="store_true",
                        help="Save retrieval example grids per label")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print("Loading model...")
    img_enc, expr_enc, label_list = load_model(args.checkpoint, device, args.model_type)
    print(f"  Labels: {label_list}")

    sys.path.insert(0, str(Path(__file__).parent))
    from build_patch_expression_dataset import PatchExpressionDataset
    import pandas as pd

    train_ds = PatchExpressionDataset(
        args.items_csv, args.expr_npz, args.scaler_npz, args.image_path,
        patch_size=args.patch_size, split="train", augment=False,
    )
    test_ds = PatchExpressionDataset(
        args.items_csv, args.expr_npz, args.scaler_npz, args.image_path,
        patch_size=args.patch_size, split="test", augment=False,
    )
    print(f"  Train: {len(train_ds)}, Test: {len(test_ds)}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=False,
                               num_workers=args.num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers, pin_memory=True)

    print("Encoding train set (gallery)...")
    z_img_train, z_expr_train, y_train = encode_all(train_loader, img_enc, expr_enc, device)
    print("Encoding test set (queries)...")
    z_img_test, z_expr_test, y_test = encode_all(test_loader, img_enc, expr_enc, device)

    print("\n── Cross-modal retrieval: expression -> image ──")
    overall = recall_at_k(z_expr_test, z_img_train, y_test, y_train)
    print(f"  Overall: {overall}")

    per_label = recall_by_label(z_expr_test, z_img_train, y_test, y_train, label_list)
    for lb, metrics in per_label.items():
        print(f"  {lb}: {metrics}")

    print("\n── Linear probe ──")
    train_z = z_img_train.numpy()
    test_z = z_img_test.numpy()
    y_tr = y_train.numpy()
    y_te = y_test.numpy()
    acc_img = linear_probe(train_z, y_tr, test_z, y_te)
    acc_expr = linear_probe(z_expr_train.numpy(), y_tr, z_expr_test.numpy(), y_te)
    acc_concat = linear_probe(
        np.concatenate([train_z, z_expr_train.numpy()], axis=1), y_tr,
        np.concatenate([test_z, z_expr_test.numpy()], axis=1), y_te,
    )
    print(f"  Image-only:       {acc_img:.4f}")
    print(f"  Expression-only:  {acc_expr:.4f}")
    print(f"  Concatenated:     {acc_concat:.4f}")
    random_baseline = 1.0 / len(label_list)
    print(f"  Random baseline:  {random_baseline:.4f}")

    metrics = {
        "retrieval_overall": overall,
        "retrieval_per_label": per_label,
        "linear_probe": {
            "image_only": acc_img,
            "expression_only": acc_expr,
            "concatenated": acc_concat,
            "random_baseline": random_baseline,
        },
        "n_train": len(train_ds),
        "n_test": len(test_ds),
        "label_list": label_list,
        "checkpoint": str(args.checkpoint),
    }
    metrics_path = args.output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2))
    print(f"\nSaved metrics → {metrics_path}")

    emb_path = args.output_dir / "embeddings.npz"
    np.savez(
        emb_path,
        z_img_train=z_img_train.numpy(), z_expr_train=z_expr_train.numpy(), y_train=y_tr,
        z_img_test=z_img_test.numpy(), z_expr_test=z_expr_test.numpy(), y_test=y_te,
        label_list=np.array(label_list),
    )
    print(f"Saved embeddings → {emb_path}")

    if args.save_examples:
        print("\nSaving retrieval example grids...")
        from PIL import Image as PILImage
        PILImage.MAX_IMAGE_PIXELS = None
        image = PILImage.open(args.image_path).convert("RGB")
        examples_dir = args.output_dir / "retrieval_examples"
        examples_dir.mkdir(exist_ok=True)
        save_retrieval_examples(
            z_query=z_expr_test,
            z_gallery=z_img_train,
            query_labels=y_test,
            gallery_labels=y_train,
            label_list=label_list,
            gallery_df=train_ds.df,
            image=image,
            patch_size=args.patch_size,
            output_dir=examples_dir,
        )


if __name__ == "__main__":
    main()
