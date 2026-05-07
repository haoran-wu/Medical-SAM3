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

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    n_genes   = ckpt["n_genes"]
    a         = ckpt["args"]
    embed_dim = a.get("embed_dim", 128)
    dropout   = a.get("dropout", 0.3)
    image_backbone = a.get("image_backbone", ckpt.get("image_backbone", "resnet50"))

    if model_type == "classification":
        from train_classification import ImageEncoder, ExpressionEncoder
        img_enc  = ImageEncoder(
            embed_dim=embed_dim,
            freeze_backbone=False,
            image_backbone=image_backbone,
        ).to(device)
        expr_enc = ExpressionEncoder(n_genes=n_genes, embed_dim=embed_dim, dropout=dropout).to(device)
    else:
        from train_contrastive import ImageEncoder, ExpressionEncoder
        img_enc  = ImageEncoder(
            out_dim=embed_dim,
            freeze_backbone=False,
            image_backbone=image_backbone,
        ).to(device)
        expr_enc = ExpressionEncoder(n_genes=n_genes, out_dim=embed_dim, dropout=dropout).to(device)

    img_enc.load_state_dict(ckpt["img_enc"])
    expr_enc.load_state_dict(ckpt["expr_enc"])
    img_enc.eval()
    expr_enc.eval()
    return img_enc, expr_enc, ckpt["label_list"], a


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


def load_encoded(path: Path):
    data = np.load(path, allow_pickle=True)
    return (
        torch.from_numpy(data["z_img"]),
        torch.from_numpy(data["z_expr"]),
        torch.from_numpy(data["labels"]),
    )


def save_encoded(path: Path, z_img: torch.Tensor, z_expr: torch.Tensor, labels: torch.Tensor) -> None:
    tmp_path = path.with_name(path.name + ".tmp.npz")
    np.savez(
        tmp_path,
        z_img=z_img.numpy(),
        z_expr=z_expr.numpy(),
        labels=labels.numpy(),
    )
    tmp_path.replace(path)


def recall_at_k(z_query: torch.Tensor, z_gallery: torch.Tensor,
                query_labels: torch.Tensor, gallery_labels: torch.Tensor,
                ks=(1, 5, 10), chunk_size: int = 1024) -> dict:
    """
    For each query (expression vector), retrieve top-k from gallery (image embeddings)
    by cosine similarity. A hit = at least one retrieved item shares the query's label.
    """
    max_k = min(max(ks), z_gallery.shape[0])
    hits_by_k = {k: [] for k in ks}
    for start in range(0, z_query.shape[0], chunk_size):
        end = min(start + chunk_size, z_query.shape[0])
        sim = z_query[start:end] @ z_gallery.T
        topk_indices = sim.topk(max_k, dim=1).indices
        topk_labels = gallery_labels[topk_indices]
        q_labels = query_labels[start:end]
        for k in ks:
            k_eff = min(k, z_gallery.shape[0])
            hits = (topk_labels[:, :k_eff] == q_labels.unsqueeze(1)).any(dim=1).float()
            hits_by_k[k].append(hits.cpu())
        del sim, topk_indices, topk_labels
    results = {}
    for k in ks:
        results[f"recall@{k}"] = float(torch.cat(hits_by_k[k]).mean())
    return results


def recall_by_label(z_query, z_gallery, query_labels, gallery_labels, label_list, ks=(1, 5, 10), chunk_size: int = 1024):
    per_label = {}
    for lb_idx, lb_name in enumerate(label_list):
        mask = (query_labels == lb_idx)
        if mask.sum() == 0:
            continue
        per_label[lb_name] = recall_at_k(
            z_query[mask],
            z_gallery,
            query_labels[mask],
            gallery_labels,
            ks=ks,
            chunk_size=chunk_size,
        )
    return per_label


def fusion_retrieval_sweep(
    z_img_query: torch.Tensor,
    z_expr_query: torch.Tensor,
    z_img_gallery: torch.Tensor,
    query_labels: torch.Tensor,
    gallery_labels: torch.Tensor,
    alphas: list[float],
    ks=(1, 5, 10),
    chunk_size: int = 1024,
) -> dict:
    """Haiku-style score-level fusion: alpha * image query + (1-alpha) * expression query."""
    z_img_query = F.normalize(z_img_query, dim=-1)
    z_expr_query = F.normalize(z_expr_query, dim=-1)
    z_img_gallery = F.normalize(z_img_gallery, dim=-1)
    results = {}
    best_alpha = None
    best_recall1 = -1.0
    for alpha in alphas:
        z_query = F.normalize(alpha * z_img_query + (1.0 - alpha) * z_expr_query, dim=-1)
        metrics = recall_at_k(z_query, z_img_gallery, query_labels, gallery_labels, ks=ks, chunk_size=chunk_size)
        results[f"{alpha:.1f}"] = metrics
        if metrics.get("recall@1", -1.0) > best_recall1:
            best_alpha = alpha
            best_recall1 = metrics["recall@1"]
    return {
        "alphas": alphas,
        "results": results,
        "best_alpha_by_recall@1": best_alpha,
        "best_recall@1": best_recall1,
    }


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
    query_df,
    gallery_df,
    image,
    crop_size: int,
    output_dir: Path,
    n_examples: int = 3,
    top_k: int = 5,
) -> None:
    from PIL import Image, ImageDraw

    sim = z_query @ z_gallery.T
    w, h = image.size
    half = crop_size // 2

    def crop_from_row(row):
        cx, cy = int(round(row["hires_x"])), int(round(row["hires_y"]))
        patch = image.crop((max(0, cx - half), max(0, cy - half),
                            min(w, cx + half), min(h, cy + half)))
        if patch.size != (crop_size, crop_size):
            patch = patch.resize((crop_size, crop_size))
        return patch

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
            row_patches.append(crop_from_row(query_df.iloc[qi]))
            for gi in row_indices:
                row_patches.append(crop_from_row(gallery_df.iloc[gi]))
            rows.append(row_patches)

        n_cols = 1 + top_k
        grid = Image.new("RGB", (n_cols * crop_size, n_examples * crop_size), (200, 200, 200))
        for ri, row_patches in enumerate(rows):
            for ci, p in enumerate(row_patches):
                grid.paste(p, (ci * crop_size, ri * crop_size))
        # Red border around query column
        draw = ImageDraw.Draw(grid)
        for ri in range(len(rows)):
            draw.rectangle(
                [0, ri * crop_size, crop_size - 1, (ri + 1) * crop_size - 1],
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
    parser.add_argument("--crop-size", type=int, default=None)
    parser.add_argument("--input-size", type=int, default=None)
    parser.add_argument("--include-labels", type=str, default=None,
                        help="Comma-separated labels to keep; defaults to checkpoint training args when present")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--save-examples", action="store_true",
                        help="Save retrieval example grids per label")
    parser.add_argument("--fusion-alpha-sweep", action="store_true",
                        help="Evaluate Haiku-style image/expression fused-query retrieval")
    parser.add_argument("--retrieval-chunk-size", type=int, default=1024,
                        help="Query chunk size for top-k retrieval to avoid materializing huge similarity matrices")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print("Loading model...")
    img_enc, expr_enc, label_list, ckpt_args = load_model(args.checkpoint, device, args.model_type)
    args.crop_size = args.crop_size if args.crop_size is not None else ckpt_args.get("crop_size", args.patch_size)
    args.input_size = args.input_size if args.input_size is not None else ckpt_args.get("input_size", args.crop_size)
    include_labels_arg = args.include_labels
    if include_labels_arg is None:
        include_labels_arg = ckpt_args.get("include_labels")
    include_labels = None
    if include_labels_arg:
        include_labels = [x.strip() for x in include_labels_arg.split(",") if x.strip()]
    print(f"  Labels: {label_list}")
    print(f"  Image crop/input: {args.crop_size}/{args.input_size}")
    if include_labels is not None:
        print(f"  Evaluation labels: {include_labels}")

    sys.path.insert(0, str(Path(__file__).parent))
    from build_patch_expression_dataset import PatchExpressionDataset
    import pandas as pd

    train_ds = PatchExpressionDataset(
        args.items_csv, args.expr_npz, args.scaler_npz, args.image_path,
        patch_size=args.patch_size, crop_size=args.crop_size, input_size=args.input_size,
        split="train", include_labels=include_labels, augment=False,
    )
    test_ds = PatchExpressionDataset(
        args.items_csv, args.expr_npz, args.scaler_npz, args.image_path,
        patch_size=args.patch_size, crop_size=args.crop_size, input_size=args.input_size,
        split="test", include_labels=include_labels, augment=False,
    )
    if train_ds.label_list != label_list:
        raise ValueError(
            f"Dataset labels {train_ds.label_list} do not match checkpoint labels {label_list}. "
            "Pass --include-labels to match the trained checkpoint."
        )
    print(f"  Train: {len(train_ds)}, Test: {len(test_ds)}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=False,
                               num_workers=args.num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers, pin_memory=True)

    train_encoded_path = args.output_dir / "encoded_train.npz"
    test_encoded_path = args.output_dir / "encoded_test.npz"
    if train_encoded_path.exists():
        print(f"Loading cached train embeddings from {train_encoded_path}...")
        z_img_train, z_expr_train, y_train = load_encoded(train_encoded_path)
    else:
        print("Encoding train set (gallery)...")
        z_img_train, z_expr_train, y_train = encode_all(train_loader, img_enc, expr_enc, device)
        save_encoded(train_encoded_path, z_img_train, z_expr_train, y_train)
        print(f"Saved cached train embeddings -> {train_encoded_path}")

    if test_encoded_path.exists():
        print(f"Loading cached test embeddings from {test_encoded_path}...")
        z_img_test, z_expr_test, y_test = load_encoded(test_encoded_path)
    else:
        print("Encoding test set (queries)...")
        z_img_test, z_expr_test, y_test = encode_all(test_loader, img_enc, expr_enc, device)
        save_encoded(test_encoded_path, z_img_test, z_expr_test, y_test)
        print(f"Saved cached test embeddings -> {test_encoded_path}")

    print("\n── Cross-modal retrieval: expression -> image ──")
    overall = recall_at_k(z_expr_test, z_img_train, y_test, y_train, chunk_size=args.retrieval_chunk_size)
    print(f"  Overall: {overall}")

    per_label = recall_by_label(
        z_expr_test,
        z_img_train,
        y_test,
        y_train,
        label_list,
        chunk_size=args.retrieval_chunk_size,
    )
    for lb, metrics in per_label.items():
        print(f"  {lb}: {metrics}")

    fusion = None
    if args.fusion_alpha_sweep:
        print("\n── Fusion retrieval sweep: alpha * image + (1-alpha) * expression -> image ──")
        fusion = fusion_retrieval_sweep(
            z_img_query=z_img_test,
            z_expr_query=z_expr_test,
            z_img_gallery=z_img_train,
            query_labels=y_test,
            gallery_labels=y_train,
            alphas=[round(x * 0.1, 1) for x in range(11)],
            chunk_size=args.retrieval_chunk_size,
        )
        for alpha, metrics in fusion["results"].items():
            print(f"  alpha={alpha}: {metrics}")
        print(
            f"  Best alpha by recall@1: {fusion['best_alpha_by_recall@1']} "
            f"(recall@1={fusion['best_recall@1']:.4f})"
        )

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
        "fusion_alpha_sweep": fusion,
        "n_train": len(train_ds),
        "n_test": len(test_ds),
        "label_list": label_list,
        "include_labels": include_labels,
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
            query_df=test_ds.df,
            gallery_df=train_ds.df,
            image=image,
            crop_size=args.crop_size,
            output_dir=examples_dir,
        )


if __name__ == "__main__":
    main()
