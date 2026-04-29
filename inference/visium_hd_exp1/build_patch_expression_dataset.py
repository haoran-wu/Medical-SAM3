#!/usr/bin/env python3
"""
Build a paired H&E patch + gene expression dataset for contrastive training.

Each item: (H&E patch crop, log1p-normalized expression vector, region label)

Memory-efficient design: the expression matrix stays sparse on disk.
  - log1p is applied to stored (non-zero) values only
  - per-gene mean and std are computed from the sparse matrix without densifying
  - PatchExpressionDataset loads one row at a time in __getitem__

Outputs:
  patch_dataset/items.csv        — filtered bins with split assignment
  patch_dataset/expr_log1p.npz   — sparse log1p expression (scipy sparse format)
  patch_dataset/scaler.npz       — per-gene mean and std for z-scoring at runtime
  patch_dataset/sanity_patches.png — optional visual check

Spatial train/val/test split: right strip (10% by hires_x) = test,
top strip of remainder (15% by hires_y) = val, rest = train.

Usage:
  python build_patch_expression_dataset.py \
    --metadata output/visium_hd_exp1/exported_expression/bin_metadata_008um.csv \
    --expression output/visium_hd_exp1/exported_expression/expression_008um.npz \
    --image-path /nfs/.../Exp1/spatial/tissue_hires_image.png \
    --patch-size 64 \
    --output-dir output/visium_hd_exp1/patch_dataset/ \
    --sanity-check
"""

import argparse
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import scipy.sparse as sp


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = Path("/nfs/roberts/project/pi_xy48/hw646/Exp1")
DEFAULT_IMAGE_PATH = DATA_DIR / "spatial" / "tissue_hires_image.png"
DEFAULT_METADATA = PROJECT_ROOT / "output" / "visium_hd_exp1" / "exported_expression" / "bin_metadata_008um.csv"
DEFAULT_EXPRESSION = PROJECT_ROOT / "output" / "visium_hd_exp1" / "exported_expression" / "expression_008um.npz"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "output" / "visium_hd_exp1" / "patch_dataset"


def load_expression_sparse(path: Path) -> Tuple[sp.csr_matrix, np.ndarray, np.ndarray]:
    """Load expression NPZ (saved by export_008um_expression.py) as sparse CSR."""
    data = np.load(path, allow_pickle=True)
    X = sp.csr_matrix(
        (data["data"], data["indices"], data["indptr"]),
        shape=tuple(data["shape"]),
    )
    gene_names = data["gene_names"]
    barcodes = data["barcodes"]
    return X, gene_names, barcodes


def normalize_expression_sparse(
    X: sp.csr_matrix,
) -> Tuple[sp.csr_matrix, np.ndarray, np.ndarray]:
    """
    log1p on non-zero values only (log1p(0)=0 so zeros are unchanged).
    Compute per-gene mean and std from sparse matrix without densifying.
    Returns (X_log1p, gene_means, gene_stds).
    """
    X_log = X.copy().astype(np.float32)
    X_log.data = np.log1p(X_log.data)

    n = X_log.shape[0]
    gene_means = np.asarray(X_log.mean(axis=0)).flatten().astype(np.float32)

    X_sq = X_log.copy()
    X_sq.data = X_sq.data ** 2
    e_x2 = np.asarray(X_sq.mean(axis=0)).flatten().astype(np.float32)
    gene_stds = np.sqrt(np.maximum(e_x2 - gene_means ** 2, 0.0)).astype(np.float32) + 1e-8

    return X_log, gene_means, gene_stds


def stratified_random_split(
    df: pd.DataFrame,
    val_frac: float = 0.15,
    test_frac: float = 0.10,
    seed: int = 42,
) -> np.ndarray:
    """Stratified random split by region_label so each split has the same label proportions."""
    rng = np.random.default_rng(seed)
    splits = np.full(len(df), "train", dtype=object)
    for label in df["region_label"].unique():
        idx = np.where(df["region_label"].values == label)[0]
        rng.shuffle(idx)
        n = len(idx)
        n_test = max(1, int(n * test_frac))
        n_val  = max(1, int(n * val_frac))
        splits[idx[:n_test]]            = "test"
        splits[idx[n_test:n_test+n_val]] = "val"
        # remainder stays "train"
    return splits


def sample_patches_for_sanity_check(
    df: pd.DataFrame,
    image,
    patch_size: int,
    n: int = 16,
    output_dir: Optional[Path] = None,
) -> None:
    import math
    from PIL import Image

    sample = df.sample(min(n, len(df)), random_state=42)
    cols = 4
    rows = math.ceil(len(sample) / cols)
    grid = Image.new("RGB", (cols * patch_size, rows * patch_size), (255, 255, 255))
    w, h = image.size
    half = patch_size // 2
    for i, (_, row) in enumerate(sample.iterrows()):
        cx, cy = int(round(row["hires_x"])), int(round(row["hires_y"]))
        x0, y0 = max(0, cx - half), max(0, cy - half)
        x1, y1 = min(w, x0 + patch_size), min(h, y0 + patch_size)
        patch = image.crop((x0, y0, x1, y1))
        if patch.size != (patch_size, patch_size):
            patch = patch.resize((patch_size, patch_size), Image.Resampling.BILINEAR)
        grid.paste(patch, ((i % cols) * patch_size, (i // cols) * patch_size))

    if output_dir is not None:
        out = output_dir / "sanity_patches.png"
        grid.save(out)
        print(f"  Saved sanity patches → {out}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build paired patch+expression dataset.")
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--expression", type=Path, default=DEFAULT_EXPRESSION)
    parser.add_argument("--image-path", type=Path, default=DEFAULT_IMAGE_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--patch-size", type=int, default=64)
    parser.add_argument("--min-region-bins", type=int, default=100)
    parser.add_argument("--sanity-check", action="store_true")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading bin metadata...")
    df = pd.read_csv(args.metadata)
    print(f"  {len(df)} bins total")
    print(f"  Label distribution:\n{df['region_label'].value_counts().to_string()}")

    df = df[df["region_label"] != "background"].reset_index(drop=True)
    label_counts = df["region_label"].value_counts()
    keep_labels = label_counts[label_counts >= args.min_region_bins].index
    df = df[df["region_label"].isin(keep_labels)].reset_index(drop=True)
    print(f"\n  After filtering: {len(df)} bins, labels: {keep_labels.tolist()}")

    print("\nLoading expression matrix (sparse, no densification)...")
    X, gene_names, expr_barcodes = load_expression_sparse(args.expression)
    print(f"  Shape: {X.shape}, nnz: {X.nnz}, sparsity: {1 - X.nnz / (X.shape[0]*X.shape[1]):.3f}")

    # Align to filtered df
    expr_bc_idx = {b: i for i, b in enumerate(expr_barcodes)}
    row_idx = df["barcode"].map(expr_bc_idx)
    missing = row_idx.isna().sum()
    if missing:
        print(f"  WARNING: dropping {missing} bins missing from expression matrix")
        df = df[row_idx.notna()].reset_index(drop=True)
        row_idx = row_idx[row_idx.notna()]
    X = X[row_idx.astype(int).values]
    print(f"  Aligned shape: {X.shape}")

    print("\nNormalizing (log1p + computing per-gene mean/std, stays sparse)...")
    X_log, gene_means, gene_stds = normalize_expression_sparse(X)
    print(f"  gene_means range: {gene_means.min():.4f} .. {gene_means.max():.4f}")
    print(f"  gene_stds range:  {gene_stds.min():.4f} .. {gene_stds.max():.4f}")

    print("\nAssigning stratified random splits...")
    df["split"] = stratified_random_split(df)
    print(f"  {df['split'].value_counts().to_dict()}")

    if args.sanity_check:
        print("\nLoading image for sanity patch check...")
        from PIL import Image
        Image.MAX_IMAGE_PIXELS = None
        image = Image.open(args.image_path).convert("RGB")
        sample_patches_for_sanity_check(df, image, args.patch_size, output_dir=args.output_dir)

    items_path = args.output_dir / "items.csv"
    df[["barcode", "hires_x", "hires_y", "region_label", "split"]].to_csv(
        items_path, index=True, index_label="item_idx"
    )
    print(f"\nSaved items.csv ({len(df)} rows) → {items_path}")

    expr_path = args.output_dir / "expr_log1p.npz"
    sp.save_npz(str(expr_path), X_log.tocsr())
    print(f"Saved sparse log1p expression → {expr_path}  ({expr_path.stat().st_size / 1e6:.1f} MB)")

    scaler_path = args.output_dir / "scaler.npz"
    np.savez(scaler_path, gene_means=gene_means, gene_stds=gene_stds, gene_names=gene_names)
    print(f"Saved scaler → {scaler_path}")

    print(f"\nDone. n_genes={X_log.shape[1]}, n_labels={df['region_label'].nunique()}, patch_size={args.patch_size}")


# ─── PyTorch Dataset ──────────────────────────────────────────────────────────

try:
    import torch
    from torch.utils.data import Dataset
    import torchvision.transforms as T

    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD  = [0.229, 0.224, 0.225]

    class PatchExpressionDataset(Dataset):
        """
        Yields (patch [3×H×W], expr_vec [n_genes], label_idx).

        Expression matrix is kept sparse on disk; each __getitem__ extracts one
        row, converts to dense, and applies z-score with precomputed scaler.
        The full hires image is loaded into memory once (≈63 MB).
        """

        def __init__(
            self,
            items_csv: Path,
            expr_npz: Path,
            scaler_npz: Path,
            image_path: Path,
            patch_size: int = 64,
            split: Optional[str] = None,
            augment: bool = False,
        ) -> None:
            from PIL import Image

            self.patch_size = patch_size

            df = pd.read_csv(items_csv, index_col="item_idx")
            if split is not None:
                df = df[df["split"] == split]
                # Keep item_idx as the index (do NOT reset) — it is used to
                # index into expr_npz which is aligned to the original item order.
            self.df = df  # index = original item_idx, used in __getitem__

            self.X = sp.load_npz(str(expr_npz))  # sparse [total_n_items, n_genes]

            scaler = np.load(scaler_npz)
            self.gene_means = scaler["gene_means"].astype(np.float32)
            self.gene_stds  = scaler["gene_stds"].astype(np.float32)

            self.label_list = sorted(df["region_label"].unique().tolist())
            self.label_to_idx = {lb: i for i, lb in enumerate(self.label_list)}
            self.labels_idx = np.array([self.label_to_idx[lb] for lb in df["region_label"]])

            Image.MAX_IMAGE_PIXELS = None
            self.image = Image.open(image_path).convert("RGB")
            self.img_w, self.img_h = self.image.size

            normalize = T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
            if augment:
                self.transform = T.Compose([
                    T.RandomHorizontalFlip(),
                    T.RandomVerticalFlip(),
                    T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
                    T.ToTensor(),
                    normalize,
                ])
            else:
                self.transform = T.Compose([T.ToTensor(), normalize])

        def __len__(self) -> int:
            return len(self.df)

        def __getitem__(self, idx: int):
            from PIL import Image

            row = self.df.iloc[idx]
            cx, cy = int(round(row["hires_x"])), int(round(row["hires_y"]))
            half = self.patch_size // 2
            x0 = max(0, cx - half);  y0 = max(0, cy - half)
            x1 = min(self.img_w, x0 + self.patch_size)
            y1 = min(self.img_h, y0 + self.patch_size)
            patch = self.image.crop((x0, y0, x1, y1))
            if patch.size != (self.patch_size, self.patch_size):
                patch = patch.resize((self.patch_size, self.patch_size), Image.Resampling.BILINEAR)

            patch_tensor = self.transform(patch)

            # sparse row → dense → z-score
            # self.df.index[idx] = original item_idx (row in the full matrix)
            orig_idx = self.df.index[idx]
            expr = np.asarray(self.X[orig_idx].todense()).flatten().astype(np.float32)
            expr = (expr - self.gene_means) / self.gene_stds
            expr_tensor = torch.from_numpy(expr)

            return patch_tensor, expr_tensor, int(self.labels_idx[idx])

        @property
        def n_genes(self) -> int:
            return self.X.shape[1]

        @property
        def n_classes(self) -> int:
            return len(self.label_list)

except ImportError:
    pass


if __name__ == "__main__":
    main()
