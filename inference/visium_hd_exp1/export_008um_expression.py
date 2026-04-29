#!/usr/bin/env python3
"""
Export VisiumHD 8um bin metadata and gene expression matrix.

For each in-tissue bin, records:
  barcode, fullres_x, fullres_y, hires_x, hires_y, region_label

Region label is assigned by looking up the bin's hires position in the
binary region masks produced by render_geojson_masks.py.

Outputs:
  bin_metadata_008um.csv      — per-bin spatial + label table
  expression_008um.npz        — sparse expression matrix (X), gene_names, barcodes
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = Path("/vast/palmer/pi/xiting_yan/hw568/collections_spatial_datasets/VisiumHD_Human_Lung/Exp1")
BIN_DIR = DATA_DIR / "binned_outputs" / "square_008um"
DEFAULT_MASKS_DIR = PROJECT_ROOT / "output" / "visium_hd_exp1" / "region_masks"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "output" / "visium_hd_exp1" / "exported_expression"


def load_scalef(data_dir: Path) -> float:
    scalef_path = data_dir / "spatial" / "scalefactors_json.json"
    if scalef_path.exists():
        data = json.loads(scalef_path.read_text())
        return float(data["tissue_hires_scalef"])
    bin_scalef_path = data_dir / "binned_outputs" / "square_008um" / "spatial" / "scalefactors_json.json"
    data = json.loads(bin_scalef_path.read_text())
    return float(data["tissue_hires_scalef"])


def load_tissue_positions(bin_dir: Path) -> pd.DataFrame:
    parquet_path = bin_dir / "spatial" / "tissue_positions.parquet"
    csv_path = bin_dir / "spatial" / "tissue_positions.csv"
    if parquet_path.exists():
        return pd.read_parquet(parquet_path)
    if csv_path.exists():
        return pd.read_csv(csv_path)
    raise FileNotFoundError(f"No tissue_positions file in {bin_dir / 'spatial'}")


def load_expression(bin_dir: Path):
    """Return (adata, gene_names, barcodes) using scanpy."""
    import scanpy as sc  # type: ignore
    h5_path = bin_dir / "filtered_feature_bc_matrix.h5"
    if not h5_path.exists():
        raise FileNotFoundError(f"Expression H5 not found: {h5_path}")
    adata = sc.read_10x_h5(str(h5_path))
    adata.var_names_make_unique()
    gene_names = np.array(adata.var_names.tolist())
    barcodes = np.array(adata.obs_names.tolist())
    X = adata.X  # sparse matrix, shape [n_bins x n_genes]
    return X, gene_names, barcodes


def assign_region_labels(
    hires_x: np.ndarray,
    hires_y: np.ndarray,
    masks_dir: Path,
) -> np.ndarray:
    """
    For each bin, look up its hires pixel position in each label mask PNG.
    Returns an array of region label strings (or 'background' if no mask covers it).
    """
    from PIL import Image  # type: ignore

    Image.MAX_IMAGE_PIXELS = None

    summary_path = masks_dir / "region_summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(
            f"region_summary.json not found in {masks_dir}. "
            "Run render_geojson_masks.py first."
        )
    summary = json.loads(summary_path.read_text())
    label_entries = summary["labels"]

    n = len(hires_x)
    labels = np.full(n, "background", dtype=object)

    hx = np.round(hires_x).astype(int)
    hy = np.round(hires_y).astype(int)

    for entry in label_entries:
        mask_path = Path(entry["mask_path"])
        if not mask_path.exists():
            print(f"  WARNING: mask not found: {mask_path}")
            continue
        mask = np.array(Image.open(mask_path)) > 0  # bool array [H, W]
        h, w = mask.shape

        valid = (hx >= 0) & (hx < w) & (hy >= 0) & (hy < h)
        in_mask = valid & mask[np.clip(hy, 0, h - 1), np.clip(hx, 0, w - 1)]
        labels[in_mask] = entry["label"]

    return labels


def main() -> None:
    parser = argparse.ArgumentParser(description="Export VisiumHD 8um bin metadata + expression.")
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR,
                        help="Root Exp1 data directory (contains spatial/ and binned_outputs/)")
    parser.add_argument("--masks-dir", type=Path, default=DEFAULT_MASKS_DIR,
                        help="Directory with region mask PNGs + region_summary.json")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--in-tissue-only", action="store_true", default=True,
                        help="Keep only in-tissue bins (default: True)")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    bin_dir = args.data_dir / "binned_outputs" / "square_008um"

    print("Loading scale factor...")
    scalef = load_scalef(args.data_dir)
    print(f"  tissue_hires_scalef = {scalef}")

    print("Loading tissue positions...")
    positions = load_tissue_positions(bin_dir)
    print(f"  Total bins: {len(positions)}")

    # Space Ranger column names vary slightly; normalize them
    col_map = {}
    for col in positions.columns:
        low = col.lower()
        if "barcode" in low:
            col_map["barcode"] = col
        elif "in_tissue" in low:
            col_map["in_tissue"] = col
        elif "col" in low and ("fullres" in low or "pxl" in low):
            col_map["fullres_x"] = col
        elif "row" in low and ("fullres" in low or "pxl" in low):
            col_map["fullres_y"] = col

    if "barcode" not in col_map:
        # barcode may be the index
        positions = positions.reset_index()
        for col in positions.columns:
            if "barcode" in col.lower():
                col_map["barcode"] = col
                break

    required = ["barcode", "in_tissue", "fullres_x", "fullres_y"]
    missing = [k for k in required if k not in col_map]
    if missing:
        print(f"  Columns found: {positions.columns.tolist()}")
        raise KeyError(f"Could not map required columns: {missing}")

    df = pd.DataFrame({
        "barcode": positions[col_map["barcode"]].values,
        "in_tissue": positions[col_map["in_tissue"]].values.astype(int),
        "fullres_x": positions[col_map["fullres_x"]].values.astype(float),
        "fullres_y": positions[col_map["fullres_y"]].values.astype(float),
    })

    if args.in_tissue_only:
        df = df[df["in_tissue"] == 1].reset_index(drop=True)
        print(f"  In-tissue bins: {len(df)}")

    df["hires_x"] = df["fullres_x"] * scalef
    df["hires_y"] = df["fullres_y"] * scalef

    print("Loading expression matrix (this may take a minute)...")
    X, gene_names, expr_barcodes = load_expression(bin_dir)
    print(f"  Expression shape: {X.shape} ({len(gene_names)} genes)")

    # Align positions to expression matrix by barcode
    expr_barcode_index = {b: i for i, b in enumerate(expr_barcodes)}
    expr_indices = df["barcode"].map(expr_barcode_index)
    valid_mask = expr_indices.notna()
    n_missing = (~valid_mask).sum()
    if n_missing > 0:
        print(f"  WARNING: {n_missing} bins have no expression entry; dropping them")
    df = df[valid_mask].reset_index(drop=True)
    expr_indices = expr_indices[valid_mask].astype(int).values

    X_aligned = X[expr_indices]  # reorder rows to match df

    print("Assigning region labels from masks...")
    df["region_label"] = assign_region_labels(
        df["hires_x"].values, df["hires_y"].values, args.masks_dir
    )
    label_counts = df["region_label"].value_counts()
    print("  Region label counts:")
    for label, count in label_counts.items():
        print(f"    {label}: {count}")

    metadata_path = args.output_dir / "bin_metadata_008um.csv"
    df.to_csv(metadata_path, index=False)
    print(f"\nSaved bin metadata ({len(df)} bins) → {metadata_path}")

    expr_path = args.output_dir / "expression_008um.npz"
    if sp.issparse(X_aligned):
        X_csr = X_aligned.tocsr()
        np.savez(
            expr_path,
            data=X_csr.data,
            indices=X_csr.indices,
            indptr=X_csr.indptr,
            shape=np.array(X_csr.shape),
            gene_names=gene_names,
            barcodes=df["barcode"].values,
        )
    else:
        np.savez(expr_path, X=X_aligned, gene_names=gene_names, barcodes=df["barcode"].values)
    print(f"Saved expression matrix → {expr_path}")


if __name__ == "__main__":
    main()
