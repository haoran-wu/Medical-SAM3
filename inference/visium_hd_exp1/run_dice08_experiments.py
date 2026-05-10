#!/usr/bin/env python3
"""Try stronger H&E + molecular models for VisiumHD Exp1 region masks.

The script deliberately reports both:
- full-slide fit metrics: can we reconstruct the current labeled slide?
- spatial block holdout metrics: is the rule likely to generalize within slide?

The second number is the one to trust for scientific confidence.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from scipy import ndimage as ndi
from skimage import color, filters, morphology
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.metrics import confusion_matrix


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_HE_IMAGE = PROJECT_ROOT / "output" / "visium_hd_exp1" / "assets" / "tissue_hires_image.png"
DEFAULT_LABEL_IMAGE = PROJECT_ROOT / "output" / "visium_hd_exp1" / "molecular_prior_experiments" / "dx-60_dy80" / "ficture_factor_label_image_he.npy"
DEFAULT_TARGET_DIRS = [
    PROJECT_ROOT / "output" / "visium_hd_exp1" / "sam3_runs" / "base_sam3_multipoint" / "masks",
    PROJECT_ROOT / "output" / "visium_hd_exp1" / "sam3_runs" / "base_sam3_multipoint_remaining_dev" / "masks",
]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "output" / "visium_hd_exp1" / "dice08_experiments" / "he_molecular_extratrees_1536"


LABEL_ORDER = [
    "lung_bronchiola",
    "erythorocytes",
    "immune_infiltration",
    "lung_alveoli_normal_adjacent",
    "lung_vessels",
    "pigment",
    "stroma",
    "tumor",
]

LABEL_COLORS = {
    "lung_bronchiola": (31, 119, 180),
    "erythorocytes": (255, 127, 14),
    "immune_infiltration": (44, 160, 44),
    "lung_alveoli_normal_adjacent": (148, 103, 189),
    "lung_vessels": (140, 86, 75),
    "pigment": (127, 127, 127),
    "stroma": (188, 189, 34),
    "tumor": (23, 190, 207),
}


def slugify(value: str) -> str:
    out = "".join(ch if ch.isalnum() else "_" for ch in value.lower()).strip("_")
    while "__" in out:
        out = out.replace("__", "_")
    return out or "unlabeled"


def resize_array(arr: np.ndarray, shape_hw: Tuple[int, int], resample: int) -> np.ndarray:
    h, w = shape_hw
    return np.array(Image.fromarray(arr).resize((w, h), resample=resample))


def max_side_shape(shape_hw: Tuple[int, int], max_side: int) -> Tuple[int, int]:
    h, w = shape_hw
    if max(h, w) <= max_side:
        return h, w
    scale = max_side / float(max(h, w))
    return max(1, int(round(h * scale))), max(1, int(round(w * scale)))


def load_targets(target_dirs: Sequence[Path], shape_hw: Tuple[int, int]) -> Dict[str, np.ndarray]:
    targets: Dict[str, np.ndarray] = {}
    for target_dir in target_dirs:
        for path in sorted(target_dir.glob("*_target.png")):
            label = path.name.split("_", 1)[1].removesuffix("_target.png")
            mask = np.array(Image.open(path).convert("L")) > 127
            if mask.shape != shape_hw:
                mask = resize_array((mask.astype(np.uint8) * 255), shape_hw, Image.Resampling.NEAREST) > 127
            targets[label] = mask
    return {label: targets[label] for label in LABEL_ORDER if label in targets}


def mask_metrics(pred: np.ndarray, gt: np.ndarray) -> Dict[str, float]:
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    inter = int(np.logical_and(pred, gt).sum())
    pred_sum = int(pred.sum())
    gt_sum = int(gt.sum())
    union = int(np.logical_or(pred, gt).sum())
    return {
        "dice": 2.0 * inter / (pred_sum + gt_sum) if pred_sum + gt_sum else 0.0,
        "iou": inter / union if union else 0.0,
        "precision": inter / pred_sum if pred_sum else 0.0,
        "recall": inter / gt_sum if gt_sum else 0.0,
        "pred_pixels": float(pred_sum),
        "gt_pixels": float(gt_sum),
    }


def make_multiclass_target(targets: Dict[str, np.ndarray], tissue_mask: np.ndarray) -> Tuple[np.ndarray, Dict[int, str]]:
    y = np.zeros(tissue_mask.shape, dtype=np.uint8)
    id_to_label = {0: "background"}
    # Later labels overwrite earlier overlaps; tumor/stroma are intentionally late.
    for idx, label in enumerate(LABEL_ORDER, start=1):
        if label not in targets:
            continue
        y[targets[label]] = idx
        id_to_label[idx] = label
    y[~tissue_mask] = 0
    return y, id_to_label


def build_features(he: np.ndarray, label_image: np.ndarray, tissue_mask: np.ndarray) -> np.ndarray:
    he_float = he.astype(np.float32) / 255.0
    hsv = color.rgb2hsv(he_float).astype(np.float32)
    lab = color.rgb2lab(he_float).astype(np.float32)
    gray = color.rgb2gray(he_float).astype(np.float32)
    sobel = filters.sobel(gray).astype(np.float32)
    h, w = gray.shape
    yy, xx = np.indices((h, w), dtype=np.float32)
    xx /= max(1, w - 1)
    yy /= max(1, h - 1)
    factor = np.clip(label_image, -1, 11)
    factor_one_hot = np.zeros((h, w, 13), dtype=np.float32)
    factor_one_hot.reshape(-1, 13)[(factor.reshape(-1) + 1).astype(np.int16).clip(0, 12),] = 0.0
    flat_factor = (factor.reshape(-1) + 1).astype(np.int16).clip(0, 12)
    factor_one_hot.reshape(-1, 13)[np.arange(h * w), flat_factor] = 1.0
    local_factor_support = ndi.uniform_filter((label_image >= 0).astype(np.float32), size=9)
    tissue_smooth = ndi.uniform_filter(tissue_mask.astype(np.float32), size=15)
    tissue_distance = ndi.distance_transform_edt(tissue_mask).astype(np.float32)
    if tissue_distance.max() > 0:
        tissue_distance /= tissue_distance.max()

    multiscale_parts = []
    for sigma in (2.0, 6.0, 16.0):
        multiscale_parts.append(ndi.gaussian_filter(gray, sigma=sigma).astype(np.float32)[..., None])
        multiscale_parts.append(ndi.gaussian_filter(sobel, sigma=sigma).astype(np.float32)[..., None])
        for channel in range(3):
            multiscale_parts.append(ndi.gaussian_filter(lab[..., channel], sigma=sigma).astype(np.float32)[..., None])

    factor_context_parts = []
    for size in (9, 31, 91):
        for factor_id in range(12):
            factor_context_parts.append(
                ndi.uniform_filter((label_image == factor_id).astype(np.float32), size=size)[..., None]
            )

    features = np.dstack(
        [
            he_float,
            hsv,
            lab / np.array([100.0, 128.0, 128.0], dtype=np.float32),
            gray[..., None],
            sobel[..., None],
            xx[..., None],
            yy[..., None],
            local_factor_support[..., None],
            tissue_smooth[..., None],
            tissue_distance[..., None],
            factor_one_hot,
            *multiscale_parts,
            *factor_context_parts,
        ]
    ).astype(np.float32)
    return features


def sample_indices(
    y: np.ndarray,
    tissue_mask: np.ndarray,
    labels: Iterable[int],
    max_per_class: int,
    rng: np.random.Generator,
    allowed: np.ndarray | None = None,
) -> np.ndarray:
    if allowed is None:
        allowed = np.ones_like(tissue_mask, dtype=bool)
    idxs: List[np.ndarray] = []
    for class_id in labels:
        mask = (y == class_id) & allowed
        if class_id == 0:
            mask &= tissue_mask
        coords = np.flatnonzero(mask)
        if coords.size == 0:
            continue
        n = min(max_per_class, coords.size)
        idxs.append(rng.choice(coords, size=n, replace=False))
    return np.concatenate(idxs) if idxs else np.array([], dtype=np.int64)


def predict_in_chunks(model, features: np.ndarray, chunk_size: int) -> np.ndarray:
    flat = features.reshape(-1, features.shape[-1])
    out = np.zeros(flat.shape[0], dtype=np.uint8)
    for start in range(0, flat.shape[0], chunk_size):
        stop = min(flat.shape[0], start + chunk_size)
        out[start:stop] = model.predict(flat[start:stop]).astype(np.uint8)
    return out.reshape(features.shape[:2])


def clean_binary_mask(mask: np.ndarray, min_area: int, close_radius: int, fill_holes: int) -> np.ndarray:
    out = mask.astype(bool)
    if close_radius > 0:
        out = morphology.binary_closing(out, morphology.disk(close_radius))
    if min_area > 0:
        out = morphology.remove_small_objects(out, min_size=min_area)
    if fill_holes > 0:
        out = ndi.binary_fill_holes(out)
        out = morphology.remove_small_holes(out, area_threshold=fill_holes)
    return out.astype(bool)


def factor_subset_sweep(
    label_image: np.ndarray,
    targets: Dict[str, np.ndarray],
    output_dir: Path,
) -> pd.DataFrame:
    rows = []
    for label, gt in targets.items():
        best = None
        for subset_bits in range(1, 1 << 12):
            factors = [factor for factor in range(12) if subset_bits & (1 << factor)]
            mask = np.isin(label_image, factors)
            metrics = mask_metrics(mask, gt)
            row = {
                "label": label,
                "factors": ",".join(str(f) for f in factors),
                "n_factors": len(factors),
                **metrics,
            }
            if best is None or row["dice"] > best["dice"]:
                best = row
        if best is not None:
            rows.append(best)
    df = pd.DataFrame(rows).sort_values("dice", ascending=False)
    df.to_csv(output_dir / "factor_subset_oracle_upper_bound.csv", index=False)
    return df


def postprocess_label_predictions(
    pred: np.ndarray,
    id_to_label: Dict[int, str],
    targets: Dict[str, np.ndarray],
    output_dir: Path,
) -> Tuple[pd.DataFrame, Dict[str, np.ndarray]]:
    rows = []
    best_masks: Dict[str, np.ndarray] = {}
    for class_id, label in id_to_label.items():
        if class_id == 0 or label not in targets:
            continue
        raw = pred == class_id
        best_row = {"label": label, "variant": "raw", **mask_metrics(raw, targets[label])}
        best_mask = raw
        for min_area in (0, 100, 500, 1500, 4000):
            for close_radius in (0, 2, 4, 8, 12):
                for fill_holes in (0, 500, 2500, 10000):
                    mask = clean_binary_mask(raw, min_area=min_area, close_radius=close_radius, fill_holes=fill_holes)
                    metrics = mask_metrics(mask, targets[label])
                    if metrics["dice"] > best_row["dice"]:
                        best_row = {
                            "label": label,
                            "variant": f"min_area={min_area};close={close_radius};holes={fill_holes}",
                            **metrics,
                        }
                        best_mask = mask
        rows.append(best_row)
        best_masks[label] = best_mask
    df = pd.DataFrame(rows).sort_values("dice", ascending=False)
    df.to_csv(output_dir / "supervised_postprocessed_metrics.csv", index=False)
    return df, best_masks


def make_overlay(he: np.ndarray, masks: Dict[str, np.ndarray], alpha: float = 0.42) -> np.ndarray:
    out = he.astype(np.float32).copy()
    for label, mask in masks.items():
        color_rgb = np.array(LABEL_COLORS.get(label, (255, 255, 0)), dtype=np.float32)
        idx = mask.astype(bool)
        out[idx] = (1.0 - alpha) * out[idx] + alpha * color_rgb
    return np.clip(out, 0, 255).astype(np.uint8)


def save_label_panels(
    he: np.ndarray,
    masks: Dict[str, np.ndarray],
    targets: Dict[str, np.ndarray],
    output_dir: Path,
) -> None:
    panel_dir = output_dir / "panels"
    mask_dir = output_dir / "masks"
    panel_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)
    for label, pred in masks.items():
        if label not in targets:
            continue
        gt = targets[label]
        color_rgb = LABEL_COLORS.get(label, (255, 255, 0))
        Image.fromarray(pred.astype(np.uint8) * 255).save(mask_dir / f"{slugify(label)}_prediction.png")
        fig, axes = plt.subplots(1, 3, figsize=(13, 5))
        axes[0].imshow(make_overlay(he, {label: pred}, alpha=0.45))
        axes[0].set_title("Prediction")
        axes[1].imshow(make_overlay(he, {label: gt}, alpha=0.45))
        axes[1].set_title("Target")
        overlap = np.zeros((*gt.shape, 3), dtype=np.uint8)
        overlap[np.logical_and(pred, gt)] = (255, 255, 255)
        overlap[np.logical_and(pred, ~gt)] = color_rgb
        overlap[np.logical_and(~pred, gt)] = (230, 57, 70)
        axes[2].imshow(he)
        axes[2].imshow(overlap, alpha=(overlap.sum(axis=-1) > 0) * 0.75)
        axes[2].set_title("White=TP, color=FP, red=FN")
        for ax in axes:
            ax.axis("off")
        fig.suptitle(label)
        fig.tight_layout()
        fig.savefig(panel_dir / f"{slugify(label)}.png", dpi=170, bbox_inches="tight")
        plt.close(fig)


def spatial_holdout_experiment(
    features: np.ndarray,
    y: np.ndarray,
    tissue_mask: np.ndarray,
    id_to_label: Dict[int, str],
    targets: Dict[str, np.ndarray],
    output_dir: Path,
    rng: np.random.Generator,
    *,
    grid_rows: int,
    grid_cols: int,
    max_per_class: int,
    n_estimators: int,
) -> pd.DataFrame:
    h, w = y.shape
    block_y = np.minimum((np.arange(h)[:, None] * grid_rows) // h, grid_rows - 1)
    block_x = np.minimum((np.arange(w)[None, :] * grid_cols) // w, grid_cols - 1)
    block_id = block_y * grid_cols + block_x
    classes = sorted(id_to_label.keys())
    rows = []
    for holdout in range(grid_rows * grid_cols):
        val = block_id == holdout
        train = ~val
        train_idx = sample_indices(y, tissue_mask, classes, max_per_class, rng, allowed=train)
        if train_idx.size == 0:
            continue
        model = ExtraTreesClassifier(
            n_estimators=n_estimators,
            max_depth=None,
            min_samples_leaf=2,
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=1000 + holdout,
        )
        model.fit(features.reshape(-1, features.shape[-1])[train_idx], y.reshape(-1)[train_idx])
        val_flat = np.flatnonzero(val)
        val_pred = np.zeros(val_flat.size, dtype=np.uint8)
        flat_features = features.reshape(-1, features.shape[-1])
        for start in range(0, val_flat.size, 200_000):
            idx = val_flat[start : start + 200_000]
            val_pred[start : start + idx.size] = model.predict(flat_features[idx]).astype(np.uint8)
        pred_block = np.zeros_like(y)
        pred_block.reshape(-1)[val_flat] = val_pred
        for class_id, label in id_to_label.items():
            if class_id == 0 or label not in targets:
                continue
            gt = targets[label] & val
            if gt.sum() == 0:
                continue
            metrics = mask_metrics((pred_block == class_id) & val, gt)
            rows.append({"holdout_block": holdout, "label": label, **metrics})
        print(f"  spatial holdout {holdout + 1}/{grid_rows * grid_cols}: train pixels={train_idx.size}")
    df = pd.DataFrame(rows)
    if not df.empty:
        df.to_csv(output_dir / "spatial_block_holdout_metrics.csv", index=False)
        summary = df.groupby("label", as_index=False).agg(
            mean_dice=("dice", "mean"),
            median_dice=("dice", "median"),
            min_dice=("dice", "min"),
            n_blocks=("dice", "size"),
        )
        summary.to_csv(output_dir / "spatial_block_holdout_summary.csv", index=False)
    return df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Dice 0.8-oriented H&E+molecular experiments.")
    parser.add_argument("--he-image", type=Path, default=DEFAULT_HE_IMAGE)
    parser.add_argument("--label-image", type=Path, default=DEFAULT_LABEL_IMAGE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max-side", type=int, default=1536)
    parser.add_argument("--max-per-class", type=int, default=65000)
    parser.add_argument("--n-estimators", type=int, default=260)
    parser.add_argument("--chunk-size", type=int, default=350000)
    parser.add_argument("--skip-holdout", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(42)

    Image.MAX_IMAGE_PIXELS = None
    he_full = np.array(Image.open(args.he_image).convert("RGB"))
    label_full = np.load(args.label_image).astype(np.int16)
    shape = max_side_shape(he_full.shape[:2], args.max_side)
    he = resize_array(he_full, shape, Image.Resampling.BILINEAR)
    label_image = resize_array(label_full.astype(np.int16), shape, Image.Resampling.NEAREST).astype(np.int16)
    tissue_mask = np.any(he < 245, axis=-1)
    tissue_mask = morphology.remove_small_objects(tissue_mask, min_size=500)
    tissue_mask = ndi.binary_fill_holes(tissue_mask)
    targets = load_targets(DEFAULT_TARGET_DIRS, shape)
    y, id_to_label = make_multiclass_target(targets, tissue_mask)

    print("=" * 72)
    print("Dice 0.8-oriented H&E + molecular experiments")
    print("=" * 72)
    print(f"Image shape: {he.shape[0]} x {he.shape[1]}")
    print(f"Labels: {', '.join(targets)}")
    print(f"Output: {args.output_dir}")

    factor_upper = factor_subset_sweep(label_image, targets, args.output_dir)
    print("Best factor-subset upper bound:")
    print(factor_upper[["label", "factors", "dice", "precision", "recall"]].to_string(index=False))

    features = build_features(he, label_image, tissue_mask)
    classes = sorted(id_to_label.keys())
    train_idx = sample_indices(y, tissue_mask, classes, args.max_per_class, rng)
    print(f"Training full-slide ExtraTrees on {train_idx.size} sampled pixels and {features.shape[-1]} features")
    model = ExtraTreesClassifier(
        n_estimators=args.n_estimators,
        max_depth=None,
        min_samples_leaf=1,
        class_weight="balanced_subsample",
        n_jobs=-1,
        random_state=42,
    )
    flat_features = features.reshape(-1, features.shape[-1])
    model.fit(flat_features[train_idx], y.reshape(-1)[train_idx])
    pred = predict_in_chunks(model, features, args.chunk_size)
    pred[~tissue_mask] = 0

    raw_rows = []
    raw_masks = {}
    for class_id, label in id_to_label.items():
        if class_id == 0 or label not in targets:
            continue
        mask = pred == class_id
        raw_masks[label] = mask
        raw_rows.append({"label": label, **mask_metrics(mask, targets[label])})
    raw_df = pd.DataFrame(raw_rows).sort_values("dice", ascending=False)
    raw_df.to_csv(args.output_dir / "supervised_raw_metrics.csv", index=False)
    post_df, best_masks = postprocess_label_predictions(pred, id_to_label, targets, args.output_dir)
    save_label_panels(he, best_masks, targets, args.output_dir)
    Image.fromarray(make_overlay(he, best_masks)).save(args.output_dir / "all_label_predictions_overlay.png")
    Image.fromarray(make_overlay(he, targets, alpha=0.35)).save(args.output_dir / "all_targets_overlay.png")

    holdout_df = pd.DataFrame()
    if not args.skip_holdout:
        print("Running spatial block holdout...")
        holdout_df = spatial_holdout_experiment(
            features,
            y,
            tissue_mask,
            id_to_label,
            targets,
            args.output_dir,
            rng,
            grid_rows=3,
            grid_cols=3,
            max_per_class=max(4000, args.max_per_class // 5),
            n_estimators=max(80, args.n_estimators // 2),
        )

    summary = {
        "image_shape": list(he.shape[:2]),
        "n_features": int(features.shape[-1]),
        "labels": list(targets.keys()),
        "train_sample_pixels": int(train_idx.size),
        "factor_subset_upper_bound_csv": str(args.output_dir / "factor_subset_oracle_upper_bound.csv"),
        "supervised_postprocessed_csv": str(args.output_dir / "supervised_postprocessed_metrics.csv"),
        "spatial_holdout_csv": str(args.output_dir / "spatial_block_holdout_metrics.csv") if not holdout_df.empty else None,
    }
    (args.output_dir / "experiment_summary.json").write_text(json.dumps(summary, indent=2))

    lines = ["# Dice 0.8 Experiment Results", ""]
    lines += ["## Factor subset oracle upper bound", ""]
    lines.append("| label | factors | Dice | Precision | Recall |")
    lines.append("|---|---|---:|---:|---:|")
    for _, row in factor_upper.iterrows():
        lines.append(f"| {row['label']} | {row['factors']} | {row['dice']:.3f} | {row['precision']:.3f} | {row['recall']:.3f} |")
    lines += ["", "## Full-slide H&E+molecular ExtraTrees, postprocessed", ""]
    lines.append("| label | Dice | Precision | Recall | variant |")
    lines.append("|---|---:|---:|---:|---|")
    for _, row in post_df.iterrows():
        lines.append(f"| {row['label']} | {row['dice']:.3f} | {row['precision']:.3f} | {row['recall']:.3f} | {row['variant']} |")
    if not holdout_df.empty:
        summary_df = pd.read_csv(args.output_dir / "spatial_block_holdout_summary.csv")
        lines += ["", "## Spatial block holdout", ""]
        lines.append("| label | mean Dice | median Dice | min Dice | blocks |")
        lines.append("|---|---:|---:|---:|---:|")
        for _, row in summary_df.iterrows():
            lines.append(
                f"| {row['label']} | {row['mean_dice']:.3f} | {row['median_dice']:.3f} | {row['min_dice']:.3f} | {int(row['n_blocks'])} |"
            )
    (args.output_dir / "RESULTS.md").write_text("\n".join(lines) + "\n")
    print("Full-slide postprocessed:")
    print(post_df[["label", "dice", "precision", "recall", "variant"]].to_string(index=False))
    if not holdout_df.empty:
        print("Spatial holdout summary:")
        print(pd.read_csv(args.output_dir / "spatial_block_holdout_summary.csv").to_string(index=False))
    print(f"Wrote results to {args.output_dir}")


if __name__ == "__main__":
    main()
