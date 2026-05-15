#!/usr/bin/env python3
"""Spatial train/validation/test evaluation for the Visium HD pixel classifier.

This script is intentionally stricter than the earlier full-slide teacher-mask
experiments.  A spatial block is held out as test, a different block is used
only for model/threshold selection, and all remaining blocks are used for
training.  The annotation is therefore used as supervision only inside the
training blocks and as an untouched evaluation target inside test blocks.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
from PIL import Image
from scipy import ndimage as ndi
from skimage import morphology
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier

from run_unified_pixel_classifier_sweep import (
    COLORS,
    DEFAULT_ANNOT,
    DEFAULT_FACTOR,
    DEFAULT_HE,
    DEFAULT_SUMMARY,
    LABEL_ORDER,
    component_filter,
    load_targets,
    make_classifier,
    make_features,
    make_training_label_image,
    max_side_shape,
    metrics,
    overlay,
    predict_proba_image,
    resize,
    slugify,
    tissue_mask,
)


SPLIT_NAMES = ("train", "validation", "test")


def feature_names(use_xy: bool, use_factor: bool, n_factors: int) -> List[str]:
    names = [
        "rgb_r",
        "rgb_g",
        "rgb_b",
        "lab_l",
        "lab_a",
        "lab_b",
        "hsv_h",
        "hsv_s",
        "hsv_v",
        "hed_h",
        "hed_e",
        "hed_d",
        "sobel_gray",
        "gaussian_gray_sigma1",
        "gaussian_gray_sigma3",
        "red_minus_greenblue",
        "one_minus_value",
    ]
    if use_factor:
        names.append("ficture_factor_id")
        names.extend([f"ficture_gene_score_{label}" for label in LABEL_ORDER])
    if use_xy:
        names.extend(["x_normalized", "y_normalized"])
    return names


def make_spatial_blocks(shape_hw: Tuple[int, int], rows: int, cols: int) -> np.ndarray:
    h, w = shape_hw
    by = np.minimum((np.arange(h)[:, None] * rows) // h, rows - 1)
    bx = np.minimum((np.arange(w)[None, :] * cols) // w, cols - 1)
    return (by * cols + bx).astype(np.int16)


def sample_training_from_split(
    features: np.ndarray,
    label_img: np.ndarray,
    tissue: np.ndarray,
    split_mask: np.ndarray,
    rng: np.random.Generator,
    max_pos_per_class: int,
    max_bg: int,
) -> Tuple[np.ndarray, np.ndarray, List[Dict[str, object]]]:
    flat_x = features.reshape(-1, features.shape[-1])
    flat_y = label_img.ravel()
    allowed = (tissue & split_mask).ravel()
    xs = []
    ys = []
    rows: List[Dict[str, object]] = []
    for class_id, label in enumerate(LABEL_ORDER, start=1):
        coords_all = np.flatnonzero((flat_y == class_id) & allowed)
        coords = coords_all
        if coords.size > max_pos_per_class:
            coords = rng.choice(coords, size=max_pos_per_class, replace=False)
        if coords.size:
            xs.append(flat_x[coords])
            ys.append(np.full(coords.size, class_id, dtype=np.int16))
        rows.append(
            {
                "class_id": class_id,
                "label": label,
                "available_pixels": int(coords_all.size),
                "sampled_pixels": int(coords.size),
            }
        )
    bg_all = np.flatnonzero((flat_y == 0) & allowed)
    bg = bg_all
    if bg.size > max_bg:
        bg = rng.choice(bg, size=max_bg, replace=False)
    if bg.size:
        xs.append(flat_x[bg])
        ys.append(np.zeros(bg.size, dtype=np.int16))
    rows.append(
        {
            "class_id": 0,
            "label": "background",
            "available_pixels": int(bg_all.size),
            "sampled_pixels": int(bg.size),
        }
    )
    if not xs:
        return np.empty((0, features.shape[-1]), dtype=np.float32), np.empty((0,), dtype=np.int16), rows
    return np.concatenate(xs, axis=0), np.concatenate(ys, axis=0), rows


def split_pixel_counts(
    label_img: np.ndarray,
    tissue: np.ndarray,
    split_masks: Dict[str, np.ndarray],
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for split, mask in split_masks.items():
        allowed = tissue & mask
        rows.append(
            {
                "split": split,
                "class_id": 0,
                "label": "background",
                "pixels": int(((label_img == 0) & allowed).sum()),
            }
        )
        for class_id, label in enumerate(LABEL_ORDER, start=1):
            rows.append(
                {
                    "split": split,
                    "class_id": class_id,
                    "label": label,
                    "pixels": int(((label_img == class_id) & allowed).sum()),
                }
            )
    return rows


def clean_prediction(
    raw: np.ndarray,
    min_area: int,
    max_area: int,
    eccentricity_max: float,
    close_radius: int,
    dilate_radius: int,
    fill_holes: int,
) -> np.ndarray:
    out = raw.astype(bool)
    if close_radius > 0:
        out = morphology.binary_closing(out, morphology.disk(close_radius))
    if dilate_radius > 0:
        out = morphology.binary_dilation(out, morphology.disk(dilate_radius))
    out = component_filter(out, min_area, max_area, eccentricity_max)
    if fill_holes > 0:
        out = ndi.binary_fill_holes(out)
        out = morphology.remove_small_holes(out, area_threshold=fill_holes)
    return out.astype(bool)


def candidate_masks_for_label(
    probs: np.ndarray,
    argmax: np.ndarray,
    tissue: np.ndarray,
    label: str,
    class_id: int,
    threshold_source: np.ndarray,
    percentiles: Iterable[float],
    close_radii: Iterable[int],
    dilate_radii: Iterable[int],
    fill_holes_values: Iterable[int],
    min_area: int,
    max_area: int,
    eccentricity_max: float,
) -> Iterable[Tuple[Dict[str, object], np.ndarray]]:
    arg_raw = (argmax == class_id) & tissue
    arg_mask = clean_prediction(arg_raw, min_area, max_area, eccentricity_max, 0, 0, 0)
    yield (
        {
            "label": label,
            "mode": "argmax",
            "percentile": -1.0,
            "threshold": -1.0,
            "close_radius": 0,
            "dilate_radius": 0,
            "fill_holes": 0,
        },
        arg_mask,
    )

    p = probs[..., class_id]
    vals = p[threshold_source]
    if vals.size == 0:
        return
    for pct in percentiles:
        thr = float(np.percentile(vals, pct))
        raw = (p >= thr) & tissue
        for close_radius in close_radii:
            for dilate_radius in dilate_radii:
                for fill_holes in fill_holes_values:
                    pred = clean_prediction(
                        raw,
                        min_area,
                        max_area,
                        eccentricity_max,
                        close_radius,
                        dilate_radius,
                        fill_holes,
                    )
                    yield (
                        {
                            "label": label,
                            "mode": "threshold",
                            "percentile": float(pct),
                            "threshold": thr,
                            "close_radius": int(close_radius),
                            "dilate_radius": int(dilate_radius),
                            "fill_holes": int(fill_holes),
                        },
                        pred,
                    )


def choose_on_validation_and_test(
    probs: np.ndarray,
    tissue: np.ndarray,
    targets: Dict[str, np.ndarray],
    split_masks: Dict[str, np.ndarray],
    args: argparse.Namespace,
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]], Dict[str, np.ndarray]]:
    argmax = probs.argmax(axis=-1)
    val_rows: List[Dict[str, object]] = []
    test_rows: List[Dict[str, object]] = []
    best_test_masks: Dict[str, np.ndarray] = {}
    threshold_source = tissue & split_masks["validation"]
    for class_id, label in enumerate(LABEL_ORDER, start=1):
        gt_val = targets[label] & split_masks["validation"]
        gt_test = targets[label] & split_masks["test"]
        best_val_row = None
        best_mask = None
        for params, pred in candidate_masks_for_label(
            probs,
            argmax,
            tissue,
            label,
            class_id,
            threshold_source,
            args.percentiles,
            args.close_radii,
            args.dilate_radii,
            args.fill_holes,
            args.min_area,
            args.max_area,
            args.eccentricity_max,
        ):
            row = {
                **params,
                **metrics(pred & split_masks["validation"], gt_val),
            }
            val_rows.append(row)
            if best_val_row is None or row["dice"] > best_val_row["dice"]:
                best_val_row = row
                best_mask = pred
        if best_val_row is None or best_mask is None:
            continue
        test_rows.append(
            {
                **{key: best_val_row[key] for key in ("label", "mode", "percentile", "threshold", "close_radius", "dilate_radius", "fill_holes")},
                "validation_dice_selected": float(best_val_row["dice"]),
                **metrics(best_mask & split_masks["test"], gt_test),
            }
        )
        best_test_masks[label] = best_mask & split_masks["test"]
    return val_rows, test_rows, best_test_masks


def multiclass_accuracy_rows(
    probs: np.ndarray,
    label_img: np.ndarray,
    tissue: np.ndarray,
    split_masks: Dict[str, np.ndarray],
) -> List[Dict[str, object]]:
    pred = probs.argmax(axis=-1).astype(np.int16)
    rows: List[Dict[str, object]] = []
    for split, mask in split_masks.items():
        allowed = tissue & mask
        y_true = label_img[allowed]
        y_pred = pred[allowed]
        rows.append(
            {
                "split": split,
                "label": "all_tissue_pixels",
                "accuracy": float((y_true == y_pred).mean()) if y_true.size else 0.0,
                "pixels": int(y_true.size),
            }
        )
        for class_id, label in enumerate(["background", *LABEL_ORDER]):
            cls = y_true == class_id
            if cls.sum() == 0:
                continue
            rows.append(
                {
                    "split": split,
                    "label": label,
                    "accuracy": float((y_pred[cls] == class_id).mean()),
                    "pixels": int(cls.sum()),
                }
            )
    return rows


def save_feature_importance(model, names: List[str], output_path: Path) -> None:
    importances = getattr(model, "feature_importances_", None)
    if importances is None and hasattr(model, "named_steps"):
        importances = getattr(model.named_steps.get("extratreesclassifier"), "feature_importances_", None)
    if importances is None:
        return
    rows = []
    for idx, value in enumerate(importances):
        rows.append({"feature_index": idx, "feature": names[idx] if idx < len(names) else f"feature_{idx}", "importance": float(value)})
    pd.DataFrame(rows).sort_values("importance", ascending=False).to_csv(output_path, index=False)


def make_eval_classifier(args: argparse.Namespace, seed: int):
    if args.classifier == "extratrees":
        return ExtraTreesClassifier(
            n_estimators=args.n_estimators,
            max_depth=args.max_depth if args.max_depth > 0 else None,
            min_samples_leaf=args.min_samples_leaf,
            class_weight="balanced",
            n_jobs=-1,
            random_state=seed,
        )
    if args.classifier == "rf":
        return RandomForestClassifier(
            n_estimators=args.n_estimators,
            max_depth=args.max_depth if args.max_depth > 0 else 24,
            min_samples_leaf=args.min_samples_leaf,
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=seed,
        )
    return make_classifier(args.classifier, seed)


def run(args: argparse.Namespace) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    Image.MAX_IMAGE_PIXELS = None
    he_full = np.array(Image.open(args.he_image).convert("RGB"))
    shape = max_side_shape(he_full.shape[:2], args.max_side)
    he = resize(he_full, shape, Image.Resampling.BILINEAR)
    factor_full = np.load(args.factor_image)
    factors = resize(factor_full.astype(np.int16), shape, Image.Resampling.NEAREST).astype(np.int16)
    tissue = tissue_mask(he)
    targets = load_targets(args.summary_path, shape)
    features = make_features(he, factors, args.factor_annotation, tissue, use_xy=args.use_xy, use_factor=not args.no_factor)
    label_img = make_training_label_image(targets, tissue)
    blocks = make_spatial_blocks(shape, args.grid_rows, args.grid_cols)
    n_blocks = args.grid_rows * args.grid_cols
    folds = list(range(n_blocks)) if args.fold < 0 else [args.fold]
    percentiles = tuple(float(x) for x in args.percentiles.split(",") if x.strip())
    args.percentiles = percentiles

    all_count_rows: List[Dict[str, object]] = []
    all_train_rows: List[Dict[str, object]] = []
    all_val_rows: List[Dict[str, object]] = []
    all_test_rows: List[Dict[str, object]] = []
    all_accuracy_rows: List[Dict[str, object]] = []
    best_overlay_masks: Dict[str, np.ndarray] = {}
    best_fold_by_label: Dict[str, float] = {}

    valid = factors >= 0
    n_factors = int(factors[valid].max()) + 1 if valid.any() else 1
    names = feature_names(args.use_xy, not args.no_factor, n_factors)
    (args.output_dir / "feature_names.json").write_text(json.dumps(names, indent=2))

    for test_block in folds:
        val_block = (test_block + args.validation_offset) % n_blocks
        if val_block == test_block:
            val_block = (test_block + 1) % n_blocks
        split_masks = {
            "test": blocks == test_block,
            "validation": blocks == val_block,
            "train": (blocks != test_block) & (blocks != val_block),
        }
        fold_dir = args.output_dir / f"fold_test{test_block:02d}_val{val_block:02d}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        rng = np.random.default_rng(args.seed + test_block)
        x_train, y_train, train_rows = sample_training_from_split(
            features,
            label_img,
            tissue,
            split_masks["train"],
            rng,
            args.max_pos_per_class,
            args.max_bg,
        )
        if x_train.size == 0:
            continue
        model = make_eval_classifier(args, args.seed + test_block)
        model.fit(x_train, y_train)
        probs = predict_proba_image(model, features, len(LABEL_ORDER) + 1, args.chunk)
        val_rows, test_rows, fold_test_masks = choose_on_validation_and_test(probs, tissue, targets, split_masks, args)
        accuracy_rows = multiclass_accuracy_rows(probs, label_img, tissue, split_masks)

        for row in split_pixel_counts(label_img, tissue, split_masks):
            all_count_rows.append({"fold": test_block, "test_block": test_block, "validation_block": val_block, **row})
        for row in train_rows:
            all_train_rows.append({"fold": test_block, "test_block": test_block, "validation_block": val_block, **row})
        for row in val_rows:
            all_val_rows.append({"fold": test_block, "test_block": test_block, "validation_block": val_block, **row})
        for row in test_rows:
            all_test_rows.append({"fold": test_block, "test_block": test_block, "validation_block": val_block, **row})
            label = str(row["label"])
            dice = float(row["dice"])
            if label not in best_fold_by_label or dice > best_fold_by_label[label]:
                best_fold_by_label[label] = dice
                best_overlay_masks[label] = fold_test_masks[label]
        for row in accuracy_rows:
            all_accuracy_rows.append({"fold": test_block, "test_block": test_block, "validation_block": val_block, **row})

        save_feature_importance(model, names, fold_dir / "feature_importance.csv")
        Image.fromarray(overlay(he, fold_test_masks)).save(fold_dir / "test_prediction_overlay.png")
        for label, mask in fold_test_masks.items():
            Image.fromarray((mask.astype(np.uint8) * 255)).save(fold_dir / f"{slugify(label)}_test_prediction.png")
        print(
            f"fold test={test_block} val={val_block}: train_sample={x_train.shape[0]} "
            f"test_mean_dice={np.mean([float(r['dice']) for r in test_rows]) if test_rows else 0:.4f}"
        )

    counts_df = pd.DataFrame(all_count_rows)
    train_df = pd.DataFrame(all_train_rows)
    val_df = pd.DataFrame(all_val_rows)
    test_df = pd.DataFrame(all_test_rows)
    acc_df = pd.DataFrame(all_accuracy_rows)
    counts_df.to_csv(args.output_dir / "spatial_split_pixel_counts.csv", index=False)
    train_df.to_csv(args.output_dir / "spatial_split_training_sample_counts.csv", index=False)
    val_df.to_csv(args.output_dir / "spatial_split_validation_parameter_sweep.csv", index=False)
    test_df.to_csv(args.output_dir / "spatial_split_test_metrics.csv", index=False)
    acc_df.to_csv(args.output_dir / "spatial_split_multiclass_accuracy.csv", index=False)

    if not test_df.empty:
        positive_test = test_df[test_df["gt_pixels"] > 0].copy()
        summary = positive_test.groupby("label", as_index=False).agg(
            mean_test_dice=("dice", "mean"),
            median_test_dice=("dice", "median"),
            min_test_dice=("dice", "min"),
            max_test_dice=("dice", "max"),
            mean_precision=("precision", "mean"),
            mean_recall=("recall", "mean"),
            positive_test_folds=("dice", "size"),
        )
        summary.sort_values("mean_test_dice", ascending=False).to_csv(args.output_dir / "spatial_split_test_summary.csv", index=False)
        Image.fromarray(overlay(he, best_overlay_masks)).save(args.output_dir / "best_test_block_per_label_overlay.png")

    manifest = {
        "he_image": str(args.he_image),
        "factor_image": str(args.factor_image),
        "factor_annotation": str(args.factor_annotation),
        "summary_path": str(args.summary_path),
        "output_dir": str(args.output_dir),
        "shape": list(shape),
        "grid_rows": args.grid_rows,
        "grid_cols": args.grid_cols,
        "classifier": args.classifier,
        "n_estimators": args.n_estimators,
        "max_depth": args.max_depth,
        "min_samples_leaf": args.min_samples_leaf,
        "use_xy": args.use_xy,
        "use_factor": not args.no_factor,
        "max_pos_per_class": args.max_pos_per_class,
        "max_bg": args.max_bg,
        "percentiles": list(args.percentiles),
        "close_radii": list(args.close_radii),
        "dilate_radii": list(args.dilate_radii),
        "fill_holes": list(args.fill_holes),
        "note": "Validation block selects thresholds/postprocessing; reported test metrics are from disjoint spatial blocks.",
    }
    (args.output_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2))


def parse_int_list(value: str) -> Tuple[int, ...]:
    return tuple(int(x) for x in value.split(",") if x.strip())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Strict spatial train/validation/test pixel-classifier evaluation.")
    parser.add_argument("--he-image", type=Path, default=DEFAULT_HE)
    parser.add_argument("--summary-path", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--factor-image", type=Path, default=DEFAULT_FACTOR)
    parser.add_argument("--factor-annotation", type=Path, default=DEFAULT_ANNOT)
    parser.add_argument("--output-dir", type=Path, default=Path("output/visium_hd_exp1/classification_spatial_split_eval/et_factor_xy_1024"))
    parser.add_argument("--max-side", type=int, default=1024)
    parser.add_argument("--grid-rows", type=int, default=3)
    parser.add_argument("--grid-cols", type=int, default=3)
    parser.add_argument("--fold", type=int, default=-1, help="Run one test block; default runs every block.")
    parser.add_argument("--validation-offset", type=int, default=1)
    parser.add_argument("--classifier", choices=["extratrees", "rf", "hgb", "logreg"], default="extratrees")
    parser.add_argument("--n-estimators", type=int, default=360)
    parser.add_argument("--max-depth", type=int, default=0, help="0 means no limit for ExtraTrees, RF keeps its local default unless set.")
    parser.add_argument("--min-samples-leaf", type=int, default=2)
    parser.add_argument("--use-xy", action="store_true")
    parser.add_argument("--no-factor", action="store_true")
    parser.add_argument("--max-pos-per-class", type=int, default=25000)
    parser.add_argument("--max-bg", type=int, default=80000)
    parser.add_argument("--chunk", type=int, default=450000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--percentiles", default="55,60,65,70,75,80,85,90,92,94,96,98")
    parser.add_argument("--close-radii", type=parse_int_list, default=(0, 2, 4, 6))
    parser.add_argument("--dilate-radii", type=parse_int_list, default=(0, 1, 2, 4))
    parser.add_argument("--fill-holes", type=parse_int_list, default=(0, 500, 2500))
    parser.add_argument("--min-area", type=int, default=80)
    parser.add_argument("--max-area", type=int, default=5000000)
    parser.add_argument("--eccentricity-max", type=float, default=0.995)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
