#!/usr/bin/env python3
"""Pixel-level micro-class classifier for weak small structures.

This is a calibration/upper-bound experiment on the single annotated sample:
train lightweight color/stain/texture classifiers for erythrocytes, pigment,
and immune infiltration, then sweep probability thresholds and component
filters. It is meant to test whether the remaining failure is feature
separability rather than SAM proposal quality.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from PIL import Image
from skimage import color, filters, morphology, measure
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_HE = PROJECT_ROOT / "output/visium_hd_exp1/assets/tissue_hires_image.png"
DEFAULT_SUMMARY = PROJECT_ROOT / "output/visium_hd_exp1/sam3_local_region_summary/region_summary.json"

LABELS = ["erythorocytes", "pigment", "immune_infiltration"]
COLORS = {"erythorocytes": (255, 127, 14), "pigment": (127, 127, 127), "immune_infiltration": (44, 160, 44)}


def max_side_shape(shape_hw: Tuple[int, int], max_side: int) -> Tuple[int, int]:
    h, w = shape_hw
    scale = min(1.0, float(max_side) / max(h, w))
    return max(1, int(round(h * scale))), max(1, int(round(w * scale)))


def resize(arr: np.ndarray, shape_hw: Tuple[int, int], resample: Image.Resampling) -> np.ndarray:
    h, w = shape_hw
    if arr.shape[:2] == (h, w):
        return arr
    return np.array(Image.fromarray(arr).resize((w, h), resample))


def resolve_path(path_text: str, base_dir: Path) -> Path:
    path = Path(path_text)
    if path.exists():
        return path
    candidate = base_dir / path_text
    if candidate.exists():
        return candidate
    return Path.cwd() / path_text


def read_mask(path: Path, shape_hw: Tuple[int, int]) -> np.ndarray:
    return np.array(Image.open(path).convert("L").resize((shape_hw[1], shape_hw[0]), Image.Resampling.NEAREST)) > 127


def load_targets(summary_path: Path, shape_hw: Tuple[int, int]) -> Dict[str, np.ndarray]:
    summary = json.loads(summary_path.read_text())
    targets = {}
    for item in summary["labels"]:
        slug = item.get("slug") or item["label"]
        if slug in LABELS:
            targets[slug] = read_mask(resolve_path(item["mask_path"], summary_path.parent), shape_hw)
    return targets


def tissue_mask(he: np.ndarray) -> np.ndarray:
    rgb = he.astype(np.float32) / 255.0
    hsv = color.rgb2hsv(rgb)
    gray = color.rgb2gray(rgb)
    mask = (hsv[..., 1] > 0.04) & (gray < 0.97)
    mask = morphology.remove_small_objects(mask, 512)
    return morphology.binary_closing(mask, morphology.disk(3))


def zscore(arr: np.ndarray, tissue: np.ndarray) -> np.ndarray:
    vals = arr[tissue]
    mean = float(vals.mean()) if vals.size else 0.0
    std = max(float(vals.std()), 1e-6) if vals.size else 1.0
    return np.clip((arr - mean) / std, -6, 6).astype(np.float32)


def make_features(he: np.ndarray, tissue: np.ndarray, use_xy: bool) -> np.ndarray:
    rgb = he.astype(np.float32) / 255.0
    lab = color.rgb2lab(rgb).astype(np.float32)
    hsv = color.rgb2hsv(rgb).astype(np.float32)
    hed = color.rgb2hed(rgb).astype(np.float32)
    gray = color.rgb2gray(rgb).astype(np.float32)
    sobel = filters.sobel(gray).astype(np.float32)
    gauss1 = filters.gaussian(gray, sigma=1).astype(np.float32)
    gauss3 = filters.gaussian(gray, sigma=3).astype(np.float32)
    yy, xx = np.indices(gray.shape, dtype=np.float32)
    channels = [
        rgb[..., 0],
        rgb[..., 1],
        rgb[..., 2],
        lab[..., 0] / 100.0,
        lab[..., 1] / 128.0,
        lab[..., 2] / 128.0,
        hsv[..., 0],
        hsv[..., 1],
        hsv[..., 2],
        hed[..., 0],
        hed[..., 1],
        hed[..., 2],
        sobel,
        gauss1,
        gauss3,
        rgb[..., 0] - 0.5 * (rgb[..., 1] + rgb[..., 2]),
        1.0 - hsv[..., 2],
    ]
    if use_xy:
        channels.extend([xx / max(1, gray.shape[1] - 1), yy / max(1, gray.shape[0] - 1)])
    channels = [zscore(ch, tissue) for ch in channels]
    return np.stack(channels, axis=-1)


def metrics(pred: np.ndarray, gt: np.ndarray) -> Dict[str, float]:
    inter = int(np.logical_and(pred, gt).sum())
    ps = int(pred.sum())
    gs = int(gt.sum())
    union = int(np.logical_or(pred, gt).sum())
    return {
        "dice": 2 * inter / (ps + gs) if ps + gs else 0.0,
        "iou": inter / union if union else 0.0,
        "precision": inter / ps if ps else 0.0,
        "recall": inter / gs if gs else 0.0,
        "pred_pixels": float(ps),
        "gt_pixels": float(gs),
    }


def sample_training(
    features: np.ndarray,
    targets: Dict[str, np.ndarray],
    tissue: np.ndarray,
    rng: np.random.Generator,
    max_pos: int,
    max_bg: int,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    label_names = ["background"] + LABELS
    occupied = np.zeros(tissue.shape, dtype=bool)
    xs = []
    ys = []
    for class_id, label in enumerate(LABELS, start=1):
        mask = targets[label] & tissue
        occupied |= mask
        coords = np.flatnonzero(mask.ravel())
        if coords.size > max_pos:
            coords = rng.choice(coords, size=max_pos, replace=False)
        xs.append(features.reshape(-1, features.shape[-1])[coords])
        ys.append(np.full(coords.size, class_id, dtype=np.int32))
    bg = tissue & ~occupied
    bg_coords = np.flatnonzero(bg.ravel())
    if bg_coords.size > max_bg:
        bg_coords = rng.choice(bg_coords, size=max_bg, replace=False)
    xs.append(features.reshape(-1, features.shape[-1])[bg_coords])
    ys.append(np.zeros(bg_coords.size, dtype=np.int32))
    return np.concatenate(xs, axis=0), np.concatenate(ys, axis=0), label_names


def make_classifier(kind: str, seed: int):
    if kind == "rf":
        return RandomForestClassifier(n_estimators=240, max_depth=20, min_samples_leaf=3, class_weight="balanced_subsample", n_jobs=-1, random_state=seed)
    if kind == "logreg":
        return make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, class_weight="balanced", n_jobs=1, multi_class="auto"))
    return ExtraTreesClassifier(n_estimators=320, max_depth=None, min_samples_leaf=2, class_weight="balanced", n_jobs=-1, random_state=seed)


def component_filter(mask: np.ndarray, min_area: int, max_area: int, eccentricity_max: float) -> np.ndarray:
    labels = measure.label(mask)
    out = np.zeros(mask.shape, dtype=bool)
    for region in measure.regionprops(labels):
        if region.area < min_area or region.area > max_area:
            continue
        if region.eccentricity > eccentricity_max:
            continue
        out[labels == region.label] = True
    return out


def predict_proba_image(model, features: np.ndarray, n_classes: int, chunk: int) -> np.ndarray:
    flat = features.reshape(-1, features.shape[-1])
    out = np.zeros((flat.shape[0], n_classes), dtype=np.float32)
    for start in range(0, flat.shape[0], chunk):
        stop = min(flat.shape[0], start + chunk)
        proba = model.predict_proba(flat[start:stop])
        classes = getattr(model, "classes_", None)
        if classes is None and hasattr(model, "named_steps"):
            classes = model.named_steps["logisticregression"].classes_
        for col, class_id in enumerate(classes):
            out[start:stop, int(class_id)] = proba[:, col]
    return out.reshape(features.shape[:2] + (n_classes,))


def overlay(he: np.ndarray, masks: Dict[str, np.ndarray]) -> np.ndarray:
    out = he.astype(np.float32).copy()
    for label, mask in masks.items():
        out[mask] = 0.5 * out[mask] + 0.5 * np.array(COLORS[label], dtype=np.float32)
    return np.clip(out, 0, 255).astype(np.uint8)


def run(args: argparse.Namespace) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    he_full = np.array(Image.open(args.he_image).convert("RGB"))
    shape = max_side_shape(he_full.shape[:2], args.max_side)
    he = resize(he_full, shape, Image.Resampling.BILINEAR)
    tissue = tissue_mask(he)
    targets = load_targets(args.summary_path, shape)
    features = make_features(he, tissue, args.use_xy)
    rng = np.random.default_rng(args.seed)
    x_train, y_train, label_names = sample_training(features, targets, tissue, rng, args.max_pos, args.max_bg)
    model = make_classifier(args.classifier, args.seed)
    model.fit(x_train, y_train)
    probs = predict_proba_image(model, features, len(label_names), args.chunk)
    percentiles = [float(x) for x in args.percentiles.split(",")]
    rows = []
    best_masks: Dict[str, np.ndarray] = {}
    mask_dir = args.output_dir / "masks"
    mask_dir.mkdir(parents=True, exist_ok=True)
    for class_id, label in enumerate(LABELS, start=1):
        if args.labels and label not in args.labels.split(","):
            continue
        p = probs[..., class_id]
        vals = p[tissue]
        best_row = None
        best_mask = None
        for pct in percentiles:
            thr = float(np.percentile(vals, pct))
            raw = (p >= thr) & tissue
            for close_radius in args.close_radii:
                closed = morphology.binary_closing(raw, morphology.disk(close_radius)) if close_radius > 0 else raw
                for dilate_radius in args.dilate_radii:
                    pred = morphology.binary_dilation(closed, morphology.disk(dilate_radius)) if dilate_radius > 0 else closed
                    pred = component_filter(pred, args.min_area, args.max_area, args.eccentricity_max)
                    row = {
                        "label": label,
                        "classifier": args.classifier,
                        "percentile": pct,
                        "threshold": thr,
                        "close_radius": close_radius,
                        "dilate_radius": dilate_radius,
                        **metrics(pred, targets[label]),
                    }
                    rows.append(row)
                    if best_row is None or row["dice"] > best_row["dice"]:
                        best_row = row
                        best_mask = pred
        if best_mask is not None:
            best_masks[label] = best_mask
            Image.fromarray(best_mask.astype(np.uint8) * 255).save(mask_dir / f"{label}_micro_classifier_best.png")
    df = pd.DataFrame(rows)
    df.to_csv(args.output_dir / "micro_classifier_sweep_metrics.csv", index=False)
    best = df.sort_values("dice", ascending=False).groupby("label", as_index=False).head(1)
    best.to_csv(args.output_dir / "micro_classifier_best_by_label.csv", index=False)
    Image.fromarray(overlay(he, best_masks)).save(args.output_dir / "micro_classifier_overlay.png")
    lines = ["# Micro Pixel Classifier Sweep", "", "| label | Dice | precision | recall | classifier | pct | close | dilate |", "|---|---:|---:|---:|---|---:|---:|---:|"]
    for _, row in best.sort_values("dice", ascending=False).iterrows():
        lines.append(
            f"| {row['label']} | {row['dice']:.3f} | {row['precision']:.3f} | {row['recall']:.3f} | "
            f"{row['classifier']} | {row['percentile']:.1f} | {int(row['close_radius'])} | {int(row['dilate_radius'])} |"
        )
    (args.output_dir / "README.md").write_text("\n".join(lines) + "\n")
    print(best.sort_values("dice", ascending=False).to_string(index=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pixel classifier sweep for micro classes.")
    parser.add_argument("--he-image", type=Path, default=DEFAULT_HE)
    parser.add_argument("--summary-path", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-side", type=int, default=1536)
    parser.add_argument("--classifier", choices=["extratrees", "rf", "logreg"], default="extratrees")
    parser.add_argument("--labels", default="")
    parser.add_argument("--percentiles", default="80,85,90,92,94,95,96,97,98,98.5,99,99.3,99.5")
    parser.add_argument("--max-pos", type=int, default=60000)
    parser.add_argument("--max-bg", type=int, default=180000)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--chunk", type=int, default=500000)
    parser.add_argument("--use-xy", action="store_true")
    parser.add_argument("--min-area", type=int, default=2)
    parser.add_argument("--max-area", type=int, default=4096)
    parser.add_argument("--eccentricity-max", type=float, default=0.995)
    parser.add_argument("--close-radii", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--dilate-radii", type=int, nargs="+", default=[0, 1, 2, 3])
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
