#!/usr/bin/env python3
"""Unified multiclass pixel classifier for all Visium HD Exp1 labels.

This keeps one model for every class: background + 8 tissue/cell-type labels.
Features combine H&E color/stain/texture and projected FICTURE/gene-prior
channels, but prediction is made by a single classifier.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from PIL import Image
from skimage import color, filters, measure, morphology
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_HE = PROJECT_ROOT / "output/visium_hd_exp1/assets/tissue_hires_image.png"
DEFAULT_SUMMARY = PROJECT_ROOT / "output/visium_hd_exp1/sam3_local_region_summary/region_summary.json"
DEFAULT_FACTOR = PROJECT_ROOT / "output/visium_hd_exp1/molecular_prior_experiments/dx-60_dy80/ficture_factor_label_image_he.npy"
DEFAULT_ANNOT = PROJECT_ROOT / "output/visium_hd_exp1/molecular_prior_experiments/dx-60_dy80/factor_annotation.csv"

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

TRAIN_PRIORITY = [
    "pigment",
    "erythorocytes",
    "immune_infiltration",
    "lung_bronchiola",
    "lung_vessels",
    "lung_alveoli_normal_adjacent",
    "tumor",
    "stroma",
]

COLORS = {
    "lung_bronchiola": (31, 119, 180),
    "erythorocytes": (255, 127, 14),
    "immune_infiltration": (44, 160, 44),
    "lung_alveoli_normal_adjacent": (148, 103, 189),
    "lung_vessels": (140, 86, 75),
    "pigment": (127, 127, 127),
    "stroma": (188, 189, 34),
    "tumor": (23, 190, 207),
}


def slugify(label: str) -> str:
    return "".join(ch.lower() if ch.isalnum() else "_" for ch in label).strip("_")


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
        slug = item.get("slug") or slugify(item["label"])
        if slug in LABEL_ORDER:
            targets[slug] = read_mask(resolve_path(item["mask_path"], summary_path.parent), shape_hw)
    return targets


def load_gene_factor_scores(path: Path, n_factors: int) -> Dict[str, np.ndarray]:
    scores: Dict[str, np.ndarray] = {label: np.zeros(n_factors, dtype=np.float64) for label in LABEL_ORDER}
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            factor = int(row["factor"])
            if factor < 0 or factor >= n_factors:
                continue
            label_scores = json.loads(row.get("label_scores_json") or "{}")
            for label, score in label_scores.items():
                slug = slugify(label)
                if slug in scores:
                    scores[slug][factor] = max(scores[slug][factor], float(score))
    for label, vec in list(scores.items()):
        if vec.max() > 0:
            scores[label] = vec / vec.max()
    return scores


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


def make_features(
    he: np.ndarray,
    factors: np.ndarray,
    factor_annotation: Path,
    tissue: np.ndarray,
    *,
    use_xy: bool,
    use_factor: bool,
) -> np.ndarray:
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
    if use_factor:
        valid = factors >= 0
        n_factors = int(factors[valid].max()) + 1 if valid.any() else 1
        factor_norm = np.zeros(factors.shape, dtype=np.float32)
        factor_norm[valid] = factors[valid] / max(1, n_factors - 1)
        channels.append(factor_norm)
        gene_scores = load_gene_factor_scores(factor_annotation, n_factors)
        for label in LABEL_ORDER:
            score = np.zeros(factors.shape, dtype=np.float32)
            score[valid] = gene_scores[label][factors[valid]]
            channels.append(score)
    if use_xy:
        channels.extend([xx / max(1, gray.shape[1] - 1), yy / max(1, gray.shape[0] - 1)])
    channels = [zscore(ch.astype(np.float32), tissue) for ch in channels]
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


def make_training_label_image(targets: Dict[str, np.ndarray], tissue: np.ndarray) -> np.ndarray:
    label_img = np.zeros(tissue.shape, dtype=np.int16)
    for class_id, label in enumerate(LABEL_ORDER, start=1):
        if label in targets:
            label_img[targets[label] & tissue] = class_id
    # Re-apply priority so small classes are not swallowed in overlap areas.
    for label in TRAIN_PRIORITY:
        class_id = LABEL_ORDER.index(label) + 1
        label_img[targets[label] & tissue] = class_id
    return label_img


def sample_training(
    features: np.ndarray,
    label_img: np.ndarray,
    tissue: np.ndarray,
    rng: np.random.Generator,
    max_pos_per_class: int,
    max_bg: int,
) -> Tuple[np.ndarray, np.ndarray]:
    flat_x = features.reshape(-1, features.shape[-1])
    flat_y = label_img.ravel()
    xs = []
    ys = []
    for class_id in range(1, len(LABEL_ORDER) + 1):
        coords = np.flatnonzero(flat_y == class_id)
        if coords.size > max_pos_per_class:
            coords = rng.choice(coords, size=max_pos_per_class, replace=False)
        xs.append(flat_x[coords])
        ys.append(np.full(coords.size, class_id, dtype=np.int16))
    bg = tissue.ravel() & (flat_y == 0)
    bg_coords = np.flatnonzero(bg)
    if bg_coords.size > max_bg:
        bg_coords = rng.choice(bg_coords, size=max_bg, replace=False)
    xs.append(flat_x[bg_coords])
    ys.append(np.zeros(bg_coords.size, dtype=np.int16))
    return np.concatenate(xs, axis=0), np.concatenate(ys, axis=0)


def make_classifier(kind: str, seed: int):
    if kind == "rf":
        return RandomForestClassifier(n_estimators=260, max_depth=24, min_samples_leaf=2, class_weight="balanced_subsample", n_jobs=-1, random_state=seed)
    if kind == "hgb":
        return HistGradientBoostingClassifier(max_iter=240, learning_rate=0.08, l2_regularization=0.02, random_state=seed)
    if kind == "logreg":
        return make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, class_weight="balanced", n_jobs=1))
    return ExtraTreesClassifier(n_estimators=360, max_depth=None, min_samples_leaf=2, class_weight="balanced", n_jobs=-1, random_state=seed)


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


def overlay(he: np.ndarray, masks: Dict[str, np.ndarray]) -> np.ndarray:
    out = he.astype(np.float32).copy()
    for label in LABEL_ORDER:
        mask = masks.get(label)
        if mask is None:
            continue
        out[mask] = 0.55 * out[mask] + 0.45 * np.array(COLORS[label], dtype=np.float32)
    return np.clip(out, 0, 255).astype(np.uint8)


def run(args: argparse.Namespace) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    he_full = np.array(Image.open(args.he_image).convert("RGB"))
    shape = max_side_shape(he_full.shape[:2], args.max_side)
    he = resize(he_full, shape, Image.Resampling.BILINEAR)
    factor_full = np.load(args.factor_image)
    factors = resize(factor_full.astype(np.int16), shape, Image.Resampling.NEAREST).astype(np.int16)
    tissue = tissue_mask(he)
    targets = load_targets(args.summary_path, shape)
    features = make_features(he, factors, args.factor_annotation, tissue, use_xy=args.use_xy, use_factor=not args.no_factor)
    label_img = make_training_label_image(targets, tissue)
    rng = np.random.default_rng(args.seed)
    x_train, y_train = sample_training(features, label_img, tissue, rng, args.max_pos_per_class, args.max_bg)
    model = make_classifier(args.classifier, args.seed)
    model.fit(x_train, y_train)
    probs = predict_proba_image(model, features, len(LABEL_ORDER) + 1, args.chunk)

    argmax = probs.argmax(axis=-1)
    percentiles = [float(x) for x in args.percentiles.split(",")]
    rows: List[Dict[str, object]] = []
    best_masks: Dict[str, np.ndarray] = {}
    argmax_masks: Dict[str, np.ndarray] = {}
    mask_dir = args.output_dir / "masks"
    mask_dir.mkdir(parents=True, exist_ok=True)

    for class_id, label in enumerate(LABEL_ORDER, start=1):
        arg_pred = (argmax == class_id) & tissue
        arg_pred = component_filter(arg_pred, args.min_area, args.max_area, args.eccentricity_max)
        argmax_masks[label] = arg_pred
        rows.append({"label": label, "mode": "argmax", "percentile": -1.0, "threshold": -1.0, "close_radius": 0, "dilate_radius": 0, **metrics(arg_pred, targets[label])})
        p = probs[..., class_id]
        vals = p[tissue]
        best_row = rows[-1]
        best_mask = arg_pred
        for pct in percentiles:
            thr = float(np.percentile(vals, pct))
            raw = (p >= thr) & tissue
            for close_radius in args.close_radii:
                closed = morphology.binary_closing(raw, morphology.disk(close_radius)) if close_radius > 0 else raw
                for dilate_radius in args.dilate_radii:
                    pred = morphology.binary_dilation(closed, morphology.disk(dilate_radius)) if dilate_radius > 0 else closed
                    pred = component_filter(pred, args.min_area, args.max_area, args.eccentricity_max)
                    row = {"label": label, "mode": "threshold", "percentile": pct, "threshold": thr, "close_radius": close_radius, "dilate_radius": dilate_radius, **metrics(pred, targets[label])}
                    rows.append(row)
                    if row["dice"] > best_row["dice"]:
                        best_row = row
                        best_mask = pred
        best_masks[label] = best_mask
        Image.fromarray(arg_pred.astype(np.uint8) * 255).save(mask_dir / f"{label}_unified_argmax_mask.png")
        Image.fromarray(best_mask.astype(np.uint8) * 255).save(mask_dir / f"{label}_unified_best_mask.png")

    df = pd.DataFrame(rows)
    df.to_csv(args.output_dir / "unified_pixel_classifier_metrics.csv", index=False)
    best = df.sort_values("dice", ascending=False).groupby("label", as_index=False).head(1)
    best.to_csv(args.output_dir / "unified_pixel_classifier_best_by_label.csv", index=False)
    Image.fromarray(overlay(he, argmax_masks)).save(args.output_dir / "unified_argmax_overlay.png")
    Image.fromarray(overlay(he, best_masks)).save(args.output_dir / "unified_best_overlay.png")

    lines = ["# Unified Pixel Classifier", ""]
    lines.append(f"classifier={args.classifier}, max_side={args.max_side}, use_xy={args.use_xy}, use_factor={not args.no_factor}")
    lines += ["", "| label | Dice | precision | recall | mode | pct | close | dilate |", "|---|---:|---:|---:|---|---:|---:|---:|"]
    for _, row in best.sort_values("dice", ascending=False).iterrows():
        lines.append(
            f"| {row['label']} | {row['dice']:.3f} | {row['precision']:.3f} | {row['recall']:.3f} | "
            f"{row['mode']} | {row['percentile']:.1f} | {int(row['close_radius'])} | {int(row['dilate_radius'])} |"
        )
    lines += ["", "## Argmax Only", "", "| label | Dice | precision | recall |", "|---|---:|---:|---:|"]
    arg_df = df[df["mode"] == "argmax"]
    for _, row in arg_df.sort_values("dice", ascending=False).iterrows():
        lines.append(f"| {row['label']} | {row['dice']:.3f} | {row['precision']:.3f} | {row['recall']:.3f} |")
    (args.output_dir / "README.md").write_text("\n".join(lines) + "\n")
    print(best.sort_values("dice", ascending=False).to_string(index=False))
    print("\nARGMAX")
    print(arg_df.sort_values("dice", ascending=False).to_string(index=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Unified multiclass pixel classifier.")
    parser.add_argument("--he-image", type=Path, default=DEFAULT_HE)
    parser.add_argument("--summary-path", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--factor-image", type=Path, default=DEFAULT_FACTOR)
    parser.add_argument("--factor-annotation", type=Path, default=DEFAULT_ANNOT)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-side", type=int, default=1536)
    parser.add_argument("--classifier", choices=["extratrees", "rf", "hgb", "logreg"], default="extratrees")
    parser.add_argument("--percentiles", default="70,75,80,85,90,92,94,95,96,97,98,98.5,99,99.3,99.5")
    parser.add_argument("--max-pos-per-class", type=int, default=60000)
    parser.add_argument("--max-bg", type=int, default=220000)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--chunk", type=int, default=450000)
    parser.add_argument("--use-xy", action="store_true")
    parser.add_argument("--no-factor", action="store_true")
    parser.add_argument("--min-area", type=int, default=2)
    parser.add_argument("--max-area", type=int, default=600000)
    parser.add_argument("--eccentricity-max", type=float, default=0.995)
    parser.add_argument("--close-radii", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--dilate-radii", type=int, nargs="+", default=[0, 1, 2, 3])
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
