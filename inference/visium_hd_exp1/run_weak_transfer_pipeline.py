#!/usr/bin/env python3
"""Weak-transfer pseudo-segmentation for the 1-annotated + N-unannotated setup.

This pipeline is designed for the real data constraint:
- one annotated VisiumHD sample
- many unannotated samples

It learns weak label evidence from the annotated sample, avoids spatial x/y
features by default, and exports pseudo masks plus confidence/uncertainty maps.
When targets are available for a sample, it reports metrics; otherwise it only
reports confidence summaries.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
from scipy import ndimage as ndi
from skimage import color, filters, morphology
from sklearn.ensemble import ExtraTreesClassifier


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "output" / "visium_hd_exp1" / "weak_transfer_pseudoseg"
DEFAULT_ANNOTATED_HE = PROJECT_ROOT / "output" / "visium_hd_exp1" / "assets" / "tissue_hires_image.png"
DEFAULT_ANNOTATED_FACTOR = PROJECT_ROOT / "output" / "visium_hd_exp1" / "molecular_prior_experiments" / "dx-60_dy80" / "ficture_factor_label_image_he.npy"
DEFAULT_SAMPLE_MANIFEST = DEFAULT_OUTPUT_DIR / "sample_manifest.csv"
TARGET_DIRS = [
    PROJECT_ROOT / "output" / "visium_hd_exp1" / "sam3_runs" / "base_sam3_multipoint" / "masks",
    PROJECT_ROOT / "output" / "visium_hd_exp1" / "sam3_runs" / "base_sam3_multipoint_remaining_dev" / "masks",
]

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

LABEL_COLORS: Dict[str, Tuple[int, int, int]] = {
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
    if max_side <= 0 or max(h, w) <= max_side:
        return h, w
    scale = max_side / float(max(h, w))
    return max(1, int(round(h * scale))), max(1, int(round(w * scale)))


def load_targets(shape_hw: Tuple[int, int]) -> Dict[str, np.ndarray]:
    targets: Dict[str, np.ndarray] = {}
    for target_dir in TARGET_DIRS:
        for path in sorted(target_dir.glob("*_target.png")):
            label = path.name.split("_", 1)[1].removesuffix("_target.png")
            mask = np.array(Image.open(path).convert("L")) > 127
            if mask.shape != shape_hw:
                mask = resize_array(mask.astype(np.uint8) * 255, shape_hw, Image.Resampling.NEAREST) > 127
            targets[label] = mask
    return {label: targets[label] for label in LABEL_ORDER if label in targets}


def make_multiclass_target(targets: Dict[str, np.ndarray], tissue_mask: np.ndarray) -> Tuple[np.ndarray, Dict[int, str]]:
    y = np.zeros(tissue_mask.shape, dtype=np.uint8)
    id_to_label = {0: "background"}
    for idx, label in enumerate(LABEL_ORDER, start=1):
        if label in targets:
            y[targets[label]] = idx
            id_to_label[idx] = label
    y[~tissue_mask] = 0
    return y, id_to_label


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


def estimate_tissue_mask(he: np.ndarray) -> np.ndarray:
    tissue = np.any(he < 245, axis=-1)
    tissue = morphology.remove_small_objects(tissue, min_size=500)
    tissue = ndi.binary_fill_holes(tissue)
    return tissue.astype(bool)


def build_transfer_features(he: np.ndarray, label_image: np.ndarray, tissue_mask: np.ndarray) -> np.ndarray:
    he_float = he.astype(np.float32) / 255.0
    hsv = color.rgb2hsv(he_float).astype(np.float32)
    lab = color.rgb2lab(he_float).astype(np.float32)
    gray = color.rgb2gray(he_float).astype(np.float32)
    sobel = filters.sobel(gray).astype(np.float32)
    h, w = gray.shape

    factor = np.clip(label_image, -1, 11)
    factor_one_hot = np.zeros((h, w, 13), dtype=np.float32)
    flat_factor = (factor.reshape(-1) + 1).astype(np.int16).clip(0, 12)
    factor_one_hot.reshape(-1, 13)[np.arange(h * w), flat_factor] = 1.0

    parts = [
        he_float,
        hsv,
        lab / np.array([100.0, 128.0, 128.0], dtype=np.float32),
        gray[..., None],
        sobel[..., None],
        factor_one_hot,
    ]
    for sigma in (2.0, 6.0, 16.0):
        parts.append(ndi.gaussian_filter(gray, sigma=sigma).astype(np.float32)[..., None])
        parts.append(ndi.gaussian_filter(sobel, sigma=sigma).astype(np.float32)[..., None])
        for channel in range(3):
            parts.append(ndi.gaussian_filter(lab[..., channel], sigma=sigma).astype(np.float32)[..., None])
    for size in (9, 31, 91):
        for factor_id in range(12):
            parts.append(ndi.uniform_filter((label_image == factor_id).astype(np.float32), size=size)[..., None])
    parts.append(ndi.uniform_filter(tissue_mask.astype(np.float32), size=15)[..., None])
    return np.dstack(parts).astype(np.float32)


def sample_training_indices(
    y: np.ndarray,
    tissue_mask: np.ndarray,
    max_per_class: int,
    rng: np.random.Generator,
) -> np.ndarray:
    idxs = []
    for class_id in sorted(np.unique(y)):
        mask = y == class_id
        if class_id == 0:
            mask &= tissue_mask
        coords = np.flatnonzero(mask)
        if coords.size == 0:
            continue
        n = min(max_per_class, coords.size)
        idxs.append(rng.choice(coords, size=n, replace=False))
    return np.concatenate(idxs) if idxs else np.array([], dtype=np.int64)


def predict_proba_in_chunks(
    model: ExtraTreesClassifier,
    features: np.ndarray,
    chunk_size: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    flat = features.reshape(-1, features.shape[-1])
    class_ids = model.classes_.astype(np.uint8)
    pred = np.zeros(flat.shape[0], dtype=np.uint8)
    conf = np.zeros(flat.shape[0], dtype=np.float32)
    margin = np.zeros(flat.shape[0], dtype=np.float32)
    dense_proba = np.zeros((flat.shape[0], int(class_ids.max()) + 1), dtype=np.float32)
    for start in range(0, flat.shape[0], chunk_size):
        stop = min(flat.shape[0], start + chunk_size)
        proba = model.predict_proba(flat[start:stop]).astype(np.float32)
        dense_proba[start:stop, class_ids] = proba
        order = np.argsort(proba, axis=1)
        best_idx = order[:, -1]
        second_idx = order[:, -2] if proba.shape[1] > 1 else order[:, -1]
        pred[start:stop] = class_ids[best_idx]
        conf[start:stop] = proba[np.arange(stop - start), best_idx]
        margin[start:stop] = proba[np.arange(stop - start), best_idx] - proba[np.arange(stop - start), second_idx]
    confidence = np.minimum(conf.reshape(features.shape[:2]), 1.0)
    margin_image = np.minimum(np.maximum(margin.reshape(features.shape[:2]), 0.0), 1.0)
    return (
        pred.reshape(features.shape[:2]),
        np.minimum(confidence * 0.65 + margin_image * 0.35, 1.0),
        dense_proba.reshape((*features.shape[:2], dense_proba.shape[-1])),
    )


def estimate_label_factor_support(y: np.ndarray, label_image: np.ndarray, id_to_label: Dict[int, str]) -> Dict[int, set[int]]:
    support: Dict[int, set[int]] = {}
    for class_id, label in id_to_label.items():
        if class_id == 0:
            continue
        factors, counts = np.unique(label_image[y == class_id], return_counts=True)
        rows = [(int(factor), int(count)) for factor, count in zip(factors, counts) if factor >= 0]
        rows.sort(key=lambda item: item[1], reverse=True)
        total = sum(count for _, count in rows)
        keep = set()
        acc = 0
        for factor, count in rows:
            keep.add(factor)
            acc += count
            if total > 0 and acc / total >= 0.85:
                break
        support[class_id] = keep
    return support


def clean_prediction(
    pred: np.ndarray,
    confidence: np.ndarray,
    tissue_mask: np.ndarray,
    min_confidence: float,
    *,
    proba: np.ndarray | None = None,
    factor: np.ndarray | None = None,
    label_factor_support: Dict[int, set[int]] | None = None,
    rare_rescue: bool = False,
) -> np.ndarray:
    out = pred.copy()
    class_thresholds = {
        2: min(min_confidence, 0.12),  # erythorocytes
        6: min(min_confidence, 0.06),  # pigment
    }
    if proba is not None:
        for class_id, threshold in class_thresholds.items():
            if class_id >= proba.shape[-1]:
                continue
            weak = (pred == class_id) & (proba[..., class_id] >= threshold)
            out[weak] = class_id
    out[(confidence < min_confidence) | (~tissue_mask)] = 0
    if rare_rescue and proba is not None and factor is not None and label_factor_support is not None:
        rescue_specs = {
            2: 0.08,  # erythorocytes
            6: 0.035,  # pigment
        }
        for class_id, threshold in rescue_specs.items():
            if class_id >= proba.shape[-1]:
                continue
            support = label_factor_support.get(class_id, set())
            factor_ok = np.isin(factor, list(support)) if support else np.ones_like(tissue_mask, dtype=bool)
            rescue = (proba[..., class_id] >= threshold) & factor_ok & tissue_mask
            out[rescue] = class_id
    cleaned = np.zeros_like(out, dtype=np.uint8)
    for class_id in sorted(np.unique(out)):
        if class_id == 0:
            continue
        mask = out == class_id
        min_size = 12 if class_id in (2, 6) else 100
        mask = morphology.remove_small_objects(mask, min_size=min_size)
        close_radius = 0 if class_id in (2, 6) else 1
        if close_radius > 0:
            mask = morphology.binary_closing(mask, morphology.disk(close_radius))
        cleaned[mask] = class_id
    return cleaned


def calibrate_class_thresholds(
    proba: np.ndarray,
    y: np.ndarray,
    tissue_mask: np.ndarray,
    id_to_label: Dict[int, str],
    *,
    min_threshold: float = 0.02,
    max_threshold: float = 0.85,
) -> Dict[int, float]:
    thresholds: Dict[int, float] = {}
    grid = np.concatenate(
        [
            np.linspace(min_threshold, 0.18, 17),
            np.linspace(0.20, max_threshold, 27),
        ]
    )
    for class_id, label in id_to_label.items():
        if class_id == 0 or class_id >= proba.shape[-1]:
            continue
        gt = y == class_id
        best_threshold = 0.5
        best_dice = -1.0
        for threshold in grid:
            pred = (proba[..., class_id] >= threshold) & tissue_mask
            pred = morphology.remove_small_objects(pred, min_size=12 if class_id in (2, 6) else 100)
            metric = mask_metrics(pred, gt)
            if metric["dice"] > best_dice:
                best_dice = metric["dice"]
                best_threshold = float(threshold)
        thresholds[class_id] = best_threshold
    return thresholds


def apply_class_thresholds(
    proba: np.ndarray,
    thresholds: Dict[int, float],
    tissue_mask: np.ndarray,
    *,
    weights: Dict[int, float] | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    score = np.zeros(proba.shape, dtype=np.float32)
    weights = weights or {}
    for class_id, threshold in thresholds.items():
        if class_id >= proba.shape[-1] or threshold <= 0:
            continue
        score[..., class_id] = (proba[..., class_id] / threshold) * weights.get(class_id, 1.0)
    score[..., 0] = 0.0
    pred = np.argmax(score, axis=-1).astype(np.uint8)
    confidence = np.max(score, axis=-1).astype(np.float32)
    pred[(confidence < 1.0) | (~tissue_mask)] = 0
    confidence = np.minimum(confidence / 3.0, 1.0)
    cleaned = np.zeros_like(pred, dtype=np.uint8)
    for class_id in sorted(np.unique(pred)):
        if class_id == 0:
            continue
        mask = pred == class_id
        mask = morphology.remove_small_objects(mask, min_size=12 if class_id in (2, 6) else 100)
        if class_id not in (2, 6):
            mask = morphology.binary_closing(mask, morphology.disk(1))
        cleaned[mask] = class_id
    return cleaned, confidence


def calibrate_competition_weights(
    proba: np.ndarray,
    y: np.ndarray,
    tissue_mask: np.ndarray,
    id_to_label: Dict[int, str],
    thresholds: Dict[int, float],
    *,
    rounds: int = 3,
) -> Tuple[Dict[int, float], List[Dict[str, float]]]:
    """Tune class weights against the final non-overlap multiclass objective."""
    class_ids = [class_id for class_id in sorted(thresholds) if class_id in id_to_label]
    weights = {class_id: 1.0 for class_id in class_ids}
    candidates = np.array([0.35, 0.45, 0.58, 0.75, 0.9, 1.0, 1.15, 1.35, 1.6, 1.9, 2.3, 2.8], dtype=np.float32)

    def evaluate(current_weights: Dict[int, float]) -> Tuple[float, List[Dict[str, float]]]:
        pred, _ = apply_class_thresholds(proba, thresholds, tissue_mask, weights=current_weights)
        rows = []
        for class_id in class_ids:
            label = id_to_label[class_id]
            rows.append({"label": label, **mask_metrics(pred == class_id, y == class_id)})
        dice_values = np.array([row["dice"] for row in rows], dtype=np.float32)
        rare = np.array(
            [row["dice"] for row in rows if row["label"] in {"erythorocytes", "pigment", "immune_infiltration"}],
            dtype=np.float32,
        )
        min_dice = float(dice_values.min()) if dice_values.size else 0.0
        mean_dice = float(dice_values.mean()) if dice_values.size else 0.0
        rare_mean = float(rare.mean()) if rare.size else mean_dice
        objective = 0.55 * min_dice + 0.30 * mean_dice + 0.15 * rare_mean
        return objective, rows

    best_objective, best_rows = evaluate(weights)
    for _ in range(rounds):
        improved = False
        for class_id in class_ids:
            class_best = weights[class_id]
            class_objective = best_objective
            class_rows = best_rows
            for candidate in candidates:
                trial = dict(weights)
                trial[class_id] = float(candidate)
                objective, rows = evaluate(trial)
                if objective > class_objective + 1e-6:
                    class_best = float(candidate)
                    class_objective = objective
                    class_rows = rows
            if class_best != weights[class_id]:
                weights[class_id] = class_best
                best_objective = class_objective
                best_rows = class_rows
                improved = True
        if not improved:
            break
    return weights, best_rows


def overlay(he: np.ndarray, label_index: np.ndarray, id_to_label: Dict[int, str], alpha: float = 0.42) -> np.ndarray:
    out = he.astype(np.float32).copy()
    for class_id, label in id_to_label.items():
        if class_id == 0:
            continue
        color_rgb = np.array(LABEL_COLORS.get(label, (255, 255, 0)), dtype=np.float32)
        idx = label_index == class_id
        out[idx] = (1.0 - alpha) * out[idx] + alpha * color_rgb
    return np.clip(out, 0, 255).astype(np.uint8)


def save_uncertain_tiles(
    he: np.ndarray,
    pred: np.ndarray,
    confidence: np.ndarray,
    id_to_label: Dict[int, str],
    outdir: Path,
    tile_size: int,
    top_k: int,
) -> pd.DataFrame:
    outdir.mkdir(parents=True, exist_ok=True)
    h, w = confidence.shape
    rows = []
    for y0 in range(0, h, tile_size):
        for x0 in range(0, w, tile_size):
            y1 = min(h, y0 + tile_size)
            x1 = min(w, x0 + tile_size)
            tile_pred = pred[y0:y1, x0:x1]
            fg = tile_pred > 0
            if fg.sum() < 50:
                continue
            tile_conf = confidence[y0:y1, x0:x1][fg]
            rows.append(
                {
                    "x0": x0,
                    "y0": y0,
                    "x1": x1,
                    "y1": y1,
                    "mean_confidence": float(tile_conf.mean()),
                    "foreground_pixels": int(fg.sum()),
                }
            )
    df = pd.DataFrame(rows).sort_values("mean_confidence", ascending=True).head(top_k)
    for idx, row in df.reset_index(drop=True).iterrows():
        x0, y0, x1, y1 = (int(row[k]) for k in ("x0", "y0", "x1", "y1"))
        tile_he = he[y0:y1, x0:x1]
        tile_pred = pred[y0:y1, x0:x1]
        tile_overlay = overlay(tile_he, tile_pred, id_to_label)
        canvas = Image.new("RGB", (tile_he.shape[1] * 2, tile_he.shape[0]), "white")
        canvas.paste(Image.fromarray(tile_he), (0, 0))
        canvas.paste(Image.fromarray(tile_overlay), (tile_he.shape[1], 0))
        canvas.save(outdir / f"uncertain_tile_{idx:02d}_conf_{row['mean_confidence']:.3f}.png")
    df.to_csv(outdir / "uncertain_tiles.csv", index=False)
    return df


def create_default_manifest(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(
        [
            {
                "sample_id": "visium_hd_exp1_annotated",
                "he_path": str(DEFAULT_ANNOTATED_HE),
                "factor_label_path": str(DEFAULT_ANNOTATED_FACTOR),
                "has_annotation": True,
            }
        ]
    )
    df.to_csv(path, index=False)


def load_sample(he_path: Path, factor_path: Path, max_side: int) -> Tuple[np.ndarray, np.ndarray, float]:
    he_full = np.array(Image.open(he_path).convert("RGB"))
    factor_full = np.load(factor_path).astype(np.int16)
    shape = max_side_shape(he_full.shape[:2], max_side)
    scale = shape[0] / float(he_full.shape[0])
    he = resize_array(he_full, shape, Image.Resampling.BILINEAR)
    factor = resize_array(factor_full.astype(np.int16), shape, Image.Resampling.NEAREST).astype(np.int16)
    return he, factor, scale


def main() -> None:
    parser = argparse.ArgumentParser(description="Run weak-transfer pseudo-segmentation.")
    parser.add_argument("--sample-manifest", type=Path, default=DEFAULT_SAMPLE_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max-side", type=int, default=1536)
    parser.add_argument("--max-per-class", type=int, default=45000)
    parser.add_argument("--n-estimators", type=int, default=180)
    parser.add_argument("--min-confidence", type=float, default=0.42)
    parser.add_argument("--chunk-size", type=int, default=300000)
    parser.add_argument("--rare-rescue", action="store_true")
    parser.add_argument("--calibrate-thresholds", action="store_true")
    parser.add_argument("--calibrate-competition", action="store_true")
    args = parser.parse_args()

    if not args.sample_manifest.exists():
        create_default_manifest(args.sample_manifest)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(20260510)

    annotated_he, annotated_factor, _ = load_sample(DEFAULT_ANNOTATED_HE, DEFAULT_ANNOTATED_FACTOR, args.max_side)
    annotated_tissue = estimate_tissue_mask(annotated_he)
    targets = load_targets(annotated_he.shape[:2])
    y, id_to_label = make_multiclass_target(targets, annotated_tissue)
    label_factor_support = estimate_label_factor_support(y, annotated_factor, id_to_label)
    features = build_transfer_features(annotated_he, annotated_factor, annotated_tissue)
    train_idx = sample_training_indices(y, annotated_tissue, args.max_per_class, rng)
    if train_idx.size == 0:
        raise RuntimeError("No annotated training pixels were available.")

    print("=" * 72)
    print("Weak-transfer pseudo-segmentation")
    print("=" * 72)
    print(f"Training pixels: {train_idx.size}")
    print(f"Features: {features.shape[-1]}")
    print(f"Manifest: {args.sample_manifest}")

    model = ExtraTreesClassifier(
        n_estimators=args.n_estimators,
        min_samples_leaf=2,
        class_weight="balanced_subsample",
        n_jobs=-1,
        random_state=20260510,
    )
    model.fit(features.reshape(-1, features.shape[-1])[train_idx], y.reshape(-1)[train_idx])
    calibrated_thresholds: Dict[int, float] = {}
    calibrated_weights: Dict[int, float] = {}
    if args.calibrate_thresholds:
        print("Calibrating per-class thresholds on the annotated sample...")
        _, _, annotated_proba = predict_proba_in_chunks(model, features, args.chunk_size)
        calibrated_thresholds = calibrate_class_thresholds(annotated_proba, y, annotated_tissue, id_to_label)
        (args.output_dir / "calibrated_thresholds.json").write_text(
            json.dumps({id_to_label[k]: v for k, v in calibrated_thresholds.items()}, indent=2)
        )
        print(json.dumps({id_to_label[k]: v for k, v in calibrated_thresholds.items()}, indent=2))
        if args.calibrate_competition:
            print("Calibrating class competition weights on the final multiclass masks...")
            calibrated_weights, calibration_rows = calibrate_competition_weights(
                annotated_proba,
                y,
                annotated_tissue,
                id_to_label,
                calibrated_thresholds,
            )
            (args.output_dir / "calibrated_competition_weights.json").write_text(
                json.dumps({id_to_label[k]: v for k, v in calibrated_weights.items()}, indent=2)
            )
            pd.DataFrame(calibration_rows).to_csv(args.output_dir / "calibration_multiclass_metrics.csv", index=False)
            print(json.dumps({id_to_label[k]: v for k, v in calibrated_weights.items()}, indent=2))

    manifest = pd.read_csv(args.sample_manifest)
    sample_rows = []
    for _, item in manifest.iterrows():
        sample_id = slugify(str(item["sample_id"]))
        sample_out = args.output_dir / sample_id
        (sample_out / "masks").mkdir(parents=True, exist_ok=True)
        he, factor, scale = load_sample(Path(item["he_path"]), Path(item["factor_label_path"]), args.max_side)
        tissue = estimate_tissue_mask(he)
        sample_features = build_transfer_features(he, factor, tissue)
        pred_raw, confidence, proba = predict_proba_in_chunks(model, sample_features, args.chunk_size)
        if calibrated_thresholds:
            pred, confidence = apply_class_thresholds(
                proba,
                calibrated_thresholds,
                tissue,
                weights=calibrated_weights,
            )
        else:
            pred = clean_prediction(
                pred_raw,
                confidence,
                tissue,
                args.min_confidence,
                proba=proba,
                factor=factor,
                label_factor_support=label_factor_support,
                rare_rescue=args.rare_rescue,
            )

        Image.fromarray(pred.astype(np.uint8)).save(sample_out / "pseudo_multiclass_label_index.png")
        Image.fromarray(np.clip(confidence * 255, 0, 255).astype(np.uint8)).save(sample_out / "confidence_map.png")
        Image.fromarray(overlay(he, pred, id_to_label)).save(sample_out / "pseudo_overlay.png")
        for class_id, label in id_to_label.items():
            if class_id == 0:
                continue
            Image.fromarray((pred == class_id).astype(np.uint8) * 255).save(sample_out / "masks" / f"{label}_pseudo_mask.png")

        uncertainty_df = save_uncertain_tiles(
            he,
            pred,
            confidence,
            id_to_label,
            sample_out / "uncertain_review_tiles",
            tile_size=256,
            top_k=24,
        )

        metrics_rows = []
        has_annotation = bool(item.get("has_annotation", False))
        if has_annotation:
            sample_targets = load_targets(he.shape[:2])
            for class_id, label in id_to_label.items():
                if class_id == 0 or label not in sample_targets:
                    continue
                metrics_rows.append({"label": label, **mask_metrics(pred == class_id, sample_targets[label])})
            pd.DataFrame(metrics_rows).to_csv(sample_out / "metrics_if_annotation_available.csv", index=False)

        confidence_summary = {
            "sample_id": sample_id,
            "scale_from_original": scale,
            "mean_foreground_confidence": float(confidence[pred > 0].mean()) if np.any(pred > 0) else 0.0,
            "median_foreground_confidence": float(np.median(confidence[pred > 0])) if np.any(pred > 0) else 0.0,
            "foreground_pixels": int((pred > 0).sum()),
            "n_uncertain_tiles": int(len(uncertainty_df)),
            "has_annotation": has_annotation,
        }
        if metrics_rows:
            metrics_df = pd.DataFrame(metrics_rows)
            confidence_summary["min_dice_if_annotation_available"] = float(metrics_df["dice"].min())
            confidence_summary["mean_dice_if_annotation_available"] = float(metrics_df["dice"].mean())
        (sample_out / "sample_summary.json").write_text(json.dumps(confidence_summary, indent=2))
        sample_rows.append(confidence_summary)
        print(json.dumps(confidence_summary, indent=2))

    summary_df = pd.DataFrame(sample_rows)
    summary_df.to_csv(args.output_dir / "weak_transfer_summary.csv", index=False)
    lines = ["# Weak-transfer Pseudo-segmentation Results", ""]
    lines.append("This run is designed for one annotated sample plus unlabeled samples. Metrics are only meaningful for samples with annotation.")
    lines.append("")
    lines.append("| sample | fg pixels | mean confidence | median confidence | annotation | min Dice if available |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for _, row in summary_df.iterrows():
        min_dice = row.get("min_dice_if_annotation_available", np.nan)
        lines.append(
            f"| {row['sample_id']} | {int(row['foreground_pixels'])} | "
            f"{row['mean_foreground_confidence']:.3f} | {row['median_foreground_confidence']:.3f} | "
            f"{bool(row['has_annotation'])} | {min_dice:.3f} |"
        )
    (args.output_dir / "README.md").write_text("\n".join(lines) + "\n")
    print(f"Wrote weak-transfer outputs to {args.output_dir}")


if __name__ == "__main__":
    main()
