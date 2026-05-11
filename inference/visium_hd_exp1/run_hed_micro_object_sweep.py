#!/usr/bin/env python3
"""HED/color-deconvolution micro-object sweeps for weak classes.

SAM and molecular priors are currently weak for erythrocytes and pigment. This
script follows the pathology image-processing playbook instead: use H&E stain
separation, HSV/RGB color contrasts, connected-component constraints, and
small morphology sweeps to build class-specific micro masks.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
from PIL import Image
from skimage import color, filters, measure, morphology


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_HE = PROJECT_ROOT / "output/visium_hd_exp1/assets/tissue_hires_image.png"
DEFAULT_SUMMARY = PROJECT_ROOT / "output/visium_hd_exp1/sam3_local_region_summary/region_summary.json"

LABELS = ["erythorocytes", "pigment", "immune_infiltration"]
COLORS = {
    "erythorocytes": (255, 127, 14),
    "pigment": (127, 127, 127),
    "immune_infiltration": (44, 160, 44),
}


def max_side_shape(shape_hw: Tuple[int, int], max_side: int) -> Tuple[int, int]:
    h, w = shape_hw
    scale = min(1.0, float(max_side) / max(h, w))
    return max(1, int(round(h * scale))), max(1, int(round(w * scale)))


def resize_image(arr: np.ndarray, shape_hw: Tuple[int, int], resample: Image.Resampling) -> np.ndarray:
    h, w = shape_hw
    if arr.shape[:2] == (h, w):
        return arr
    return np.array(Image.fromarray(arr).resize((w, h), resample))


def read_mask(path: Path, shape_hw: Tuple[int, int]) -> np.ndarray:
    img = Image.open(path).convert("L")
    return np.array(img.resize((shape_hw[1], shape_hw[0]), Image.Resampling.NEAREST)) > 127


def resolve_path(path_text: str, base_dir: Path) -> Path:
    path = Path(path_text)
    if path.exists():
        return path
    candidate = base_dir / path_text
    if candidate.exists():
        return candidate
    return Path.cwd() / path_text


def load_targets(summary_path: Path, shape_hw: Tuple[int, int]) -> Dict[str, np.ndarray]:
    summary = json.loads(summary_path.read_text())
    targets = {}
    for item in summary["labels"]:
        slug = item.get("slug") or item["label"]
        if slug in LABELS:
            targets[slug] = read_mask(resolve_path(item["mask_path"], summary_path.parent), shape_hw)
    return targets


def metrics(pred: np.ndarray, gt: np.ndarray) -> Dict[str, float]:
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


def tissue_mask(he: np.ndarray) -> np.ndarray:
    rgb = he.astype(np.float32) / 255.0
    hsv = color.rgb2hsv(rgb)
    gray = color.rgb2gray(rgb)
    mask = (hsv[..., 1] > 0.05) & (gray < 0.96)
    mask = morphology.remove_small_objects(mask, 256)
    return morphology.binary_closing(mask, morphology.disk(3))


def robust_z(x: np.ndarray, mask: np.ndarray) -> np.ndarray:
    vals = x[mask]
    med = float(np.median(vals)) if vals.size else 0.0
    mad = float(np.median(np.abs(vals - med))) if vals.size else 1.0
    scale = max(1e-6, 1.4826 * mad)
    return np.clip((x - med) / scale, -8, 8)


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


def feature_maps(he: np.ndarray, tissue: np.ndarray) -> Dict[str, np.ndarray]:
    rgb = he.astype(np.float32) / 255.0
    hsv = color.rgb2hsv(rgb)
    hed = color.rgb2hed(rgb)
    gray = color.rgb2gray(rgb)
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    maps = {
        "hematoxylin": hed[..., 0],
        "eosin": hed[..., 1],
        "dab_residual": hed[..., 2],
        "sat": hsv[..., 1],
        "val": hsv[..., 2],
        "dark": 1.0 - hsv[..., 2],
        "red_excess": r - 0.5 * (g + b),
        "brown_excess": 0.5 * (r + g) - b,
        "blue_excess": b - 0.5 * (r + g),
        "sobel": filters.sobel(gray),
    }
    return {name: robust_z(arr.astype(np.float32), tissue) for name, arr in maps.items()}


def score_for_label(label: str, maps: Dict[str, np.ndarray], mode: str) -> np.ndarray:
    if label == "erythorocytes":
        if mode == "strict":
            return 1.2 * maps["eosin"] + 1.0 * maps["red_excess"] + 0.4 * maps["sat"] - 0.4 * maps["hematoxylin"]
        if mode == "dark":
            return 0.9 * maps["eosin"] + 0.8 * maps["red_excess"] + 0.6 * maps["dark"]
        return maps["eosin"] + maps["red_excess"] + 0.5 * maps["sat"]
    if label == "pigment":
        if mode == "brown":
            return 1.1 * maps["dark"] + 0.9 * maps["brown_excess"] + 0.4 * maps["sat"]
        if mode == "black":
            return 1.4 * maps["dark"] + 0.4 * maps["hematoxylin"] + 0.2 * maps["sobel"]
        return maps["dark"] + 0.5 * maps["brown_excess"] + 0.4 * maps["sat"]
    if mode == "nuclear":
        return 1.2 * maps["hematoxylin"] + 0.5 * maps["sobel"] - 0.2 * maps["eosin"]
    if mode == "cluster":
        return maps["hematoxylin"] + 0.6 * maps["blue_excess"] + 0.4 * maps["sobel"]
    return maps["hematoxylin"] + 0.4 * maps["sobel"]


def overlay(he: np.ndarray, masks: Dict[str, np.ndarray]) -> np.ndarray:
    out = he.astype(np.float32).copy()
    for label, mask in masks.items():
        color_rgb = np.array(COLORS[label], dtype=np.float32)
        out[mask] = 0.5 * out[mask] + 0.5 * color_rgb
    return np.clip(out, 0, 255).astype(np.uint8)


def run(args: argparse.Namespace) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    he_full = np.array(Image.open(args.he_image).convert("RGB"))
    shape = max_side_shape(he_full.shape[:2], args.max_side)
    he = resize_image(he_full, shape, Image.Resampling.BILINEAR)
    tissue = tissue_mask(he)
    targets = load_targets(args.summary_path, shape)
    maps = feature_maps(he, tissue)

    percentiles = [float(x) for x in args.percentiles.split(",")]
    rows: List[Dict[str, object]] = []
    best_masks: Dict[str, np.ndarray] = {}
    mask_dir = args.output_dir / "masks"
    mask_dir.mkdir(parents=True, exist_ok=True)
    for label in args.labels.split(","):
        label = label.strip()
        if label not in targets:
            continue
        best_row = None
        best_mask = None
        for mode in args.modes.split(","):
            score = score_for_label(label, maps, mode.strip())
            vals = score[tissue]
            for pct in percentiles:
                threshold = float(np.percentile(vals, pct))
                raw = (score >= threshold) & tissue
                for close_radius in args.close_radii:
                    closed = morphology.binary_closing(raw, morphology.disk(close_radius)) if close_radius > 0 else raw
                    for dilate_radius in args.dilate_radii:
                        pred = morphology.binary_dilation(closed, morphology.disk(dilate_radius)) if dilate_radius > 0 else closed
                        pred = component_filter(pred, args.min_area, args.max_area, args.eccentricity_max)
                        row = {
                            "label": label,
                            "mode": mode.strip(),
                            "percentile": pct,
                            "threshold": threshold,
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
            Image.fromarray(best_mask.astype(np.uint8) * 255).save(mask_dir / f"{label}_hed_micro_best.png")
    df = pd.DataFrame(rows)
    df.to_csv(args.output_dir / "hed_micro_sweep_metrics.csv", index=False)
    best = df.sort_values("dice", ascending=False).groupby("label", as_index=False).head(1)
    best.to_csv(args.output_dir / "hed_micro_best_by_label.csv", index=False)
    Image.fromarray(overlay(he, best_masks)).save(args.output_dir / "hed_micro_overlay.png")

    lines = ["# HED Micro-object Sweep", "", "| label | Dice | precision | recall | mode | pct | close | dilate |", "|---|---:|---:|---:|---|---:|---:|---:|"]
    for _, row in best.sort_values("dice", ascending=False).iterrows():
        lines.append(
            f"| {row['label']} | {row['dice']:.3f} | {row['precision']:.3f} | {row['recall']:.3f} | "
            f"{row['mode']} | {row['percentile']:.1f} | {int(row['close_radius'])} | {int(row['dilate_radius'])} |"
        )
    (args.output_dir / "README.md").write_text("\n".join(lines) + "\n")
    print(best.sort_values("dice", ascending=False).to_string(index=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="HED/color micro-object segmentation sweep.")
    parser.add_argument("--he-image", type=Path, default=DEFAULT_HE)
    parser.add_argument("--summary-path", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-side", type=int, default=1536)
    parser.add_argument("--labels", default="erythorocytes,pigment,immune_infiltration")
    parser.add_argument("--modes", default="default,strict,dark,brown,black,nuclear,cluster")
    parser.add_argument("--percentiles", default="95,96,97,98,98.5,99,99.3,99.5,99.7,99.85")
    parser.add_argument("--min-area", type=int, default=4)
    parser.add_argument("--max-area", type=int, default=4096)
    parser.add_argument("--eccentricity-max", type=float, default=0.98)
    parser.add_argument("--close-radii", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--dilate-radii", type=int, nargs="+", default=[0, 1, 2, 3])
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
