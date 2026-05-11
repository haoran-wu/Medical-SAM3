#!/usr/bin/env python3
"""Edge-aware watershed from molecular priors.

Use projected FICTURE/gene prior scores as markers and H&E image gradients as
barriers. This tests a graph/region-growing alternative to both SAM proposals
and independent superpixel classification.
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
from skimage import color, filters, morphology, segmentation


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_HE = PROJECT_ROOT / "output/visium_hd_exp1/assets/tissue_hires_image.png"
DEFAULT_FACTOR = PROJECT_ROOT / "output/visium_hd_exp1/molecular_prior_experiments/dx-60_dy80/ficture_factor_label_image_he.npy"
DEFAULT_ANNOT = PROJECT_ROOT / "output/visium_hd_exp1/molecular_prior_experiments/dx-60_dy80/factor_annotation.csv"
DEFAULT_SUMMARY = PROJECT_ROOT / "output/visium_hd_exp1/sam3_local_region_summary/region_summary.json"

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


def resize_image(arr: np.ndarray, shape_hw: Tuple[int, int], resample: Image.Resampling) -> np.ndarray:
    h, w = shape_hw
    if arr.shape[:2] == (h, w):
        return arr
    return np.array(Image.fromarray(arr).resize((w, h), resample))


def read_mask(path: Path, shape_hw: Tuple[int, int]) -> np.ndarray:
    return np.array(Image.open(path).convert("L").resize((shape_hw[1], shape_hw[0]), Image.Resampling.NEAREST)) > 127


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
    out = {}
    for item in summary["labels"]:
        label = item.get("slug") or slugify(item["label"])
        if label in LABEL_ORDER:
            out[label] = read_mask(resolve_path(item["mask_path"], summary_path.parent), shape_hw)
    return out


def load_gene_factor_scores(path: Path, n_factors: int) -> Dict[str, np.ndarray]:
    scores: Dict[str, np.ndarray] = {}
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            factor = int(row["factor"])
            if factor < 0 or factor >= n_factors:
                continue
            label_scores = json.loads(row.get("label_scores_json") or "{}")
            for label, score in label_scores.items():
                slug = slugify(label)
                scores.setdefault(slug, np.zeros(n_factors, dtype=np.float64))
                scores[slug][factor] = max(scores[slug][factor], float(score))
    for slug, vec in list(scores.items()):
        if vec.max() > 0:
            scores[slug] = vec / vec.max()
    return scores


def tissue_mask(he: np.ndarray) -> np.ndarray:
    rgb = he.astype(np.float32) / 255.0
    hsv = color.rgb2hsv(rgb)
    gray = color.rgb2gray(rgb)
    mask = (hsv[..., 1] > 0.04) & (gray < 0.97)
    mask = morphology.remove_small_objects(mask, 512)
    return morphology.binary_closing(mask, morphology.disk(3))


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


def make_priors(factors: np.ndarray, gene_scores: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    priors = {}
    valid = factors >= 0
    for label in LABEL_ORDER:
        if label not in gene_scores:
            continue
        score = np.zeros(factors.shape, dtype=np.float32)
        score[valid] = gene_scores[label][factors[valid]]
        if score.max() > 0:
            score = score / score.max()
        priors[label] = score
    return priors


def make_markers(
    priors: Dict[str, np.ndarray],
    tissue: np.ndarray,
    seed_quantile: float,
    margin: float,
    min_seed_area: int,
) -> np.ndarray:
    stack = np.stack([priors[label] for label in LABEL_ORDER], axis=0)
    best_idx = np.argmax(stack, axis=0)
    best = np.max(stack, axis=0)
    second = np.partition(stack, -2, axis=0)[-2]
    markers = np.zeros(tissue.shape, dtype=np.int32)
    for idx, label in enumerate(LABEL_ORDER, start=1):
        vals = priors[label][tissue]
        thr = float(np.percentile(vals, seed_quantile)) if vals.size else 1.0
        seed = tissue & (best_idx == idx - 1) & (best >= thr) & ((best - second) >= margin)
        seed = morphology.remove_small_objects(seed, min_seed_area)
        markers[seed] = idx
    return markers


def overlay(he: np.ndarray, masks: Dict[str, np.ndarray]) -> np.ndarray:
    out = he.astype(np.float32).copy()
    for label, mask in masks.items():
        color_rgb = np.array(COLORS[label], dtype=np.float32)
        out[mask] = 0.55 * out[mask] + 0.45 * color_rgb
    return np.clip(out, 0, 255).astype(np.uint8)


def run(args: argparse.Namespace) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    he_full = np.array(Image.open(args.he_image).convert("RGB"))
    shape = max_side_shape(he_full.shape[:2], args.max_side)
    he = resize_image(he_full, shape, Image.Resampling.BILINEAR)
    factor_full = np.load(args.factor_image)
    factors = resize_image(factor_full.astype(np.int16), shape, Image.Resampling.NEAREST).astype(np.int16)
    tissue = tissue_mask(he)
    targets = load_targets(args.summary_path, shape)
    n_factors = int(factors[factors >= 0].max()) + 1 if np.any(factors >= 0) else 1
    gene_scores = load_gene_factor_scores(args.factor_annotation, n_factors)
    priors = make_priors(factors, gene_scores)

    rgb = he.astype(np.float32) / 255.0
    gray = color.rgb2gray(rgb)
    lab = color.rgb2lab(rgb)
    gradient = filters.sobel(gray)
    if args.lab_edge_weight > 0:
        gradient = gradient + args.lab_edge_weight * (filters.sobel(lab[..., 1]) + filters.sobel(lab[..., 2]))
    gradient = filters.gaussian(gradient, sigma=args.gradient_sigma)
    if gradient.max() > 0:
        gradient = gradient / gradient.max()

    markers = make_markers(priors, tissue, args.seed_quantile, args.margin, args.min_seed_area)
    labels_img = segmentation.watershed(gradient, markers=markers, mask=tissue, compactness=args.compactness)
    masks = {}
    rows = []
    mask_dir = args.output_dir / "masks"
    mask_dir.mkdir(parents=True, exist_ok=True)
    for idx, label in enumerate(LABEL_ORDER, start=1):
        pred = labels_img == idx
        if args.close_radius > 0:
            pred = morphology.binary_closing(pred, morphology.disk(args.close_radius))
        if args.min_area > 0:
            pred = morphology.remove_small_objects(pred, args.min_area)
        masks[label] = pred
        Image.fromarray(pred.astype(np.uint8) * 255).save(mask_dir / f"{label}_watershed_mask.png")
        rows.append({"label": label, **metrics(pred, targets[label])})

    df = pd.DataFrame(rows)
    df.to_csv(args.output_dir / "watershed_metrics.csv", index=False)
    Image.fromarray(overlay(he, masks)).save(args.output_dir / "watershed_overlay.png")
    lines = ["# Gene-prior Edge Watershed", ""]
    lines.append(
        f"max_side={args.max_side}, seed_quantile={args.seed_quantile}, margin={args.margin}, "
        f"gradient_sigma={args.gradient_sigma}, compactness={args.compactness}"
    )
    lines += ["", "| label | Dice | precision | recall |", "|---|---:|---:|---:|"]
    for _, row in df.sort_values("dice", ascending=False).iterrows():
        lines.append(f"| {row['label']} | {row['dice']:.3f} | {row['precision']:.3f} | {row['recall']:.3f} |")
    (args.output_dir / "README.md").write_text("\n".join(lines) + "\n")
    print(df.sort_values("dice", ascending=False).to_string(index=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Gene prior marker watershed on H&E edges.")
    parser.add_argument("--he-image", type=Path, default=DEFAULT_HE)
    parser.add_argument("--factor-image", type=Path, default=DEFAULT_FACTOR)
    parser.add_argument("--factor-annotation", type=Path, default=DEFAULT_ANNOT)
    parser.add_argument("--summary-path", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-side", type=int, default=1536)
    parser.add_argument("--seed-quantile", type=float, default=85.0)
    parser.add_argument("--margin", type=float, default=0.05)
    parser.add_argument("--min-seed-area", type=int, default=64)
    parser.add_argument("--min-area", type=int, default=128)
    parser.add_argument("--close-radius", type=int, default=1)
    parser.add_argument("--gradient-sigma", type=float, default=1.0)
    parser.add_argument("--lab-edge-weight", type=float, default=0.25)
    parser.add_argument("--compactness", type=float, default=0.0)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
