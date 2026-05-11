#!/usr/bin/env python3
"""Superpixel H&E + gene/FICTURE segmentation for VisiumHD Exp1.

This is an intentionally different line from dense SAM prompting. It segments
the H&E into superpixels, scores each superpixel with:

- gene-derived FICTURE factor label scores,
- label prototypes from H&E color/texture,
- label prototypes from factor histograms,
- optional spatial centroid priors,

then assigns non-overlapping labels by competition. SAM can later refine these
superpixel regions, but this script tests whether molecular + H&E evidence can
choose semantic tissue regions more cleanly than raw SAM candidates.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from scipy import ndimage as ndi
from skimage import color, filters, morphology, segmentation


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_HE = PROJECT_ROOT / "output" / "visium_hd_exp1" / "assets" / "tissue_hires_image.png"
DEFAULT_FACTOR = PROJECT_ROOT / "output" / "visium_hd_exp1" / "molecular_prior_experiments" / "dx-60_dy80" / "ficture_factor_label_image_he.npy"
DEFAULT_ANNOT = PROJECT_ROOT / "output" / "visium_hd_exp1" / "molecular_prior_experiments" / "dx-60_dy80" / "factor_annotation.csv"
DEFAULT_SUMMARY = PROJECT_ROOT / "output" / "visium_hd_exp1" / "sam3_local_region_summary" / "region_summary.json"

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


def slugify(label: str) -> str:
    out = "".join(ch.lower() if ch.isalnum() else "_" for ch in label).strip("_")
    while "__" in out:
        out = out.replace("__", "_")
    return out


def resize_image(image: np.ndarray, shape_hw: Tuple[int, int], resample: int) -> np.ndarray:
    h, w = shape_hw
    return np.array(Image.fromarray(image).resize((w, h), resample=resample))


def max_side_shape(shape_hw: Tuple[int, int], max_side: int) -> Tuple[int, int]:
    h, w = shape_hw
    if max_side <= 0 or max(h, w) <= max_side:
        return h, w
    scale = max_side / float(max(h, w))
    return max(1, int(round(h * scale))), max(1, int(round(w * scale)))


def load_mask(path: Path, shape_hw: Tuple[int, int]) -> np.ndarray:
    mask = np.array(Image.open(path).convert("L")) > 127
    if mask.shape != shape_hw:
        mask = resize_image((mask.astype(np.uint8) * 255), shape_hw, Image.Resampling.NEAREST) > 127
    return mask


def resolve_path(path_text: str, base_dir: Path) -> Path:
    path = Path(path_text)
    if path.exists():
        return path
    candidate = base_dir / path_text
    if candidate.exists():
        return candidate
    return path


def load_targets(summary_path: Path, shape_hw: Tuple[int, int]) -> Dict[str, np.ndarray]:
    summary = json.loads(summary_path.read_text())
    targets = {}
    for item in summary["labels"]:
        label = item["label"]
        if label not in LABEL_ORDER:
            continue
        targets[label] = load_mask(resolve_path(item["mask_path"], summary_path.parent), shape_hw)
    return {label: targets[label] for label in LABEL_ORDER if label in targets}


def load_gene_factor_scores(path: Path, n_factors: int) -> Dict[str, np.ndarray]:
    scores: Dict[str, np.ndarray] = {label: np.zeros(n_factors, dtype=np.float64) for label in LABEL_ORDER}
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
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


def tissue_mask_from_he(he: np.ndarray) -> np.ndarray:
    hsv = color.rgb2hsv(he.astype(np.float32) / 255.0)
    gray = color.rgb2gray(he.astype(np.float32) / 255.0)
    tissue = (hsv[..., 1] > 0.045) & (gray < 0.94)
    tissue = morphology.remove_small_objects(tissue, min_size=max(256, tissue.size // 20000))
    tissue = morphology.binary_closing(tissue, morphology.disk(5))
    return tissue


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


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    an = float(np.linalg.norm(a))
    bn = float(np.linalg.norm(b))
    if an <= 1e-12 or bn <= 1e-12:
        return 0.0
    return float(np.dot(a, b) / (an * bn))


def factor_hist(factors: np.ndarray, mask: np.ndarray, n_factors: int) -> np.ndarray:
    vals = factors[mask & (factors >= 0)]
    hist = np.bincount(vals.astype(np.int64), minlength=n_factors).astype(np.float64) + 1e-3
    return hist / hist.sum()


def label_prototypes(
    lab: np.ndarray,
    gray: np.ndarray,
    factors: np.ndarray,
    targets: Dict[str, np.ndarray],
    n_factors: int,
) -> Dict[str, Dict[str, object]]:
    protos: Dict[str, Dict[str, object]] = {}
    yy, xx = np.indices(gray.shape, dtype=np.float32)
    for label, mask in targets.items():
        if not mask.any():
            continue
        pix = lab[mask]
        gy, gx = yy[mask], xx[mask]
        protos[label] = {
            "lab_mean": pix.mean(axis=0),
            "lab_std": np.maximum(pix.std(axis=0), 1.0),
            "gray_mean": float(gray[mask].mean()),
            "gray_std": max(float(gray[mask].std()), 0.02),
            "factor_hist": factor_hist(factors, mask, n_factors),
            "centroid": np.array([gx.mean() / max(1, gray.shape[1] - 1), gy.mean() / max(1, gray.shape[0] - 1)]),
            "area_frac": float(mask.mean()),
        }
    return protos


def superpixel_table(
    he: np.ndarray,
    lab: np.ndarray,
    gray: np.ndarray,
    sobel: np.ndarray,
    factors: np.ndarray,
    segments: np.ndarray,
    tissue: np.ndarray,
    n_factors: int,
) -> pd.DataFrame:
    rows = []
    yy, xx = np.indices(gray.shape, dtype=np.float32)
    for seg_id in np.unique(segments[tissue]):
        mask = (segments == seg_id) & tissue
        area = int(mask.sum())
        if area <= 0:
            continue
        hist = factor_hist(factors, mask, n_factors)
        rows.append(
            {
                "segment": int(seg_id),
                "area": area,
                "area_frac": float(area / tissue.size),
                "x": float(xx[mask].mean() / max(1, gray.shape[1] - 1)),
                "y": float(yy[mask].mean() / max(1, gray.shape[0] - 1)),
                "lab_l": float(lab[..., 0][mask].mean()),
                "lab_a": float(lab[..., 1][mask].mean()),
                "lab_b": float(lab[..., 2][mask].mean()),
                "gray": float(gray[mask].mean()),
                "sobel": float(sobel[mask].mean()),
                **{f"factor_{i}": float(hist[i]) for i in range(n_factors)},
            }
        )
    return pd.DataFrame(rows)


def score_segments(
    table: pd.DataFrame,
    protos: Dict[str, Dict[str, object]],
    gene_scores: Dict[str, np.ndarray],
    *,
    w_gene: float,
    w_factor: float,
    w_he: float,
    w_spatial: float,
    min_score: float,
) -> Dict[int, str]:
    assignments: Dict[int, str] = {}
    factor_cols = [col for col in table.columns if col.startswith("factor_")]
    for _, row in table.iterrows():
        seg_id = int(row["segment"])
        seg_lab = np.array([row["lab_l"], row["lab_a"], row["lab_b"]], dtype=np.float64)
        seg_hist = row[factor_cols].to_numpy(dtype=np.float64)
        seg_xy = np.array([row["x"], row["y"]], dtype=np.float64)
        best_label = ""
        best_score = min_score
        for label, proto in protos.items():
            gene_score = float(np.dot(seg_hist, gene_scores.get(label, np.zeros_like(seg_hist))))
            factor_score = cosine(seg_hist, proto["factor_hist"])
            color_z = float(np.mean(np.abs((seg_lab - proto["lab_mean"]) / proto["lab_std"])))
            he_score = math.exp(-0.5 * color_z)
            dist = float(np.linalg.norm(seg_xy - proto["centroid"]))
            spatial_score = math.exp(-3.5 * dist)
            score = w_gene * gene_score + w_factor * factor_score + w_he * he_score + w_spatial * spatial_score
            if score > best_score:
                best_score = score
                best_label = label
        if best_label:
            assignments[seg_id] = best_label
    return assignments


def assignments_to_masks(
    segments: np.ndarray,
    assignments: Dict[int, str],
    shape_hw: Tuple[int, int],
    min_area: int,
    close_radius: int,
) -> Dict[str, np.ndarray]:
    masks = {label: np.zeros(shape_hw, dtype=bool) for label in LABEL_ORDER}
    for seg_id, label in assignments.items():
        masks[label] |= segments == seg_id
    for label, mask in list(masks.items()):
        out = mask
        if close_radius > 0:
            out = morphology.binary_closing(out, morphology.disk(close_radius))
        if min_area > 0:
            out = morphology.remove_small_objects(out, min_size=min_area)
        masks[label] = out.astype(bool)
    return masks


def make_overlay(he: np.ndarray, masks: Dict[str, np.ndarray]) -> np.ndarray:
    out = he.astype(np.float32).copy()
    for label in LABEL_ORDER:
        mask = masks.get(label)
        if mask is None or not mask.any():
            continue
        color_rgb = np.array(LABEL_COLORS[label], dtype=np.float32)
        out[mask] = 0.55 * out[mask] + 0.45 * color_rgb
    return np.clip(out, 0, 255).astype(np.uint8)


def run_one(args: argparse.Namespace) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    he_full = np.array(Image.open(args.he_image).convert("RGB"))
    shape = max_side_shape(he_full.shape[:2], args.max_side)
    he = resize_image(he_full, shape, Image.Resampling.BILINEAR)
    factor_full = np.load(args.factor_image)
    factors = resize_image(factor_full.astype(np.int16), shape, Image.Resampling.NEAREST).astype(np.int16)
    tissue = tissue_mask_from_he(he)
    targets = load_targets(args.summary_path, shape)

    he_float = he.astype(np.float32) / 255.0
    lab = color.rgb2lab(he_float).astype(np.float32)
    gray = color.rgb2gray(he_float).astype(np.float32)
    sobel = filters.sobel(gray).astype(np.float32)
    n_factors = int(factors[factors >= 0].max()) + 1 if np.any(factors >= 0) else 1
    gene_scores = load_gene_factor_scores(args.factor_annotation, n_factors)
    protos = label_prototypes(lab, gray, factors, targets, n_factors)

    segments = segmentation.slic(
        he,
        n_segments=args.n_segments,
        compactness=args.compactness,
        sigma=args.sigma,
        start_label=1,
        mask=tissue,
        convert2lab=True,
    )
    table = superpixel_table(he, lab, gray, sobel, factors, segments, tissue, n_factors)
    table.to_csv(args.output_dir / "superpixel_features.csv", index=False)
    assignments = score_segments(
        table,
        protos,
        gene_scores,
        w_gene=args.w_gene,
        w_factor=args.w_factor,
        w_he=args.w_he,
        w_spatial=args.w_spatial,
        min_score=args.min_score,
    )
    masks = assignments_to_masks(segments, assignments, shape, args.min_area, args.close_radius)
    mask_dir = args.output_dir / "masks"
    mask_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for label in LABEL_ORDER:
        if label not in targets:
            continue
        pred = masks.get(label, np.zeros(shape, dtype=bool))
        Image.fromarray(pred.astype(np.uint8) * 255).save(mask_dir / f"{label}_superpixel_mask.png")
        rows.append({"label": label, **mask_metrics(pred, targets[label])})
    metrics = pd.DataFrame(rows)
    metrics.to_csv(args.output_dir / "superpixel_metrics.csv", index=False)
    Image.fromarray(make_overlay(he, masks)).save(args.output_dir / "superpixel_overlay.png")
    boundaries = segmentation.mark_boundaries(he, segments, color=(0, 1, 1), mode="thick")[:, :, :3]
    Image.fromarray(np.clip(boundaries * 255, 0, 255).astype(np.uint8)).save(args.output_dir / "superpixel_boundaries.png")

    lines = ["# Superpixel Gene + H&E Segmentation", ""]
    lines.append(
        f"Parameters: n_segments={args.n_segments}, compactness={args.compactness}, sigma={args.sigma}, "
        f"weights=gene:{args.w_gene}/factor:{args.w_factor}/he:{args.w_he}/spatial:{args.w_spatial}, min_score={args.min_score}"
    )
    lines += ["", "| label | Dice | IoU | Recall | Precision |", "|---|---:|---:|---:|---:|"]
    for _, row in metrics.sort_values("dice", ascending=False).iterrows():
        lines.append(f"| {row['label']} | {row['dice']:.3f} | {row['iou']:.3f} | {row['recall']:.3f} | {row['precision']:.3f} |")
    (args.output_dir / "README.md").write_text("\n".join(lines) + "\n")
    print(metrics.sort_values("dice", ascending=False).to_string(index=False))
    print(f"Output: {args.output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Superpixel H&E + gene/FICTURE segmentation.")
    parser.add_argument("--he-image", type=Path, default=DEFAULT_HE)
    parser.add_argument("--factor-image", type=Path, default=DEFAULT_FACTOR)
    parser.add_argument("--factor-annotation", type=Path, default=DEFAULT_ANNOT)
    parser.add_argument("--summary-path", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-side", type=int, default=1536)
    parser.add_argument("--n-segments", type=int, default=9000)
    parser.add_argument("--compactness", type=float, default=8.0)
    parser.add_argument("--sigma", type=float, default=1.0)
    parser.add_argument("--w-gene", type=float, default=0.35)
    parser.add_argument("--w-factor", type=float, default=0.25)
    parser.add_argument("--w-he", type=float, default=0.30)
    parser.add_argument("--w-spatial", type=float, default=0.10)
    parser.add_argument("--min-score", type=float, default=0.35)
    parser.add_argument("--min-area", type=int, default=128)
    parser.add_argument("--close-radius", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    run_one(parse_args())


if __name__ == "__main__":
    main()
