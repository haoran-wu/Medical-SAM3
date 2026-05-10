#!/usr/bin/env python3
"""Refine reference-map candidates with SAM3.

This script uses connected components from the FICTURE/reference color map as
candidate regions, maps their boxes onto a target image, then prompts SAM3 with
those boxes. The target image can be the reference map itself or the real H&E.
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
MPL_CONFIG_DIR = PROJECT_ROOT / "output" / ".mplconfig"
MPL_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CONFIG_DIR))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from skimage.color import rgb2gray, rgb2hsv
from skimage.measure import label, regionprops
from skimage.morphology import binary_closing, binary_dilation, disk, remove_small_objects
from skimage.transform import AffineTransform, warp

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from sam3_inference import SAM3Model, resize_mask
from sam3_baseline_utils import ensure_dirs, make_overlay, metrics_to_dict, normalize_prediction, save_mask


DEFAULT_CANDIDATE_ROOT = PROJECT_ROOT / "output" / "visium_hd_exp1" / "reference_map_candidates" / "hex12_k12"
DEFAULT_REFERENCE_MAP = Path(
    "/Users/haoranwu/Desktop/visiumhd_exp1_pixel_cell_type_result/reference_aligned/"
    "hex_12.k12.pixel.rot90.reference_color_mapped.png"
)
DEFAULT_HE_IMAGE = PROJECT_ROOT / "output" / "visium_hd_exp1" / "assets" / "tissue_hires_image.png"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "output" / "visium_hd_exp1" / "sam3_reference_candidate_refine"


def load_transform_json(path: Optional[Path]) -> Optional[np.ndarray]:
    if path is None:
        return None
    data = json.loads(path.read_text())
    matrix = data.get("full_matrix_ref_to_he") or data.get("matrix")
    if matrix is None:
        raise ValueError(f"No full_matrix_ref_to_he or matrix field found in {path}")
    arr = np.asarray(matrix, dtype=np.float64)
    if arr.shape != (3, 3):
        raise ValueError(f"Expected a 3x3 transform matrix in {path}, got {arr.shape}")
    return arr


def parse_list(value: str) -> List[int]:
    parsed = ast.literal_eval(value)
    return [int(v) for v in parsed]


def load_candidates(path: Path, limit: int, factors: Optional[set[int]], sort_by: str) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            factor = int(row["factor"])
            if factors is not None and factor not in factors:
                continue
            row["factor"] = factor
            row["rank_within_factor"] = int(row["rank_within_factor"])
            row["area"] = int(row["area"])
            row["bbox_xyxy"] = parse_list(row["bbox_xyxy"])
            row["rgb"] = parse_list(row["rgb"])
            row["weight"] = float(row["weight"])
            rows.append(row)
    if sort_by == "area":
        rows.sort(key=lambda item: int(item["area"]), reverse=True)
    return rows[:limit] if limit > 0 else rows


def map_bbox_stretch(
    bbox: Sequence[int],
    source_size: Tuple[int, int],
    target_size: Tuple[int, int],
    margin: float,
) -> Tuple[int, int, int, int]:
    src_w, src_h = source_size
    tgt_w, tgt_h = target_size
    x0, y0, x1, y1 = [float(v) for v in bbox]
    x0 = x0 / src_w * tgt_w
    x1 = x1 / src_w * tgt_w
    y0 = y0 / src_h * tgt_h
    y1 = y1 / src_h * tgt_h
    w = x1 - x0
    h = y1 - y0
    x0 -= w * margin
    x1 += w * margin
    y0 -= h * margin
    y1 += h * margin
    return (
        max(0, int(round(x0))),
        max(0, int(round(y0))),
        min(tgt_w - 1, int(round(x1))),
        min(tgt_h - 1, int(round(y1))),
    )


def map_bbox_affine(
    bbox: Sequence[int],
    matrix_ref_to_target_full: np.ndarray,
    target_size: Tuple[int, int],
    resize_scale: float,
    margin: float,
) -> Tuple[int, int, int, int]:
    tgt_w, tgt_h = target_size
    x0, y0, x1, y1 = [float(v) for v in bbox]
    corners = np.array(
        [
            [x0, y0, 1.0],
            [x1, y0, 1.0],
            [x1, y1, 1.0],
            [x0, y1, 1.0],
        ]
    )
    mapped = (matrix_ref_to_target_full @ corners.T).T
    mapped_xy = mapped[:, :2] / mapped[:, 2:3]
    mapped_xy *= resize_scale
    mx0, my0 = mapped_xy.min(axis=0)
    mx1, my1 = mapped_xy.max(axis=0)
    w = mx1 - mx0
    h = my1 - my0
    mx0 -= w * margin
    mx1 += w * margin
    my0 -= h * margin
    my1 += h * margin
    return (
        max(0, int(round(mx0))),
        max(0, int(round(my0))),
        min(tgt_w - 1, int(round(mx1))),
        min(tgt_h - 1, int(round(my1))),
    )


def bbox_from_mask(mask: np.ndarray, margin: float) -> Optional[Tuple[int, int, int, int]]:
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return None
    x0, x1 = float(xs.min()), float(xs.max())
    y0, y1 = float(ys.min()), float(ys.max())
    w = max(1.0, x1 - x0)
    h = max(1.0, y1 - y0)
    x0 -= w * margin
    x1 += w * margin
    y0 -= h * margin
    y1 += h * margin
    height, width = mask.shape
    return (
        max(0, int(round(x0))),
        max(0, int(round(y0))),
        min(width - 1, int(round(x1))),
        min(height - 1, int(round(y1))),
    )


def estimate_tissue_mask(image: np.ndarray) -> np.ndarray:
    hsv = rgb2hsv(image / 255.0)
    gray = rgb2gray(image / 255.0)
    tissue = (hsv[:, :, 1] > 0.045) & (gray < 0.94)
    tissue = remove_small_objects(tissue, min_size=max(64, image.size // 10000))
    tissue = binary_closing(tissue, disk(5))
    return tissue.astype(bool)


def warp_reference_mask_to_target(
    mask_ref: np.ndarray,
    matrix_ref_to_target_full: np.ndarray,
    target_shape: Tuple[int, int],
    resize_scale: float,
) -> np.ndarray:
    scale_to_resized = np.array(
        [[resize_scale, 0.0, 0.0], [0.0, resize_scale, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    matrix = scale_to_resized @ matrix_ref_to_target_full
    tform = AffineTransform(matrix=matrix)
    warped = warp(
        mask_ref.astype(np.float32),
        inverse_map=tform.inverse,
        output_shape=target_shape,
        order=0,
        preserve_range=True,
    )
    return warped > 0.5


def keep_component_near_prior(
    pred_mask: np.ndarray,
    prior_mask: Optional[np.ndarray],
    tissue_mask: Optional[np.ndarray],
    prior_dilate_radius: int,
) -> np.ndarray:
    refined = pred_mask.astype(bool)
    if tissue_mask is not None:
        refined = refined & tissue_mask
    if prior_mask is None or not prior_mask.any() or not refined.any():
        return refined.astype(np.uint8)

    prior_context = binary_dilation(prior_mask.astype(bool), disk(max(1, prior_dilate_radius)))
    labeled = label(refined)
    best_label = 0
    best_overlap = 0
    for region in regionprops(labeled):
        component = labeled == region.label
        overlap = int(np.logical_and(component, prior_context).sum())
        if overlap > best_overlap:
            best_label = int(region.label)
            best_overlap = overlap

    if best_label == 0:
        return np.logical_and(refined, prior_context).astype(np.uint8)
    component = labeled == best_label
    # Keep SAM3's morphology, but prevent large off-prior leaks.
    refined = component & prior_context
    return refined.astype(np.uint8)


def draw_panel(
    image: np.ndarray,
    factor: int,
    candidate_rank: int,
    bbox: Tuple[int, int, int, int],
    pred_mask: np.ndarray,
    target_mask: Optional[np.ndarray],
    metrics: Optional[Dict[str, float]],
    output_path: Path,
) -> None:
    n_cols = 4 if target_mask is not None else 3
    fig, axes = plt.subplots(1, n_cols, figsize=(5 * n_cols, 5))
    if n_cols == 1:
        axes = [axes]
    x0, y0, x1, y1 = bbox

    axes[0].imshow(image)
    axes[0].add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, edgecolor="cyan", linewidth=2))
    axes[0].set_title(f"Factor {factor}, rank {candidate_rank}\nMapped box prompt")
    axes[0].axis("off")

    axes[1].imshow(make_overlay(image, pred_mask, (0, 220, 255), alpha=0.50))
    axes[1].set_title(f"SAM3 refined mask\npixels={int(pred_mask.sum())}")
    axes[1].axis("off")

    if target_mask is not None:
        axes[2].imshow(make_overlay(image, target_mask, (255, 190, 0), alpha=0.50))
        axes[2].set_title("Reference component target")
        axes[2].axis("off")
        axes[3].imshow(image)
        target_view = np.ma.masked_where(target_mask == 0, target_mask)
        pred_view = np.ma.masked_where(pred_mask == 0, pred_mask)
        axes[3].imshow(target_view, alpha=0.35, cmap="autumn")
        axes[3].imshow(pred_view, alpha=0.35, cmap="winter")
        if metrics:
            axes[3].set_title(
                f"Target vs refined\nDice={metrics['dice']:.3f}, IoU={metrics['iou']:.3f}, R={metrics['recall']:.3f}"
            )
        else:
            axes[3].set_title("Target vs refined")
        axes[3].axis("off")
    else:
        axes[2].imshow(image)
        pred_view = np.ma.masked_where(pred_mask == 0, pred_mask)
        axes[2].imshow(pred_view, alpha=0.45, cmap="winter")
        axes[2].set_title("Overlay on target image")
        axes[2].axis("off")

    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Refine reference-map candidates with SAM3 box prompts.")
    parser.add_argument("--candidate-root", type=Path, default=DEFAULT_CANDIDATE_ROOT)
    parser.add_argument("--reference-map", type=Path, default=DEFAULT_REFERENCE_MAP)
    parser.add_argument("--target-image", type=Path, default=DEFAULT_HE_IMAGE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--device", choices=["cuda", "mps", "cpu"], default=None)
    parser.add_argument("--max-side", type=int, default=2048, help="Resize target image longest side. 0 disables.")
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument(
        "--candidate-sort",
        choices=["area", "csv"],
        default="area",
        help="Sort candidates by reference area, or preserve CSV row order.",
    )
    parser.add_argument("--factors", type=str, default=None, help="Optional comma-separated factor IDs.")
    parser.add_argument("--box-margin", type=float, default=0.06)
    parser.add_argument("--evaluate-reference-target", action="store_true")
    parser.add_argument(
        "--transform-json",
        type=Path,
        default=None,
        help="Optional JSON containing full_matrix_ref_to_he. If set, candidate boxes are affine-mapped instead of stretched.",
    )
    parser.add_argument("--use-affine-prior-mask", action="store_true")
    parser.add_argument("--clip-to-tissue", action="store_true")
    parser.add_argument("--prior-dilate-radius", type=int, default=48)
    args = parser.parse_args()

    csv_path = args.candidate_root / "candidate_components.csv"
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)
    if not args.target_image.exists():
        raise FileNotFoundError(args.target_image)

    factors = {int(v.strip()) for v in args.factors.split(",")} if args.factors else None
    candidates = load_candidates(csv_path, args.limit, factors, args.candidate_sort)
    if not candidates:
        raise ValueError("No candidate rows selected.")

    output_dir = args.output_dir
    masks_dir = output_dir / "masks"
    panels_dir = output_dir / "panels"
    ensure_dirs((output_dir, masks_dir, panels_dir))

    reference = Image.open(args.reference_map).convert("RGB")
    source_size = reference.size
    image = np.array(Image.open(args.target_image).convert("RGB"))
    original_target_size = (image.shape[1], image.shape[0])
    max_side = None if args.max_side == 0 else args.max_side
    resize_scale = 1.0
    if max_side is not None and max(image.shape[:2]) > max_side:
        resize_scale = max_side / float(max(image.shape[:2]))
        new_w = max(1, int(round(image.shape[1] * resize_scale)))
        new_h = max(1, int(round(image.shape[0] * resize_scale)))
        image = np.array(Image.fromarray(image).resize((new_w, new_h), resample=Image.Resampling.BILINEAR))

    target_size = (image.shape[1], image.shape[0])
    sam3 = SAM3Model(confidence_threshold=0.1, checkpoint_path=args.checkpoint, device=args.device)
    state = sam3.encode_image(image)

    report: Dict[str, object] = {
        "candidate_root": str(args.candidate_root),
        "reference_map": str(args.reference_map),
        "target_image": str(args.target_image),
        "checkpoint": args.checkpoint,
        "source_size": list(source_size),
        "original_target_size": list(original_target_size),
        "target_size": list(target_size),
        "resize_scale": resize_scale,
        "box_margin": args.box_margin,
        "candidate_sort": args.candidate_sort,
        "evaluate_reference_target": args.evaluate_reference_target,
        "transform_json": str(args.transform_json) if args.transform_json else None,
        "mapping_mode": "affine" if args.transform_json else "stretch",
        "use_affine_prior_mask": args.use_affine_prior_mask,
        "clip_to_tissue": args.clip_to_tissue,
        "prior_dilate_radius": args.prior_dilate_radius,
        "candidates": [],
    }

    affine_ref_to_target = load_transform_json(args.transform_json)
    tissue_mask = estimate_tissue_mask(image) if args.clip_to_tissue else None
    union = np.zeros(image.shape[:2], dtype=np.uint8)
    for idx, cand in enumerate(candidates):
        affine_prior_mask = None
        if affine_ref_to_target is not None and args.use_affine_prior_mask:
            mask_path = args.candidate_root / str(cand["mask_path"])
            if mask_path.exists():
                mask_ref = np.array(Image.open(mask_path).convert("L")) > 127
                affine_prior_mask = warp_reference_mask_to_target(
                    mask_ref, affine_ref_to_target, image.shape[:2], resize_scale
                )
                if tissue_mask is not None:
                    affine_prior_mask = affine_prior_mask & binary_dilation(tissue_mask, disk(3))
                mapped_from_mask = bbox_from_mask(affine_prior_mask, args.box_margin)
                if mapped_from_mask is None:
                    print(f"Skipping candidate {idx}: affine prior mask is empty on target.")
                    continue
                mapped = mapped_from_mask
            else:
                mapped = map_bbox_affine(cand["bbox_xyxy"], affine_ref_to_target, target_size, resize_scale, args.box_margin)
        elif affine_ref_to_target is not None:
            mapped = map_bbox_affine(cand["bbox_xyxy"], affine_ref_to_target, target_size, resize_scale, args.box_margin)
        else:
            mapped = map_bbox_stretch(cand["bbox_xyxy"], source_size, target_size, args.box_margin)
        if mapped[2] <= mapped[0] or mapped[3] <= mapped[1]:
            print(f"Skipping candidate {idx}: invalid mapped box {mapped}.")
            continue
        pred = normalize_prediction(sam3.predict_box(state, mapped, image.shape[:2]), image.shape[:2])
        pred = keep_component_near_prior(pred, affine_prior_mask, tissue_mask, args.prior_dilate_radius)
        union = np.logical_or(union, pred).astype(np.uint8)

        target_mask = None
        metrics = None
        if affine_prior_mask is not None:
            target_mask = affine_prior_mask.astype(np.uint8)
            metrics = metrics_to_dict(pred, target_mask)
        elif args.evaluate_reference_target:
            mask_path = args.candidate_root / str(cand["mask_path"])
            if mask_path.exists():
                target_mask_ref = np.array(Image.open(mask_path).convert("L")) > 127
                target_mask = resize_mask(target_mask_ref.astype(np.uint8), image.shape[:2]).astype(np.uint8)
                metrics = metrics_to_dict(pred, target_mask)

        stem = f"{idx:03d}_factor_{int(cand['factor']):02d}_rank_{int(cand['rank_within_factor']):02d}"
        save_mask(pred, masks_dir / f"{stem}_sam3_refined.png")
        draw_panel(
            image=image,
            factor=int(cand["factor"]),
            candidate_rank=int(cand["rank_within_factor"]),
            bbox=mapped,
            pred_mask=pred,
            target_mask=target_mask,
            metrics=metrics,
            output_path=panels_dir / f"{stem}.png",
        )
        item = {
            "candidate_index": idx,
            "factor": int(cand["factor"]),
            "rank_within_factor": int(cand["rank_within_factor"]),
            "candidate_area_reference": int(cand["area"]),
            "reference_bbox_xyxy": cand["bbox_xyxy"],
            "mapped_bbox_xyxy": list(mapped),
            "prediction_pixels": int(pred.sum()),
            "mask_path": str((masks_dir / f"{stem}_sam3_refined.png").relative_to(output_dir)),
            "panel_path": str((panels_dir / f"{stem}.png").relative_to(output_dir)),
            "top_genes_specific": cand.get("top_genes_specific"),
        }
        if metrics:
            item["metrics_vs_resized_reference_component"] = metrics
        report["candidates"].append(item)
        print(
            f"{stem}: box={mapped}, pred_pixels={int(pred.sum())}"
            + (f", Dice={metrics['dice']:.3f}" if metrics else "")
        )

    save_mask(union, output_dir / "sam3_refined_union.png")
    Image.fromarray(make_overlay(image, union, (0, 220, 255), alpha=0.45)).save(output_dir / "sam3_refined_union_overlay.png")
    (output_dir / "refine_report.json").write_text(json.dumps(report, indent=2))
    print(f"Saved outputs to {output_dir}")


if __name__ == "__main__":
    main()
