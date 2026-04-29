#!/usr/bin/env python3
"""
Run tile-based multi-point SAM3 inference on Visium HD Exp1 region masks.

For each selected region label:
- split the hires image into overlapping tiles
- on each tile that contains target pixels, sample positive and negative points
- run SAM3 / Medical-SAM3 on the tile
- stitch tile predictions back into a full-image mask

This is designed for large irregular tissue regions where one full-region bbox is
too coarse and aggressive whole-image downsampling would lose detail.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
MPL_CONFIG_DIR = PROJECT_ROOT / "output" / ".mplconfig"
MPL_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CONFIG_DIR))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from sam3_inference import SAM3Model
from sam3_baseline_utils import (
    DATA_DIR,
    ensure_dirs,
    load_binary_mask,
    make_overlay,
    metrics_to_dict,
    normalize_prediction,
    parse_label_filter,
    resolve_mask_path,
    save_mask,
    select_label_records,
    slugify,
)


DEFAULT_IMAGE_PATH = DATA_DIR / "spatial" / "tissue_hires_image.png"
DEFAULT_SUMMARY_PATH = PROJECT_ROOT / "output" / "visium_hd_exp1" / "region_masks" / "region_summary.json"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "output" / "visium_hd_exp1" / "sam3_runs" / "multipoint"
DEFAULT_LABELS = "tumor,stroma,immune infiltration"


def resolve_summary_path(summary_path: Optional[Path], masks_dir: Optional[Path]) -> Path:
    if summary_path is not None:
        return summary_path
    if masks_dir is not None:
        return masks_dir / "region_summary.json"
    return DEFAULT_SUMMARY_PATH


def make_tile_starts(length: int, tile_size: int, overlap: int) -> List[int]:
    if tile_size <= 0:
        raise ValueError("tile_size must be positive")
    if overlap < 0 or overlap >= tile_size:
        raise ValueError("tile_overlap must satisfy 0 <= overlap < tile_size")

    if length <= tile_size:
        return [0]

    step = tile_size - overlap
    starts = list(range(0, max(1, length - tile_size + 1), step))
    last = length - tile_size
    if starts[-1] != last:
        starts.append(last)
    return starts


def make_tiles(image_shape: Tuple[int, int], tile_size: int, overlap: int) -> List[Tuple[int, int, int, int]]:
    height, width = image_shape
    y_starts = make_tile_starts(height, tile_size, overlap)
    x_starts = make_tile_starts(width, tile_size, overlap)
    tiles: List[Tuple[int, int, int, int]] = []
    for y0 in y_starts:
        for x0 in x_starts:
            y1 = min(height, y0 + tile_size)
            x1 = min(width, x0 + tile_size)
            tiles.append((x0, y0, x1, y1))
    return tiles


def mask_bbox(mask: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def expand_box(
    box: Tuple[int, int, int, int],
    shape: Tuple[int, int],
    margin_fraction: float,
) -> Tuple[int, int, int, int]:
    x0, y0, x1, y1 = box
    height, width = shape
    dx = int(round((x1 - x0 + 1) * margin_fraction))
    dy = int(round((y1 - y0 + 1) * margin_fraction))
    return (
        max(0, x0 - dx),
        max(0, y0 - dy),
        min(width - 1, x1 + dx),
        min(height - 1, y1 + dy),
    )


def sample_points_from_tile_mask(
    tile_mask: np.ndarray,
    n_pos: int,
    n_neg: int,
    rng: np.random.Generator,
    negative_margin: float,
) -> Tuple[List[Tuple[int, int]], List[int]]:
    ys_pos, xs_pos = np.where(tile_mask > 0)
    if len(xs_pos) == 0:
        return [], []

    pos_coords = np.stack([xs_pos, ys_pos], axis=1)
    n_pos_eff = min(n_pos, len(pos_coords))
    pos_keep = np.sort(rng.choice(len(pos_coords), size=n_pos_eff, replace=False))
    pos_coords = pos_coords[pos_keep]

    points: List[Tuple[int, int]] = [(int(x), int(y)) for x, y in pos_coords]
    labels: List[int] = [1] * len(points)

    if n_neg <= 0:
        return points, labels

    bbox = mask_bbox(tile_mask)
    if bbox is not None:
        x0, y0, x1, y1 = expand_box(bbox, tile_mask.shape, negative_margin)
        crop = tile_mask[y0 : y1 + 1, x0 : x1 + 1]
        ys_neg, xs_neg = np.where(crop == 0)
        if len(xs_neg) > 0:
            neg_coords = np.stack([xs_neg + x0, ys_neg + y0], axis=1)
        else:
            ys_neg, xs_neg = np.where(tile_mask == 0)
            neg_coords = np.stack([xs_neg, ys_neg], axis=1) if len(xs_neg) > 0 else np.zeros((0, 2), dtype=int)
    else:
        neg_coords = np.zeros((0, 2), dtype=int)

    if len(neg_coords) == 0:
        return points, labels

    n_neg_eff = min(n_neg, len(neg_coords))
    neg_keep = np.sort(rng.choice(len(neg_coords), size=n_neg_eff, replace=False))
    neg_coords = neg_coords[neg_keep]
    points.extend((int(x), int(y)) for x, y in neg_coords)
    labels.extend([0] * len(neg_coords))
    return points, labels


def draw_global_points(
    image: np.ndarray,
    positive_points: Sequence[Tuple[int, int]],
    negative_points: Sequence[Tuple[int, int]],
    radius: int = 5,
) -> np.ndarray:
    canvas = Image.fromarray(image.copy())
    draw = ImageDraw.Draw(canvas)
    for x, y in positive_points:
        draw.ellipse([x - radius, y - radius, x + radius, y + radius], fill=(255, 99, 71), outline=(255, 99, 71))
    for x, y in negative_points:
        draw.ellipse([x - radius, y - radius, x + radius, y + radius], fill=(65, 105, 225), outline=(65, 105, 225))
    return np.array(canvas)


def save_panel(
    image: np.ndarray,
    label: str,
    gt_mask: np.ndarray,
    pred_mask: np.ndarray,
    point_preview: np.ndarray,
    tile_coverage: np.ndarray,
    metrics: Dict[str, float],
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.flatten()

    axes[0].imshow(point_preview)
    axes[0].set_title(f"{label}\nPositive points = red, negative points = blue")
    axes[0].axis("off")

    axes[1].imshow(make_overlay(image, gt_mask, (255, 140, 0)))
    axes[1].set_title("GeoJSON target mask")
    axes[1].axis("off")

    axes[2].imshow(make_overlay(image, pred_mask, (60, 179, 113)))
    axes[2].set_title(
        f"Stitched prediction\nDice={metrics['dice']:.3f}, IoU={metrics['iou']:.3f}, Recall={metrics['recall']:.3f}"
    )
    axes[2].axis("off")

    coverage_view = np.ma.masked_where(tile_coverage == 0, tile_coverage)
    axes[3].imshow(image)
    axes[3].imshow(coverage_view, cmap="magma", alpha=0.65)
    axes[3].set_title("Tile vote count")
    axes[3].axis("off")

    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run tile-based multi-point SAM3 inference on Exp1.")
    parser.add_argument("--image-path", type=Path, default=DEFAULT_IMAGE_PATH)
    parser.add_argument("--summary-path", type=Path, default=None, help="Path to region_summary.json.")
    parser.add_argument("--masks-dir", type=Path, default=None, help="Directory containing region masks and region_summary.json.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--device", choices=["cuda", "mps", "cpu"], default=None)
    parser.add_argument("--labels", type=str, default=DEFAULT_LABELS, help="Comma-separated region labels to include.")
    parser.add_argument("--tile-size", type=int, default=1024)
    parser.add_argument("--tile-overlap", type=int, default=128)
    parser.add_argument("--n-pos", type=int, default=10)
    parser.add_argument("--n-neg", type=int, default=5)
    parser.add_argument("--negative-margin", type=float, default=0.10, help="Margin around tile-local bbox when sampling negative points.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--stitch-threshold", type=float, default=0.5)
    args = parser.parse_args()

    summary_path = resolve_summary_path(args.summary_path, args.masks_dir)
    if not args.image_path.exists():
        raise FileNotFoundError(f"Image not found: {args.image_path}")
    if not summary_path.exists():
        raise FileNotFoundError(f"Region summary not found: {summary_path}")

    output_dir = args.output_dir
    masks_dir = output_dir / "masks"
    overlays_dir = output_dir / "overlays"
    panels_dir = output_dir / "panels"
    ensure_dirs((output_dir, masks_dir, overlays_dir, panels_dir))

    summary = json.loads(summary_path.read_text())
    image = np.array(Image.open(args.image_path).convert("RGB"))
    image_h, image_w = image.shape[:2]
    tiles = make_tiles((image_h, image_w), args.tile_size, args.tile_overlap)

    include_labels = parse_label_filter(args.labels)
    records = select_label_records(summary["labels"], include_labels=include_labels)
    if not records:
        available = ", ".join(str(item["label"]) for item in summary["labels"])
        raise ValueError(f"No matching labels found. Available labels: {available}")

    print("=" * 60)
    print("Visium HD Exp1 multi-point SAM3 baseline")
    print("=" * 60)
    print(f"Image: {args.image_path}")
    print(f"Summary: {summary_path}")
    print(f"Checkpoint: {args.checkpoint or 'default SAM3 from Hugging Face'}")
    print(f"Device: {args.device or 'auto'}")
    print(f"Tiles: {len(tiles)} ({args.tile_size}px with {args.tile_overlap}px overlap)")
    print(f"Labels: {[str(item['label']) for _, item in records]}")
    print(f"Output: {output_dir}")

    sam3 = SAM3Model(confidence_threshold=0.1, checkpoint_path=args.checkpoint, device=args.device)

    experiment = {
        "image_path": str(args.image_path),
        "summary_path": str(summary_path),
        "checkpoint": args.checkpoint,
        "device": args.device,
        "image_shape": [int(image_h), int(image_w)],
        "tile_size": args.tile_size,
        "tile_overlap": args.tile_overlap,
        "n_tiles": len(tiles),
        "n_pos": args.n_pos,
        "n_neg": args.n_neg,
        "negative_margin": args.negative_margin,
        "stitch_threshold": args.stitch_threshold,
        "labels": [],
    }

    for label_index, (idx, item) in enumerate(records):
        label = str(item["label"])
        slug = str(item.get("slug", slugify(label)))
        gt_mask = load_binary_mask(resolve_mask_path(PROJECT_ROOT, summary_path, str(item["mask_path"])))
        if gt_mask.shape != (image_h, image_w):
            raise ValueError(f"Mask shape mismatch for {label}: {gt_mask.shape} vs {(image_h, image_w)}")

        accum = np.zeros((image_h, image_w), dtype=np.float32)
        counts = np.zeros((image_h, image_w), dtype=np.uint16)
        global_pos_points: List[Tuple[int, int]] = []
        global_neg_points: List[Tuple[int, int]] = []
        tile_records: List[Dict[str, object]] = []

        for tile_idx, (x0, y0, x1, y1) in enumerate(tiles):
            tile_mask = gt_mask[y0:y1, x0:x1]
            if tile_mask.sum() == 0:
                continue

            tile_image = image[y0:y1, x0:x1]
            rng = np.random.default_rng(args.seed + label_index * 100000 + tile_idx)
            points_local, point_labels = sample_points_from_tile_mask(
                tile_mask,
                n_pos=args.n_pos,
                n_neg=args.n_neg,
                rng=rng,
                negative_margin=args.negative_margin,
            )
            if not points_local:
                continue

            inference_state = sam3.encode_image(tile_image)
            pred_tile = normalize_prediction(
                sam3.predict_points(inference_state, points_local, point_labels, tile_image.shape[:2]),
                tile_image.shape[:2],
            )

            accum[y0:y1, x0:x1] += pred_tile.astype(np.float32)
            counts[y0:y1, x0:x1] += 1

            for (x_local, y_local), point_label in zip(points_local, point_labels):
                global_point = (int(x0 + x_local), int(y0 + y_local))
                if point_label == 1:
                    global_pos_points.append(global_point)
                else:
                    global_neg_points.append(global_point)

            tile_records.append(
                {
                    "tile_xyxy": [int(x0), int(y0), int(x1), int(y1)],
                    "n_positive_points": int(sum(1 for value in point_labels if value == 1)),
                    "n_negative_points": int(sum(1 for value in point_labels if value == 0)),
                    "tile_target_positive_pixels": int(tile_mask.sum()),
                    "tile_prediction_positive_pixels": int(pred_tile.sum()),
                }
            )

        pred_soft = np.zeros((image_h, image_w), dtype=np.float32)
        valid = counts > 0
        pred_soft[valid] = accum[valid] / counts[valid].astype(np.float32)
        pred_mask = (pred_soft >= args.stitch_threshold).astype(np.uint8)
        metrics = metrics_to_dict(pred_mask, gt_mask)

        stem = f"{idx + 1:02d}_{slug}"
        point_preview = draw_global_points(image, global_pos_points, global_neg_points)
        save_mask(gt_mask, masks_dir / f"{stem}_target.png")
        save_mask(pred_mask, masks_dir / f"{stem}_pred_multipoint.png")
        save_mask((counts > 0).astype(np.uint8), masks_dir / f"{stem}_tile_coverage_binary.png")
        Image.fromarray(make_overlay(image, gt_mask, (255, 140, 0))).save(overlays_dir / f"{stem}_target.png")
        Image.fromarray(make_overlay(image, pred_mask, (60, 179, 113))).save(overlays_dir / f"{stem}_pred_multipoint.png")
        Image.fromarray(point_preview).save(overlays_dir / f"{stem}_point_preview.png")
        save_panel(
            image=image,
            label=label,
            gt_mask=gt_mask,
            pred_mask=pred_mask,
            point_preview=point_preview,
            tile_coverage=counts,
            metrics=metrics,
            output_path=panels_dir / f"{stem}.png",
        )

        record = {
            "label": label,
            "slug": slug,
            "target_positive_pixels": int(gt_mask.sum()),
            "prediction_positive_pixels": int(pred_mask.sum()),
            "n_prompted_tiles": int(len(tile_records)),
            "n_positive_points_total": int(len(global_pos_points)),
            "n_negative_points_total": int(len(global_neg_points)),
            "multipoint_metrics": metrics,
            "tile_records": tile_records,
        }
        experiment["labels"].append(record)

        print(f"\nLabel: {label}")
        print(
            f"  Dice={metrics['dice']:.3f}, IoU={metrics['iou']:.3f}, Recall={metrics['recall']:.3f}, "
            f"tiles={record['n_prompted_tiles']}, pos_points={record['n_positive_points_total']}, "
            f"neg_points={record['n_negative_points_total']}"
        )

    (output_dir / "experiment_summary.json").write_text(json.dumps(experiment, indent=2))
    print(f"\nSaved outputs to: {output_dir}")


if __name__ == "__main__":
    main()
