#!/usr/bin/env python3
"""
Generate SAM3 candidate masks for Visium HD Exp1.

This is the first SaLIP-style stage for this project: use SAM3 as a
high-recall candidate generator before any H&E-expression biology ranking.

SAM3 does not expose the exact SAM "automatic mask generator" API in this repo,
so this script approximates segment-everything with dense positive point
prompts over a downsampled H&E image. Each point produces one candidate mask.
The output answers the first question:

    Does the SAM3 candidate pool contain masks that cover the target regions?

It optionally evaluates the candidate pool against existing GeoJSON-derived
region masks by reporting union recall and best single-candidate Dice.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

PROJECT_ROOT = Path(os.environ.get("PROJECT_ROOT_OVERRIDE", str(Path(__file__).resolve().parent.parent.parent)))
MPL_CONFIG_DIR = PROJECT_ROOT / "output" / ".mplconfig"
MPL_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CONFIG_DIR))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(PROJECT_ROOT / "inference"))
sys.path.insert(0, str(PROJECT_ROOT / "inference" / "visium_hd_exp1"))

from metrics import compute_all_metrics
from sam3_inference import SAM3Model, generate_bbox_from_mask, resize_mask
from sam3_baseline_utils import (
    DATA_DIR,
    ensure_dirs,
    load_binary_mask,
    make_overlay,
    parse_label_filter,
    resolve_mask_path,
    save_mask,
    select_label_records,
    slugify,
)


DEFAULT_IMAGE_PATH = DATA_DIR / "spatial" / "tissue_hires_image.png"
DEFAULT_SUMMARY_PATHS = [
    PROJECT_ROOT / "output" / "visium_hd_exp1" / "region_masks" / "region_summary.json",
    PROJECT_ROOT / "output_visium_hd_exp1_legacy" / "visium_hd_exp1" / "region_masks" / "region_summary.json",
    PROJECT_ROOT / "output_visium_hd_exp1_legacy" / "visium_hd_exp1" / "region_masks_flip_y" / "region_summary.json",
]
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "visium_hd_exp1"
    / "sam3_candidate_proposals"
    / "dense_points_smoke"
)


def resolve_default_summary() -> Optional[Path]:
    for path in DEFAULT_SUMMARY_PATHS:
        if path.exists():
            return path
    return None


def resize_image_and_masks(
    image: np.ndarray,
    masks: Sequence[np.ndarray],
    max_side: Optional[int],
) -> Tuple[np.ndarray, List[np.ndarray], float]:
    if max_side is None:
        return image, [m.astype(np.uint8) for m in masks], 1.0

    height, width = image.shape[:2]
    current_max = max(height, width)
    if current_max <= max_side:
        return image, [m.astype(np.uint8) for m in masks], 1.0

    scale = max_side / float(current_max)
    new_h = max(1, int(round(height * scale)))
    new_w = max(1, int(round(width * scale)))
    resized_image = np.array(
        Image.fromarray(image).resize((new_w, new_h), resample=Image.Resampling.BILINEAR)
    )
    resized_masks = [resize_mask(m.astype(np.uint8), (new_h, new_w)).astype(np.uint8) for m in masks]
    return resized_image, resized_masks, scale


def make_grid_points(
    image_shape: Tuple[int, int],
    grid_step: int,
    border: int,
    max_points: Optional[int],
    tissue_mask: Optional[np.ndarray] = None,
) -> List[Tuple[int, int]]:
    height, width = image_shape
    xs = list(range(border, max(border + 1, width - border), grid_step))
    ys = list(range(border, max(border + 1, height - border), grid_step))
    points = [(int(x), int(y)) for y in ys for x in xs]
    if tissue_mask is not None:
        points = [(x, y) for x, y in points if 0 <= y < height and 0 <= x < width and tissue_mask[y, x] > 0]
    if max_points is not None:
        points = points[:max_points]
    return points


def make_sliding_boxes(
    image_shape: Tuple[int, int],
    box_size: int,
    stride: int,
    max_boxes: Optional[int],
) -> List[Tuple[int, int, int, int]]:
    height, width = image_shape
    ys = list(range(0, max(1, height - box_size + 1), stride))
    xs = list(range(0, max(1, width - box_size + 1), stride))
    if not ys or ys[-1] != max(0, height - box_size):
        ys.append(max(0, height - box_size))
    if not xs or xs[-1] != max(0, width - box_size):
        xs.append(max(0, width - box_size))

    boxes: List[Tuple[int, int, int, int]] = []
    for y in ys:
        for x in xs:
            x2 = min(width - 1, x + box_size)
            y2 = min(height - 1, y + box_size)
            if x2 > x and y2 > y:
                boxes.append((int(x), int(y), int(x2), int(y2)))
    if max_boxes is not None:
        boxes = boxes[:max_boxes]
    return boxes


def simple_tissue_mask(image: np.ndarray, threshold: int = 245) -> np.ndarray:
    """Remove near-white/near-black background for point placement."""
    rgb = image.astype(np.int16)
    near_white = np.all(rgb > threshold, axis=2)
    near_black = np.all(rgb < 8, axis=2)
    return (~near_white & ~near_black).astype(np.uint8)


def metrics_to_dict(pred: np.ndarray, gt: np.ndarray) -> Dict[str, float]:
    m = compute_all_metrics(pred.astype(np.uint8), gt.astype(np.uint8))
    return {
        "dice": float(m.dice),
        "iou": float(m.iou),
        "precision": float(m.precision),
        "recall": float(m.recall),
        "psnr": float(m.psnr),
        "ssim": float(m.ssim),
    }


def mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return float(inter / union) if union else 0.0


def nms_candidates(candidates: List[Dict[str, object]], iou_threshold: float) -> List[Dict[str, object]]:
    if iou_threshold <= 0 or not candidates:
        return candidates
    ordered = sorted(candidates, key=lambda item: int(item["area"]), reverse=True)
    kept: List[Dict[str, object]] = []
    for cand in ordered:
        mask = cand["mask"]
        if all(mask_iou(mask, kept_item["mask"]) < iou_threshold for kept_item in kept):
            kept.append(cand)
    return kept


def draw_prompt_preview(
    image: np.ndarray,
    points: Sequence[Tuple[int, int]],
    boxes: Sequence[Tuple[int, int, int, int]],
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(image)
    if points:
        xs = [x for x, _ in points]
        ys = [y for _, y in points]
        ax.scatter(xs, ys, s=10, c="cyan", edgecolors="black", linewidths=0.2)
    for x1, y1, x2, y2 in boxes:
        ax.add_patch(
            plt.Rectangle(
                (x1, y1),
                x2 - x1,
                y2 - y1,
                fill=False,
                edgecolor="cyan",
                linewidth=0.6,
                alpha=0.30,
            )
        )
    if boxes:
        title = f"Dense box prompts ({len(boxes)} boxes)"
    else:
        title = f"Dense point prompts ({len(points)} points)"
    ax.set_title(title)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def candidate_overlay(image: np.ndarray, candidates: Sequence[Dict[str, object]], alpha: float = 0.42) -> np.ndarray:
    rng = np.random.default_rng(13)
    canvas = image.astype(np.float32).copy()
    for cand in candidates:
        mask = cand["mask"].astype(bool)
        color = rng.integers(40, 255, size=3).astype(np.float32)
        canvas[mask] = (1 - alpha) * canvas[mask] + alpha * color
    return np.clip(canvas, 0, 255).astype(np.uint8)


def save_candidate_contact_sheet(
    image: np.ndarray,
    candidates: Sequence[Dict[str, object]],
    output_path: Path,
    limit: int,
) -> None:
    shown = list(candidates[:limit])
    if not shown:
        return
    cols = min(5, len(shown))
    rows = int(np.ceil(len(shown) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.2, rows * 3.2))
    axes_arr = np.atleast_1d(axes).reshape(rows, cols)
    for ax in axes_arr.flat:
        ax.axis("off")
    for idx, cand in enumerate(shown):
        ax = axes_arr.flat[idx]
        mask = cand["mask"].astype(np.uint8)
        ax.imshow(make_overlay(image, mask, (0, 220, 255), alpha=0.48))
        bbox = cand.get("bbox_xyxy")
        ax.set_title(
            f"#{idx} area={int(cand['area'])}\npoint={cand.get('point_xy')} box={bbox}",
            fontsize=8,
        )
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def draw_label_panel(
    image: np.ndarray,
    label: str,
    gt_mask: np.ndarray,
    union_mask: np.ndarray,
    best_mask: np.ndarray,
    best_point: Optional[Tuple[int, int]],
    union_metrics: Dict[str, float],
    best_metrics: Dict[str, float],
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(1, 4, figsize=(18, 5))
    axes[0].imshow(make_overlay(image, gt_mask, (255, 140, 0), alpha=0.50))
    axes[0].set_title(f"{label}\nGeoJSON target")
    axes[0].axis("off")
    axes[1].imshow(make_overlay(image, union_mask, (0, 220, 255), alpha=0.45))
    axes[1].set_title(
        f"Union of candidates\nRecall={union_metrics['recall']:.3f}, Dice={union_metrics['dice']:.3f}"
    )
    axes[1].axis("off")
    axes[2].imshow(make_overlay(image, best_mask, (30, 190, 90), alpha=0.55))
    axes[2].set_title(
        f"Best single candidate\nDice={best_metrics['dice']:.3f}, Recall={best_metrics['recall']:.3f}"
    )
    axes[2].axis("off")
    axes[3].imshow(image)
    if best_point is not None:
        axes[3].scatter([best_point[0]], [best_point[1]], s=90, c="cyan", edgecolors="black")
    axes[3].set_title(f"Best point prompt: {best_point}")
    axes[3].axis("off")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def write_candidate_csv(candidates: Sequence[Dict[str, object]], path: Path) -> None:
    fieldnames = [
        "candidate_id",
        "prompt_index",
        "prompt_type",
        "point_x",
        "point_y",
        "prompt_box_x1",
        "prompt_box_y1",
        "prompt_box_x2",
        "prompt_box_y2",
        "area",
        "bbox_x1",
        "bbox_y1",
        "bbox_x2",
        "bbox_y2",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for idx, cand in enumerate(candidates):
            bbox = cand.get("bbox_xyxy") or [None, None, None, None]
            point = cand.get("point_xy") or [None, None]
            prompt_box = cand.get("prompt_box_xyxy") or [None, None, None, None]
            writer.writerow(
                {
                    "candidate_id": idx,
                    "prompt_index": cand.get("prompt_index"),
                    "prompt_type": cand.get("prompt_type"),
                    "point_x": point[0],
                    "point_y": point[1],
                    "prompt_box_x1": prompt_box[0],
                    "prompt_box_y1": prompt_box[1],
                    "prompt_box_x2": prompt_box[2],
                    "prompt_box_y2": prompt_box[3],
                    "area": int(cand["area"]),
                    "bbox_x1": bbox[0],
                    "bbox_y1": bbox[1],
                    "bbox_x2": bbox[2],
                    "bbox_y2": bbox[3],
                }
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate SAM3 candidate masks for Visium HD Exp1.")
    parser.add_argument("--image-path", type=Path, default=DEFAULT_IMAGE_PATH)
    parser.add_argument("--summary-path", type=Path, default=resolve_default_summary())
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--device", choices=["cuda", "mps", "cpu"], default=None)
    parser.add_argument("--labels", type=str, default=None, help="Optional comma-separated labels to evaluate.")
    parser.add_argument("--max-side", type=int, default=1024, help="Downsample longest side for first candidate test. 0 disables.")
    parser.add_argument("--grid-step", type=int, default=96)
    parser.add_argument("--border", type=int, default=48)
    parser.add_argument("--max-points", type=int, default=0)
    parser.add_argument("--proposal-mode", choices=["point", "box"], default="point")
    parser.add_argument(
        "--text-prompt",
        type=str,
        default=None,
        help="Optional label-specific text prompt. If set, use joint text+box/text+point prompts.",
    )
    parser.add_argument("--box-size", type=int, default=192)
    parser.add_argument("--box-stride", type=int, default=96)
    parser.add_argument("--max-boxes", type=int, default=0)
    parser.add_argument(
        "--prompt-start",
        type=int,
        default=0,
        help="Start offset into the generated point/box prompt list. Used for resumable chunk jobs.",
    )
    parser.add_argument(
        "--prompt-count",
        type=int,
        default=0,
        help="Number of prompts to process after --prompt-start. 0 means process all remaining prompts.",
    )
    parser.add_argument("--min-area-frac", type=float, default=0.0002)
    parser.add_argument("--max-area-frac", type=float, default=0.35)
    parser.add_argument("--nms-iou", type=float, default=0.92)
    parser.add_argument("--save-mask-limit", type=int, default=80)
    parser.add_argument("--contact-sheet-limit", type=int, default=25)
    parser.add_argument("--no-tissue-filter", action="store_true")
    parser.add_argument(
        "--skip-label-eval",
        action="store_true",
        help="Skip per-label Dice/Precision/Recall panels. Useful for chunk jobs that will be merged later.",
    )
    args = parser.parse_args()

    if not args.image_path.exists():
        raise FileNotFoundError(f"Image not found: {args.image_path}")

    output_dir = args.output_dir
    masks_dir = output_dir / "candidate_masks"
    panels_dir = output_dir / "label_panels"
    ensure_dirs((output_dir, masks_dir, panels_dir))

    image = np.array(Image.open(args.image_path).convert("RGB"))
    summary = None
    records = []
    gt_masks: List[np.ndarray] = []
    summary_path = args.summary_path
    if summary_path is not None and summary_path.exists() and not args.skip_label_eval:
        summary = json.loads(summary_path.read_text())
        records = select_label_records(summary["labels"], include_labels=parse_label_filter(args.labels))
        gt_masks = [
            load_binary_mask(resolve_mask_path(PROJECT_ROOT, summary_path, str(item["mask_path"])))
            for _, item in records
        ]
    elif args.labels and not args.skip_label_eval:
        raise FileNotFoundError(f"Summary path not found for label evaluation: {summary_path}")

    max_side = None if args.max_side == 0 else args.max_side
    image, gt_masks, resize_scale = resize_image_and_masks(image, gt_masks, max_side)
    image_h, image_w = image.shape[:2]
    image_area = image_h * image_w

    tissue = None if args.no_tissue_filter else simple_tissue_mask(image)
    max_points = None if args.max_points <= 0 else args.max_points
    max_boxes = None if args.max_boxes <= 0 else args.max_boxes
    points: List[Tuple[int, int]] = []
    boxes: List[Tuple[int, int, int, int]] = []
    if args.proposal_mode == "point":
        points = make_grid_points(
            (image_h, image_w),
            grid_step=args.grid_step,
            border=args.border,
            max_points=max_points,
            tissue_mask=tissue,
        )
    else:
        boxes = make_sliding_boxes((image_h, image_w), args.box_size, args.box_stride, max_boxes)

    print("=" * 72)
    print("Visium HD Exp1 SAM3 candidate proposal generation")
    print("=" * 72)
    print(f"Image: {args.image_path}")
    print(f"Summary: {summary_path if summary_path else 'none'}")
    print(f"Checkpoint: {args.checkpoint or 'default SAM3 from HuggingFace'}")
    print(f"Device: {args.device or 'auto'}")
    print(f"Image shape: {image_h} x {image_w}; resize_scale={resize_scale:.5f}")
    print(f"Proposal mode: {args.proposal_mode}")
    print(f"Text prompt: {args.text_prompt or 'none'}")
    print(f"Grid step: {args.grid_step}; border: {args.border}; points={len(points)}")
    print(f"Box size: {args.box_size}; box stride: {args.box_stride}; boxes={len(boxes)}")
    print(f"Area filter: {args.min_area_frac:.5f} to {args.max_area_frac:.2f} of image")
    print(f"Output: {output_dir}")

    draw_prompt_preview(image, points, boxes, output_dir / "dense_prompt_grid.png")

    sam3 = SAM3Model(confidence_threshold=0.1, checkpoint_path=args.checkpoint, device=args.device)
    inference_state = sam3.encode_image(image)

    min_area = int(round(args.min_area_frac * image_area))
    max_area = int(round(args.max_area_frac * image_area))
    raw_candidates: List[Dict[str, object]] = []
    rejected_small = 0
    rejected_large = 0
    empty = 0

    prompts: List[Dict[str, object]] = []
    if args.proposal_mode == "point":
        prompts = [{"prompt_type": "point", "point_xy": point} for point in points]
    else:
        prompts = [{"prompt_type": "box", "prompt_box_xyxy": box} for box in boxes]
    total_prompts = len(prompts)
    prompt_start = max(0, int(args.prompt_start))
    if args.prompt_count and args.prompt_count > 0:
        prompt_end = min(total_prompts, prompt_start + int(args.prompt_count))
    else:
        prompt_end = total_prompts
    prompts = [
        {**prompt, "prompt_index": prompt_start + offset}
        for offset, prompt in enumerate(prompts[prompt_start:prompt_end])
    ]
    print(
        f"Prompt slice: start={prompt_start}, end={prompt_end}, "
        f"count={len(prompts)}, total={total_prompts}",
        flush=True,
    )

    for idx, prompt in enumerate(prompts, start=1):
        if prompt["prompt_type"] == "point":
            point = prompt["point_xy"]
            if args.text_prompt:
                pred = sam3.predict_points_text(inference_state, [point], [1], args.text_prompt, (image_h, image_w))
            else:
                pred = sam3.predict_points(inference_state, [point], [1], (image_h, image_w))
        else:
            if args.text_prompt:
                pred = sam3.predict_box_text(
                    inference_state,
                    prompt["prompt_box_xyxy"],
                    args.text_prompt,
                    (image_h, image_w),
                )
            else:
                pred = sam3.predict_box(inference_state, prompt["prompt_box_xyxy"], (image_h, image_w))
        if pred is None:
            empty += 1
            continue
        if pred.shape != (image_h, image_w):
            pred = resize_mask(pred, (image_h, image_w)).astype(np.uint8)
        else:
            pred = pred.astype(np.uint8)
        area = int(pred.sum())
        if area <= 0:
            empty += 1
            continue
        if area < min_area:
            rejected_small += 1
            continue
        if area > max_area:
            rejected_large += 1
            continue
        bbox = generate_bbox_from_mask(pred)
        candidate = {
            "prompt_type": prompt["prompt_type"],
            "prompt_index": int(prompt.get("prompt_index", idx - 1)),
            "area": area,
            "bbox_xyxy": list(bbox) if bbox is not None else None,
            "mask": pred,
        }
        if prompt["prompt_type"] == "point":
            point = prompt["point_xy"]
            candidate["point_xy"] = [int(point[0]), int(point[1])]
        else:
            candidate["prompt_box_xyxy"] = [int(v) for v in prompt["prompt_box_xyxy"]]
            x1, y1, x2, y2 = prompt["prompt_box_xyxy"]
            candidate["point_xy"] = [int(round((x1 + x2) / 2)), int(round((y1 + y2) / 2))]
        raw_candidates.append(candidate)
        if idx % 25 == 0 or idx == len(prompts):
            print(
                f"  processed {idx}/{len(prompts)} prompts; "
                f"raw candidates={len(raw_candidates)}, empty={empty}, "
                f"small={rejected_small}, large={rejected_large}"
            )

    candidates = nms_candidates(raw_candidates, args.nms_iou)
    candidates = sorted(candidates, key=lambda item: int(item["area"]), reverse=True)
    union_mask = np.zeros((image_h, image_w), dtype=np.uint8)
    for cand in candidates:
        union_mask = np.logical_or(union_mask, cand["mask"]).astype(np.uint8)

    print(f"Raw candidates after area filter: {len(raw_candidates)}")
    print(f"Candidates after NMS: {len(candidates)}")
    print(f"Union coverage: {int(union_mask.sum())} pixels ({union_mask.sum() / image_area:.3f} of resized image)")

    save_mask(union_mask, output_dir / "candidate_union_mask.png")
    Image.fromarray(make_overlay(image, union_mask, (0, 220, 255), alpha=0.45)).save(output_dir / "candidate_union_overlay.png")
    Image.fromarray(candidate_overlay(image, candidates[: args.save_mask_limit])).save(output_dir / "candidate_colored_overlay.png")
    save_candidate_contact_sheet(image, candidates, output_dir / "candidate_contact_sheet.png", args.contact_sheet_limit)

    for idx, cand in enumerate(candidates[: args.save_mask_limit]):
        save_mask(cand["mask"], masks_dir / f"candidate_{idx:03d}.png")
    write_candidate_csv(candidates, output_dir / "candidate_metadata.csv")

    label_results = []
    for (_, item), gt_mask in zip(records, gt_masks):
        label = str(item["label"])
        slug = str(item.get("slug", slugify(label)))
        best_mask = np.zeros_like(gt_mask, dtype=np.uint8)
        best_metrics = metrics_to_dict(best_mask, gt_mask)
        best_idx = None
        best_point = None
        for cand_idx, cand in enumerate(candidates):
            cand_metrics = metrics_to_dict(cand["mask"], gt_mask)
            if cand_metrics["dice"] > best_metrics["dice"]:
                best_metrics = cand_metrics
                best_mask = cand["mask"]
                best_idx = cand_idx
                best_point = tuple(cand["point_xy"])
        union_metrics = metrics_to_dict(union_mask, gt_mask)
        label_result = {
            "label": label,
            "slug": slug,
            "target_pixels": int(gt_mask.sum()),
            "union_metrics": union_metrics,
            "best_candidate_metrics": best_metrics,
            "best_candidate_index": best_idx,
            "best_point_xy": list(best_point) if best_point is not None else None,
            "best_candidate_pixels": int(best_mask.sum()),
        }
        label_results.append(label_result)
        draw_label_panel(
            image=image,
            label=label,
            gt_mask=gt_mask,
            union_mask=union_mask,
            best_mask=best_mask,
            best_point=best_point,
            union_metrics=union_metrics,
            best_metrics=best_metrics,
            output_path=panels_dir / f"{slug}.png",
        )
        print(
            f"Label {label}: union recall={union_metrics['recall']:.3f}, "
            f"best Dice={best_metrics['dice']:.3f}, best recall={best_metrics['recall']:.3f}, "
            f"best_idx={best_idx}"
        )

    report = {
        "image_path": str(args.image_path),
        "summary_path": str(summary_path) if summary_path else None,
        "checkpoint": args.checkpoint,
        "device": args.device,
        "max_side": max_side,
        "resize_scale": resize_scale,
        "image_shape": [int(image_h), int(image_w)],
        "proposal_mode": args.proposal_mode,
        "text_prompt": args.text_prompt,
        "prompt_start": prompt_start,
        "prompt_end": prompt_end,
        "n_total_prompts": total_prompts,
        "n_processed_prompts": len(prompts),
        "skip_label_eval": bool(args.skip_label_eval),
        "grid_step": args.grid_step,
        "border": args.border,
        "n_points": len(points),
        "box_size": args.box_size,
        "box_stride": args.box_stride,
        "n_boxes": len(boxes),
        "n_empty": empty,
        "n_rejected_small": rejected_small,
        "n_rejected_large": rejected_large,
        "n_raw_candidates": len(raw_candidates),
        "n_candidates_after_nms": len(candidates),
        "nms_iou": args.nms_iou,
        "min_area": min_area,
        "max_area": max_area,
        "union_pixels": int(union_mask.sum()),
        "union_fraction": float(union_mask.sum() / image_area),
        "labels": label_results,
    }
    (output_dir / "candidate_report.json").write_text(json.dumps(report, indent=2))
    print(f"Saved candidate report: {output_dir / 'candidate_report.json'}")


if __name__ == "__main__":
    main()
