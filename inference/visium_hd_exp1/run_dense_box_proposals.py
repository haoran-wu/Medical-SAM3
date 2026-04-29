#!/usr/bin/env python3
"""
Run dense box-proposal SAM3 baselines on Visium HD Exp1 region masks.

This slides boxes over the whole image. Each box is run independently. For each
region label, the script reports:
- the union of all non-empty box proposals
- the best single box proposal
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
MPL_CONFIG_DIR = PROJECT_ROOT / "output" / ".mplconfig"
MPL_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CONFIG_DIR))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from PIL import Image

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
    resize_image_and_masks,
    save_mask,
    select_label_records,
    slugify,
)


DEFAULT_IMAGE_PATH = DATA_DIR / "spatial" / "tissue_hires_image.png"
DEFAULT_REGION_SUMMARY = PROJECT_ROOT / "output" / "visium_hd_exp1" / "region_masks" / "region_summary.json"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "output" / "visium_hd_exp1" / "sam3_runs" / "dense_box_proposals"


def make_sliding_boxes(
    image_shape: Tuple[int, int],
    box_size: int,
    stride: int,
    max_boxes: Optional[int],
) -> List[Tuple[int, int, int, int]]:
    height, width = image_shape
    boxes: List[Tuple[int, int, int, int]] = []

    ys = list(range(0, max(1, height - box_size + 1), stride))
    xs = list(range(0, max(1, width - box_size + 1), stride))
    if not ys or ys[-1] != max(0, height - box_size):
        ys.append(max(0, height - box_size))
    if not xs or xs[-1] != max(0, width - box_size):
        xs.append(max(0, width - box_size))

    for y in ys:
        for x in xs:
            x2 = min(width - 1, x + box_size)
            y2 = min(height - 1, y + box_size)
            if x2 > x and y2 > y:
                boxes.append((int(x), int(y), int(x2), int(y2)))

    if max_boxes is not None:
        boxes = boxes[:max_boxes]
    return boxes


def save_box_preview(image: np.ndarray, boxes: List[Tuple[int, int, int, int]], output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(image)
    for x1, y1, x2, y2 in boxes:
        ax.add_patch(
            patches.Rectangle(
                (x1, y1),
                x2 - x1,
                y2 - y1,
                fill=False,
                edgecolor="cyan",
                linewidth=0.6,
                alpha=0.30,
            )
        )
    ax.set_title(f"Dense sliding boxes ({len(boxes)} boxes)")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_label_panel(
    image: np.ndarray,
    label: str,
    gt_mask: np.ndarray,
    union_mask: np.ndarray,
    best_mask: np.ndarray,
    best_box: Optional[Tuple[int, int, int, int]],
    union_metrics: Dict[str, float],
    best_metrics: Dict[str, float],
    color: Tuple[int, int, int],
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(1, 4, figsize=(18, 5))
    axes[0].imshow(make_overlay(image, gt_mask, color))
    axes[0].set_title(f"{label}\nTarget")
    axes[0].axis("off")

    axes[1].imshow(make_overlay(image, union_mask, color))
    axes[1].set_title(
        f"Union of box proposals\nDice={union_metrics['dice']:.3f}, Recall={union_metrics['recall']:.3f}"
    )
    axes[1].axis("off")

    axes[2].imshow(make_overlay(image, best_mask, color))
    axes[2].set_title(
        f"Best single box proposal\nDice={best_metrics['dice']:.3f}, IoU={best_metrics['iou']:.3f}"
    )
    axes[2].axis("off")

    axes[3].imshow(image)
    if best_box is not None:
        x1, y1, x2, y2 = best_box
        axes[3].add_patch(
            patches.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor="red", linewidth=3)
        )
    axes[3].set_title(f"Best box: {best_box}")
    axes[3].axis("off")

    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run dense box-proposal SAM3 baselines on Visium HD Exp1.")
    parser.add_argument("--image-path", type=Path, default=DEFAULT_IMAGE_PATH)
    parser.add_argument("--summary-path", type=Path, default=DEFAULT_REGION_SUMMARY)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--device", choices=["cuda", "mps", "cpu"], default=None)
    parser.add_argument("--labels", type=str, default=None, help="Comma-separated region labels to include.")
    parser.add_argument("--max-side", type=int, default=1024, help="Resize longest side for inference. Use 0 to disable.")
    parser.add_argument("--box-size", type=int, default=256)
    parser.add_argument("--stride", type=int, default=128)
    parser.add_argument("--max-boxes", type=int, default=0)
    args = parser.parse_args()

    if not args.image_path.exists():
        raise FileNotFoundError(f"Image not found: {args.image_path}")
    if not args.summary_path.exists():
        raise FileNotFoundError(f"Region summary not found: {args.summary_path}")

    output_dir = args.output_dir
    masks_dir = output_dir / "masks"
    overlays_dir = output_dir / "overlays"
    panels_dir = output_dir / "panels"
    ensure_dirs((output_dir, masks_dir, overlays_dir, panels_dir))

    summary = json.loads(args.summary_path.read_text())
    image = np.array(Image.open(args.image_path).convert("RGB"))

    include_labels = parse_label_filter(args.labels)
    records = select_label_records(summary["labels"], include_labels=include_labels)
    if not records:
        available = ", ".join(str(item["label"]) for item in summary["labels"])
        raise ValueError(f"No matching labels found. Available labels: {available}")

    masks = [
        load_binary_mask(resolve_mask_path(PROJECT_ROOT, args.summary_path, str(item["mask_path"])))
        for _, item in records
    ]

    max_side = None if args.max_side == 0 else args.max_side
    image, masks, resize_scale = resize_image_and_masks(image, masks, max_side)
    image_h, image_w = image.shape[:2]

    max_boxes = None if args.max_boxes <= 0 else args.max_boxes
    boxes = make_sliding_boxes((image_h, image_w), args.box_size, args.stride, max_boxes)

    print("=" * 60)
    print("Visium HD Exp1 dense box-proposal SAM3 baseline")
    print("=" * 60)
    print(f"Image: {args.image_path}")
    print(f"Checkpoint: {args.checkpoint or 'default SAM3 from Hugging Face'}")
    print(f"Image shape for inference: {image_h} x {image_w}")
    print(f"Box size: {args.box_size}; stride: {args.stride}; boxes: {len(boxes)}")
    print(f"Output: {output_dir}")

    save_box_preview(image, boxes, output_dir / "dense_box_grid.png")

    sam3 = SAM3Model(confidence_threshold=0.1, checkpoint_path=args.checkpoint, device=args.device)
    inference_state = sam3.encode_image(image)

    proposal_masks: List[np.ndarray] = []
    proposal_boxes: List[Tuple[int, int, int, int]] = []
    union_mask = np.zeros((image_h, image_w), dtype=np.uint8)

    for idx, box in enumerate(boxes, start=1):
        pred = normalize_prediction(sam3.predict_box(inference_state, box, (image_h, image_w)), (image_h, image_w))
        if pred.sum() > 0:
            proposal_masks.append(pred)
            proposal_boxes.append(box)
            union_mask = np.logical_or(union_mask, pred).astype(np.uint8)

        if idx % 25 == 0 or idx == len(boxes):
            print(f"  Processed {idx}/{len(boxes)} boxes; non-empty proposals={len(proposal_masks)}")

    save_mask(union_mask, masks_dir / "all_box_proposals_union.png")
    Image.fromarray(make_overlay(image, union_mask, (0, 220, 255))).save(overlays_dir / "all_box_proposals_union.png")

    experiment = {
        "image_path": str(args.image_path),
        "summary_path": str(args.summary_path),
        "checkpoint": args.checkpoint,
        "device": args.device,
        "max_side": max_side,
        "resize_scale": resize_scale,
        "image_shape": [int(image_h), int(image_w)],
        "box_size": args.box_size,
        "stride": args.stride,
        "n_boxes": int(len(boxes)),
        "n_nonempty_proposals": int(len(proposal_masks)),
        "union_positive_pixels": int(union_mask.sum()),
        "labels": [],
    }

    for (idx, item), gt_mask in zip(records, masks):
        label = str(item["label"])
        color = tuple(int(v) for v in item.get("color_rgb", [230, 57, 70]))
        best_mask = np.zeros_like(gt_mask, dtype=np.uint8)
        best_metrics = metrics_to_dict(best_mask, gt_mask)
        best_box = None
        best_proposal_index = None

        for proposal_index, (box, pred) in enumerate(zip(proposal_boxes, proposal_masks)):
            candidate_metrics = metrics_to_dict(pred, gt_mask)
            if candidate_metrics["dice"] > best_metrics["dice"]:
                best_metrics = candidate_metrics
                best_mask = pred
                best_box = box
                best_proposal_index = proposal_index

        union_metrics = metrics_to_dict(union_mask, gt_mask)
        stem = f"{idx + 1:02d}_{slugify(label)}"

        save_mask(best_mask, masks_dir / f"{stem}_best_box_proposal.png")
        Image.fromarray(make_overlay(image, best_mask, color)).save(overlays_dir / f"{stem}_best_box_proposal.png")
        save_label_panel(
            image=image,
            label=label,
            gt_mask=gt_mask,
            union_mask=union_mask,
            best_mask=best_mask,
            best_box=best_box,
            union_metrics=union_metrics,
            best_metrics=best_metrics,
            color=color,
            output_path=panels_dir / f"{stem}.png",
        )

        experiment["labels"].append(
            {
                "label": label,
                "slug": str(item.get("slug", slugify(label))),
                "target_positive_pixels": int(gt_mask.sum()),
                "union_metrics": union_metrics,
                "best_single_box_metrics": best_metrics,
                "best_box_xyxy": list(best_box) if best_box is not None else None,
                "best_proposal_index": best_proposal_index,
                "best_prediction_positive_pixels": int(best_mask.sum()),
            }
        )

        print(f"\nLabel: {label}")
        print(
            f"  Union: Dice={union_metrics['dice']:.3f}, IoU={union_metrics['iou']:.3f}, "
            f"Recall={union_metrics['recall']:.3f}, pixels={int(union_mask.sum())}"
        )
        print(
            f"  Best single box: {best_box}, Dice={best_metrics['dice']:.3f}, "
            f"IoU={best_metrics['iou']:.3f}, Recall={best_metrics['recall']:.3f}, "
            f"pixels={int(best_mask.sum())}"
        )

    (output_dir / "experiment_summary.json").write_text(json.dumps(experiment, indent=2))
    print(f"\nSaved outputs to: {output_dir}")


if __name__ == "__main__":
    main()
