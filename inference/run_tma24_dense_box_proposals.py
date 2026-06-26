#!/usr/bin/env python3
"""
Run a dense box-proposal experiment on TMA24 pseudo-masks.

This approximates a box-based proposal sweep by placing sliding-window boxes
over the image. Each box is run independently, giving a set of candidate masks.
For each TMA24 label, the script reports:
- the union of all box masks against the pseudo-mask
- the best single box-mask proposal against the pseudo-mask
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MPL_CONFIG_DIR = PROJECT_ROOT / "output" / ".mplconfig"
MPL_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CONFIG_DIR))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))

from metrics import compute_all_metrics
from sam3_inference import SAM3Model, resize_mask


DEFAULT_IMAGE_PATH = PROJECT_ROOT / "examples" / "legacy_examples" / "tma24" / "example1.jpg"
DEFAULT_PSEUDOMASK_DIR = PROJECT_ROOT / "output" / "00_FINAL_tma24_example1_scale_0p55_shiftX_neg120_shiftY_620"
DEFAULT_SUMMARY_PATH = DEFAULT_PSEUDOMASK_DIR / "summary.json"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "output" / "tma24_dense_box_proposals"


def load_binary_mask(path: Path) -> np.ndarray:
    return (np.array(Image.open(path).convert("L")) > 127).astype(np.uint8)


def resolve_summary_asset_path(summary_path: Path, asset_path: str) -> Path:
    candidate = Path(asset_path)
    if candidate.is_absolute() and candidate.exists():
        return candidate
    project_candidate = PROJECT_ROOT / candidate
    if project_candidate.exists():
        return project_candidate
    fallback = summary_path.parent / candidate.parent.name / candidate.name
    if fallback.exists():
        return fallback
    return project_candidate


def resize_image_and_masks(
    image: np.ndarray,
    masks: List[np.ndarray],
    max_side: Optional[int],
) -> Tuple[np.ndarray, List[np.ndarray], float]:
    if max_side is None:
        return image, masks, 1.0

    image_h, image_w = image.shape[:2]
    current_max = max(image_h, image_w)
    if current_max <= max_side:
        return image, masks, 1.0

    scale = max_side / float(current_max)
    new_h = max(1, int(round(image_h * scale)))
    new_w = max(1, int(round(image_w * scale)))
    resized_image = np.array(
        Image.fromarray(image).resize((new_w, new_h), resample=Image.Resampling.BILINEAR)
    )
    resized_masks = [resize_mask(mask.astype(np.uint8), (new_h, new_w)).astype(np.uint8) for mask in masks]
    return resized_image, resized_masks, scale


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


def metrics_to_dict(pred: np.ndarray, gt: np.ndarray) -> Dict[str, float]:
    m = compute_all_metrics(pred, gt)
    return {
        "dice": float(m.dice),
        "iou": float(m.iou),
        "precision": float(m.precision),
        "recall": float(m.recall),
        "psnr": float(m.psnr),
        "ssim": float(m.ssim),
    }


def make_overlay(image: np.ndarray, mask: np.ndarray, color: Tuple[int, int, int], alpha: float = 0.45) -> np.ndarray:
    base = image.astype(np.float32).copy()
    mask_bool = mask.astype(bool)
    color_arr = np.array(color, dtype=np.float32)
    base[mask_bool] = (1.0 - alpha) * base[mask_bool] + alpha * color_arr
    return np.clip(base, 0, 255).astype(np.uint8)


def save_mask(mask: np.ndarray, path: Path) -> None:
    Image.fromarray(mask.astype(np.uint8) * 255).save(path)


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
                linewidth=0.7,
                alpha=0.35,
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
    parser = argparse.ArgumentParser(description="Run dense box-proposal TMA24 experiment.")
    parser.add_argument("--image-path", type=Path, default=DEFAULT_IMAGE_PATH)
    parser.add_argument("--summary-path", type=Path, default=DEFAULT_SUMMARY_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--device", choices=["cuda", "mps", "cpu"], default=None)
    parser.add_argument("--max-side", type=int, default=1024)
    parser.add_argument("--eval-source", choices=["disk", "region"], default="region")
    parser.add_argument("--box-size", type=int, default=256)
    parser.add_argument("--stride", type=int, default=128)
    parser.add_argument("--max-boxes", type=int, default=0)
    args = parser.parse_args()

    if not args.image_path.exists():
        raise FileNotFoundError(f"Image not found: {args.image_path}")
    if not args.summary_path.exists():
        raise FileNotFoundError(f"Pseudo-mask summary not found: {args.summary_path}")

    output_dir = args.output_dir
    masks_dir = output_dir / "masks"
    overlays_dir = output_dir / "overlays"
    panels_dir = output_dir / "panels"
    for path in (output_dir, masks_dir, overlays_dir, panels_dir):
        path.mkdir(parents=True, exist_ok=True)

    summary = json.loads(args.summary_path.read_text())
    image = np.array(Image.open(args.image_path).convert("RGB"))
    eval_key = "disk_mask_path" if args.eval_source == "disk" else "region_mask_path"
    label_records = summary["labels"]
    eval_masks = [load_binary_mask(resolve_summary_asset_path(args.summary_path, item[eval_key])) for item in label_records]
    max_side = None if args.max_side == 0 else args.max_side
    image, eval_masks, resize_scale = resize_image_and_masks(image, eval_masks, max_side)
    image_h, image_w = image.shape[:2]

    max_boxes = None if args.max_boxes <= 0 else args.max_boxes
    boxes = make_sliding_boxes((image_h, image_w), args.box_size, args.stride, max_boxes)

    print("=" * 60)
    print("TMA24 dense box-proposal experiment")
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
        pred = sam3.predict_box(inference_state, box, (image_h, image_w))
        if pred is None:
            pred = np.zeros((image_h, image_w), dtype=np.uint8)
        elif pred.shape != (image_h, image_w):
            pred = resize_mask(pred, (image_h, image_w)).astype(np.uint8)
        else:
            pred = pred.astype(np.uint8)

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
        "eval_source": args.eval_source,
        "box_size": args.box_size,
        "stride": args.stride,
        "n_boxes": int(len(boxes)),
        "n_nonempty_proposals": int(len(proposal_masks)),
        "union_positive_pixels": int(union_mask.sum()),
        "labels": [],
    }

    for idx, (item, gt_mask) in enumerate(zip(label_records, eval_masks)):
        label = item["label"]
        color = tuple(int(v) for v in item.get("color_rgb", [230, 57, 70]))
        best_mask = np.zeros_like(gt_mask, dtype=np.uint8)
        best_metrics = metrics_to_dict(best_mask, gt_mask)
        best_box = None
        best_proposal_index = None

        for proposal_index, (box, pred) in enumerate(zip(proposal_boxes, proposal_masks)):
            m = metrics_to_dict(pred, gt_mask)
            if m["dice"] > best_metrics["dice"]:
                best_metrics = m
                best_mask = pred
                best_box = box
                best_proposal_index = proposal_index

        union_metrics = metrics_to_dict(union_mask, gt_mask)
        stem = f"{idx + 1:02d}_{label.lower().replace(' ', '_')}"
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
