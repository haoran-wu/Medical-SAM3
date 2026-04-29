#!/usr/bin/env python3
"""
Run whole-region SAM3 prompt baselines on Visium HD Exp1 region masks.

For each annotated region label, this script evaluates:
- a full-region-derived box prompt
- a text prompt
- a joint box + text prompt
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, Tuple

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

from sam3_inference import SAM3Model, generate_bbox_from_mask
from sam3_baseline_utils import (
    DATA_DIR,
    ensure_dirs,
    expand_bbox,
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
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "output" / "visium_hd_exp1" / "sam3_runs" / "whole_region_prompts"


def save_panel(
    image: np.ndarray,
    label: str,
    text_prompt: str,
    bbox: Tuple[int, int, int, int],
    gt_mask: np.ndarray,
    pred_box: np.ndarray,
    pred_text: np.ndarray,
    pred_joint: np.ndarray,
    metrics_box: Dict[str, float],
    metrics_text: Dict[str, float],
    metrics_joint: Dict[str, float],
    color: Tuple[int, int, int],
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()

    axes[0].imshow(image)
    x_min, y_min, x_max, y_max = bbox
    axes[0].add_patch(
        patches.Rectangle(
            (x_min, y_min),
            x_max - x_min,
            y_max - y_min,
            linewidth=2.5,
            edgecolor=np.array(color) / 255.0,
            facecolor="none",
        )
    )
    axes[0].set_title(f"{label}\nBox prompt + text: {text_prompt}")
    axes[0].axis("off")

    axes[1].imshow(make_overlay(image, gt_mask, color))
    axes[1].set_title("Region mask target")
    axes[1].axis("off")

    axes[2].imshow(make_overlay(image, pred_box, color))
    axes[2].set_title(
        f"Box-only prediction\nDice={metrics_box['dice']:.3f}, IoU={metrics_box['iou']:.3f}, Recall={metrics_box['recall']:.3f}"
    )
    axes[2].axis("off")

    axes[3].imshow(make_overlay(image, pred_text, color))
    axes[3].set_title(
        f"Text-only prediction\nDice={metrics_text['dice']:.3f}, IoU={metrics_text['iou']:.3f}, Recall={metrics_text['recall']:.3f}"
    )
    axes[3].axis("off")

    axes[4].imshow(make_overlay(image, pred_joint, color))
    axes[4].set_title(
        f"Box + text prediction\nDice={metrics_joint['dice']:.3f}, IoU={metrics_joint['iou']:.3f}, Recall={metrics_joint['recall']:.3f}"
    )
    axes[4].axis("off")

    comparison = np.zeros_like(gt_mask, dtype=np.uint8)
    comparison[gt_mask.astype(bool)] = 1
    comparison[pred_box.astype(bool)] = 2
    comparison[pred_text.astype(bool)] = 3
    comparison[pred_joint.astype(bool)] = 4
    axes[5].imshow(comparison, cmap="viridis")
    axes[5].set_title("GT vs predictions\n1=GT 2=Box 3=Text 4=Joint")
    axes[5].axis("off")

    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run whole-region SAM3 baselines on Visium HD Exp1.")
    parser.add_argument("--image-path", type=Path, default=DEFAULT_IMAGE_PATH)
    parser.add_argument("--summary-path", type=Path, default=DEFAULT_REGION_SUMMARY)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--device", choices=["cuda", "mps", "cpu"], default=None)
    parser.add_argument("--labels", type=str, default=None, help="Comma-separated region labels to include.")
    parser.add_argument("--max-side", type=int, default=2048, help="Resize longest side for inference. Use 0 to disable.")
    parser.add_argument("--box-margin", type=float, default=0.04, help="Fractional margin added around the full-region box.")
    parser.add_argument("--text-template", type=str, default="{label}", help="Text prompt template. Use {label} as placeholder.")
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

    experiment = {
        "image_path": str(args.image_path),
        "summary_path": str(args.summary_path),
        "checkpoint": args.checkpoint,
        "device": args.device,
        "max_side": max_side,
        "resize_scale": resize_scale,
        "image_shape": [int(image_h), int(image_w)],
        "box_margin": args.box_margin,
        "text_template": args.text_template,
        "labels": [],
    }

    print("=" * 60)
    print("Visium HD Exp1 whole-region SAM3 baseline")
    print("=" * 60)
    print(f"Image: {args.image_path}")
    print(f"Summary: {args.summary_path}")
    print(f"Checkpoint: {args.checkpoint or 'default SAM3 from Hugging Face'}")
    print(f"Device: {args.device or 'auto'}")
    print(f"Output: {output_dir}")
    print(f"Image shape for inference: {image_h} x {image_w}")
    print(f"Labels: {len(records)}")

    sam3 = SAM3Model(confidence_threshold=0.1, checkpoint_path=args.checkpoint, device=args.device)
    inference_state = sam3.encode_image(image)

    for (idx, item), gt_mask in zip(records, masks):
        label = str(item["label"])
        color = tuple(int(v) for v in item.get("color_rgb", [230, 57, 70]))
        text_prompt = args.text_template.format(label=label)
        stem = f"{idx + 1:02d}_{slugify(label)}"

        bbox = generate_bbox_from_mask(gt_mask)
        if bbox is None:
            print(f"\nLabel: {label}")
            print("  Skipped: no positive pixels in region mask.")
            continue
        bbox = expand_bbox(bbox, gt_mask.shape, args.box_margin)

        pred_box = normalize_prediction(sam3.predict_box(inference_state, bbox, gt_mask.shape), gt_mask.shape)
        pred_text = normalize_prediction(sam3.predict_text(inference_state, text_prompt), gt_mask.shape)
        pred_joint = normalize_prediction(sam3.predict_box_text(inference_state, bbox, text_prompt, gt_mask.shape), gt_mask.shape)

        metrics_box = metrics_to_dict(pred_box, gt_mask)
        metrics_text = metrics_to_dict(pred_text, gt_mask)
        metrics_joint = metrics_to_dict(pred_joint, gt_mask)

        save_mask(gt_mask, masks_dir / f"{stem}_target.png")
        save_mask(pred_box, masks_dir / f"{stem}_pred_box.png")
        save_mask(pred_text, masks_dir / f"{stem}_pred_text.png")
        save_mask(pred_joint, masks_dir / f"{stem}_pred_box_text.png")
        Image.fromarray(make_overlay(image, gt_mask, color)).save(overlays_dir / f"{stem}_target.png")
        Image.fromarray(make_overlay(image, pred_box, color)).save(overlays_dir / f"{stem}_pred_box.png")
        Image.fromarray(make_overlay(image, pred_text, color)).save(overlays_dir / f"{stem}_pred_text.png")
        Image.fromarray(make_overlay(image, pred_joint, color)).save(overlays_dir / f"{stem}_pred_box_text.png")

        save_panel(
            image=image,
            label=label,
            text_prompt=text_prompt,
            bbox=bbox,
            gt_mask=gt_mask,
            pred_box=pred_box,
            pred_text=pred_text,
            pred_joint=pred_joint,
            metrics_box=metrics_box,
            metrics_text=metrics_text,
            metrics_joint=metrics_joint,
            color=color,
            output_path=panels_dir / f"{stem}.png",
        )

        record = {
            "label": label,
            "slug": str(item.get("slug", slugify(label))),
            "text_prompt": text_prompt,
            "box_xyxy": [int(v) for v in bbox],
            "target_positive_pixels": int(gt_mask.sum()),
            "box_prediction_positive_pixels": int(pred_box.sum()),
            "text_prediction_positive_pixels": int(pred_text.sum()),
            "box_text_prediction_positive_pixels": int(pred_joint.sum()),
            "box_prompt_metrics": metrics_box,
            "text_prompt_metrics": metrics_text,
            "box_text_prompt_metrics": metrics_joint,
        }
        experiment["labels"].append(record)

        print(f"\nLabel: {label}")
        print(f"  Text: {text_prompt}")
        print(f"  Box: {record['box_xyxy']}")
        print(
            f"  Box-only: Dice={metrics_box['dice']:.3f}, IoU={metrics_box['iou']:.3f}, "
            f"Recall={metrics_box['recall']:.3f}, pixels={record['box_prediction_positive_pixels']}"
        )
        print(
            f"  Text-only: Dice={metrics_text['dice']:.3f}, IoU={metrics_text['iou']:.3f}, "
            f"Recall={metrics_text['recall']:.3f}, pixels={record['text_prediction_positive_pixels']}"
        )
        print(
            f"  Box+text: Dice={metrics_joint['dice']:.3f}, IoU={metrics_joint['iou']:.3f}, "
            f"Recall={metrics_joint['recall']:.3f}, pixels={record['box_text_prediction_positive_pixels']}"
        )

    (output_dir / "experiment_summary.json").write_text(json.dumps(experiment, indent=2))
    print(f"\nSaved outputs to: {output_dir}")


if __name__ == "__main__":
    main()
