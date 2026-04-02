#!/usr/bin/env python3
"""
Run a left-to-right transfer experiment on TMA24 pseudo-masks.

Workflow:
- take the final pseudo-mask for each label
- split it into left-half prompt region and right-half evaluation region
- derive a bounding box from the left-half mask
- run SAM3 with:
  1. left-half box prompt
  2. text-only prompt
  3. left-half box + text joint prompt
- evaluate predictions only on the right half of the image

This provides a minimal test of whether information from the left side can help
recover similar structures on the right side.
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
from sam3_inference import SAM3Model, generate_bbox_from_mask, resize_mask


DEFAULT_IMAGE_PATH = PROJECT_ROOT / "example1.jpg"
DEFAULT_PSEUDOMASK_DIR = PROJECT_ROOT / "output" / "00_FINAL_tma24_example1_scale_0p55_shiftX_neg120_shiftY_620"
DEFAULT_SUMMARY_PATH = DEFAULT_PSEUDOMASK_DIR / "summary.json"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "output" / "tma24_left_to_right_transfer"

COLORS: List[Tuple[float, float, float]] = [
    (230 / 255.0, 57 / 255.0, 70 / 255.0),
    (42 / 255.0, 157 / 255.0, 143 / 255.0),
    (106 / 255.0, 76 / 255.0, 147 / 255.0),
]


def load_binary_mask(path: Path) -> np.ndarray:
    return (np.array(Image.open(path).convert("L")) > 127).astype(np.uint8)


def split_mask_left_right(mask: np.ndarray, split_x: int) -> Tuple[np.ndarray, np.ndarray]:
    left = np.zeros_like(mask, dtype=np.uint8)
    right = np.zeros_like(mask, dtype=np.uint8)
    left[:, :split_x] = mask[:, :split_x]
    right[:, split_x:] = mask[:, split_x:]
    return left, right


def mask_right_half(mask: np.ndarray, split_x: int) -> np.ndarray:
    out = np.zeros_like(mask, dtype=np.uint8)
    out[:, split_x:] = mask[:, split_x:]
    return out


def make_overlay(image: np.ndarray, mask: np.ndarray, color: Tuple[float, float, float], alpha: float = 0.45) -> np.ndarray:
    base = image.astype(np.float32).copy()
    mask_bool = mask.astype(bool)
    color_arr = np.array(color, dtype=np.float32) * 255.0
    base[mask_bool] = (1.0 - alpha) * base[mask_bool] + alpha * color_arr
    return np.clip(base, 0, 255).astype(np.uint8)


def save_mask(mask: np.ndarray, path: Path) -> None:
    Image.fromarray(mask.astype(np.uint8) * 255).save(path)


def save_overlay(overlay: np.ndarray, path: Path) -> None:
    Image.fromarray(overlay).save(path)


def save_panel(
    image: np.ndarray,
    label: str,
    left_mask: np.ndarray,
    right_mask: np.ndarray,
    bbox: Optional[Tuple[int, int, int, int]],
    pred_box_right: np.ndarray,
    pred_text_right: np.ndarray,
    pred_joint_full: np.ndarray,
    pred_joint_right: np.ndarray,
    box_metrics: Dict[str, float],
    text_metrics: Dict[str, float],
    joint_metrics: Dict[str, float],
    output_path: Path,
    color: Tuple[float, float, float],
) -> None:
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.flatten()

    axes[0].imshow(image)
    if bbox is not None:
        x_min, y_min, x_max, y_max = bbox
        rect = patches.Rectangle(
            (x_min, y_min),
            x_max - x_min,
            y_max - y_min,
            linewidth=2.5,
            edgecolor=(1.0, 0.2, 0.2),
            facecolor="none",
        )
        axes[0].add_patch(rect)
    axes[0].set_title(f"{label}\nLeft-half box prompt")
    axes[0].axis("off")

    axes[1].imshow(left_mask, cmap="gray")
    axes[1].set_title("Left pseudo-mask")
    axes[1].axis("off")

    axes[2].imshow(right_mask, cmap="gray")
    axes[2].set_title("Right pseudo-mask (eval GT)")
    axes[2].axis("off")

    axes[3].imshow(make_overlay(image, pred_box_right, color))
    axes[3].set_title(
        "Box prompt prediction (right half)\n"
        f"Dice={box_metrics['dice']:.3f}, IoU={box_metrics['iou']:.3f}, Recall={box_metrics['recall']:.3f}"
    )
    axes[3].axis("off")

    axes[4].imshow(make_overlay(image, pred_text_right, color))
    axes[4].set_title(
        "Text prompt prediction (right half)\n"
        f"Dice={text_metrics['dice']:.3f}, IoU={text_metrics['iou']:.3f}, Recall={text_metrics['recall']:.3f}"
    )
    axes[4].axis("off")

    axes[5].imshow(make_overlay(image, pred_joint_full, color))
    axes[5].set_title("Joint box+text prediction\n(full image overlay)")
    axes[5].axis("off")

    combined = np.zeros_like(right_mask, dtype=np.uint8)
    combined[right_mask.astype(bool)] = 1
    combined[pred_box_right.astype(bool)] = 2
    combined[pred_text_right.astype(bool)] = 3
    combined[pred_joint_right.astype(bool)] = 4
    axes[6].imshow(make_overlay(image, pred_joint_right, color))
    axes[6].set_title(
        "Joint box+text (right half)\n"
        f"Dice={joint_metrics['dice']:.3f}, IoU={joint_metrics['iou']:.3f}, Recall={joint_metrics['recall']:.3f}"
    )
    axes[6].axis("off")

    axes[7].imshow(combined, cmap="viridis")
    axes[7].set_title("Right GT vs predictions\n1=GT 2=Box 3=Text 4=Joint")
    axes[7].axis("off")

    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def metrics_to_dict(pred: np.ndarray, gt: np.ndarray) -> Dict[str, float]:
    m = compute_all_metrics(pred, gt)
    return {
        "dice": float(m.dice),
        "iou": float(m.iou),
        "psnr": float(m.psnr),
        "ssim": float(m.ssim),
        "precision": float(m.precision),
        "recall": float(m.recall),
    }


def serialize_bbox(bbox: Optional[Tuple[int, int, int, int]]) -> Optional[List[int]]:
    if bbox is None:
        return None
    return [int(v) for v in bbox]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run left-to-right transfer experiment using TMA24 pseudo-masks.")
    parser.add_argument("--image-path", type=Path, default=DEFAULT_IMAGE_PATH)
    parser.add_argument("--summary-path", type=Path, default=DEFAULT_SUMMARY_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--split-fraction", type=float, default=0.5, help="Vertical split position as fraction of image width.")
    parser.add_argument("--label", type=str, default=None, help="Run only a single label from the pseudo-mask summary.")
    args = parser.parse_args()

    if not args.image_path.exists():
        raise FileNotFoundError(f"Image not found: {args.image_path}")
    if not args.summary_path.exists():
        raise FileNotFoundError(f"Pseudo-mask summary not found: {args.summary_path}")

    summary = json.loads(args.summary_path.read_text())
    image = np.array(Image.open(args.image_path).convert("RGB"))
    image_h, image_w = image.shape[:2]
    split_x = int(round(image_w * args.split_fraction))

    selected_label = args.label.strip() if args.label else None

    output_dir = args.output_dir
    if selected_label:
        label_stem = selected_label.lower().replace(" ", "_")
        output_dir = output_dir.parent / f"{output_dir.name}_{label_stem}"
    masks_dir = output_dir / "masks"
    overlays_dir = output_dir / "overlays"
    panels_dir = output_dir / "panels"
    output_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)
    overlays_dir.mkdir(parents=True, exist_ok=True)
    panels_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("TMA24 left-to-right transfer experiment")
    print("=" * 60)
    print(f"Image: {args.image_path}")
    print(f"Pseudo-mask summary: {args.summary_path}")
    print(f"Checkpoint: {args.checkpoint or 'default SAM3 from Hugging Face'}")
    print(f"Split x: {split_x} / {image_w}")
    if selected_label:
        print(f"Selected label: {selected_label}")

    sam3 = SAM3Model(confidence_threshold=0.1, checkpoint_path=args.checkpoint)
    inference_state = sam3.encode_image(image)

    experiment = {
        "image_path": str(args.image_path),
        "summary_path": str(args.summary_path),
        "checkpoint": args.checkpoint,
        "split_fraction": args.split_fraction,
        "split_x": split_x,
        "labels": [],
    }

    matched_any = False

    for idx, item in enumerate(summary["labels"]):
        label = item["label"]
        if selected_label and label != selected_label:
            continue

        matched_any = True
        color = COLORS[idx % len(COLORS)]
        region_mask_path = PROJECT_ROOT / item["region_mask_path"]
        gt_mask = load_binary_mask(region_mask_path)
        left_mask, right_mask = split_mask_left_right(gt_mask, split_x)
        bbox = generate_bbox_from_mask(left_mask)

        pred_box = np.zeros_like(gt_mask, dtype=np.uint8)
        if bbox is not None:
            pred_box_raw = sam3.predict_box(inference_state, bbox, gt_mask.shape)
            if pred_box_raw is not None:
                if pred_box_raw.shape != gt_mask.shape:
                    pred_box_raw = resize_mask(pred_box_raw, gt_mask.shape)
                pred_box = pred_box_raw.astype(np.uint8)

        pred_text = np.zeros_like(gt_mask, dtype=np.uint8)
        pred_text_raw = sam3.predict_text(inference_state, label)
        if pred_text_raw is not None:
            if pred_text_raw.shape != gt_mask.shape:
                pred_text_raw = resize_mask(pred_text_raw, gt_mask.shape)
            pred_text = pred_text_raw.astype(np.uint8)

        pred_joint = np.zeros_like(gt_mask, dtype=np.uint8)
        if bbox is not None:
            pred_joint_raw = sam3.predict_box_text(inference_state, bbox, label, gt_mask.shape)
            if pred_joint_raw is not None:
                if pred_joint_raw.shape != gt_mask.shape:
                    pred_joint_raw = resize_mask(pred_joint_raw, gt_mask.shape)
                pred_joint = pred_joint_raw.astype(np.uint8)

        pred_box_right = mask_right_half(pred_box, split_x)
        pred_text_right = mask_right_half(pred_text, split_x)
        pred_joint_right = mask_right_half(pred_joint, split_x)

        box_metrics = metrics_to_dict(pred_box_right, right_mask)
        text_metrics = metrics_to_dict(pred_text_right, right_mask)
        joint_metrics = metrics_to_dict(pred_joint_right, right_mask)

        stem = f"{idx + 1:02d}_{label.lower().replace(' ', '_')}"

        save_mask(left_mask, masks_dir / f"{stem}_left_prompt_mask.png")
        save_mask(right_mask, masks_dir / f"{stem}_right_eval_mask.png")
        save_mask(pred_box_right, masks_dir / f"{stem}_pred_box_right.png")
        save_mask(pred_text_right, masks_dir / f"{stem}_pred_text_right.png")
        save_mask(pred_joint, masks_dir / f"{stem}_pred_joint_full.png")
        save_mask(pred_joint_right, masks_dir / f"{stem}_pred_joint_right.png")
        save_overlay(make_overlay(image, pred_box_right, color), overlays_dir / f"{stem}_pred_box_right.png")
        save_overlay(make_overlay(image, pred_text_right, color), overlays_dir / f"{stem}_pred_text_right.png")
        save_overlay(make_overlay(image, pred_joint, color), overlays_dir / f"{stem}_pred_joint_full.png")
        save_overlay(make_overlay(image, pred_joint_right, color), overlays_dir / f"{stem}_pred_joint_right.png")
        save_panel(
            image=image,
            label=label,
            left_mask=left_mask,
            right_mask=right_mask,
            bbox=bbox,
            pred_box_right=pred_box_right,
            pred_text_right=pred_text_right,
            pred_joint_full=pred_joint,
            pred_joint_right=pred_joint_right,
            box_metrics=box_metrics,
            text_metrics=text_metrics,
            joint_metrics=joint_metrics,
            output_path=panels_dir / f"{stem}.png",
            color=color,
        )

        experiment["labels"].append(
            {
                "label": label,
                "left_bbox_xyxy": serialize_bbox(bbox),
                "left_positive_pixels": int(left_mask.sum()),
                "right_positive_pixels": int(right_mask.sum()),
                "box_prompt_metrics_right_half": box_metrics,
                "text_prompt_metrics_right_half": text_metrics,
                "joint_box_text_metrics_right_half": joint_metrics,
            }
        )

        print(f"\nLabel: {label}")
        print(f"  Left bbox: {bbox}")
        print(
            f"  Right-half box prompt: Dice={box_metrics['dice']:.3f}, "
            f"IoU={box_metrics['iou']:.3f}, Recall={box_metrics['recall']:.3f}"
        )
        print(
            f"  Right-half text prompt: Dice={text_metrics['dice']:.3f}, "
            f"IoU={text_metrics['iou']:.3f}, Recall={text_metrics['recall']:.3f}"
        )
        print(
            f"  Right-half joint box+text: Dice={joint_metrics['dice']:.3f}, "
            f"IoU={joint_metrics['iou']:.3f}, Recall={joint_metrics['recall']:.3f}"
        )

    if selected_label and not matched_any:
        available = ", ".join(item["label"] for item in summary["labels"])
        raise ValueError(f"Label not found in summary: {selected_label}. Available labels: {available}")

    (output_dir / "experiment_summary.json").write_text(json.dumps(experiment, indent=2))
    print("\nSaved outputs to:", output_dir)


if __name__ == "__main__":
    main()
