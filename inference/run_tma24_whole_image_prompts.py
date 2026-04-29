#!/usr/bin/env python3
"""
Run whole-image TMA24 prompt experiments.

This script tests whether SAM3 can recover each TMA24 pseudo-mask label on the
full image using:
- a whole-region box prompt
- a text prompt
- a joint box + text prompt
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
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "output" / "tma24_whole_image_prompts"


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


def expand_bbox(
    bbox: Tuple[int, int, int, int],
    image_shape: Tuple[int, int],
    margin_fraction: float,
) -> Tuple[int, int, int, int]:
    if margin_fraction <= 0:
        return bbox

    x_min, y_min, x_max, y_max = bbox
    height, width = image_shape
    box_w = x_max - x_min
    box_h = y_max - y_min
    dx = int(round(box_w * margin_fraction))
    dy = int(round(box_h * margin_fraction))
    return (
        max(0, x_min - dx),
        max(0, y_min - dy),
        min(width - 1, x_max + dx),
        min(height - 1, y_max + dy),
    )


def make_overlay(image: np.ndarray, mask: np.ndarray, color: Tuple[int, int, int], alpha: float = 0.45) -> np.ndarray:
    base = image.astype(np.float32).copy()
    mask_bool = mask.astype(bool)
    color_arr = np.array(color, dtype=np.float32)
    base[mask_bool] = (1.0 - alpha) * base[mask_bool] + alpha * color_arr
    return np.clip(base, 0, 255).astype(np.uint8)


def draw_box(image: np.ndarray, bbox: Tuple[int, int, int, int], color: Tuple[int, int, int]) -> np.ndarray:
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(image)
    x_min, y_min, x_max, y_max = bbox
    rect = patches.Rectangle(
        (x_min, y_min),
        x_max - x_min,
        y_max - y_min,
        linewidth=2.5,
        edgecolor=np.array(color) / 255.0,
        facecolor="none",
    )
    ax.add_patch(rect)
    ax.axis("off")
    fig.tight_layout(pad=0)
    fig.canvas.draw()
    boxed = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
    plt.close(fig)
    return boxed


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


def save_mask(mask: np.ndarray, path: Path) -> None:
    Image.fromarray(mask.astype(np.uint8) * 255).save(path)


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
    axes[1].set_title("Pseudo-mask target")
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
    parser = argparse.ArgumentParser(description="Run whole-image TMA24 text/box prompt experiments.")
    parser.add_argument("--image-path", type=Path, default=DEFAULT_IMAGE_PATH)
    parser.add_argument("--summary-path", type=Path, default=DEFAULT_SUMMARY_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--device", choices=["cuda", "mps", "cpu"], default=None)
    parser.add_argument("--label", type=str, default=None, help="Run one label only.")
    parser.add_argument("--max-side", type=int, default=2048, help="Resize longest side for inference. Use 0 to disable.")
    parser.add_argument("--box-source", choices=["disk", "region"], default="region", help="Pseudo-mask variant used to derive the full-image box.")
    parser.add_argument("--eval-source", choices=["disk", "region"], default="region", help="Pseudo-mask variant used as evaluation target.")
    parser.add_argument("--box-margin", type=float, default=0.04, help="Fractional margin added around the generated full-image box.")
    parser.add_argument("--text-template", type=str, default="{label}", help="Text prompt template. Use {label} as the label placeholder.")
    args = parser.parse_args()

    if not args.image_path.exists():
        raise FileNotFoundError(f"Image not found: {args.image_path}")
    if not args.summary_path.exists():
        raise FileNotFoundError(f"Pseudo-mask summary not found: {args.summary_path}")

    output_dir = args.output_dir
    if args.label:
        output_dir = output_dir.parent / f"{output_dir.name}_{args.label.lower().replace(' ', '_')}"
    masks_dir = output_dir / "masks"
    overlays_dir = output_dir / "overlays"
    panels_dir = output_dir / "panels"
    for path in (output_dir, masks_dir, overlays_dir, panels_dir):
        path.mkdir(parents=True, exist_ok=True)

    summary = json.loads(args.summary_path.read_text())
    image = np.array(Image.open(args.image_path).convert("RGB"))

    selected_label = args.label.strip() if args.label else None
    records = []
    for idx, item in enumerate(summary["labels"]):
        if selected_label and item["label"] != selected_label:
            continue
        records.append((idx, item))

    if selected_label and not records:
        available = ", ".join(item["label"] for item in summary["labels"])
        raise ValueError(f"Label not found: {selected_label}. Available labels: {available}")

    eval_key = "disk_mask_path" if args.eval_source == "disk" else "region_mask_path"
    box_key = "disk_mask_path" if args.box_source == "disk" else "region_mask_path"
    eval_masks = [load_binary_mask(resolve_summary_asset_path(args.summary_path, item[eval_key])) for _, item in records]
    box_masks = [load_binary_mask(resolve_summary_asset_path(args.summary_path, item[box_key])) for _, item in records]

    max_side = None if args.max_side == 0 else args.max_side
    image, resized_masks, resize_scale = resize_image_and_masks(image, eval_masks + box_masks, max_side)
    eval_masks = resized_masks[: len(eval_masks)]
    box_masks = resized_masks[len(eval_masks):]
    image_h, image_w = image.shape[:2]

    experiment = {
        "image_path": str(args.image_path),
        "summary_path": str(args.summary_path),
        "checkpoint": args.checkpoint,
        "device": args.device,
        "max_side": max_side,
        "resize_scale": resize_scale,
        "image_shape": [int(image_h), int(image_w)],
        "box_source": args.box_source,
        "eval_source": args.eval_source,
        "box_margin": args.box_margin,
        "text_template": args.text_template,
        "labels": [],
    }

    print("=" * 60)
    print("TMA24 whole-image prompt experiment")
    print("=" * 60)
    print(f"Image: {args.image_path}")
    print(f"Summary: {args.summary_path}")
    print(f"Checkpoint: {args.checkpoint or 'default SAM3 from Hugging Face'}")
    print(f"Device: {args.device or 'auto'}")
    print(f"Output: {output_dir}")
    print(f"Image shape for inference: {image_h} x {image_w}")
    print(f"Box source: {args.box_source}; eval source: {args.eval_source}")

    sam3 = SAM3Model(confidence_threshold=0.1, checkpoint_path=args.checkpoint, device=args.device)
    inference_state = sam3.encode_image(image)

    for (idx, item), gt_mask, box_mask in zip(records, eval_masks, box_masks):
        label = item["label"]
        text_prompt = args.text_template.format(label=label)
        color = tuple(int(v) for v in item.get("color_rgb", [230, 57, 70]))
        stem = f"{idx + 1:02d}_{label.lower().replace(' ', '_')}"

        bbox = generate_bbox_from_mask(box_mask)
        if bbox is None:
            print(f"\nLabel: {label}")
            print("  Skipped: no positive pixels in box mask.")
            continue
        bbox = expand_bbox(bbox, gt_mask.shape, args.box_margin)

        pred_box = np.zeros_like(gt_mask, dtype=np.uint8)
        pred_box_raw = sam3.predict_box(inference_state, bbox, gt_mask.shape)
        if pred_box_raw is not None:
            if pred_box_raw.shape != gt_mask.shape:
                pred_box_raw = resize_mask(pred_box_raw, gt_mask.shape)
            pred_box = pred_box_raw.astype(np.uint8)

        pred_text = np.zeros_like(gt_mask, dtype=np.uint8)
        pred_text_raw = sam3.predict_text(inference_state, text_prompt)
        if pred_text_raw is not None:
            if pred_text_raw.shape != gt_mask.shape:
                pred_text_raw = resize_mask(pred_text_raw, gt_mask.shape)
            pred_text = pred_text_raw.astype(np.uint8)

        pred_joint = np.zeros_like(gt_mask, dtype=np.uint8)
        pred_joint_raw = sam3.predict_box_text(inference_state, bbox, text_prompt, gt_mask.shape)
        if pred_joint_raw is not None:
            if pred_joint_raw.shape != gt_mask.shape:
                pred_joint_raw = resize_mask(pred_joint_raw, gt_mask.shape)
            pred_joint = pred_joint_raw.astype(np.uint8)

        metrics_box = metrics_to_dict(pred_box, gt_mask)
        metrics_text = metrics_to_dict(pred_text, gt_mask)
        metrics_joint = metrics_to_dict(pred_joint, gt_mask)

        save_mask(gt_mask, masks_dir / f"{stem}_target_{args.eval_source}.png")
        save_mask(pred_box, masks_dir / f"{stem}_pred_box.png")
        save_mask(pred_text, masks_dir / f"{stem}_pred_text.png")
        save_mask(pred_joint, masks_dir / f"{stem}_pred_box_text.png")
        Image.fromarray(make_overlay(image, gt_mask, color)).save(overlays_dir / f"{stem}_target_{args.eval_source}.png")
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
    print("\nSaved outputs to:", output_dir)


if __name__ == "__main__":
    main()
