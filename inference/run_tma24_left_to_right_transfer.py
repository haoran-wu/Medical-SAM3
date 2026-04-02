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
from sam3_inference import SAM3Model, resize_mask


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
) -> Tuple[np.ndarray, List[np.ndarray]]:
    if max_side is None:
        return image, masks

    image_h, image_w = image.shape[:2]
    current_max = max(image_h, image_w)
    if current_max <= max_side:
        return image, masks

    scale = max_side / float(current_max)
    new_h = max(1, int(round(image_h * scale)))
    new_w = max(1, int(round(image_w * scale)))
    resized_image = np.array(
        Image.fromarray(image).resize((new_w, new_h), resample=Image.Resampling.BILINEAR)
    )
    resized_masks = [resize_mask(mask.astype(np.uint8), (new_h, new_w)).astype(np.uint8) for mask in masks]
    return resized_image, resized_masks


def split_mask_left_right(mask: np.ndarray, split_x: int) -> Tuple[np.ndarray, np.ndarray]:
    left = np.zeros_like(mask, dtype=np.uint8)
    right = np.zeros_like(mask, dtype=np.uint8)
    left[:, :split_x] = mask[:, :split_x]
    right[:, split_x:] = mask[:, split_x:]
    return left, right


def generate_component_bboxes_from_mask(
    mask: np.ndarray,
    min_component_pixels: int = 250,
) -> List[Tuple[int, int, int, int]]:
    """
    Extract one bbox per connected component from a binary prompt mask.
    """
    height, width = mask.shape
    visited = np.zeros_like(mask, dtype=bool)
    boxes: List[Tuple[int, int, int, int, int]] = []

    for y in range(height):
        for x in range(width):
            if not mask[y, x] or visited[y, x]:
                continue

            stack = [(y, x)]
            visited[y, x] = True
            area = 0
            x_min = x_max = x
            y_min = y_max = y

            while stack:
                cy, cx = stack.pop()
                area += 1
                x_min = min(x_min, cx)
                x_max = max(x_max, cx)
                y_min = min(y_min, cy)
                y_max = max(y_max, cy)

                for ny, nx in ((cy - 1, cx), (cy + 1, cx), (cy, cx - 1), (cy, cx + 1)):
                    if 0 <= ny < height and 0 <= nx < width and mask[ny, nx] and not visited[ny, nx]:
                        visited[ny, nx] = True
                        stack.append((ny, nx))

            if area >= min_component_pixels:
                boxes.append((area, x_min, y_min, x_max, y_max))

    boxes.sort(reverse=True)
    return [(x_min, y_min, x_max, y_max) for _, x_min, y_min, x_max, y_max in boxes]


def sample_points_from_mask(
    mask: np.ndarray,
    max_points: int = 32,
) -> Tuple[List[Tuple[int, int]], List[int]]:
    """
    Sample a deterministic subset of positive points from a prompt mask.
    """
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return [], []

    coords = np.stack([xs, ys], axis=1)
    if len(coords) > max_points:
        rng = np.random.default_rng(0)
        keep = np.sort(rng.choice(len(coords), size=max_points, replace=False))
        coords = coords[keep]

    points = [(int(x), int(y)) for x, y in coords]
    labels = [1] * len(points)
    return points, labels


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
    bboxes: List[Tuple[int, int, int, int]],
    point_coords: List[Tuple[int, int]],
    prompt_kind: str,
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
    if prompt_kind == "box":
        for x_min, y_min, x_max, y_max in bboxes:
            rect = patches.Rectangle(
                (x_min, y_min),
                x_max - x_min,
                y_max - y_min,
                linewidth=2.5,
                edgecolor=(1.0, 0.2, 0.2),
                facecolor="none",
            )
            axes[0].add_patch(rect)
        axes[0].set_title(f"{label}\nLeft-half multi-box prompt ({len(bboxes)} boxes)")
    else:
        if point_coords:
            xs = [x for x, _ in point_coords]
            ys = [y for _, y in point_coords]
            axes[0].scatter(xs, ys, s=22, c=[(1.0, 0.2, 0.2)], edgecolors="white", linewidths=0.4)
        axes[0].set_title(f"{label}\nLeft-half dense-point prompt ({len(point_coords)} points)")
    axes[0].axis("off")

    axes[1].imshow(left_mask, cmap="gray")
    axes[1].set_title("Left pseudo-mask")
    axes[1].axis("off")

    axes[2].imshow(right_mask, cmap="gray")
    axes[2].set_title("Right pseudo-mask (eval GT)")
    axes[2].axis("off")

    axes[3].imshow(make_overlay(image, pred_box_right, color))
    axes[3].set_title(
        f"{prompt_kind.capitalize()} prompt prediction (right half)\n"
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


def serialize_bboxes(bboxes: List[Tuple[int, int, int, int]]) -> List[List[int]]:
    return [[int(v) for v in bbox] for bbox in bboxes]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run left-to-right transfer experiment using TMA24 pseudo-masks.")
    parser.add_argument("--image-path", type=Path, default=DEFAULT_IMAGE_PATH)
    parser.add_argument("--summary-path", type=Path, default=DEFAULT_SUMMARY_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--split-fraction", type=float, default=0.5, help="Vertical split position as fraction of image width.")
    parser.add_argument("--label", type=str, default=None, help="Run only a single label from the pseudo-mask summary.")
    parser.add_argument("--max-side", type=int, default=None, help="Resize image and pseudo-masks so the longest side is at most this many pixels.")
    parser.add_argument("--prompt-source", choices=["disk", "region"], default="disk", help="Which pseudo-mask variant to use when deriving left-side prompt boxes.")
    parser.add_argument("--component-min-pixels", type=int, default=250, help="Minimum connected-component area kept when converting the left prompt mask into multiple boxes.")
    parser.add_argument("--prompt-kind", choices=["box", "points"], default="box", help="Prompt geometry to derive from the left-half pseudo-mask.")
    parser.add_argument("--max-points", type=int, default=32, help="Maximum number of positive prompt points when using --prompt-kind points.")
    args = parser.parse_args()

    if not args.image_path.exists():
        raise FileNotFoundError(f"Image not found: {args.image_path}")
    if not args.summary_path.exists():
        raise FileNotFoundError(f"Pseudo-mask summary not found: {args.summary_path}")

    summary = json.loads(args.summary_path.read_text())
    image = np.array(Image.open(args.image_path).convert("RGB"))

    selected_label = args.label.strip() if args.label else None

    output_dir = args.output_dir
    if selected_label:
        label_stem = selected_label.lower().replace(" ", "_")
        output_dir = output_dir.parent / f"{output_dir.name}_{label_stem}"
    if args.prompt_kind != "box":
        output_dir = output_dir.parent / f"{output_dir.name}_{args.prompt_kind}"
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
    if selected_label:
        print(f"Selected label: {selected_label}")
    if args.max_side:
        print(f"Max side resize: {args.max_side}")
    print(f"Prompt source: {args.prompt_source}")
    print(f"Min component pixels: {args.component_min_pixels}")
    print(f"Prompt kind: {args.prompt_kind}")
    if args.prompt_kind == "points":
        print(f"Max points: {args.max_points}")

    experiment = {
        "image_path": str(args.image_path),
        "summary_path": str(args.summary_path),
        "checkpoint": args.checkpoint,
        "split_fraction": args.split_fraction,
        "max_side": args.max_side,
        "prompt_source": args.prompt_source,
        "component_min_pixels": args.component_min_pixels,
        "prompt_kind": args.prompt_kind,
        "max_points": args.max_points,
        "labels": [],
    }

    matched_any = False
    label_records: List[Tuple[int, Dict[str, object]]] = []

    for idx, item in enumerate(summary["labels"]):
        label = item["label"]
        if selected_label and label != selected_label:
            continue

        matched_any = True
        label_records.append((idx, item))

    if selected_label and not matched_any:
        available = ", ".join(item["label"] for item in summary["labels"])
        raise ValueError(f"Label not found in summary: {selected_label}. Available labels: {available}")

    eval_masks: List[np.ndarray] = []
    prompt_masks: List[np.ndarray] = []
    for _, item in label_records:
        region_mask_path = resolve_summary_asset_path(args.summary_path, item["region_mask_path"])
        prompt_path_key = "disk_mask_path" if args.prompt_source == "disk" else "region_mask_path"
        prompt_mask_path = resolve_summary_asset_path(args.summary_path, item[prompt_path_key])
        eval_masks.append(load_binary_mask(region_mask_path))
        prompt_masks.append(load_binary_mask(prompt_mask_path))

    image, resized_masks = resize_image_and_masks(image, eval_masks + prompt_masks, args.max_side)
    eval_masks = resized_masks[: len(eval_masks)]
    prompt_masks = resized_masks[len(eval_masks):]
    image_h, image_w = image.shape[:2]
    split_x = int(round(image_w * args.split_fraction))
    experiment["split_x"] = split_x
    experiment["image_shape"] = [int(image_h), int(image_w)]
    print(f"Split x: {split_x} / {image_w}")

    sam3 = SAM3Model(confidence_threshold=0.1, checkpoint_path=args.checkpoint)
    inference_state = sam3.encode_image(image)

    for (idx, item), gt_mask, prompt_mask in zip(label_records, eval_masks, prompt_masks):
        label = item["label"]
        color = COLORS[idx % len(COLORS)]
        left_prompt_mask, _ = split_mask_left_right(prompt_mask, split_x)
        _, right_mask = split_mask_left_right(gt_mask, split_x)
        bboxes = generate_component_bboxes_from_mask(left_prompt_mask, min_component_pixels=args.component_min_pixels)
        point_coords, point_labels = sample_points_from_mask(left_prompt_mask, max_points=args.max_points)

        pred_box = np.zeros_like(gt_mask, dtype=np.uint8)
        if args.prompt_kind == "box" and bboxes:
            pred_box_raw = sam3.predict_boxes(inference_state, bboxes, gt_mask.shape)
            if pred_box_raw is not None:
                if pred_box_raw.shape != gt_mask.shape:
                    pred_box_raw = resize_mask(pred_box_raw, gt_mask.shape)
                pred_box = pred_box_raw.astype(np.uint8)
        elif args.prompt_kind == "points" and point_coords:
            pred_box_raw = sam3.predict_points(inference_state, point_coords, point_labels, gt_mask.shape)
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
        if args.prompt_kind == "box" and bboxes:
            pred_joint_raw = sam3.predict_boxes_text(inference_state, bboxes, label, gt_mask.shape)
            if pred_joint_raw is not None:
                if pred_joint_raw.shape != gt_mask.shape:
                    pred_joint_raw = resize_mask(pred_joint_raw, gt_mask.shape)
                pred_joint = pred_joint_raw.astype(np.uint8)
        elif args.prompt_kind == "points" and point_coords:
            pred_joint_raw = sam3.predict_points_text(inference_state, point_coords, point_labels, label, gt_mask.shape)
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

        save_mask(left_prompt_mask, masks_dir / f"{stem}_left_prompt_mask.png")
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
            left_mask=left_prompt_mask,
            right_mask=right_mask,
            bboxes=bboxes,
            point_coords=point_coords,
            prompt_kind=args.prompt_kind,
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
                "left_bboxes_xyxy": serialize_bboxes(bboxes),
                "n_left_boxes": int(len(bboxes)),
                "left_prompt_points_xy": [[int(x), int(y)] for x, y in point_coords],
                "n_left_points": int(len(point_coords)),
                "left_positive_pixels": int(left_prompt_mask.sum()),
                "right_positive_pixels": int(right_mask.sum()),
                "box_prompt_metrics_right_half": box_metrics,
                "text_prompt_metrics_right_half": text_metrics,
                "joint_box_text_metrics_right_half": joint_metrics,
            }
        )

        print(f"\nLabel: {label}")
        if args.prompt_kind == "box":
            print(f"  Left prompt boxes: {bboxes}")
        else:
            print(f"  Left prompt points: {point_coords[:10]}{' ...' if len(point_coords) > 10 else ''}")
        print(
            f"  Right-half {args.prompt_kind} prompt: Dice={box_metrics['dice']:.3f}, "
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

    (output_dir / "experiment_summary.json").write_text(json.dumps(experiment, indent=2))
    print("\nSaved outputs to:", output_dir)


if __name__ == "__main__":
    main()
