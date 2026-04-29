#!/usr/bin/env python3
"""
Run Medical-SAM3 inference on a single image using one or more box prompts.

This script keeps the SAM3 box-prompt setting simple:
- input is one full image
- prompt is one or more bounding boxes
- output is a whole-image prediction mask and overlay
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MPL_CONFIG_DIR = PROJECT_ROOT / "output" / ".mplconfig"
MPL_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CONFIG_DIR))

import matplotlib
matplotlib.use("Agg")
import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).parent))

from sam3_inference import SAM3Model, resize_mask


def parse_boxes(boxes_text: str) -> List[Tuple[int, int, int, int]]:
    """
    Parse 'x1,y1,x2,y2;x1,y1,x2,y2' into a list of boxes.
    """
    boxes: List[Tuple[int, int, int, int]] = []
    for item in boxes_text.split(";"):
        item = item.strip()
        if not item:
            continue
        parts = [int(v.strip()) for v in item.split(",")]
        if len(parts) != 4:
            raise ValueError(f"Invalid box: {item}")
        x1, y1, x2, y2 = parts
        if x2 <= x1 or y2 <= y1:
            raise ValueError(f"Box must satisfy x2>x1 and y2>y1: {item}")
        boxes.append((x1, y1, x2, y2))
    if not boxes:
        raise ValueError("No valid boxes were provided.")
    return boxes


def load_rgb_image(image_path: Path, max_side: int) -> Tuple[np.ndarray, dict]:
    Image.MAX_IMAGE_PIXELS = None
    with Image.open(image_path) as img:
        original_size = img.size
        rgb_img = img.convert("RGB")

        resize_scale = 1.0
        if max_side and max(rgb_img.size) > max_side:
            resize_scale = max_side / float(max(rgb_img.size))
            resized_size = (
                max(1, int(round(rgb_img.size[0] * resize_scale))),
                max(1, int(round(rgb_img.size[1] * resize_scale))),
            )
            rgb_img = rgb_img.resize(resized_size, Image.Resampling.BILINEAR)

        image = np.array(rgb_img)

    metadata = {
        "original_size": original_size,
        "inference_size": (image.shape[1], image.shape[0]),
        "resize_scale": resize_scale,
    }
    return image, metadata


def scale_boxes(boxes: List[Tuple[int, int, int, int]], scale: float) -> List[Tuple[int, int, int, int]]:
    if scale == 1.0:
        return boxes
    scaled = []
    for x1, y1, x2, y2 in boxes:
        scaled.append(
            (
                int(round(x1 * scale)),
                int(round(y1 * scale)),
                int(round(x2 * scale)),
                int(round(y2 * scale)),
            )
        )
    return scaled


def make_overlay(image: np.ndarray, mask: np.ndarray, color=(255, 99, 71), alpha: float = 0.45) -> np.ndarray:
    base = image.astype(np.float32).copy()
    mask_bool = mask.astype(bool)
    color_arr = np.array(color, dtype=np.float32)
    base[mask_bool] = (1.0 - alpha) * base[mask_bool] + alpha * color_arr
    return np.clip(base, 0, 255).astype(np.uint8)


def draw_boxes(image: np.ndarray, boxes: List[Tuple[int, int, int, int]], color=(255, 99, 71), width: int = 8) -> np.ndarray:
    boxed = Image.fromarray(image.copy())
    draw = ImageDraw.Draw(boxed)
    for x1, y1, x2, y2 in boxes:
        draw.rectangle([x1, y1, x2, y2], outline=color, width=width)
    return np.array(boxed)


def save_mask(mask: np.ndarray, output_path: Path) -> None:
    Image.fromarray((mask.astype(np.uint8) * 255)).save(output_path)


def save_overlay(overlay: np.ndarray, output_path: Path) -> None:
    Image.fromarray(overlay).save(output_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Medical-SAM3 on a single image with one or more box prompts.")
    parser.add_argument("--image-path", type=Path, required=True, help="Path to the input image.")
    parser.add_argument("--boxes", type=str, required=True, help="Semicolon-separated XYXY boxes, e.g. '10,20,30,40;50,60,80,90'")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for outputs.")
    parser.add_argument("--checkpoint", type=str, default=None, help="Optional custom checkpoint path.")
    parser.add_argument("--max-side", type=int, default=2048, help="Resize longest image side to this size. Use 0 to disable.")
    args = parser.parse_args()

    if not args.image_path.exists():
        raise FileNotFoundError(f"Image not found: {args.image_path}")

    original_boxes = parse_boxes(args.boxes)
    image, image_metadata = load_rgb_image(args.image_path, args.max_side)
    inference_boxes = scale_boxes(original_boxes, image_metadata["resize_scale"])

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    input_with_boxes = draw_boxes(image, inference_boxes)

    sam3 = SAM3Model(confidence_threshold=0.1, checkpoint_path=args.checkpoint)
    inference_state = sam3.encode_image(image)
    pred_mask = sam3.predict_boxes(inference_state, inference_boxes, image.shape[:2])

    if pred_mask is None:
        pred_mask = np.zeros(image.shape[:2], dtype=np.uint8)
    elif pred_mask.shape != image.shape[:2]:
        pred_mask = resize_mask(pred_mask, image.shape[:2])

    overlay = make_overlay(image, pred_mask)

    input_path = output_dir / "input_with_boxes.png"
    mask_path = output_dir / "pred_mask_box.png"
    overlay_path = output_dir / "pred_overlay_box.png"
    Image.fromarray(input_with_boxes).save(input_path)
    save_mask(pred_mask, mask_path)
    save_overlay(overlay, overlay_path)

    summary = {
        "image_path": str(args.image_path),
        "checkpoint": args.checkpoint,
        "image_metadata": image_metadata,
        "original_boxes_xyxy": [list(map(int, box)) for box in original_boxes],
        "inference_boxes_xyxy": [list(map(int, box)) for box in inference_boxes],
        "positive_pixels": int(pred_mask.sum()),
        "coverage_ratio": float(pred_mask.sum() / float(pred_mask.size)),
        "input_with_boxes": str(input_path),
        "pred_mask_box": str(mask_path),
        "pred_overlay_box": str(overlay_path),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    print("=" * 60)
    print("Single-image box inference complete")
    print(f"Input with boxes: {input_path}")
    print(f"Prediction mask: {mask_path}")
    print(f"Prediction overlay: {overlay_path}")
    print(f"Positive pixels: {summary['positive_pixels']}")
    print(f"Coverage ratio: {summary['coverage_ratio']:.6f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
