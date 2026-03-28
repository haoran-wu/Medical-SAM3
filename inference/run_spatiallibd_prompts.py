#!/usr/bin/env python3
"""
Run text-prompted Medical-SAM3 inference on a single spatialLIBD TIFF image.

This script is for the current workflow in this repo:
- input image: data/spatialLIBD/151673/tissue_hires_image.png
- prompts: dorsolateral prefrontal cortex layers + white matter

It does not require ground-truth masks. The output is a set of predicted masks,
per-prompt overlays, and a JSON summary.
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Iterable, List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MPL_CONFIG_DIR = PROJECT_ROOT / "output" / ".mplconfig"
MPL_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CONFIG_DIR))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

from sam3_inference import SAM3Model, resize_mask


DEFAULT_IMAGE_PATH = PROJECT_ROOT / "data" / "spatialLIBD" / "151673" / "tissue_hires_image.png"
DEFAULT_PROMPTS_PATH = Path(__file__).parent / "prompts" / "spatiallibd_dlpfc_layers.txt"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "output" / "spatialLIBD_151673"
DEFAULT_MAX_SIDE = 2048
OVERLAY_COLORS = [
    (230, 57, 70),
    (29, 53, 87),
    (69, 123, 157),
    (42, 157, 143),
    (233, 196, 106),
    (244, 162, 97),
    (106, 76, 147),
]


def slugify(value: str) -> str:
    """Convert a prompt into a filesystem-friendly stem."""
    value = value.lower().strip()
    value = re.sub(r"[^a-z0-9]+", "_", value)
    return value.strip("_")


def load_prompts(prompts_path: Path) -> List[str]:
    """Load one prompt per line from a text file."""
    prompts = [line.strip() for line in prompts_path.read_text().splitlines() if line.strip()]
    if not prompts:
        raise ValueError(f"No prompts found in {prompts_path}")
    return prompts


def load_rgb_image(image_path: Path, max_side: int) -> Tuple[np.ndarray, dict]:
    """Load a TIFF/PNG/JPEG image and optionally resize it for manageable inference."""
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


def make_overlay(image: np.ndarray, mask: np.ndarray, color: Tuple[int, int, int], alpha: float = 0.45) -> np.ndarray:
    """Blend a binary mask onto an RGB image."""
    base = image.astype(np.float32).copy()
    mask_bool = mask.astype(bool)
    color_arr = np.array(color, dtype=np.float32)
    base[mask_bool] = (1.0 - alpha) * base[mask_bool] + alpha * color_arr
    return np.clip(base, 0, 255).astype(np.uint8)


def save_mask(mask: np.ndarray, output_path: Path) -> None:
    """Write a binary mask as a black/white PNG."""
    Image.fromarray((mask.astype(np.uint8) * 255)).save(output_path)


def save_overlay(overlay: np.ndarray, output_path: Path) -> None:
    """Write an RGB overlay PNG."""
    Image.fromarray(overlay).save(output_path)


def save_contact_sheet(image: np.ndarray, overlays: Iterable[dict], output_path: Path) -> None:
    """Save a simple grid showing the source image and all prompt overlays."""
    overlays = list(overlays)
    total_panels = len(overlays) + 1
    cols = 2
    rows = int(np.ceil(total_panels / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(12, 5 * rows))
    axes = np.array(axes).reshape(-1)

    axes[0].imshow(image)
    axes[0].set_title("Source TIFF")
    axes[0].axis("off")

    for idx, item in enumerate(overlays, start=1):
        axes[idx].imshow(item["overlay"])
        axes[idx].set_title(f"{item['prompt']}\ncoverage={item['coverage_ratio']:.4f}")
        axes[idx].axis("off")

    for idx in range(total_panels, len(axes)):
        axes[idx].axis("off")

    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Medical-SAM3 on a single spatialLIBD TIFF image.")
    parser.add_argument(
        "--image-path",
        type=Path,
        default=DEFAULT_IMAGE_PATH,
        help="Path to the input TIFF image.",
    )
    parser.add_argument(
        "--prompts-path",
        type=Path,
        default=DEFAULT_PROMPTS_PATH,
        help="Text file with one prompt per line.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for masks, overlays, and metadata.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Optional custom checkpoint path for Medical-SAM3.",
    )
    parser.add_argument(
        "--max-side",
        type=int,
        default=DEFAULT_MAX_SIDE,
        help="Resize the input so its longest side is at most this many pixels. Use 0 to disable resizing.",
    )
    args = parser.parse_args()

    if not args.image_path.exists():
        raise FileNotFoundError(f"Image not found: {args.image_path}")
    if not args.prompts_path.exists():
        raise FileNotFoundError(f"Prompts file not found: {args.prompts_path}")

    prompts = load_prompts(args.prompts_path)

    print("=" * 60)
    print("Medical-SAM3 spatialLIBD text-prompt inference")
    print("=" * 60)
    print(f"Image: {args.image_path}")
    print(f"Prompts: {args.prompts_path}")
    print(f"Checkpoint: {args.checkpoint or 'default SAM3 from HuggingFace'}")
    print(f"Max side: {args.max_side or 'disabled'}")

    image, image_metadata = load_rgb_image(args.image_path, args.max_side)
    print(f"Original size: {image_metadata['original_size']}")
    print(f"Inference size: {image_metadata['inference_size']}")
    print(f"Resize scale: {image_metadata['resize_scale']:.6f}")

    output_dir = args.output_dir
    masks_dir = output_dir / "masks"
    overlays_dir = output_dir / "overlays"
    output_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)
    overlays_dir.mkdir(parents=True, exist_ok=True)

    sam3 = SAM3Model(confidence_threshold=0.1, checkpoint_path=args.checkpoint)
    inference_state = sam3.encode_image(image)

    summary = {
        "image_path": str(args.image_path),
        "prompts_path": str(args.prompts_path),
        "checkpoint": args.checkpoint,
        "image_metadata": image_metadata,
        "predictions": [],
    }
    contact_sheet_items = []

    for idx, prompt in enumerate(prompts):
        print(f"\n[{idx + 1}/{len(prompts)}] Prompt: {prompt}")
        pred_mask = sam3.predict_text(inference_state, prompt)
        if pred_mask is None:
            pred_mask = np.zeros(image.shape[:2], dtype=np.uint8)
        elif pred_mask.shape != image.shape[:2]:
            pred_mask = resize_mask(pred_mask, image.shape[:2])

        positive_pixels = int(pred_mask.sum())
        coverage_ratio = positive_pixels / float(pred_mask.size)
        prompt_stem = f"{idx + 1:02d}_{slugify(prompt)}"
        mask_path = masks_dir / f"{prompt_stem}.png"
        overlay_path = overlays_dir / f"{prompt_stem}.png"

        overlay = make_overlay(image, pred_mask, OVERLAY_COLORS[idx % len(OVERLAY_COLORS)])
        save_mask(pred_mask, mask_path)
        save_overlay(overlay, overlay_path)

        summary["predictions"].append(
            {
                "prompt": prompt,
                "mask_path": str(mask_path),
                "overlay_path": str(overlay_path),
                "positive_pixels": positive_pixels,
                "coverage_ratio": coverage_ratio,
            }
        )
        contact_sheet_items.append(
            {
                "prompt": prompt,
                "overlay": overlay,
                "coverage_ratio": coverage_ratio,
            }
        )
        print(f"  Positive pixels: {positive_pixels}")
        print(f"  Coverage ratio: {coverage_ratio:.6f}")

    metadata_path = output_dir / "predictions.json"
    metadata_path.write_text(json.dumps(summary, indent=2))

    contact_sheet_path = output_dir / "prompt_overlays.png"
    save_contact_sheet(image, contact_sheet_items, contact_sheet_path)

    print("\n" + "=" * 60)
    print("Inference complete")
    print(f"Masks: {masks_dir}")
    print(f"Overlays: {overlays_dir}")
    print(f"Summary: {metadata_path}")
    print(f"Contact sheet: {contact_sheet_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
