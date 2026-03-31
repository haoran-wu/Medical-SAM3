#!/usr/bin/env python3
"""
Render a single multicolor overlay image from per-prompt Medical-SAM3 masks.
"""

import argparse
import json
from pathlib import Path
from typing import List, Sequence, Tuple

from PIL import Image, ImageDraw, ImageFont


PALETTE: List[Tuple[int, int, int]] = [
    (230, 25, 75),
    (60, 180, 75),
    (255, 225, 25),
    (0, 130, 200),
    (245, 130, 48),
    (145, 30, 180),
    (70, 240, 240),
    (240, 50, 230),
    (210, 245, 60),
    (250, 190, 190),
    (0, 128, 128),
    (230, 190, 255),
    (170, 110, 40),
    (128, 128, 0),
    (128, 0, 0),
    (0, 0, 128),
]


def load_font(font_name: str, size: int) -> ImageFont.ImageFont:
    """Best-effort font loading with a safe fallback."""
    try:
        return ImageFont.truetype(font_name, size)
    except OSError:
        return ImageFont.load_default()


def resolve_mask_path(output_dir: Path, mask_path_str: str) -> Path:
    """Resolve a mask path from JSON, even if it was generated on another machine."""
    mask_path = Path(mask_path_str)
    if mask_path.exists():
        return mask_path
    candidate = output_dir / "masks" / mask_path.name
    if candidate.exists():
        return candidate
    raise FileNotFoundError(f"Mask not found: {mask_path_str}")


def load_predictions(predictions_path: Path, hide_empty_prompts: bool) -> dict:
    """Load prediction metadata and optionally drop empty masks from the legend."""
    data = json.loads(predictions_path.read_text())
    predictions = data["predictions"]
    if hide_empty_prompts:
        predictions = [item for item in predictions if item["positive_pixels"] > 0]
    if not predictions:
        raise ValueError("No predictions left to render.")
    data["predictions"] = predictions
    return data


def render_multicolor_overlay(
    image_path: Path,
    predictions_path: Path,
    output_path: Path,
    *,
    max_display_height: int = 1800,
    alpha: int = 110,
    hide_empty_prompts: bool = False,
    title: str | None = None,
) -> Path:
    """Create a single image with all prompt masks overlaid in distinct colors."""
    data = load_predictions(predictions_path, hide_empty_prompts)
    predictions = data["predictions"]
    output_dir = predictions_path.parent

    base = Image.open(image_path).convert("RGBA")
    orig_w, orig_h = base.size
    scale = min(1.0, max_display_height / orig_h) if max_display_height > 0 else 1.0
    display_size = (int(orig_w * scale), int(orig_h * scale))
    base_display = base.resize(display_size, Image.Resampling.LANCZOS)

    overlay = Image.new("RGBA", display_size, (0, 0, 0, 0))
    for pred, color in zip(predictions, PALETTE):
        mask_path = resolve_mask_path(output_dir, pred["mask_path"])
        mask = Image.open(mask_path).convert("L")
        if mask.size != display_size:
            mask = mask.resize(display_size, Image.Resampling.NEAREST)
        layer = Image.new("RGBA", display_size, color + (0,))
        layer.putalpha(mask.point(lambda pixel, a=alpha: a if pixel > 0 else 0))
        overlay = Image.alpha_composite(overlay, layer)

    combined = Image.alpha_composite(base_display, overlay)

    legend_width = 900
    title_font = load_font("/System/Library/Fonts/Supplemental/Arial Bold.ttf", 34)
    text_font = load_font("/System/Library/Fonts/Supplemental/Arial.ttf", 24)
    small_font = load_font("/System/Library/Fonts/Supplemental/Arial.ttf", 20)

    if title is None:
        title = f"All Prompt Predictions: {image_path.name}"

    x0 = display_size[0] + 35
    y = 30
    legend_top = y
    row_height = 110
    legend_bottom = y + 55 + 45 + 50 + len(predictions) * row_height
    canvas_height = max(display_size[1], legend_bottom + 30)

    canvas = Image.new("RGBA", (display_size[0] + legend_width, canvas_height), (255, 255, 255, 255))
    image_y = max(0, (canvas_height - display_size[1]) // 2)
    canvas.alpha_composite(combined, (0, image_y))

    draw = ImageDraw.Draw(canvas)
    draw.text((x0, y), title, fill=(20, 20, 20), font=title_font)
    y += 55
    draw.text((x0, y), "Each color is one text prompt.", fill=(70, 70, 70), font=text_font)
    y += 45
    draw.text((x0, y), "Overlaps blend visually where masks overlap.", fill=(70, 70, 70), font=small_font)
    y += 50

    for idx, (pred, color) in enumerate(zip(predictions, PALETTE), start=1):
        box_y = y + (idx - 1) * row_height
        draw.rounded_rectangle(
            [x0, box_y, x0 + 34, box_y + 34],
            radius=6,
            fill=color + (255,),
            outline=(30, 30, 30),
        )
        prompt = pred["prompt"]
        pixels = pred["positive_pixels"]
        coverage = pred["coverage_ratio"] * 100
        draw.text((x0 + 50, box_y - 2), f"{idx}. {prompt}", fill=(20, 20, 20), font=text_font)
        draw.text(
            (x0 + 50, box_y + 34),
            f"predicted pixels: {pixels:,}   coverage: {coverage:.2f}%",
            fill=(70, 70, 70),
            font=small_font,
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render one multicolor overlay from Medical-SAM3 prompt masks.")
    parser.add_argument("--image-path", type=Path, required=True, help="Path to the source image.")
    parser.add_argument(
        "--predictions-path",
        type=Path,
        required=True,
        help="Path to predictions.json generated by run_spatiallibd_prompts.py.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=None,
        help="Where to write the combined overlay. Defaults next to predictions.json.",
    )
    parser.add_argument(
        "--max-display-height",
        type=int,
        default=1800,
        help="Resize the preview so the displayed image is no taller than this many pixels.",
    )
    parser.add_argument(
        "--alpha",
        type=int,
        default=110,
        help="Mask overlay alpha from 0 to 255.",
    )
    parser.add_argument(
        "--hide-empty-prompts",
        action="store_true",
        help="Exclude prompts with zero predicted pixels from the final legend and overlay.",
    )
    parser.add_argument(
        "--title",
        type=str,
        default=None,
        help="Optional custom title shown above the legend.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path = args.output_path or args.predictions_path.parent / "prompt_multicolor_overlay.png"
    written = render_multicolor_overlay(
        image_path=args.image_path,
        predictions_path=args.predictions_path,
        output_path=output_path,
        max_display_height=args.max_display_height,
        alpha=args.alpha,
        hide_empty_prompts=args.hide_empty_prompts,
        title=args.title,
    )
    print(f"Wrote combined overlay: {written}")


if __name__ == "__main__":
    main()
