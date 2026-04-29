#!/usr/bin/env python3
"""
Rasterize VisiumHD GeoJSON annotations onto the hires tissue image.

The Exp1 annotations appear to use a bottom-left coordinate origin, so this
script flips y by default with: image_y = image_height - 1 - geojson_y.
"""

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_DIR = Path("/vast/palmer/pi/xiting_yan/hw568/collections_spatial_datasets/VisiumHD_Human_Lung/Exp1")
DEFAULT_IMAGE_PATH = DEFAULT_DATA_DIR / "spatial" / "tissue_hires_image.png"
DEFAULT_GEOJSON_PATH = DEFAULT_DATA_DIR / "VisiumHD_Exp1.geojson"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "output" / "visiumhd_human_lung_exp1_geojson"

PALETTE: List[Tuple[int, int, int]] = [
    (31, 119, 180),
    (255, 127, 14),
    (44, 160, 44),
    (148, 103, 189),
    (140, 86, 75),
    (127, 127, 127),
    (188, 189, 34),
    (23, 190, 207),
    (214, 39, 40),
    (227, 119, 194),
    (70, 130, 180),
    (255, 187, 120),
]

LABEL_KEYS = (
    "classification.name",
    "classification",
    "name",
    "label",
    "class",
    "annotation",
    "cell_type",
    "type",
    "object_type",
)


def slugify(value: str) -> str:
    value = value.lower().strip()
    value = re.sub(r"[^a-z0-9]+", "_", value)
    return value.strip("_") or "unlabeled"


def load_font(font_name: str, size: int) -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype(font_name, size)
    except OSError:
        return ImageFont.load_default()


def nested_get(mapping: Dict[str, Any], dotted_key: str) -> Any:
    value: Any = mapping
    for part in dotted_key.split("."):
        if not isinstance(value, dict) or part not in value:
            return None
        value = value[part]
    return value


def stringify_label(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, dict):
        for key in ("name", "label", "value", "type"):
            if key in value:
                return stringify_label(value[key])
        return None
    if isinstance(value, (str, int, float)):
        text = str(value).strip()
        return text if text else None
    return None


def infer_label(feature: Dict[str, Any], label_key: str | None) -> str:
    props = feature.get("properties") or {}
    if label_key:
        value = nested_get(props, label_key) if "." in label_key else props.get(label_key)
        label = stringify_label(value)
        if label:
            return label
        raise KeyError(f"Label key '{label_key}' not found or empty in feature properties.")

    for key in LABEL_KEYS:
        value = nested_get(props, key) if "." in key else props.get(key)
        label = stringify_label(value)
        if label:
            return label

    for value in props.values():
        label = stringify_label(value)
        if label:
            return label

    return "unlabeled"


def iter_positions(geometry: Dict[str, Any]) -> Iterable[Tuple[float, float]]:
    coords = geometry.get("coordinates")
    geom_type = geometry.get("type")
    if geom_type == "Polygon":
        for ring in coords or []:
            for point in ring:
                if len(point) >= 2:
                    yield float(point[0]), float(point[1])
    elif geom_type == "MultiPolygon":
        for polygon in coords or []:
            for ring in polygon:
                for point in ring:
                    if len(point) >= 2:
                        yield float(point[0]), float(point[1])
    elif geom_type == "GeometryCollection":
        for item in geometry.get("geometries") or []:
            yield from iter_positions(item)


def transform_point(x: float, y: float, image_height: int, flip_y: bool) -> Tuple[int, int]:
    if flip_y:
        y = image_height - 1 - y
    return int(round(x)), int(round(y))


def transform_ring(
    ring: Sequence[Sequence[float]],
    image_height: int,
    flip_y: bool,
) -> List[Tuple[int, int]]:
    return [transform_point(float(point[0]), float(point[1]), image_height, flip_y) for point in ring if len(point) >= 2]


def draw_polygon_geometry(
    draw: ImageDraw.ImageDraw,
    geometry: Dict[str, Any],
    image_height: int,
    fill: int,
    flip_y: bool,
) -> None:
    geom_type = geometry.get("type")
    coords = geometry.get("coordinates")
    if geom_type == "Polygon":
        draw_polygon(draw, coords or [], image_height, fill, flip_y)
    elif geom_type == "MultiPolygon":
        for polygon in coords or []:
            draw_polygon(draw, polygon, image_height, fill, flip_y)
    elif geom_type == "GeometryCollection":
        for item in geometry.get("geometries") or []:
            draw_polygon_geometry(draw, item, image_height, fill, flip_y)


def draw_polygon(
    draw: ImageDraw.ImageDraw,
    rings: Sequence[Sequence[Sequence[float]]],
    image_height: int,
    fill: int,
    flip_y: bool,
) -> None:
    if not rings:
        return
    exterior = transform_ring(rings[0], image_height, flip_y)
    if len(exterior) >= 3:
        draw.polygon(exterior, fill=fill)
    for hole in rings[1:]:
        interior = transform_ring(hole, image_height, flip_y)
        if len(interior) >= 3:
            draw.polygon(interior, fill=0)


def make_overlay(image: Image.Image, mask: Image.Image, color: Tuple[int, int, int], alpha: int) -> Image.Image:
    base = image.convert("RGBA")
    layer = Image.new("RGBA", image.size, color + (0,))
    layer.putalpha(mask.point(lambda pixel: alpha if pixel > 0 else 0))
    return Image.alpha_composite(base, layer).convert("RGB")


def save_combined_overlay(
    image: Image.Image,
    label_masks: List[Dict[str, Any]],
    output_path: Path,
    alpha: int,
    max_display_height: int,
) -> None:
    scale = min(1.0, max_display_height / image.height) if max_display_height > 0 else 1.0
    display_size = (max(1, int(round(image.width * scale))), max(1, int(round(image.height * scale))))
    base = image.convert("RGBA").resize(display_size, Image.Resampling.LANCZOS)

    overlay = Image.new("RGBA", display_size, (0, 0, 0, 0))
    for item in label_masks:
        mask = item["mask"]
        if mask.size != display_size:
            mask = mask.resize(display_size, Image.Resampling.NEAREST)
        layer = Image.new("RGBA", display_size, item["color"] + (0,))
        layer.putalpha(mask.point(lambda pixel, a=alpha: a if pixel > 0 else 0))
        overlay = Image.alpha_composite(overlay, layer)

    combined = Image.alpha_composite(base, overlay)

    legend_width = 760
    title_font = load_font("/System/Library/Fonts/Supplemental/Arial Bold.ttf", 32)
    text_font = load_font("/System/Library/Fonts/Supplemental/Arial.ttf", 24)
    small_font = load_font("/System/Library/Fonts/Supplemental/Arial.ttf", 19)
    row_height = 74
    legend_height = 130 + len(label_masks) * row_height
    canvas_height = max(display_size[1], legend_height + 30)
    canvas = Image.new("RGBA", (display_size[0] + legend_width, canvas_height), (255, 255, 255, 255))
    canvas.alpha_composite(combined, (0, max(0, (canvas_height - display_size[1]) // 2)))

    draw = ImageDraw.Draw(canvas)
    x0 = display_size[0] + 34
    y = 28
    draw.text((x0, y), "VisiumHD Exp1 GeoJSON", fill=(20, 20, 20), font=title_font)
    y += 48
    draw.text((x0, y), "Rasterized annotations on hires image", fill=(75, 75, 75), font=small_font)
    y += 48
    for idx, item in enumerate(label_masks, start=1):
        row_y = y + (idx - 1) * row_height
        draw.rounded_rectangle((x0, row_y, x0 + 32, row_y + 32), radius=6, fill=item["color"] + (255,))
        label = item["label"]
        coverage = item["positive_pixels"] / float(image.width * image.height) * 100.0
        draw.text((x0 + 48, row_y - 2), f"{idx}. {label}", fill=(20, 20, 20), font=text_font)
        draw.text((x0 + 48, row_y + 30), f"{item['feature_count']} features, coverage {coverage:.2f}%", fill=(80, 80, 80), font=small_font)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.convert("RGB").save(output_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Rasterize VisiumHD Exp1 GeoJSON annotations.")
    parser.add_argument("--image-path", type=Path, default=DEFAULT_IMAGE_PATH)
    parser.add_argument("--geojson-path", type=Path, default=DEFAULT_GEOJSON_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--label-key", type=str, default=None, help="Feature property key to use as label, e.g. classification.name.")
    parser.add_argument("--no-flip-y", action="store_true", help="Disable the default y-axis flip.")
    parser.add_argument("--alpha", type=int, default=115, help="Overlay alpha from 0 to 255.")
    parser.add_argument("--max-display-height", type=int, default=1800, help="Display height for combined preview; 0 disables resizing.")
    args = parser.parse_args()

    if not args.image_path.exists():
        raise FileNotFoundError(f"Image not found: {args.image_path}")
    if not args.geojson_path.exists():
        raise FileNotFoundError(f"GeoJSON not found: {args.geojson_path}")

    Image.MAX_IMAGE_PIXELS = None
    image = Image.open(args.image_path).convert("RGB")
    data = json.loads(args.geojson_path.read_text())
    features = data.get("features", [])
    if not features:
        raise ValueError(f"No features found in {args.geojson_path}")

    flip_y = not args.no_flip_y
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    geometry_counts: Counter[str] = Counter()
    xs: List[float] = []
    ys: List[float] = []

    for feature in features:
        geometry = feature.get("geometry") or {}
        geometry_counts[geometry.get("type", "unknown")] += 1
        for x, y in iter_positions(geometry):
            xs.append(x)
            ys.append(y)
        grouped[infer_label(feature, args.label_key)].append(feature)

    output_dir = args.output_dir
    masks_dir = output_dir / "masks"
    overlays_dir = output_dir / "overlays"
    masks_dir.mkdir(parents=True, exist_ok=True)
    overlays_dir.mkdir(parents=True, exist_ok=True)

    label_masks: List[Dict[str, Any]] = []
    summary = {
        "image_path": str(args.image_path),
        "geojson_path": str(args.geojson_path),
        "image_size": {"width": image.width, "height": image.height},
        "flip_y": flip_y,
        "label_key": args.label_key or "auto",
        "feature_count": len(features),
        "geometry_counts": dict(geometry_counts),
        "coordinate_bounds_before_flip": {
            "x_min": min(xs) if xs else None,
            "x_max": max(xs) if xs else None,
            "y_min": min(ys) if ys else None,
            "y_max": max(ys) if ys else None,
        },
        "labels": [],
    }

    for idx, (label, label_features) in enumerate(sorted(grouped.items())):
        color = PALETTE[idx % len(PALETTE)]
        mask = Image.new("L", image.size, 0)
        draw = ImageDraw.Draw(mask)
        for feature in label_features:
            draw_polygon_geometry(draw, feature.get("geometry") or {}, image.height, 255, flip_y)

        mask_np = np.array(mask) > 0
        stem = f"{idx + 1:02d}_{slugify(label)}"
        mask_path = masks_dir / f"{stem}.png"
        overlay_path = overlays_dir / f"{stem}.png"
        mask.save(mask_path)
        make_overlay(image, mask, color, args.alpha).save(overlay_path)

        item = {
            "label": label,
            "feature_count": len(label_features),
            "positive_pixels": int(mask_np.sum()),
            "mask_path": str(mask_path),
            "overlay_path": str(overlay_path),
            "color_rgb": color,
            "mask": mask,
        }
        label_masks.append(item)
        summary["labels"].append({key: value for key, value in item.items() if key != "mask"})

    save_combined_overlay(
        image=image,
        label_masks=label_masks,
        output_path=output_dir / "combined_geojson_overlay_with_legend.png",
        alpha=args.alpha,
        max_display_height=args.max_display_height,
    )
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    print(f"Image size: {image.width} x {image.height}")
    print(f"Features: {len(features)}")
    print(f"Geometry counts: {dict(geometry_counts)}")
    print(f"Coordinate bounds before flip: {summary['coordinate_bounds_before_flip']}")
    print(f"Flip y: {flip_y}")
    print("Labels:")
    for item in summary["labels"]:
        coverage = item["positive_pixels"] / float(image.width * image.height) * 100.0
        print(f"  - {item['label']}: {item['feature_count']} features, {coverage:.2f}% coverage")
    print(f"Wrote output to: {output_dir}")


if __name__ == "__main__":
    main()
