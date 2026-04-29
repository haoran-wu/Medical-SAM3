#!/usr/bin/env python3
"""
Rasterize VisiumHD Exp1 GeoJSON annotations onto the hires tissue image.

GeoJSON coordinates are full-resolution Space Ranger coordinates.
Scale them by tissue_hires_scalef to map to tissue_hires_image.png.
The annotations also use a bottom-left origin so y is flipped by default.
"""

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = Path("/vast/palmer/pi/xiting_yan/hw568/collections_spatial_datasets/VisiumHD_Human_Lung/Exp1")
DEFAULT_IMAGE_PATH = DATA_DIR / "spatial" / "tissue_hires_image.png"
DEFAULT_GEOJSON_PATH = DATA_DIR / "VisiumHD_Exp1.geojson"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "output" / "visium_hd_exp1" / "region_masks"
DEFAULT_SCALEF = 0.13752006
DEFAULT_LABEL_KEY = "region"

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


def slugify(value: str) -> str:
    value = value.lower().strip()
    value = re.sub(r"[^a-z0-9]+", "_", value)
    return value.strip("_") or "unlabeled"


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


def infer_label(feature: Dict[str, Any], label_key: str) -> str:
    props = feature.get("properties") or {}
    value = nested_get(props, label_key) if "." in label_key else props.get(label_key)
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


def transform_ring(
    ring: Sequence[Sequence[float]],
    scalef: float,
    image_height: int,
    flip_y: bool,
) -> List[Tuple[int, int]]:
    result = []
    for point in ring:
        if len(point) < 2:
            continue
        x = int(round(float(point[0]) * scalef))
        y = float(point[1]) * scalef
        if flip_y:
            y = image_height - 1 - y
        result.append((x, int(round(y))))
    return result


def draw_polygon_geometry(
    draw: ImageDraw.ImageDraw,
    geometry: Dict[str, Any],
    scalef: float,
    image_height: int,
    fill: int,
    flip_y: bool,
) -> None:
    geom_type = geometry.get("type")
    coords = geometry.get("coordinates")
    if geom_type == "Polygon":
        _draw_polygon(draw, coords or [], scalef, image_height, fill, flip_y)
    elif geom_type == "MultiPolygon":
        for polygon in coords or []:
            _draw_polygon(draw, polygon, scalef, image_height, fill, flip_y)
    elif geom_type == "GeometryCollection":
        for item in geometry.get("geometries") or []:
            draw_polygon_geometry(draw, item, scalef, image_height, fill, flip_y)


def _draw_polygon(
    draw: ImageDraw.ImageDraw,
    rings: Sequence[Sequence[Sequence[float]]],
    scalef: float,
    image_height: int,
    fill: int,
    flip_y: bool,
) -> None:
    if not rings:
        return
    exterior = transform_ring(rings[0], scalef, image_height, flip_y)
    if len(exterior) >= 3:
        draw.polygon(exterior, fill=fill)
    for hole in rings[1:]:
        interior = transform_ring(hole, scalef, image_height, flip_y)
        if len(interior) >= 3:
            draw.polygon(interior, fill=0)


def make_overlay(image: Image.Image, mask: Image.Image, color: Tuple[int, int, int], alpha: int = 115) -> Image.Image:
    base = image.convert("RGBA")
    layer = Image.new("RGBA", image.size, color + (0,))
    layer.putalpha(mask.point(lambda p: alpha if p > 0 else 0))
    return Image.alpha_composite(base, layer).convert("RGB")


def main() -> None:
    parser = argparse.ArgumentParser(description="Rasterize VisiumHD Exp1 GeoJSON annotations.")
    parser.add_argument("--image-path", type=Path, default=DEFAULT_IMAGE_PATH)
    parser.add_argument("--geojson-path", type=Path, default=DEFAULT_GEOJSON_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--label-key", type=str, default=DEFAULT_LABEL_KEY,
                        help="Feature property key for region label (default: region)")
    parser.add_argument("--scalef", type=float, default=DEFAULT_SCALEF,
                        help="Scale factor to convert full-res GeoJSON coords to hires image coords")
    parser.add_argument("--flip-y", action="store_true",
                        help="Enable y-axis flip for bottom-left coordinate origin (off by default)")
    args = parser.parse_args()

    if not args.image_path.exists():
        raise FileNotFoundError(f"Image not found: {args.image_path}")
    if not args.geojson_path.exists():
        raise FileNotFoundError(f"GeoJSON not found: {args.geojson_path}")

    flip_y = args.flip_y

    Image.MAX_IMAGE_PIXELS = None
    image = Image.open(args.image_path).convert("RGB")
    print(f"Image size: {image.width} x {image.height}")

    data = json.loads(args.geojson_path.read_text())
    features = data.get("features", [])
    if not features:
        raise ValueError(f"No features in {args.geojson_path}")
    print(f"Features: {len(features)}")

    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    geometry_counts: Counter[str] = Counter()
    xs, ys = [], []
    for feature in features:
        geom = feature.get("geometry") or {}
        geometry_counts[geom.get("type", "unknown")] += 1
        for x, y in iter_positions(geom):
            xs.append(x)
            ys.append(y)
        grouped[infer_label(feature, args.label_key)].append(feature)

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    overlays_dir = output_dir / "overlays"
    overlays_dir.mkdir(exist_ok=True)

    summary = {
        "image_path": str(args.image_path),
        "geojson_path": str(args.geojson_path),
        "image_size": {"width": image.width, "height": image.height},
        "scalef": args.scalef,
        "flip_y": flip_y,
        "label_key": args.label_key,
        "feature_count": len(features),
        "geometry_counts": dict(geometry_counts),
        "fullres_coordinate_bounds": {
            "x_min": min(xs) if xs else None,
            "x_max": max(xs) if xs else None,
            "y_min": min(ys) if ys else None,
            "y_max": max(ys) if ys else None,
        },
        "labels": [],
    }

    print(f"\nRendering {len(grouped)} labels with scalef={args.scalef}, flip_y={flip_y}:")
    for idx, (label, label_features) in enumerate(sorted(grouped.items())):
        color = PALETTE[idx % len(PALETTE)]
        mask = Image.new("L", image.size, 0)
        draw = ImageDraw.Draw(mask)
        for feature in label_features:
            draw_polygon_geometry(draw, feature.get("geometry") or {}, args.scalef, image.height, 255, flip_y)

        mask_np = np.array(mask) > 0
        stem = slugify(label)
        mask_path = output_dir / f"{stem}.png"
        overlay_path = overlays_dir / f"{stem}_overlay.png"
        mask.save(mask_path)
        make_overlay(image, mask, color).save(overlay_path)

        coverage = mask_np.sum() / float(image.width * image.height) * 100.0
        print(f"  {label}: {len(label_features)} features, {coverage:.2f}% coverage → {mask_path.name}")

        summary["labels"].append({
            "label": label,
            "slug": stem,
            "feature_count": len(label_features),
            "positive_pixels": int(mask_np.sum()),
            "coverage_pct": round(coverage, 4),
            "mask_path": str(mask_path),
            "overlay_path": str(overlay_path),
            "color_rgb": list(color),
        })

    summary_path = output_dir / "region_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"\nWrote {len(grouped)} masks and summary to: {output_dir}")


if __name__ == "__main__":
    main()
