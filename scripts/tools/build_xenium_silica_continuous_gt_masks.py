#!/usr/bin/env python3
"""Build continuous Xenium Silica GT masks from cell-level annotation.

The source annotation is cell-level: each annotated row in ``TMA*.csv`` has a
coordinate and a ``new_annotation`` label.  This script maps those annotated
points back to the nearest Xenium cell centroid, fills the corresponding
``cell_boundaries.parquet`` polygon, then projects the filled cell regions into
the manually accepted H&E crop frame.

This output is evaluation GT only.  It is not used as FICTURE/SAM/VLM input.
"""

from __future__ import annotations

import argparse
import html
import json
import math
import os
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from scipy.spatial import cKDTree

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

import serve_xenium_manual_alignment_tool as manual_tool
import xenium_wholeslide_tif_he_translation_registration as ws_align
from render_xenium_manual_aligned_annotation_tma_panels import apply_manual_residual, load_manual_params


DEFAULT_ALIGN_ROOT = ROOT / "output/xenium_silica_wholeslide_tif_he_translation_registration_20260624"
DEFAULT_RAW_XENIUM_ROOT = Path("/Volumes/Disk/ST_annotated_datasets/tmp_Xenium")
DEFAULT_METADATA_ROOT = Path("/Volumes/Disk/ST_annotated_datasets/Xenium_Silica/with_annotation")
DEFAULT_OUT_ROOT = ROOT / "output/aaai_xenium_silica_20260623/continuous_gt_masks_cellboundary_20260626"
ANNOTATED_TMAS = [7, 24, 29, 30, 31, 34, 36, 39, 41, 42]
GT_COL = "new_annotation"
CELLTYPE_COL = "cellTypeFibSim"


def slide_for_tma(tma: int) -> str:
    return "770-1" if tma <= 24 else "771-1"


def raw_slide_for_tma(tma: int) -> str:
    return f"slide{slide_for_tma(tma)}"


def clean_label(value: object) -> str:
    text = str(value).strip()
    if text.lower() in {"", "nan", "na", "none", "null"}:
        return ""
    return text


def slugify(text: str) -> str:
    text = clean_label(text).lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_") or "unlabeled"


def rgb_tuple(hex_color: str) -> tuple[int, int, int]:
    color = hex_color.lstrip("#")
    return tuple(int(color[i : i + 2], 16) for i in (0, 2, 4))


def font(size: int) -> ImageFont.ImageFont:
    for path in [
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
    ]:
        try:
            return ImageFont.truetype(path, size)
        except Exception:
            pass
    return ImageFont.load_default()


def safe_read_csv(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def project_raw_xy_to_local_he(
    x: np.ndarray,
    y: np.ndarray,
    *,
    manifest: dict,
    crop_bbox: list[int],
    width: int,
    height: int,
    params: dict,
) -> np.ndarray:
    meta = manifest["morphology"]
    affine = np.array(manifest["selected_transform"]["affine_xy_from_morphology_to_oriented_he"], dtype=float)
    morph_xy = np.column_stack([x.astype(float) * meta["scale_x_px_per_coord"], y.astype(float) * meta["scale_y_px_per_coord"]])
    he_xy = ws_align.apply_affine_to_points(morph_xy, affine)
    local_xy = he_xy - np.array([float(crop_bbox[0]), float(crop_bbox[1])], dtype=float)
    return apply_manual_residual(local_xy, width, height, params)


def draw_point_annotation(
    he: Image.Image,
    xy: np.ndarray,
    labels: list[str],
    palette: dict[str, str],
    *,
    radius: int = 2,
) -> tuple[Image.Image, Image.Image]:
    overlay = he.convert("RGBA")
    mask = Image.new("RGBA", he.size, (255, 255, 255, 255))
    draw_overlay = ImageDraw.Draw(overlay)
    draw_mask = ImageDraw.Draw(mask)
    for (x, y), label in zip(xy, labels):
        if not label or not np.isfinite(x) or not np.isfinite(y):
            continue
        color = palette.get(label, "#999999")
        rgb = rgb_tuple(color)
        xi = int(round(float(x)))
        yi = int(round(float(y)))
        if -radius <= xi < he.width + radius and -radius <= yi < he.height + radius:
            box = (xi - radius, yi - radius, xi + radius, yi + radius)
            draw_overlay.ellipse(box, fill=(*rgb, 230))
            draw_mask.ellipse(box, fill=(*rgb, 255))
    return overlay.convert("RGB"), mask.convert("RGB")


def append_legend(image: Image.Image, counts: dict[str, int], title: str, subtitle: str) -> Image.Image:
    legend_w = 420
    pad = 22
    row_h = 34
    width, height = image.size
    out = Image.new("RGB", (width + legend_w, height), "white")
    out.paste(image.convert("RGB"), (0, 0))
    draw = ImageDraw.Draw(out)
    x0 = width
    draw.rectangle((x0, 0, width + legend_w - 1, height - 1), fill="#ffffff", outline="#cbd5e1")
    title_font = font(19)
    small_font = font(14)
    tiny_font = font(12)
    draw.text((x0 + pad, pad), title, fill="#172033", font=title_font)
    draw.text((x0 + pad, pad + 26), subtitle, fill="#64748b", font=tiny_font)
    y = pad + 64
    for label, count in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])):
        color = ws_align.base.GT_PALETTE.get(label, "#999999")
        draw.rounded_rectangle((x0 + pad, y + 3, x0 + pad + 22, y + 25), radius=3, fill=rgb_tuple(color))
        label_text = label if len(label) <= 34 else label[:31] + "..."
        draw.text((x0 + pad + 34, y), label_text, fill="#1f2937", font=small_font)
        draw.text((x0 + pad + 34, y + 17), f"{count:,} cells", fill="#64748b", font=tiny_font)
        y += row_h
        if y > height - 28:
            draw.text((x0 + pad, y), "...", fill="#64748b", font=small_font)
            break
    return out


def overlay_mask_on_he(he: Image.Image, rgb_mask: Image.Image, alpha: int = 155) -> Image.Image:
    base = he.convert("RGBA")
    arr = np.asarray(rgb_mask.convert("RGB"), dtype=np.uint8)
    nonwhite = np.any(arr < 248, axis=2)
    rgba = np.zeros((arr.shape[0], arr.shape[1], 4), dtype=np.uint8)
    rgba[:, :, :3] = arr
    rgba[:, :, 3] = (nonwhite.astype(np.uint8) * alpha)
    return Image.alpha_composite(base, Image.fromarray(rgba, "RGBA")).convert("RGB")


def read_annotated_metadata(metadata_root: Path, tma: int, annotation_col: str) -> pd.DataFrame:
    path = metadata_root / f"TMA{tma}.csv"
    df = safe_read_csv(path)
    missing = {"coordinateX", "coordinateY", annotation_col} - set(df.columns)
    if missing:
        raise ValueError(f"{path} missing columns: {sorted(missing)}")
    df[annotation_col] = df[annotation_col].map(clean_label)
    df = df[df[annotation_col] != ""].copy()
    df["coordinateX"] = pd.to_numeric(df["coordinateX"], errors="coerce")
    df["coordinateY"] = pd.to_numeric(df["coordinateY"], errors="coerce")
    df = df[np.isfinite(df["coordinateX"]) & np.isfinite(df["coordinateY"])].copy()
    return df


def match_annotation_to_cells(
    annotated: pd.DataFrame,
    cells: pd.DataFrame,
    *,
    annotation_col: str,
    max_distance: float,
) -> tuple[pd.DataFrame, dict[str, object]]:
    cell_xy = cells[["x_centroid", "y_centroid"]].to_numpy(dtype=float)
    tree = cKDTree(cell_xy)
    query_xy = annotated[["coordinateX", "coordinateY"]].to_numpy(dtype=float)
    dist, idx = tree.query(query_xy, k=1)
    matched = annotated.copy()
    matched["_nn_distance"] = dist
    matched["_xenium_cell_id"] = cells.iloc[idx]["cell_id"].to_numpy()
    matched = matched[matched["_nn_distance"] <= max_distance].copy()
    matched.sort_values("_nn_distance", inplace=True)

    conflict_counter = 0
    label_by_cell: dict[str, str] = {}
    keep_rows = []
    for _, row in matched.iterrows():
        cell_id = str(row["_xenium_cell_id"])
        label = str(row[annotation_col])
        old = label_by_cell.get(cell_id)
        if old is not None:
            if old != label:
                conflict_counter += 1
            continue
        label_by_cell[cell_id] = label
        keep_rows.append(row)
    dedup = pd.DataFrame(keep_rows)

    distances = matched["_nn_distance"].to_numpy(dtype=float)
    stats = {
        "n_annotation_rows": int(len(annotated)),
        "n_matched_rows_within_threshold": int(len(matched)),
        "n_unique_matched_cells": int(len(dedup)),
        "n_duplicate_or_conflicting_cell_matches_dropped": int(len(matched) - len(dedup)),
        "n_duplicate_conflicting_label_matches": int(conflict_counter),
        "max_match_distance": float(max_distance),
        "distance_percentiles": {
            "p50": float(np.percentile(distances, 50)) if distances.size else None,
            "p90": float(np.percentile(distances, 90)) if distances.size else None,
            "p95": float(np.percentile(distances, 95)) if distances.size else None,
            "p99": float(np.percentile(distances, 99)) if distances.size else None,
            "max": float(np.max(distances)) if distances.size else None,
        },
    }
    return dedup, stats


def load_boundaries_for_cells(raw_xenium_root: Path, raw_slide: str, cell_ids: Iterable[str]) -> pd.DataFrame:
    ids = set(str(cell_id) for cell_id in cell_ids)
    path = raw_xenium_root / raw_slide / "cell_boundaries.parquet"
    bd = pd.read_parquet(path, columns=["cell_id", "vertex_x", "vertex_y"])
    bd["cell_id"] = bd["cell_id"].astype(str)
    return bd[bd["cell_id"].isin(ids)].copy()


def draw_filled_masks(
    he: Image.Image,
    boundaries: pd.DataFrame,
    matched: pd.DataFrame,
    *,
    manifest: dict,
    crop_bbox: list[int],
    params: dict,
    annotation_col: str,
    out_dir: Path,
) -> tuple[Image.Image, Image.Image, dict[str, int], dict[str, dict[str, str]]]:
    width, height = he.size
    rgb_mask = Image.new("RGB", (width, height), "white")
    rgb_draw = ImageDraw.Draw(rgb_mask)
    label_masks: dict[str, Image.Image] = {}
    label_draws: dict[str, ImageDraw.ImageDraw] = {}
    id_to_label = {str(row["_xenium_cell_id"]): str(row[annotation_col]) for _, row in matched.iterrows()}
    counts: Counter[str] = Counter()
    outputs: dict[str, dict[str, str]] = {}

    labels_dir = out_dir / "per_class_masks"
    labels_dir.mkdir(parents=True, exist_ok=True)

    for cell_id, group in boundaries.groupby("cell_id", sort=False):
        label = id_to_label.get(str(cell_id))
        if not label:
            continue
        xy = project_raw_xy_to_local_he(
            group["vertex_x"].to_numpy(dtype=float),
            group["vertex_y"].to_numpy(dtype=float),
            manifest=manifest,
            crop_bbox=crop_bbox,
            width=width,
            height=height,
            params=params,
        )
        finite = np.isfinite(xy).all(axis=1)
        poly = [(float(x), float(y)) for x, y in xy[finite]]
        if len(poly) < 3:
            continue
        color = ws_align.base.GT_PALETTE.get(label, "#999999")
        rgb = rgb_tuple(color)
        rgb_draw.polygon(poly, fill=rgb)
        if label not in label_masks:
            label_masks[label] = Image.new("L", (width, height), 0)
            label_draws[label] = ImageDraw.Draw(label_masks[label])
        label_draws[label].polygon(poly, fill=255)
        counts[label] += 1

    for label, mask in sorted(label_masks.items()):
        slug = slugify(label)
        binary_path = labels_dir / f"{slug}_binary.png"
        color_path = labels_dir / f"{slug}_color.png"
        mask.save(binary_path)
        color = rgb_tuple(ws_align.base.GT_PALETTE.get(label, "#999999"))
        color_img = Image.new("RGB", (width, height), "white")
        arr = np.asarray(mask, dtype=np.uint8) > 0
        color_arr = np.array(color_img)
        color_arr[arr] = color
        Image.fromarray(color_arr, "RGB").save(color_path)
        outputs[label] = {
            "binary_mask": str(binary_path.relative_to(out_dir)),
            "color_mask": str(color_path.relative_to(out_dir)),
        }

    overlay = overlay_mask_on_he(he, rgb_mask)
    return rgb_mask, overlay, dict(counts), outputs


def write_tma_panel(
    he: Image.Image,
    point_overlay: Image.Image,
    fill_overlay: Image.Image,
    fill_mask: Image.Image,
    out_path: Path,
    *,
    title: str,
    subtitle: str,
) -> None:
    tiles = [
        (he, "H&E crop", "manual accepted no-margin frame"),
        (point_overlay, "Original cell-level annotation", "sparse points from metadata"),
        (fill_overlay, "Continuous GT overlay", "nearest centroid + cell boundary fill"),
        (fill_mask, "Continuous GT mask only", "H&E removed; evaluation mask"),
    ]
    tile_w = 520
    tile_h = 460
    header_h = 58
    gap = 14
    canvas = Image.new("RGB", (len(tiles) * tile_w + (len(tiles) + 1) * gap, tile_h + header_h + 86), "#edf1f6")
    draw = ImageDraw.Draw(canvas)
    title_font = font(20)
    small_font = font(13)
    draw.text((gap, 12), title, fill="#172033", font=title_font)
    draw.text((gap, 40), subtitle, fill="#526173", font=small_font)
    top = 72
    for idx, (image, heading, desc) in enumerate(tiles):
        x = gap + idx * (tile_w + gap)
        draw.rectangle((x, top, x + tile_w, top + header_h + tile_h), fill="white", outline="#cbd5e1")
        draw.text((x + 14, top + 10), heading, fill="#172033", font=title_font)
        draw.text((x + 14, top + 35), desc, fill="#64748b", font=small_font)
        thumb = image.convert("RGB").copy()
        thumb.thumbnail((tile_w, tile_h), Image.Resampling.LANCZOS)
        canvas.paste(thumb, (x + (tile_w - thumb.width) // 2, top + header_h + (tile_h - thumb.height) // 2))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)


def build_one_tma(args: argparse.Namespace, tma: int) -> dict[str, object]:
    slide = slide_for_tma(tma)
    raw_slide = raw_slide_for_tma(tma)
    tma_dir = args.out_root / f"TMA{tma:02d}_{slide}"
    image_dir = tma_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)

    manifest = manual_tool.read_manifest(args.align_root, slide)
    params = load_manual_params(args.align_root, slide, tma)
    config = manual_tool.ensure_assets(args.align_root, slide, tma=tma, pad_px=0, options=[])
    crop_bbox = config["crop_bbox_xyxy_in_oriented_he_grid"]
    he = Image.open(args.align_root / config["he"]).convert("RGB")
    width, height = he.size

    annotated = read_annotated_metadata(args.metadata_root, tma, args.annotation_col)
    cells_path = args.raw_xenium_root / raw_slide / "cells.parquet"
    cells = pd.read_parquet(cells_path, columns=["cell_id", "x_centroid", "y_centroid"])
    cells["cell_id"] = cells["cell_id"].astype(str)
    matched, match_stats = match_annotation_to_cells(
        annotated,
        cells,
        annotation_col=args.annotation_col,
        max_distance=args.max_match_distance,
    )
    boundaries = load_boundaries_for_cells(args.raw_xenium_root, raw_slide, matched["_xenium_cell_id"])

    point_xy = project_raw_xy_to_local_he(
        annotated["coordinateX"].to_numpy(dtype=float),
        annotated["coordinateY"].to_numpy(dtype=float),
        manifest=manifest,
        crop_bbox=crop_bbox,
        width=width,
        height=height,
        params=params,
    )
    point_labels = [str(label) for label in annotated[args.annotation_col].tolist()]
    point_counts = dict(Counter(point_labels))
    point_overlay, point_mask = draw_point_annotation(he, point_xy, point_labels, ws_align.base.GT_PALETTE)
    point_overlay = append_legend(point_overlay, point_counts, f"TMA{tma} point annotation", "raw cell-level points")

    fill_mask, fill_overlay, fill_counts, per_class_outputs = draw_filled_masks(
        he,
        boundaries,
        matched,
        manifest=manifest,
        crop_bbox=crop_bbox,
        params=params,
        annotation_col=args.annotation_col,
        out_dir=tma_dir,
    )
    fill_overlay_legend = append_legend(fill_overlay, fill_counts, f"TMA{tma} continuous GT", "filled Xenium cell boundaries")
    fill_mask_legend = append_legend(fill_mask, fill_counts, f"TMA{tma} GT mask only", "no H&E background")

    he_out = image_dir / f"TMA{tma:02d}_he.png"
    point_overlay_out = image_dir / f"TMA{tma:02d}_point_annotation_overlay.png"
    point_mask_out = image_dir / f"TMA{tma:02d}_point_annotation_mask_only.png"
    fill_overlay_out = image_dir / f"TMA{tma:02d}_cell_boundary_fill_overlay.png"
    fill_mask_out = image_dir / f"TMA{tma:02d}_cell_boundary_fill_mask_only.png"
    fill_overlay_legend_out = image_dir / f"TMA{tma:02d}_cell_boundary_fill_overlay_with_legend.png"
    fill_mask_legend_out = image_dir / f"TMA{tma:02d}_cell_boundary_fill_mask_only_with_legend.png"
    panel_out = image_dir / f"TMA{tma:02d}_point_vs_cellboundary_gt_panel.png"

    he.save(he_out)
    point_overlay.save(point_overlay_out)
    point_mask.save(point_mask_out)
    fill_overlay.save(fill_overlay_out)
    fill_mask.save(fill_mask_out)
    fill_overlay_legend.save(fill_overlay_legend_out)
    fill_mask_legend.save(fill_mask_legend_out)

    subtitle = (
        f"{len(annotated):,} annotated rows; {match_stats['n_unique_matched_cells']:,} unique cells filled; "
        f"NN p95={match_stats['distance_percentiles']['p95']:.2f}px; max allowed={args.max_match_distance:.1f}px"
    )
    write_tma_panel(
        he,
        point_overlay,
        fill_overlay_legend,
        fill_mask_legend,
        panel_out,
        title=f"TMA{tma} {slide}: point annotation vs continuous GT",
        subtitle=subtitle,
    )

    record = {
        "tma": tma,
        "slide": slide,
        "raw_slide": raw_slide,
        "annotation_column": args.annotation_col,
        "image_size_px": {"width": width, "height": height},
        "crop_bbox_xyxy_in_oriented_he_grid": crop_bbox,
        "manual_alignment": {
            "dx": params["dx"],
            "dy": params["dy"],
            "sx": params["sx"],
            "sy": params["sy"],
            "source_param": params["path"],
        },
        "match_stats": match_stats,
        "point_annotation_counts": point_counts,
        "filled_cell_counts": fill_counts,
        "boundary_vertices_loaded": int(len(boundaries)),
        "paths": {
            "tma_dir": str(tma_dir),
            "he": str(he_out.relative_to(args.out_root)),
            "point_overlay": str(point_overlay_out.relative_to(args.out_root)),
            "point_mask_only": str(point_mask_out.relative_to(args.out_root)),
            "filled_overlay": str(fill_overlay_out.relative_to(args.out_root)),
            "filled_mask_only": str(fill_mask_out.relative_to(args.out_root)),
            "filled_overlay_with_legend": str(fill_overlay_legend_out.relative_to(args.out_root)),
            "filled_mask_only_with_legend": str(fill_mask_legend_out.relative_to(args.out_root)),
            "comparison_panel": str(panel_out.relative_to(args.out_root)),
        },
        "per_class_outputs": per_class_outputs,
    }
    (tma_dir / "manifest.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    return record


def write_index(records: list[dict[str, object]], out_root: Path) -> None:
    rows = []
    cards = []
    for rec in records:
        stats = rec["match_stats"]
        rows.append(
            f"<tr><td>TMA{rec['tma']}</td><td>{html.escape(str(rec['slide']))}</td>"
            f"<td>{stats['n_annotation_rows']:,}</td><td>{stats['n_unique_matched_cells']:,}</td>"
            f"<td>{stats['distance_percentiles']['p95']:.2f}</td><td>{stats['distance_percentiles']['max']:.2f}</td>"
            f"<td>{rec['image_size_px']['width']} x {rec['image_size_px']['height']}</td></tr>"
        )
        paths = rec["paths"]
        cards.append(
            f"""
            <section class="card" id="tma{rec['tma']}">
              <h2>TMA{rec['tma']} <span>{html.escape(str(rec['slide']))}</span></h2>
              <p>Point annotation rows: {stats['n_annotation_rows']:,}; unique filled cells: {stats['n_unique_matched_cells']:,}; p95 nearest-cell distance: {stats['distance_percentiles']['p95']:.2f}px.</p>
              <div class="grid">
                <figure><img src="{html.escape(paths['he'])}"><figcaption>H&amp;E crop</figcaption></figure>
                <figure><img src="{html.escape(paths['point_overlay'])}"><figcaption>Original point annotation</figcaption></figure>
                <figure><img src="{html.escape(paths['filled_overlay_with_legend'])}"><figcaption>Continuous GT overlay with legend</figcaption></figure>
                <figure><img src="{html.escape(paths['filled_mask_only_with_legend'])}"><figcaption>Continuous GT mask only</figcaption></figure>
              </div>
              <p><a href="{html.escape(str(Path(paths['tma_dir']).name + '/manifest.json'))}">Per-TMA manifest</a></p>
            </section>
            """
        )
    out_root.joinpath("index.html").write_text(
        f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Xenium Silica Continuous GT Masks</title>
<style>
body{{margin:0;font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;background:#f5f7fb;color:#172033}}
header{{background:#111827;color:white;padding:24px 30px}}
main{{padding:24px 30px 50px}}
table{{border-collapse:collapse;width:100%;background:white;border:1px solid #d8dee8;margin:18px 0}}
th,td{{padding:8px 10px;border-bottom:1px solid #e5e9f0;text-align:left;font-size:13px}}
th{{background:#edf1f6}}
.card{{background:white;border:1px solid #d7dee8;border-radius:8px;margin:18px 0;padding:16px}}
.card h2{{margin:0 0 8px;font-size:20px}} .card h2 span{{font-size:14px;color:#64748b}}
.grid{{display:grid;grid-template-columns:1fr 1fr 1.35fr 1.35fr;gap:14px;align-items:start}}
figure{{margin:0}} figcaption{{font-size:12px;color:#526173;margin-top:6px}}
img{{width:100%;height:auto;border:1px solid #d8dee8;background:white}}
code{{background:#edf1f6;padding:2px 5px;border-radius:4px;color:#111827}}
@media(max-width:1200px){{.grid{{grid-template-columns:1fr}}}}
</style>
</head>
<body>
<header>
<h1>Xenium Silica Continuous Ground Truth Masks</h1>
<p>These masks are evaluation-only GT. The source <code>new_annotation</code> is cell-level, so the raw data appears as points. We map each annotated point to the nearest Xenium segmented cell and fill the corresponding <code>cell_boundaries.parquet</code> polygon to make continuous masks.</p>
</header>
<main>
<table><thead><tr><th>TMA</th><th>Slide</th><th>Annotated rows</th><th>Filled cells</th><th>NN p95 px</th><th>NN max px</th><th>H&amp;E size</th></tr></thead><tbody>
{''.join(rows)}
</tbody></table>
{''.join(cards)}
</main>
</body>
</html>
""",
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--align-root", type=Path, default=DEFAULT_ALIGN_ROOT)
    parser.add_argument("--raw-xenium-root", type=Path, default=DEFAULT_RAW_XENIUM_ROOT)
    parser.add_argument("--metadata-root", type=Path, default=DEFAULT_METADATA_ROOT)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument("--tmas", nargs="+", type=int, default=ANNOTATED_TMAS)
    parser.add_argument("--annotation-col", default=GT_COL)
    parser.add_argument("--max-match-distance", type=float, default=10.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_root.mkdir(parents=True, exist_ok=True)
    records = []
    for tma in args.tmas:
        record = build_one_tma(args, tma)
        records.append(record)
        print(
            json.dumps(
                {
                    "tma": tma,
                    "annotated": record["match_stats"]["n_annotation_rows"],
                    "filled_cells": record["match_stats"]["n_unique_matched_cells"],
                    "p95": record["match_stats"]["distance_percentiles"]["p95"],
                },
                sort_keys=True,
            ),
            flush=True,
        )
    manifest = {
        "out_root": str(args.out_root),
        "annotation_column": args.annotation_col,
        "gt_usage": "evaluation only; not used as FICTURE/SAM/VLM input",
        "input_annotation_type": "cell-level label rows in TMA*.csv",
        "continuous_mask_policy": "nearest metadata coordinate to Xenium cells.parquet centroid, then fill matching cell_boundaries.parquet polygon",
        "max_match_distance": args.max_match_distance,
        "records": records,
    }
    (args.out_root / "continuous_gt_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    pd.DataFrame(
        [
            {
                "tma": rec["tma"],
                "slide": rec["slide"],
                "annotation_rows": rec["match_stats"]["n_annotation_rows"],
                "unique_filled_cells": rec["match_stats"]["n_unique_matched_cells"],
                "match_p50": rec["match_stats"]["distance_percentiles"]["p50"],
                "match_p90": rec["match_stats"]["distance_percentiles"]["p90"],
                "match_p95": rec["match_stats"]["distance_percentiles"]["p95"],
                "match_p99": rec["match_stats"]["distance_percentiles"]["p99"],
                "match_max": rec["match_stats"]["distance_percentiles"]["max"],
                "image_width": rec["image_size_px"]["width"],
                "image_height": rec["image_size_px"]["height"],
                "comparison_panel": rec["paths"]["comparison_panel"],
            }
            for rec in records
        ]
    ).to_csv(args.out_root / "continuous_gt_summary.csv", index=False)
    write_index(records, args.out_root)
    print(json.dumps({"out_root": str(args.out_root), "index": str(args.out_root / "index.html")}, indent=2))


if __name__ == "__main__":
    main()
