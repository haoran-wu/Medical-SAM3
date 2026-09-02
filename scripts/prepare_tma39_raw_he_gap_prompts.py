#!/usr/bin/env python3
"""Create blind H&E rescue prompts where registered FICTURE has little signal.

The detector reads raw registered H&E pixels directly. It deliberately does
not use the existing H&E-SAM candidate inventory or any annotation.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage as ndi


Image.MAX_IMAGE_PIXELS = None

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "output" / "aaai_xenium_silica_20260623"
DEFAULT_HE = (
    DATA
    / "xenium_k12_expanded_candidate_inputs_strict_official_affinewarp_20260709"
    / "per_tma/TMA39/source_images/TMA39_he.png"
)
DEFAULT_FICTURE = (
    DATA
    / "xenium_k12_expanded_candidate_inputs_strict_official_affinewarp_20260709"
    / "per_tma/TMA39/source_images/TMA39_ficture_molecule_k12.png"
)
DEFAULT_OUT = (
    DATA
    / "TMA39_meeting_followup_20260813"
    / "genemap_ficture_primary_structure_20260819"
    / "independent_he_raw_gap_reprompt_20260819"
)

PROMPT_EXPANSIONS = (0.10, 0.20, 0.30, 0.40)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tma", default="TMA39")
    parser.add_argument("--he", type=Path, default=DEFAULT_HE)
    parser.add_argument("--ficture", type=Path, default=DEFAULT_FICTURE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--cell-size", type=int, default=64)
    parser.add_argument("--min-he-fraction", type=float, default=0.35)
    parser.add_argument("--max-ficture-fraction", type=float, default=0.08)
    parser.add_argument("--min-region-cells", type=int, default=2)
    parser.add_argument("--box-expansion", type=float, default=0.10)
    return parser.parse_args()


def he_foreground(image: np.ndarray) -> np.ndarray:
    pixels = image.astype(np.float32)
    intensity = pixels.mean(axis=2)
    chroma = pixels.max(axis=2) - pixels.min(axis=2)
    return (pixels.min(axis=2) < 244) & ((chroma >= 7) | (intensity < 226))


def block_fraction(mask: np.ndarray, block: int) -> np.ndarray:
    height, width = mask.shape
    grid_h = (height + block - 1) // block
    grid_w = (width + block - 1) // block
    out = np.zeros((grid_h, grid_w), dtype=np.float32)
    for row in range(grid_h):
        y0, y1 = row * block, min((row + 1) * block, height)
        for column in range(grid_w):
            x0, x1 = column * block, min((column + 1) * block, width)
            out[row, column] = float(mask[y0:y1, x0:x1].mean())
    return out


def expand_box(
    box: tuple[int, int, int, int], fraction: float, image_shape: tuple[int, int]
) -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = box
    pad_x = int(round((x1 - x0) * fraction))
    pad_y = int(round((y1 - y0) * fraction))
    height, width = image_shape
    return (
        max(0, x0 - pad_x),
        max(0, y0 - pad_y),
        min(width, x1 + pad_x),
        min(height, y1 + pad_y),
    )


def choose_positive_point(
    he: np.ndarray,
    he_mask: np.ndarray,
    ficture_signal: np.ndarray,
    box: tuple[int, int, int, int],
) -> tuple[int, int]:
    x0, y0, x1, y1 = box
    local_he = he_mask[y0:y1, x0:x1]
    local_gap = ~ficture_signal[y0:y1, x0:x1]
    support = local_he & local_gap
    if not support.any():
        support = local_he
    if not support.any():
        return ((x0 + x1) // 2, (y0 + y1) // 2)

    distance = ndi.distance_transform_edt(support)
    yy, xx = np.indices(support.shape)
    center_x = (support.shape[1] - 1) / 2.0
    center_y = (support.shape[0] - 1) / 2.0
    center_distance = np.sqrt((xx - center_x) ** 2 + (yy - center_y) ** 2)
    local_rgb = he[y0:y1, x0:x1].astype(np.float32)
    darkness = 255.0 - local_rgb.mean(axis=2)
    chroma = local_rgb.max(axis=2) - local_rgb.min(axis=2)
    stain = darkness + 0.35 * chroma
    distance_scale = max(float(distance.max()), 1.0)
    center_scale = max(float(center_distance[support].max()), 1.0)
    stain_scale = max(float(stain[support].max()), 1.0)
    # Use one annotation-free rule for every detected gap. The point should be
    # inside tissue, away from its edge, and close to the automatically detected
    # region center so a separate structure at one box edge cannot capture it.
    score = (
        distance / distance_scale
        - 0.65 * center_distance / center_scale
        + 0.15 * stain / stain_scale
    )
    score[~support] = -1.0
    py, px = np.unravel_index(int(np.argmax(score)), score.shape)
    return (x0 + int(px), y0 + int(py))


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def lighten_black(image: np.ndarray) -> np.ndarray:
    out = image.copy()
    out[np.max(out, axis=2) < 8] = (224, 229, 232)
    return out


def fit(image: np.ndarray, width: int, height: int) -> Image.Image:
    return Image.fromarray(image).resize((width, height), Image.Resampling.LANCZOS)


def draw_prompt_panel(
    image: np.ndarray,
    rows: list[dict[str, object]],
    box_key: str,
    box_color: tuple[int, int, int],
    show_point: bool = False,
) -> np.ndarray:
    canvas = Image.fromarray(image.copy())
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default(size=34)
    for row in rows:
        box = row[box_key]
        x0, y0, x1, y1 = (int(value) for value in box)
        draw.rectangle((x0, y0, x1, y1), outline=box_color, width=5)
        if show_point:
            point = (int(row["point_x"]), int(row["point_y"]))
            radius = 22
            draw.ellipse(
                (point[0] - radius, point[1] - radius, point[0] + radius, point[1] + radius),
                fill=(20, 20, 20),
                outline=(255, 255, 255),
                width=4,
            )
        draw.rounded_rectangle(
            (x0 + 10, y0 + 10, x0 + 170, y0 + 68),
            radius=8,
            fill=(255, 255, 255),
            outline=box_color,
            width=2,
        )
        draw.text((x0 + 22, y0 + 16), str(row["region_id"]), fill=(15, 24, 28), font=font)
    return np.asarray(canvas)


def build_overview(
    he: np.ndarray,
    ficture: np.ndarray,
    he_fraction: np.ndarray,
    ficture_fraction: np.ndarray,
    selected_grid: np.ndarray,
    rows: list[dict[str, object]],
    output: Path,
    cell_size: int,
    tma: str,
) -> None:
    ficture_light = lighten_black(ficture)
    he_prompts = draw_prompt_panel(he, rows, "tight_box_xyxy", (0, 125, 120))
    ficture_prompts = draw_prompt_panel(
        ficture_light, rows, "tight_box_xyxy", (220, 92, 35)
    )

    grid_rgb = np.full((*selected_grid.shape, 3), 242, dtype=np.uint8)
    grid_rgb[..., 0] = np.uint8(np.clip(255 * (1.0 - he_fraction), 0, 255))
    grid_rgb[..., 1] = np.uint8(np.clip(220 * (1.0 - ficture_fraction), 0, 255))
    grid_rgb[..., 2] = 220
    grid_rgb[selected_grid] = (0, 145, 135)
    grid_full = np.asarray(
        Image.fromarray(grid_rgb).resize(
            (he.shape[1], he.shape[0]), Image.Resampling.NEAREST
        )
    )

    panels = [
        ("1. Registered H&E with all new prompts", he_prompts),
        ("2. Registered FICTURE with the same prompts", ficture_prompts),
        ("3. Automatically detected H&E-present / FICTURE-low cells", grid_full),
    ]
    if rows:
        gap_row = max(rows, key=lambda row: int(row["selected_cell_count"]))
        gx0, gy0, gx1, gy1 = (int(value) for value in gap_row["expanded_box_xyxy"])
        pad = 220
        cx0, cy0, cx1, cy1 = (
            max(0, gx0 - pad),
            max(0, gy0 - pad),
            min(he.shape[1], gx1 + pad),
            min(he.shape[0], gy1 + pad),
        )
        local_row = {
            **gap_row,
            "tight_box_xyxy": [
                int(gap_row["tight_box_xyxy"][0]) - cx0,
                int(gap_row["tight_box_xyxy"][1]) - cy0,
                int(gap_row["tight_box_xyxy"][2]) - cx0,
                int(gap_row["tight_box_xyxy"][3]) - cy0,
            ],
            "point_x": int(gap_row["point_x"]) - cx0,
            "point_y": int(gap_row["point_y"]) - cy0,
        }
        panels.extend(
            [
                (
                    f"4. Largest detected gap on H&E: {gap_row['region_id']}",
                    draw_prompt_panel(
                        he[cy0:cy1, cx0:cx1],
                        [local_row],
                        "tight_box_xyxy",
                        (0, 125, 120),
                    ),
                ),
                (
                    f"5. The same gap on FICTURE: {gap_row['region_id']}",
                    draw_prompt_panel(
                        ficture_light[cy0:cy1, cx0:cx1],
                        [local_row],
                        "tight_box_xyxy",
                        (220, 92, 35),
                    ),
                ),
            ]
        )
    card_w, card_h = 720, 800
    columns = 3
    panel_w, panel_h = 670, 670
    canvas = Image.new("RGB", (columns * card_w + 40, 2 * card_h + 180), "white")
    draw = ImageDraw.Draw(canvas)
    title_font = ImageFont.load_default(size=38)
    label_font = ImageFont.load_default(size=27)
    body_font = ImageFont.load_default(size=22)
    draw.text(
        (24, 18),
        f"{tma} blind H&E rescue prompts from raw H&E / FICTURE disagreement",
        fill=(24, 33, 38),
        font=title_font,
    )
    draw.text(
        (24, 68),
        (
            f"Rule: {cell_size} px cells; keep H&E tissue >= 35% and FICTURE signal <= 8%. "
            "Annotation and old H&E candidates are not used."
        ),
        fill=(69, 83, 92),
        font=body_font,
    )
    draw.text(
        (24, 104),
        (
            "Teal/orange rectangle = tight box; black dot = positive point. Both are used in SAM."
            if rows
            else "No region passed the fixed detector rule; zero independent H&E prompts were created."
        ),
        fill=(69, 83, 92),
        font=body_font,
    )
    for index, (title, panel) in enumerate(panels):
        row, column = divmod(index, columns)
        x = 20 + column * card_w
        y = 150 + row * card_h
        draw.rectangle((x, y, x + 700, y + 760), outline=(205, 215, 220), width=2)
        draw.text((x + 14, y + 12), title, fill=(24, 33, 38), font=label_font)
        canvas.paste(fit(panel, panel_w, panel_h), (x + 15, y + 55))
    canvas.save(output, optimize=True)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    he = np.asarray(Image.open(args.he).convert("RGB"))
    ficture = np.asarray(Image.open(args.ficture).convert("RGB"))
    if he.shape != ficture.shape:
        raise ValueError(f"Registered inputs differ: H&E={he.shape}, FICTURE={ficture.shape}")

    he_mask = he_foreground(he)
    ficture_signal = np.max(ficture, axis=2) >= 8
    he_fraction = block_fraction(he_mask, args.cell_size)
    ficture_fraction = block_fraction(ficture_signal, args.cell_size)
    selected_grid = (he_fraction >= args.min_he_fraction) & (
        ficture_fraction <= args.max_ficture_fraction
    )
    selected_grid = ndi.binary_closing(selected_grid, structure=np.ones((3, 3), dtype=bool))

    labels, count = ndi.label(selected_grid)
    sizes = np.bincount(labels.ravel())
    objects = ndi.find_objects(labels)
    regions: list[dict[str, object]] = []
    height, width = he.shape[:2]
    for label_id, slices in enumerate(objects, start=1):
        if slices is None or int(sizes[label_id]) < args.min_region_cells:
            continue
        row_slice, column_slice = slices
        tight_box = (
            column_slice.start * args.cell_size,
            row_slice.start * args.cell_size,
            min(width, column_slice.stop * args.cell_size),
            min(height, row_slice.stop * args.cell_size),
        )
        expanded_boxes = {
            fraction: expand_box(tight_box, fraction, (height, width))
            for fraction in PROMPT_EXPANSIONS
        }
        point_x, point_y = choose_positive_point(
            he, he_mask, ficture_signal, tight_box
        )
        component = labels == label_id
        regions.append(
            {
                "source_label_id": label_id,
                "selected_cell_count": int(sizes[label_id]),
                "tight_box_xyxy": tight_box,
                "expanded_box_xyxy": expanded_boxes[args.box_expansion],
                **{
                    f"expanded{int(round(fraction * 100))}_box_xyxy": box
                    for fraction, box in expanded_boxes.items()
                },
                "point_x": point_x,
                "point_y": point_y,
                "mean_he_tissue_fraction": float(he_fraction[component].mean()),
                "mean_ficture_signal_fraction": float(
                    ficture_fraction[component].mean()
                ),
            }
        )
    regions.sort(key=lambda row: (int(row["tight_box_xyxy"][1]), int(row["tight_box_xyxy"][0])))
    for index, row in enumerate(regions, start=1):
        row["region_id"] = f"R{index}"

    audit_rows: list[dict[str, object]] = []
    for row in regions:
        audit_rows.append(
            {
                "region_id": row["region_id"],
                "selected_cell_count": row["selected_cell_count"],
                "tight_box_xyxy": json.dumps(row["tight_box_xyxy"]),
                "expanded_box_xyxy": json.dumps(row["expanded_box_xyxy"]),
                "point_x": row["point_x"],
                "point_y": row["point_y"],
                "mean_he_tissue_fraction": f"{row['mean_he_tissue_fraction']:.6f}",
                "mean_ficture_signal_fraction": f"{row['mean_ficture_signal_fraction']:.6f}",
                "selection_used_annotation": False,
                "existing_he_masks_used_for_detection": False,
            }
        )
    audit_fields = [
        "region_id",
        "selected_cell_count",
        "tight_box_xyxy",
        "expanded_box_xyxy",
        "point_x",
        "point_y",
        "mean_he_tissue_fraction",
        "mean_ficture_signal_fraction",
        "selection_used_annotation",
        "existing_he_masks_used_for_detection",
    ]
    write_csv(args.output_dir / "detected_regions.csv", audit_rows, audit_fields)

    prompt_fields = [
        "prompt_id",
        "region_id",
        "box_x1",
        "box_y1",
        "box_x2",
        "box_y2",
        "point_x",
        "point_y",
        "selection_used_annotation",
    ]
    prompt_variants = [
        ("tight_box", "tight_box_xyxy", False),
        ("tight_box_point", "tight_box_xyxy", True),
    ]
    for fraction in PROMPT_EXPANSIONS:
        percentage = int(round(fraction * 100))
        box_key = f"expanded{percentage}_box_xyxy"
        prompt_variants.extend(
            [
                (f"expanded{percentage}_box", box_key, False),
                (f"expanded{percentage}_box_point", box_key, True),
            ]
        )
    crop_prompts: dict[str, dict[str, object]] = {}
    for name, box_key, include_point in prompt_variants:
        prompt_rows: list[dict[str, object]] = []
        for row in regions:
            x0, y0, x1, y1 = row[box_key]
            prompt_rows.append(
                {
                    "prompt_id": f"{name}_{row['region_id']}",
                    "region_id": row["region_id"],
                    "box_x1": x0,
                    "box_y1": y0,
                    "box_x2": x1,
                    "box_y2": y1,
                    "point_x": row["point_x"] if include_point else "",
                    "point_y": row["point_y"] if include_point else "",
                    "selection_used_annotation": False,
                }
            )
        write_csv(args.output_dir / f"prompts_{name}.csv", prompt_rows, prompt_fields)

    crop_pad = 256
    largest: dict[str, object] | None = None
    crop_box: tuple[int, int, int, int] | None = None
    if regions:
        largest = max(regions, key=lambda row: int(row["selected_cell_count"]))
        lx0, ly0, lx1, ly1 = largest["expanded_box_xyxy"]
        crop_box = (
            max(0, int(lx0) - crop_pad),
            max(0, int(ly0) - crop_pad),
            min(width, int(lx1) + crop_pad),
            min(height, int(ly1) + crop_pad),
        )
        cx0, cy0, cx1, cy1 = crop_box
        crop_dir = args.output_dir / "largest_gap_crop"
        crop_dir.mkdir(parents=True, exist_ok=True)
        Image.fromarray(he[cy0:cy1, cx0:cx1]).save(crop_dir / "R7_he_crop.png")
        Image.fromarray(ficture[cy0:cy1, cx0:cx1]).save(
            crop_dir / "R7_ficture_crop.png"
        )
        for name, box_key, include_point in prompt_variants:
            x0, y0, x1, y1 = largest[box_key]
            crop_prompt = {
                "prompt_id": f"{name}_{largest['region_id']}",
                "region_id": largest["region_id"],
                "box_x1": int(x0) - cx0,
                "box_y1": int(y0) - cy0,
                "box_x2": int(x1) - cx0,
                "box_y2": int(y1) - cy0,
                "point_x": int(largest["point_x"]) - cx0 if include_point else "",
                "point_y": int(largest["point_y"]) - cy0 if include_point else "",
                "selection_used_annotation": False,
            }
            write_csv(
                crop_dir / f"prompts_{name}.csv",
                [crop_prompt],
                prompt_fields,
            )
            crop_prompts[name] = crop_prompt

        write_csv(
            crop_dir / "prompts_expanded20_30_40_box.csv",
            [crop_prompts[f"expanded{percentage}_box"] for percentage in (20, 30, 40)],
            prompt_fields,
        )
        write_csv(
            crop_dir / "prompts_expanded20_30_40_box_point.csv",
            [
                crop_prompts[f"expanded{percentage}_box_point"]
                for percentage in (20, 30, 40)
            ],
            prompt_fields,
        )

    region_crop_manifest: list[dict[str, object]] = []
    region_crop_root = args.output_dir / "region_crops"
    for row in regions:
        region_id = str(row["region_id"])
        x0, y0, x1, y1 = row["expanded10_box_xyxy"]
        region_crop_box = (
            max(0, int(x0) - crop_pad),
            max(0, int(y0) - crop_pad),
            min(width, int(x1) + crop_pad),
            min(height, int(y1) + crop_pad),
        )
        rx0, ry0, rx1, ry1 = region_crop_box
        region_dir = region_crop_root / region_id
        region_dir.mkdir(parents=True, exist_ok=True)
        Image.fromarray(he[ry0:ry1, rx0:rx1]).save(region_dir / "he_crop.png")
        Image.fromarray(ficture[ry0:ry1, rx0:rx1]).save(
            region_dir / "ficture_crop.png"
        )
        region_prompt = {
            "prompt_id": f"expanded10_box_{region_id}",
            "region_id": region_id,
            "box_x1": int(x0) - rx0,
            "box_y1": int(y0) - ry0,
            "box_x2": int(x1) - rx0,
            "box_y2": int(y1) - ry0,
            "point_x": "",
            "point_y": "",
            "selection_used_annotation": False,
        }
        write_csv(region_dir / "prompt_expanded10_box.csv", [region_prompt], prompt_fields)
        region_point_prompt = {
            **region_prompt,
            "prompt_id": f"expanded10_box_point_{region_id}",
            "point_x": int(row["point_x"]) - rx0,
            "point_y": int(row["point_y"]) - ry0,
        }
        write_csv(
            region_dir / "prompt_expanded10_box_point.csv",
            [region_point_prompt],
            prompt_fields,
        )
        region_crop_manifest.append(
            {
                "region_id": region_id,
                "crop_xyxy_full_canvas": list(region_crop_box),
                "prompt_box_xyxy_crop": [
                    region_prompt["box_x1"],
                    region_prompt["box_y1"],
                    region_prompt["box_x2"],
                    region_prompt["box_y2"],
                ],
                "prompt_point_xy_crop": [
                    region_point_prompt["point_x"],
                    region_point_prompt["point_y"],
                ],
                "selection_used_annotation": False,
            }
        )
    region_crop_root.mkdir(parents=True, exist_ok=True)
    (region_crop_root / "manifest.json").write_text(
        json.dumps(region_crop_manifest, indent=2), encoding="utf-8"
    )

    Image.fromarray(he_mask.astype(np.uint8) * 255).save(
        args.output_dir / "raw_he_tissue_pixels.png"
    )
    Image.fromarray(ficture_signal.astype(np.uint8) * 255).save(
        args.output_dir / "registered_ficture_signal_pixels.png"
    )
    overview = args.output_dir / f"{args.tma}_Raw_HE_Gap_Prompts_G_Figure.png"
    build_overview(
        he,
        ficture,
        he_fraction,
        ficture_fraction,
        selected_grid,
        regions,
        overview,
        args.cell_size,
        args.tma,
    )
    manifest = {
        "goal": "Find H&E tissue where registered FICTURE has little signal and create new SAM prompts.",
        "selection_used_annotation": False,
        "existing_he_masks_used_for_detection": False,
        "he_path": str(args.he.resolve()),
        "ficture_path": str(args.ficture.resolve()),
        "canvas_height_width": [height, width],
        "cell_size_px": args.cell_size,
        "min_he_tissue_fraction": args.min_he_fraction,
        "max_ficture_signal_fraction": args.max_ficture_fraction,
        "minimum_connected_cells": args.min_region_cells,
        "box_expansion_fraction": args.box_expansion,
        "detected_region_count": len(regions),
        "largest_detected_region": None
        if largest is None
        else {
            "region_id": largest["region_id"],
            "tight_box_xyxy": largest["tight_box_xyxy"],
            "point_xy": [largest["point_x"], largest["point_y"]],
            "selected_cell_count": largest["selected_cell_count"],
            "diagnostic_crop_xyxy": crop_box,
        },
        "g_figure": str(overview.resolve()),
    }
    (args.output_dir / "prompt_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
