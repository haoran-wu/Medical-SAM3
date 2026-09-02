#!/usr/bin/env python3
"""Build a self-contained TMA30 report from the fixed TMA39 workflow outputs."""

from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import html
import io
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage as ndi


Image.MAX_IMAGE_PIXELS = None

MODULE_COLORS = (
    "#0072B2",
    "#E69F00",
    "#009E73",
    "#D55E00",
    "#CC79A7",
    "#56B4E9",
    "#F0E442",
    "#5B5F97",
    "#7A9E3A",
)
SOURCE_COLORS = {
    "Same-prompt H&E supplement": "#00B8C8",
    "Independent H&E supplement": "#D81B8C",
}
GT_COLORS = {
    "airway": "#6C7FE8",
    "arteriole": "#67C978",
    "venule": "#FF8A78",
    "alveoli": "#C77DD4",
}
GT_LABELS = {
    "airway": "Airway",
    "arteriole": "Arteriole",
    "venule": "Venule",
    "alveoli": "Alveoli",
}
GT_EXPECTED = {"airway": 3, "arteriole": 4, "venule": 4, "alveoli": 3}
EXPECTED_FIXED_FACTOR_INFO_SHA256 = "f89e0e599f6451ff37dbad24ff4b1a27b8c99bb66c033f5a40743268711181c6"
INPUT_NAMES = (
    ("01_close_he.jpg", "Close H&E"),
    ("02_close_ficture.png", "Close raw FICTURE"),
    ("03_medium_he.jpg", "Medium H&E"),
    ("04_medium_ficture.png", "Medium raw FICTURE"),
    ("05_full_he.jpg", "Full H&E locator"),
    ("06_full_ficture.png", "Full raw FICTURE locator"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def image_uri(image: Image.Image, fmt: str = "JPEG", quality: int = 90) -> str:
    buffer = io.BytesIO()
    kwargs: dict[str, Any] = {}
    if fmt.upper() == "JPEG":
        kwargs = {"quality": quality, "subsampling": 0, "optimize": True}
    elif fmt.upper() == "PNG":
        kwargs = {"optimize": True}
    image.save(buffer, format=fmt, **kwargs)
    mime = "image/jpeg" if fmt.upper() == "JPEG" else "image/png"
    return f"data:{mime};base64,{base64.b64encode(buffer.getvalue()).decode('ascii')}"


def file_uri(path: Path) -> str:
    suffix = path.suffix.lower()
    mime = "image/png" if suffix == ".png" else "image/jpeg"
    return f"data:{mime};base64,{base64.b64encode(path.read_bytes()).decode('ascii')}"


def figure_uri(figure: plt.Figure, *, dpi: int = 180) -> str:
    buffer = io.BytesIO()
    figure.savefig(buffer, format="png", dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    return f"data:image/png;base64,{base64.b64encode(buffer.getvalue()).decode('ascii')}"


def hex_rgb(value: str) -> tuple[int, int, int]:
    value = value.lstrip("#")
    return tuple(int(value[index : index + 2], 16) for index in (0, 2, 4))


def load_binary(path: Path) -> np.ndarray:
    data = np.asarray(Image.open(path).convert("L"), dtype=np.uint8) > 0
    if not data.any():
        raise RuntimeError(f"Empty mask: {path}")
    return data


def resize_for_report(image: Image.Image, max_side: int, *, ficture: bool) -> Image.Image:
    copy = image.convert("RGB")
    scale = min(1.0, max_side / max(copy.size))
    size = (max(1, round(copy.width * scale)), max(1, round(copy.height * scale)))
    return copy.resize(size, Image.Resampling.NEAREST if ficture else Image.Resampling.LANCZOS)


def table(headers: list[str], rows: list[list[str]]) -> str:
    return (
        "<div class='table-wrap'><table><thead><tr>"
        + "".join(f"<th>{heading}</th>" for heading in headers)
        + "</tr></thead><tbody>"
        + "".join(
            "<tr>" + "".join(f"<td>{value}</td>" for value in row) + "</tr>"
            for row in rows
        )
        + "</tbody></table></div>"
    )


def build_gt_regions(annotation_root: Path, shape: tuple[int, int]) -> list[dict[str, Any]]:
    regions: list[dict[str, Any]] = []
    next_id = 1
    for target_class in GT_LABELS:
        mask = load_binary(annotation_root / f"TMA30_authoritative_{target_class}_mask.png")
        if mask.shape != shape:
            raise RuntimeError(f"GT shape mismatch for {target_class}")
        components, count = ndi.label(mask, structure=np.ones((3, 3), dtype=np.uint8))
        if count != GT_EXPECTED[target_class]:
            raise RuntimeError(
                f"Expected {GT_EXPECTED[target_class]} {target_class} regions, found {count}"
            )
        for number in range(1, count + 1):
            component = components == number
            regions.append(
                {
                    "region_id": next_id,
                    "target_class": target_class,
                    "number": number,
                    "name": f"{GT_LABELS[target_class]} region {number}",
                    "area": int(component.sum()),
                    "mask": component,
                }
            )
            next_id += 1
    return regions


def module_green_image(score: np.ndarray, tissue: np.ndarray, size: int = 760) -> Image.Image:
    value = np.clip(score, 0.0, 1.0)
    rgb = np.zeros((*score.shape, 3), dtype=np.uint8)
    rgb[..., 1] = np.uint8(np.round(value * 255))
    rgb[..., 0] = np.uint8(np.round(np.clip((value - 0.72) / 0.28, 0, 1) * 150))
    rgb[..., 2] = np.uint8(np.round(value * 45))
    rgb[np.logical_and(tissue, value == 0)] = (4, 12, 8)
    image = Image.fromarray(rgb)
    return image.resize((size, size), Image.Resampling.NEAREST)


def genemap_overview(scores: np.ndarray, tissue: np.ndarray) -> tuple[str, list[str]]:
    panels = [module_green_image(scores[index], tissue) for index in range(9)]
    canvas = Image.new("RGB", (2400, 2520), "white")
    draw = ImageDraw.Draw(canvas)
    try:
        title_font = ImageFont.truetype("Arial Bold.ttf", 32)
    except OSError:
        title_font = ImageFont.load_default()
    for index, panel in enumerate(panels):
        x = (index % 3) * 800 + 20
        y = (index // 3) * 840 + 55
        canvas.paste(panel, (x, y))
        draw.text(
            (x, 14 + (index // 3) * 840),
            f"Module {index + 1}",
            fill="#16252B",
            font=title_font,
        )
    return image_uri(canvas, "PNG"), [image_uri(panel, "PNG") for panel in panels]


def distribution_figure(raw_scores: np.ndarray, tissue: np.ndarray, thresholds: dict[int, float], counts: dict[int, int]) -> str:
    figure, axes = plt.subplots(3, 3, figsize=(15.5, 13.5))
    for module_index, axis in enumerate(axes.flat, start=1):
        values = raw_scores[module_index - 1][tissue]
        low, high = np.percentile(values, [0.2, 99.8])
        shown = values[np.logical_and(values >= low, values <= high)]
        axis.hist(shown, bins=55, color="#2E75B6", edgecolor="white", linewidth=0.35)
        threshold = thresholds[module_index]
        axis.axvline(threshold, color="#E53935", linestyle="--", linewidth=2.2)
        axis.set_title(
            f"Module {module_index}: red line {threshold:.3f}\n{counts[module_index]} final prompt regions",
            fontsize=12,
            fontweight="bold",
        )
        axis.set_xlabel("Module score", fontsize=10)
        axis.set_ylabel("Number of tissue bins", fontsize=10)
        axis.grid(axis="y", color="#DDE5E8", alpha=0.7, linewidth=0.7)
        axis.spines[["top", "right"]].set_visible(False)
    figure.suptitle("TMA30 GeneMap module-score distributions", fontsize=18, fontweight="bold")
    figure.tight_layout(rect=(0, 0, 1, 0.965))
    return figure_uri(figure, dpi=190)


def prompt_overlay(background: Image.Image, rows: list[dict[str, str]], *, ficture: bool) -> Image.Image:
    shown = resize_for_report(background, 1450, ficture=ficture)
    draw = ImageDraw.Draw(shown)
    sx, sy = shown.width / background.width, shown.height / background.height
    for row in rows:
        module = int(row["gene_module"])
        color = hex_rgb(MODULE_COLORS[module - 1])
        x1, y1, x2, y2 = (int(float(row[key])) for key in ("box_x1", "box_y1", "box_x2", "box_y2"))
        rectangle = (round(x1 * sx), round(y1 * sy), round(x2 * sx), round(y2 * sy))
        draw.rectangle(rectangle, outline="white", width=4)
        draw.rectangle(rectangle, outline=color, width=2)
        px, py = int(float(row["point_x"])), int(float(row["point_y"]))
        px, py = round(px * sx), round(py * sy)
        radius = 5
        draw.ellipse((px - radius, py - radius, px + radius, py + radius), fill="white", outline="#11181B", width=2)
        draw.ellipse((px - 2, py - 2, px + 2, py + 2), fill="#11181B")
    return shown


def gap_prompt_overlay(background: Image.Image, rows: list[dict[str, str]]) -> Image.Image:
    shown = resize_for_report(background, 1450, ficture=False)
    draw = ImageDraw.Draw(shown)
    sx, sy = shown.width / background.width, shown.height / background.height
    for row in rows:
        box = json.loads(row["expanded_box_xyxy"])
        rectangle = tuple(round(value * scale) for value, scale in zip(box, (sx, sy, sx, sy)))
        draw.rectangle(rectangle, outline="white", width=4)
        draw.rectangle(rectangle, outline=SOURCE_COLORS["Independent H&E supplement"], width=2)
        px, py = round(int(row["point_x"]) * sx), round(int(row["point_y"]) * sy)
        draw.ellipse((px - 5, py - 5, px + 5, py + 5), fill="white", outline="#11181B", width=2)
        draw.text((rectangle[0] + 5, max(2, rectangle[1] - 20)), row["region_id"], fill="#11181B")
    return shown


def pool_overlay(background: Image.Image, pool_rows: list[dict[str, str]], pool_root: Path) -> Image.Image:
    shown = resize_for_report(background, 1450, ficture=False)
    values = np.asarray(shown, dtype=np.float32).copy()
    outlines: list[tuple[np.ndarray, tuple[int, int, int]]] = []
    for row in pool_rows:
        mask = Image.open(pool_root / row["mask_path"]).convert("L").resize(shown.size, Image.Resampling.NEAREST)
        binary = np.asarray(mask, dtype=np.uint8) > 0
        if row["source"] == "FICTURE primary":
            color = hex_rgb(MODULE_COLORS[int(row["gene_module"]) - 1])
        else:
            color = hex_rgb(SOURCE_COLORS[row["source"]])
        values[binary] = values[binary] * 0.62 + np.asarray(color, dtype=np.float32) * 0.38
        boundary = np.logical_and(binary, ~ndi.binary_erosion(binary, iterations=2))
        outlines.append((boundary, color))
    for boundary, color in outlines:
        values[boundary] = color
    return Image.fromarray(np.uint8(np.clip(values, 0, 255)))


def gt_overlay(background: Image.Image, regions: list[dict[str, Any]]) -> Image.Image:
    shown = resize_for_report(background, 1450, ficture=False)
    values = np.asarray(shown, dtype=np.float32).copy()
    for region in regions:
        binary = np.asarray(
            Image.fromarray(np.uint8(region["mask"]) * 255).resize(
                shown.size, Image.Resampling.NEAREST
            )
        ) > 0
        color = np.asarray(hex_rgb(GT_COLORS[region["target_class"]]), dtype=np.float32)
        values[binary] = values[binary] * 0.46 + color * 0.54
        boundary = np.logical_and(binary, ~ndi.binary_erosion(binary, iterations=2))
        values[boundary] = color
    return Image.fromarray(np.uint8(np.clip(values, 0, 255)))


def evaluate_candidates(
    pool_rows: list[dict[str, str]], pool_root: Path, regions: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    candidate_rows: list[dict[str, Any]] = []
    best_by_gt: dict[int, dict[str, Any]] = {}
    for order, row in enumerate(pool_rows, start=1):
        mask = load_binary(pool_root / row["mask_path"])
        area = int(mask.sum())
        shared_by_region = [int(np.logical_and(mask, region["mask"]).sum()) for region in regions]
        best_index = int(np.argmax(shared_by_region))
        region = regions[best_index]
        best_id = int(region["region_id"])
        best_shared = shared_by_region[best_index]
        gt_area = int(region["area"])
        metrics = {
            "shared": best_shared,
            "candidate_purity": best_shared / area,
            "gt_coverage": best_shared / gt_area,
            "iou": best_shared / (area + gt_area - best_shared),
            "dice": 2 * best_shared / (area + gt_area),
        }
        record = {
            **row,
            "order": order,
            "display_id": f"TMA30 C{order:02d}",
            "best_gt_id": best_id,
            "best_gt_name": region["name"],
            "best_gt_class": region["target_class"],
            **metrics,
        }
        candidate_rows.append(record)
        for compared_region, compared_shared in zip(regions, shared_by_region, strict=True):
            compared_gt_area = int(compared_region["area"])
            compared = {
                **row,
                "order": order,
                "display_id": f"TMA30 C{order:02d}",
                "best_gt_id": int(compared_region["region_id"]),
                "best_gt_name": compared_region["name"],
                "best_gt_class": compared_region["target_class"],
                "shared": compared_shared,
                "candidate_purity": compared_shared / area,
                "gt_coverage": compared_shared / compared_gt_area,
                "iou": compared_shared / (area + compared_gt_area - compared_shared),
                "dice": 2 * compared_shared / (area + compared_gt_area),
            }
            current = best_by_gt.get(int(compared_region["region_id"]))
            if current is None or compared_shared > current["shared"]:
                best_by_gt[int(compared_region["region_id"])] = compared
    gt_best_rows: list[dict[str, Any]] = []
    for region in regions:
        candidate = best_by_gt.get(region["region_id"])
        if candidate is None:
            raise RuntimeError(f"No candidate comparison for {region['name']}")
        gt_best_rows.append({**region, "candidate": candidate})
    return candidate_rows, gt_best_rows


def square_bounds(mask: np.ndarray, minimum: int = 850, scale: float = 1.5) -> tuple[int, int, int, int]:
    yy, xx = np.nonzero(mask)
    height, width = mask.shape
    x1, x2 = int(xx.min()), int(xx.max()) + 1
    y1, y2 = int(yy.min()), int(yy.max()) + 1
    center_x, center_y = (x1 + x2) / 2, (y1 + y2) / 2
    side = min(max(minimum, round(max(x2 - x1, y2 - y1) * scale)), width, height)
    left = max(0, min(round(center_x - side / 2), width - side))
    top = max(0, min(round(center_y - side / 2), height - side))
    return left, top, left + side, top + side


def overlap_view(he: Image.Image, candidate: np.ndarray, gt: np.ndarray) -> Image.Image:
    bounds = square_bounds(np.logical_or(candidate, gt))
    x1, y1, x2, y2 = bounds
    base = np.asarray(he.crop(bounds).resize((900, 900), Image.Resampling.LANCZOS), dtype=np.float32)
    c = np.asarray(Image.fromarray(np.uint8(candidate[y1:y2, x1:x2]) * 255).resize((900, 900), Image.Resampling.NEAREST)) > 0
    g = np.asarray(Image.fromarray(np.uint8(gt[y1:y2, x1:x2]) * 255).resize((900, 900), Image.Resampling.NEAREST)) > 0
    result = base.copy()
    for mask, color in (
        (np.logical_and(c, ~g), (36, 126, 210)),
        (np.logical_and(c, g), (32, 166, 91)),
        (np.logical_and(g, ~c), (222, 61, 150)),
    ):
        result[mask] = result[mask] * 0.38 + np.asarray(color) * 0.62
    return Image.fromarray(np.uint8(np.clip(result, 0, 255)))


def mask_comparison(candidate: np.ndarray, gt: np.ndarray) -> Image.Image:
    bounds = square_bounds(np.logical_or(candidate, gt))
    x1, y1, x2, y2 = bounds
    panels = []
    for mask in (candidate, gt):
        cropped = Image.fromarray(np.uint8(mask[y1:y2, x1:x2]) * 255)
        panels.append(cropped.resize((760, 760), Image.Resampling.NEAREST).convert("RGB"))
    canvas = Image.new("RGB", (1520, 760), "white")
    canvas.paste(panels[0], (0, 0))
    canvas.paste(panels[1], (760, 0))
    return canvas


def context_comparison(he: Image.Image, ficture: Image.Image, candidate: np.ndarray) -> Image.Image:
    bounds = square_bounds(candidate, minimum=1200, scale=3.0)
    h = he.crop(bounds).resize((760, 760), Image.Resampling.LANCZOS)
    f = ficture.crop(bounds).resize((760, 760), Image.Resampling.NEAREST)
    canvas = Image.new("RGB", (1520, 760), "white")
    canvas.paste(h, (0, 0))
    canvas.paste(f, (760, 0))
    return canvas


def best_gt_histogram(rows: list[dict[str, Any]]) -> str:
    purity = np.asarray([row["candidate"]["candidate_purity"] for row in rows])
    iou = np.asarray([row["candidate"]["iou"] for row in rows])
    bins = np.linspace(0, 1, 11)
    figure, axes = plt.subplots(1, 2, figsize=(11.5, 4.5), sharey=True)
    for axis, values, color, title, xlabel in (
        (axes[0], purity, "#2379B9", "Candidate purity", "Shared area / candidate area"),
        (axes[1], iou, "#D55E00", "IoU", "Shared area / combined area"),
    ):
        axis.hist(values, bins=bins, color=color, edgecolor="white", linewidth=1.1)
        median = float(np.median(values))
        axis.axvline(median, color="#172126", linestyle="--", linewidth=2)
        axis.text(median, axis.get_ylim()[1] * 0.92, f" median {median:.3f}", fontsize=10)
        axis.set_xlim(0, 1)
        axis.set_title(title, fontsize=14, fontweight="bold")
        axis.set_xlabel(xlabel)
        axis.set_ylabel("Number of official GT regions")
        axis.grid(axis="y", color="#DDE5E8", alpha=0.8)
        axis.spines[["top", "right"]].set_visible(False)
    figure.suptitle("One best frozen candidate for each of 14 official TMA30 regions", fontsize=16, fontweight="bold")
    figure.tight_layout(rect=(0, 0, 1, 0.93))
    return figure_uri(figure)


def source_label(row: dict[str, Any]) -> str:
    if row["source"] == "FICTURE primary":
        return f"FICTURE primary · Module {row['gene_module']}"
    return row["source"]


def factor_legend(rows: list[dict[str, str]]) -> tuple[str, str]:
    table_rows = []
    prompt_lines = []
    for row in sorted(rows, key=lambda value: int(value["Factor"])):
        genes = [item.strip() for item in (row.get("TopGene_specific") or "").split(",") if item.strip()][:5]
        swatch = f"<span class='swatch' style='background:{html.escape(row['hex'])}'></span>"
        table_rows.append(
            [
                str(int(row["Factor"])),
                f"{swatch}{html.escape(row['hex'].upper())}",
                html.escape(row["Celltype2"]),
                html.escape(", ".join(genes)),
            ]
        )
        prompt_lines.append(
            f"- Factor {int(row['Factor'])}: {row['hex'].upper()} / RGB {row['RGB']}; "
            f"marker-inferred cell type: {row['Celltype2']}; marker genes: {', '.join(genes)}."
        )
    return table(["Factor", "Fixed RGB", "Marker-inferred cell type", "Example marker genes"], table_rows), "\n".join(prompt_lines)


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    bundle = args.bundle.resolve()
    results_root = args.results.resolve()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    pool_manifest = json.loads((root / "frozen_pool/run_manifest.json").read_text())
    gene_summary = json.loads((root / "genemap_scores/summary.json").read_text())
    prompt_manifest = json.loads((root / "prompts/final_sustained_edge3/adaptive_prompt_manifest.json").read_text())
    bundle_manifest = json.loads((bundle / "run_manifest.json").read_text())
    run_manifest = json.loads((results_root / "run_manifest.json").read_text())
    pool_rows = read_csv(root / "frozen_pool/final_pool.csv")
    prompt_rows = read_csv(root / "prompts/final_sustained_edge3/adaptive_box_peak_sustained_edge3_context0.csv")
    base_prompt_rows = read_csv(root / "prompts/adaptive_base/adaptive_regions.csv")
    merge_rows = read_csv(root / "prompts/final_sustained_edge3/region_merge_audit.csv")
    threshold_rows = read_csv(root / "prompts/adaptive_base/module_thresholds.csv")
    gap_rows = read_csv(root / "independent_he_gap_detection/detected_regions.csv")
    same_prompt_rows = read_csv(root / "frozen_pool/same_prompt_he_audit.csv")
    module_rows = read_csv(root / "genemap_scores/gene_modules.csv")
    factor_rows = read_csv(bundle / "assets/factor_info.csv")
    bundle_rows = json.loads((bundle / "vlm_source_rows.json").read_text())
    vlm_rows = json.loads((results_root / "vlm_results.json").read_text())

    expected = 52
    if pool_manifest["final_candidate_count"] != expected or len(pool_rows) != expected:
        raise RuntimeError("The fixed TMA30 pool must contain exactly 52 candidates")
    if prompt_manifest["prompt_count"] != 45 or len(prompt_rows) != 45:
        raise RuntimeError("The corrected TMA30 run must contain 45 final GeneMap prompts")
    if gene_summary["gene_module_count"] != 9 or gene_summary["active_gene_count"] != 120:
        raise RuntimeError("The fixed TMA39 GeneMap definition was not retained")
    if any(row["selection_used_annotation"].lower() != "false" for row in pool_rows):
        raise RuntimeError("Candidate generation used annotation")
    if bundle_manifest["candidate_count"] != expected or run_manifest["candidate_count"] != expected:
        raise RuntimeError("VLM candidate count is not 52")
    if run_manifest["status"] != "complete" or len(vlm_rows) != expected:
        raise RuntimeError("VLM run is incomplete")
    if not run_manifest["six_image_hashes_per_candidate"]:
        raise RuntimeError("VLM six-image hash audit failed")
    if [row["candidate_uid"] for row in vlm_rows] != [row["candidate_uid"] for row in bundle_rows]:
        raise RuntimeError("VLM candidate order differs from the frozen bundle")

    he_image = Image.open(bundle / "assets/TMA30_he.png").convert("RGB")
    ficture_image = Image.open(bundle / "assets/TMA30_ficture.png").convert("RGB")
    if he_image.size != ficture_image.size or he_image.size != (6072, 6332):
        raise RuntimeError("Registered TMA30 input geometry changed")
    if sha256(bundle / "assets/TMA30_ficture.png") != bundle_manifest["input_hashes"]["ficture"]:
        raise RuntimeError("FICTURE hash mismatch")
    if sha256(bundle / "assets/factor_info.csv") != bundle_manifest["input_hashes"]["factor_info"]:
        raise RuntimeError("FICTURE legend hash mismatch")
    if bundle_manifest["input_hashes"]["factor_info"] != EXPECTED_FIXED_FACTOR_INFO_SHA256:
        raise RuntimeError("TMA30 is not using the fixed TMA39 FICTURE factor/RGB legend")

    annotation_root = bundle / "assets"
    gt_regions = build_gt_regions(annotation_root, (he_image.height, he_image.width))
    candidate_metrics, gt_best_rows = evaluate_candidates(
        pool_rows, root / "frozen_pool", gt_regions
    )
    candidate_by_uid = {row["candidate_id"]: row for row in candidate_metrics}

    scores_npz = np.load(root / "genemap_scores/gene_module_scores_unclipped.npz")
    raw_scores = scores_npz["raw_modules"]
    shown_scores = scores_npz["modules"]
    tissue = scores_npz["tissue"].astype(bool)
    if raw_scores.shape != (9, 98, 92) or int(tissue.sum()) != 4525:
        raise RuntimeError("Unexpected TMA30 GeneMap grid")
    thresholds = {int(row["gene_module"]): float(row["top5_threshold"]) for row in threshold_rows}
    module_counts = Counter(int(row["gene_module"]) for row in prompt_rows)
    genemap_uri, individual_module_uris = genemap_overview(shown_scores, tissue)
    distribution_uri = distribution_figure(raw_scores, tissue, thresholds, module_counts)
    g_he_uri = image_uri(prompt_overlay(he_image, prompt_rows, ficture=False), "JPEG", 94)
    g_ficture_uri = image_uri(prompt_overlay(ficture_image, prompt_rows, ficture=True), "PNG")
    gap_uri = image_uri(gap_prompt_overlay(he_image, gap_rows), "JPEG", 94)
    final_pool_uri = image_uri(pool_overlay(he_image, pool_rows, root / "frozen_pool"), "JPEG", 94)
    gt_uri = image_uri(gt_overlay(he_image, gt_regions), "JPEG", 94)
    overview_he_uri = image_uri(resize_for_report(he_image, 1450, ficture=False), "JPEG", 94)
    overview_ficture_uri = image_uri(resize_for_report(ficture_image, 1450, ficture=True), "PNG")
    histogram_uri = best_gt_histogram(gt_best_rows)

    module_table = []
    module_source_map = {int(row["module"]): row for row in module_rows}
    for module in range(1, 10):
        source = module_source_map[module]
        top_genes = source.get("fixed_genes", source.get("active_genes", ""))
        module_table.append(
            [
                f"<span class='swatch' style='background:{MODULE_COLORS[module - 1]}'></span>Module {module}",
                html.escape(source["fixed_gene_count"]),
                html.escape(", ".join(top_genes.split(", ")[:6])),
                str(module_counts[module]),
            ]
        )
    module_table_html = table(["Fixed module", "Genes", "Example genes", "TMA30 prompt regions"], module_table)
    factor_table_html, _ = factor_legend(factor_rows)

    gt_best_payload = []
    for item in gt_best_rows:
        candidate = item["candidate"]
        gt_mask = item["mask"]
        has_overlap = candidate["shared"] > 0
        candidate_mask = (
            load_binary(root / "frozen_pool" / candidate["mask_path"])
            if has_overlap
            else np.zeros_like(gt_mask)
        )
        gt_best_payload.append(
            {
                "region": item["name"],
                "target_class": item["target_class"],
                "candidate": candidate["display_id"] if has_overlap else "No overlapping candidate",
                "source": source_label(candidate) if has_overlap else "None",
                "purity": candidate["candidate_purity"],
                "coverage": candidate["gt_coverage"],
                "iou": candidate["iou"],
                "dice": candidate["dice"],
                "overlap": image_uri(overlap_view(he_image, candidate_mask, gt_mask), "JPEG", 93),
                "masks": image_uri(mask_comparison(candidate_mask, gt_mask), "PNG"),
                "context": image_uri(
                    context_comparison(he_image, ficture_image, candidate_mask if has_overlap else gt_mask),
                    "JPEG",
                    93,
                ),
            }
        )

    vlm_by_uid = {row["candidate_uid"]: row for row in vlm_rows}
    bundle_by_uid = {row["candidate_uid"]: row for row in bundle_rows}
    candidate_payload = []
    for metrics in candidate_metrics:
        uid = metrics["candidate_id"]
        vlm = vlm_by_uid[uid]
        bundle_row = bundle_by_uid[uid]
        folder = bundle / "input_images" / bundle_row["folder"]
        image_items = [
            {"title": title, "src": file_uri(folder / filename)} for filename, title in INPUT_NAMES
        ]
        candidate_mask = load_binary(root / "frozen_pool" / metrics["mask_path"])
        has_gt_overlap = metrics["shared"] > 0
        gt_mask = (
            next(region["mask"] for region in gt_regions if region["region_id"] == metrics["best_gt_id"])
            if has_gt_overlap
            else np.zeros_like(candidate_mask)
        )
        candidate_payload.append(
            {
                "uid": uid,
                "display_id": metrics["display_id"],
                "source": source_label(metrics),
                "source_group": metrics["source"],
                "gene_module": metrics["gene_module"],
                "prediction": GT_LABELS[vlm["predicted_class"]],
                "reason": vlm["reason"],
                "scores": {GT_LABELS[key]: value for key, value in vlm["class_scores"].items()},
                "evaluation_status": vlm["evaluation_status"],
                "official_class": (
                    GT_LABELS[vlm["official_class_for_evaluation"]]
                    if vlm["evaluation_status"] == "evaluable"
                    else "Not evaluated"
                ),
                "official_region": vlm["official_annotation"],
                "has_gt_overlap": has_gt_overlap,
                "correct": vlm["class_correct"],
                "purity": metrics["candidate_purity"],
                "coverage": metrics["gt_coverage"],
                "iou": metrics["iou"],
                "dice": metrics["dice"],
                "inputs": image_items,
                "group1": image_uri(overlap_view(he_image, candidate_mask, gt_mask), "JPEG", 92),
                "group2": image_uri(mask_comparison(candidate_mask, gt_mask), "PNG"),
                "group3": image_uri(context_comparison(he_image, ficture_image, candidate_mask), "JPEG", 92),
            }
        )

    vlm_table_rows = []
    for item in candidate_payload:
        if item["evaluation_status"] == "evaluable":
            result = "Correct" if item["correct"] else "Incorrect"
            gt_value = item["official_class"]
        else:
            result = "Not evaluated"
            gt_value = "No official region covers at least 50% of this candidate"
        vlm_table_rows.append(
            [
                item["display_id"],
                html.escape(item["source"]),
                item["prediction"],
                html.escape(gt_value),
                result,
                f"{item['purity']:.3f}",
            ]
        )
    vlm_table_html = table(
        ["Candidate", "Source", "Gemma prediction", "Official class, when evaluable", "Result", "Candidate purity"],
        vlm_table_rows,
    )

    source_counts = Counter(row["source"] for row in pool_rows)
    predicted_counts = Counter(row["predicted_class"] for row in vlm_rows)
    evaluable = [row for row in vlm_rows if row["evaluation_status"] == "evaluable"]
    correct = [row for row in evaluable if row["class_correct"]]
    gt_with_overlap = sum(row["candidate"]["shared"] > 0 for row in gt_best_rows)
    gt_dice_at_least_half = sum(row["candidate"]["dice"] >= 0.50 for row in gt_best_rows)
    retained_prompts = [row for row in same_prompt_rows if row["same_prompt_he_retained"].lower() == "true"]
    merged_count = sum(row["merged"].lower() == "true" for row in merge_rows)

    source_legend = "".join(
        f"<span><i style='background:{MODULE_COLORS[index]}'></i>Module {index + 1}</span>"
        for index in range(9)
    ) + (
        f"<span><i style='background:{SOURCE_COLORS['Same-prompt H&E supplement']}'></i>Same-prompt H&amp;E</span>"
        f"<span><i style='background:{SOURCE_COLORS['Independent H&E supplement']}'></i>Independent H&amp;E</span>"
    )
    gt_legend = "".join(
        f"<span><i style='background:{color}'></i>{GT_LABELS[label]}</span>" for label, color in GT_COLORS.items()
    )

    system_prompt = (bundle / "system_prompt.txt").read_text(encoding="utf-8")
    user_prompt = (bundle / "user_prompt_template.txt").read_text(encoding="utf-8")
    example_prompt = user_prompt.format(
        candidate_id="TMA30 C01",
        ficture_rgb_reference="[the fixed 12-factor RGB legend shown above]",
        prompt_version=bundle_manifest["prompt_version"],
    )

    css = """
:root{--ink:#17252b;--muted:#58686f;--line:#d4dfe3;--soft:#f4f7f8;--accent:#087b78;--warm:#b45f21;--good:#0b7a4d;--bad:#b73730}*{box-sizing:border-box}body{margin:0;background:#fff;color:var(--ink);font:16px/1.52 Arial,Helvetica,sans-serif;letter-spacing:0}main{max-width:1500px;margin:auto;padding:34px 42px 80px}header{padding-bottom:26px}h1{font-size:36px;line-height:1.16;margin:0 0 10px}h2{font-size:27px;margin:0 0 12px}h3{font-size:19px;margin:0 0 7px}p{margin:8px 0;color:var(--muted)}section{padding:30px 0;border-top:1px solid var(--line)}.lead{font-size:18px;max-width:1120px}.status{display:inline-flex;align-items:center;gap:7px;color:var(--good);font-weight:700}.status:before{content:'';width:10px;height:10px;background:var(--good);border-radius:50%}.metrics{display:grid;grid-template-columns:repeat(6,minmax(130px,1fr));gap:1px;background:var(--line);border:1px solid var(--line);margin:22px 0}.metric{background:#fff;padding:15px}.metric b{display:block;font-size:27px;color:var(--accent)}.metric span{font-size:13px;color:var(--muted)}.flow{display:grid;grid-template-columns:repeat(5,minmax(0,1fr));gap:8px;margin:17px 0}.flow>div{border:1px solid var(--line);background:var(--soft);padding:14px}.flow b{display:block;color:var(--accent);margin-bottom:5px}.stage{display:block;margin-top:7px;font-size:12px;font-weight:700;color:var(--good)}.planned{border-top:4px solid #D79C34!important}.planned .stage{color:#92661d}.note{border-left:4px solid var(--accent);background:var(--soft);padding:13px 16px;margin:15px 0}.warning{border-left-color:#D79C34}.grid2{display:grid;grid-template-columns:1fr 1fr;gap:16px}.grid3{display:grid;grid-template-columns:repeat(3,1fr);gap:14px}.modules{display:grid;grid-template-columns:repeat(3,1fr);gap:12px}.modules button{border:1px solid var(--line);background:#fff;padding:0;cursor:zoom-in}.modules img{display:block;width:100%;image-rendering:pixelated}.module-modal{position:fixed;inset:0;background:rgba(0,0,0,.78);z-index:20;display:none;align-items:center;justify-content:center;padding:30px}.module-modal.open{display:flex}.module-modal img{max-width:90vw;max-height:88vh;image-rendering:pixelated;background:#000}.module-modal button{position:absolute;right:24px;top:18px;width:42px;height:42px;border:0;background:#fff;font-size:25px;cursor:pointer}figure{margin:0;border:1px solid var(--line);background:#fff}figure h3{padding:12px 14px;border-bottom:1px solid var(--line)}figure img{display:block;width:100%;height:auto;object-fit:contain;background:#fff}figcaption{padding:10px 13px;color:var(--muted);border-top:1px solid var(--line)}.legend{display:flex;flex-wrap:wrap;gap:8px 16px;margin:12px 0}.legend span{white-space:nowrap}.legend i,.swatch{display:inline-block;width:14px;height:14px;vertical-align:-2px;margin-right:5px;border:1px solid rgba(0,0,0,.22)}.table-wrap{overflow:auto;border:1px solid var(--line)}table{border-collapse:collapse;width:100%;min-width:760px}th,td{padding:9px 11px;text-align:left;vertical-align:top;border-bottom:1px solid var(--line)}th{font-size:13px;background:var(--soft);position:sticky;top:0}tr:last-child td{border-bottom:0}.viewer{display:grid;grid-template-columns:300px minmax(0,1fr);gap:18px}.controls{display:grid;gap:10px;margin-bottom:12px}.controls label{font-weight:700}.controls select{width:100%;display:block;margin-top:4px;padding:9px;border:1px solid #aab9bf;background:#fff;font:inherit}.list{max-height:690px;overflow:auto;border:1px solid var(--line)}.list button{display:block;width:100%;border:0;border-bottom:1px solid var(--line);background:#fff;padding:10px 12px;text-align:left;cursor:pointer}.list button:hover,.list button.active{background:#e8f3f2}.list b,.list small{display:block}.list small{color:var(--muted);margin-top:3px}.viewer-meta{display:grid;grid-template-columns:repeat(6,minmax(110px,1fr));gap:1px;background:var(--line);border:1px solid var(--line);margin:12px 0}.viewer-meta>div{background:#fff;padding:10px}.viewer-meta b,.viewer-meta span{display:block}.viewer-meta span{font-size:12px;color:var(--muted)}.input-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:12px}.input-grid figure img{aspect-ratio:1/1;object-fit:contain}.score-grid{display:grid;grid-template-columns:repeat(4,1fr);gap:8px;margin:10px 0}.score-grid div{background:var(--soft);padding:9px;text-align:center}.score-grid b{display:block;font-size:20px}.badge{display:inline-block;padding:3px 8px;border-radius:4px;font-size:12px;font-weight:700}.correct{background:#e5f5ed;color:var(--good)}.incorrect{background:#f9e8e6;color:var(--bad)}.unevaluated{background:#edf1f3;color:#52636b}details{border:1px solid var(--line);margin:10px 0}summary{padding:12px 14px;font-weight:700;cursor:pointer;background:#fff}.detail-body{padding:14px;border-top:1px solid var(--line)}pre{white-space:pre-wrap;overflow:auto;background:#f7f9fa;border:1px solid var(--line);padding:14px;font:13px/1.46 Menlo,monospace}.formula{font:15px/1.5 Menlo,monospace;background:#f7f9fa;border:1px solid var(--line);padding:11px}.footer{font-size:13px;color:var(--muted);padding-top:20px}.small{font-size:13px}.audit{font-family:Menlo,monospace;font-size:13px}.emphasis{color:var(--ink);font-weight:700}@media(max-width:1050px){main{padding:25px 20px}.metrics{grid-template-columns:repeat(3,1fr)}.flow{grid-template-columns:repeat(2,1fr)}.viewer{grid-template-columns:1fr}.list{max-height:280px}.viewer-meta{grid-template-columns:repeat(3,1fr)}}@media(max-width:700px){h1{font-size:29px}.grid2,.grid3,.modules,.input-grid{grid-template-columns:1fr}.metrics,.viewer-meta,.score-grid{grid-template-columns:repeat(2,1fr)}.flow{grid-template-columns:1fr}}
"""

    candidate_json = json.dumps(candidate_payload).replace("</", "<\\/")
    gt_json = json.dumps(gt_best_payload).replace("</", "<\\/")
    module_json = json.dumps(individual_module_uris).replace("</", "<\\/")
    content = f"""<!doctype html><html lang='en'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'><title>TMA30 Fixed SAM3 Workflow</title><style>{css}</style></head><body><main>
<header><p class='status'>Verified fixed-pipeline transfer</p><h1>TMA30: Fixed GeneMap-to-SAM3 Workflow</h1><p class='lead'>This is the corrected TMA30 run. It applies the TMA39 candidate-generation rules to a new TMA: the same 120 genes and nine GeneMap modules, the same adaptive 0%-expansion box-plus-point rule, the same registered K=12 strict-official raw FICTURE colors, the same H&amp;E supplement rules, and the same IoU 0.90 duplicate rule.</p><div class='metrics'><div class='metric'><b>9</b><span>fixed GeneMap modules</span></div><div class='metric'><b>45</b><span>TMA30 spatial prompts</span></div><div class='metric'><b>{source_counts['FICTURE primary']}</b><span>FICTURE primary masks</span></div><div class='metric'><b>{source_counts['Same-prompt H&E supplement']} + {source_counts['Independent H&E supplement']}</b><span>two H&amp;E supplement sources</span></div><div class='metric'><b>52</b><span>frozen VLM candidates</span></div><div class='metric'><b>{len(correct)}/{len(evaluable)}</b><span>Gemma correct among evaluable</span></div></div></header>

<section><h2>1. Complete workflow</h2><div class='flow'><div><b>1. Build TMA30 GeneMaps</b>Apply the fixed TMA39 gene modules to TMA30 transcripts.<span class='stage'>Completed</span></div><div><b>2. Place adaptive prompts</b>One tight box and one highest-score point for each spatial GeneMap region.<span class='stage'>Completed</span></div><div><b>3. Run paired SAM3</b>Send each prompt once to raw FICTURE and once to H&amp;E.<span class='stage'>Completed</span></div><div><b>4. Form the first pool</b>Keep all 45 FICTURE masks and three H&amp;E masks with at least 40% new area.<span class='stage'>Completed</span></div><div><b>5. Add H&amp;E-only gaps</b>Add four automatically detected H&amp;E-present/FICTURE-low structures.<span class='stage'>Completed</span></div><div><b>6. Freeze the pool</b>IoU 0.90 cleanup leaves 52 candidates; masks are not unioned.<span class='stage'>Completed</span></div><div><b>7. VLM classification</b>Gemma classifies all 52 candidates from six registered image views.<span class='stage'>Completed</span></div><div class='planned'><b>8. Group by predicted class</b>Collect accepted regions with the same predicted tissue class on one TMA.<span class='stage'>Planned next stage</span></div><div class='planned'><b>9. Second SAM refinement</b>Use all boxes and inside points from one class together in a class-specific SAM prompt.<span class='stage'>Planned next stage</span></div><div class='planned'><b>10. Assemble the map</b>Return refined class masks to their original coordinates to make the whole-TMA map.<span class='stage'>Planned next stage</span></div></div><div class='note'><b>What is fixed across TMA39 and TMA30:</b> GeneMap membership, region rules, prompt form, FICTURE/H&amp;E candidate logic, duplicate rule, six-image VLM input design, and image rendering. TMA30 has four available official tissue labels, so its classification output list is Airway, Arteriole, Venule, and Alveoli; this label list does not change how candidates are generated.</div></section>

<section><h2>2. Registered inputs and fixed FICTURE colors</h2><p>Both inputs use the same 6072 × 6332 canvas. FICTURE is the registered K=12 strict-official raw image. Its stored RGB values are not recolored or smoothed; nearest-neighbor resizing is used wherever it is displayed.</p><div class='grid2'><figure><h3>Registered H&amp;E</h3><img src='{overview_he_uri}'><figcaption>H&amp;E supplies tissue morphology.</figcaption></figure><figure><h3>Registered K=12 strict-official raw FICTURE</h3><img src='{overview_ficture_uri}' style='image-rendering:pixelated'><figcaption>The exact 12-factor RGB map supplied to SAM3 and the VLM.</figcaption></figure></div><h3>Fixed 12-factor RGB legend</h3>{factor_table_html}<div class='note'><b>RGB consistency check:</b> this factor-info file has the same SHA-256 as the fixed TMA39 legend, so each factor keeps exactly the same RGB meaning in both TMAs.</div><details><summary>Input identity audit</summary><div class='detail-body audit'>FICTURE SHA-256: {bundle_manifest['input_hashes']['ficture']}<br>Factor legend SHA-256: {bundle_manifest['input_hashes']['factor_info']}<br>H&amp;E SHA-256: {bundle_manifest['input_hashes']['he']}</div></details></section>

<section><h2>3. TMA30 GeneMap: the same nine gene groups used for TMA39</h2><p>The 120 genes are not clustered again for TMA30. Each TMA30 transcript is placed on the spatial grid, and each module score is the average standardized expression of the genes already assigned to that module. Brighter green means a higher score within that module.</p><details><summary>How the nine modules were fixed on TMA39</summary><div class='detail-body'><p>The 120 spatially variable gene maps were compared by Pearson correlation. The distance between two genes was <code>1 − correlation</code>, so genes bright in the same tissue locations were close. Average-linkage hierarchical clustering was tested with four through ten groups. Nine groups had the strongest separation among the tested choices (mean silhouette 0.485) and were highly repeatable after gene resampling, so those nine memberships were frozen and transferred to TMA30.</p></div></details><figure><h3>All nine TMA30 GeneMap modules</h3><img src='{genemap_uri}' style='image-rendering:pixelated'><figcaption>Each square is one measured GeneMap bin. The panels are rendered with nearest-neighbor scaling so the original 98 × 92 spatial grid stays sharp.</figcaption></figure><div class='modules' id='module-buttons'></div>{module_table_html}<p class='small'>Click any module image to enlarge it.</p></section>

<section><h2>4. From GeneMap scores to 45 boxes and points</h2><p>For each module, the red dashed line marks its top 5% score threshold. Bins to the right are the high-score support used to define possible region extent. Inside that support, the top 1% finds strong region centers, and the top 2.5% confirms that each center remains connected to its surrounding signal.</p><figure><h3>How high-score locations were chosen</h3><img src='{distribution_uri}'><figcaption>Horizontal axis: module score at one TMA30 tissue bin. Vertical axis: number of tissue bins in that score interval. Red line: the module-specific 95th percentile; only the highest-scoring 5% lie to its right.</figcaption></figure><div class='grid3'><div class='formula'><b>Center</b><br>top 1% score bins</div><div class='formula'><b>Connection check</b><br>connected top 2.5% support</div><div class='formula'><b>Box extent</b><br>connected top 5% support, 0% expansion</div></div><p>The first pass found {len(base_prompt_rows)} regions. Within one module, two first-pass regions are joined only when they share at least three edge-adjacent GeneMap bins. This fixed rule merged {merged_count} pairs and produced 45 final prompts.</p><div class='grid2'><figure><h3>G figure: all 45 prompts on H&amp;E</h3><img src='{g_he_uri}'><figcaption>Every colored rectangle is the exact 0%-expansion box; every black-centered white point is the highest module-score location inside that region.</figcaption></figure><figure><h3>G figure: the same prompts on raw FICTURE</h3><img src='{g_ficture_uri}' style='image-rendering:pixelated'><figcaption>The same coordinates are transferred to the registered raw FICTURE image. Colors identify GeneMap modules, not tissue classes.</figcaption></figure></div><div class='legend'>{''.join(f"<span><i style='background:{MODULE_COLORS[i]}'></i>Module {i+1}</span>" for i in range(9))}</div></section>

<section><h2>5. Paired SAM3 masks and H&amp;E supplements</h2><p>Each of the 45 prompts is sent once to raw FICTURE and once to H&amp;E. The 45 FICTURE masks form the main pool. A same-prompt H&amp;E mask is added only when at least 40% of its area lies outside the union of all 45 FICTURE masks; three H&amp;E masks pass this fixed rule.</p><div class='metrics'><div class='metric'><b>45</b><span>FICTURE SAM3 calls</span></div><div class='metric'><b>45</b><span>paired H&amp;E SAM3 calls</span></div><div class='metric'><b>{len(retained_prompts)}</b><span>same-prompt H&amp;E retained</span></div><div class='metric'><b>40%</b><span>minimum new H&amp;E area</span></div><div class='metric'><b>0</b><span>box expansion</span></div><div class='metric'><b>1</b><span>positive point per prompt</span></div></div><h3>Independent H&amp;E-present / FICTURE-low branch</h3><p>A second fixed detector searches 64 × 64 cells. A connected region is prompted when H&amp;E tissue covers at least 35% of a cell, FICTURE signal covers at most 8%, and at least six cells connect. The four TMA30 regions below use one 10%-context box and one automatic inside point.</p><figure><h3>Four independent H&amp;E gap prompts</h3><img src='{gap_uri}'><figcaption>Thin magenta boxes and automatic positive points are the real prompts used for the independent H&amp;E branch.</figcaption></figure></section>

<section><h2>6. Frozen pool before the VLM</h2><div class='count-flow formula'>45 FICTURE primary + 3 same-prompt H&amp;E + 4 independent H&amp;E = 52 final candidates</div><p>Final cleanup removes only near-identical masks with IoU at least 0.90. No masks were removed and no masks were unioned.</p><div class='grid2'><figure><h3>All 52 final candidates on H&amp;E</h3><img src='{final_pool_uri}'><figcaption>Nine colors identify the GeneMap module behind each FICTURE candidate. Cyan marks same-prompt H&amp;E; magenta marks independent H&amp;E.</figcaption></figure><figure><h3>Fourteen official high-confidence regions</h3><img src='{gt_uri}'><figcaption>GT is shown on the same H&amp;E canvas for posthoc comparison.</figcaption></figure></div><div class='legend'>{source_legend}</div><div class='legend'>{gt_legend}</div></section>

<section><h2>7. Best frozen candidate for each official region</h2><p>For each official region, every frozen candidate is compared with it, and the candidate sharing the largest number of pixels is kept as that region's best match. Candidate purity is shared area divided by candidate area. GT coverage is shared area divided by GT area. IoU is shared area divided by the total area covered by either mask. If all 52 candidates have zero shared pixels, the viewer says “No overlapping candidate” instead of assigning an unrelated mask.</p><div class='metrics'><div class='metric'><b>{gt_with_overlap}/14</b><span>GT regions with any candidate overlap</span></div><div class='metric'><b>{gt_dice_at_least_half}/14</b><span>GT regions with best Dice at least 0.50</span></div></div><figure><h3>Best-match purity and IoU</h3><img src='{histogram_uri}'><figcaption>Each bar counts official regions, not all candidate pairs. The dashed black line is the median across the 14 best matches; missed GT regions contribute zero.</figcaption></figure><div class='viewer'><aside><div class='controls'><label>Official class<select id='best-filter'><option value='all'>All four classes</option>{''.join(f"<option value='{label}'>{GT_LABELS[label]}</option>" for label in GT_LABELS)}</select></label></div><div class='list' id='best-list'></div></aside><div><h3 id='best-title'></h3><p id='best-subtitle'></p><div class='viewer-meta'><div><b id='best-source'></b><span>candidate source</span></div><div><b id='best-purity'></b><span>candidate purity</span></div><div><b id='best-coverage'></b><span>GT coverage</span></div><div><b id='best-iou'></b><span>IoU</span></div><div><b id='best-dice'></b><span>Dice</span></div><div><b>Posthoc</b><span>comparison stage</span></div></div><details open><summary>Candidate versus GT on H&amp;E</summary><div class='detail-body'><figure><img id='best-overlap'><figcaption>Blue: candidate only. Green: shared area. Magenta: GT only.</figcaption></figure></div></details><details><summary>Candidate mask and GT mask</summary><div class='detail-body'><figure><img id='best-masks'><figcaption>Candidate mask on the left; official GT region on the right.</figcaption></figure></div></details><details><summary>Registered local context</summary><div class='detail-body'><figure><img id='best-context'><figcaption>H&amp;E on the left; raw FICTURE at the same coordinates on the right.</figcaption></figure></div></details></div></div></section>

<section><h2>8. Gemma inputs and all 52 predictions</h2><p>Gemma-4-31B-it receives six images for every frozen candidate: close, medium, and full H&amp;E plus the same three registered raw FICTURE views. A thin neutral boundary identifies the candidate without replacing its real image content.</p><div class='metrics'><div class='metric'><b>52/52</b><span>candidates classified</span></div><div class='metric'><b>{len(evaluable)}</b><span>candidates meeting the 50% GT-purity evaluation rule</span></div><div class='metric'><b>{len(correct)}</b><span>correct evaluable predictions</span></div><div class='metric'><b>{len(correct)/len(evaluable):.1%}</b><span>accuracy among evaluable candidates</span></div>{''.join(f"<div class='metric'><b>{predicted_counts[label]}</b><span>predicted {GT_LABELS[label]}</span></div>" for label in GT_LABELS)}</div><p>A candidate without at least 50% overlap with one official region is marked <b>Not evaluated</b>, not incorrect, because the expert annotation contains only high-confidence regions.</p>{vlm_table_html}<details><summary>Exact VLM system prompt</summary><pre>{html.escape(system_prompt)}</pre></details><details><summary>Example VLM user prompt</summary><pre>{html.escape(example_prompt)}</pre></details><div class='viewer' style='margin-top:18px'><aside><div class='controls'><label>Source<select id='candidate-source'><option value='all'>All sources</option><option value='FICTURE primary'>FICTURE primary</option><option value='Same-prompt H&E supplement'>Same-prompt H&amp;E</option><option value='Independent H&E supplement'>Independent H&amp;E</option></select></label><label>GT evaluation<select id='candidate-evaluation'><option value='all'>All candidates</option><option value='evaluable'>Evaluable only</option><option value='not_evaluable'>Not evaluated</option></select></label></div><div class='list' id='candidate-list'></div></aside><div><h3 id='candidate-title'></h3><p id='candidate-subtitle'></p><div class='viewer-meta'><div><b id='candidate-prediction'></b><span>Gemma prediction</span></div><div><b id='candidate-gt'></b><span>official class, when evaluable</span></div><div><b id='candidate-result'></b><span>evaluation result</span></div><div><b id='candidate-purity'></b><span>candidate purity</span></div><div><b id='candidate-iou'></b><span>IoU</span></div><div><b id='candidate-dice'></b><span>Dice</span></div></div><div class='score-grid' id='candidate-scores'></div><p id='candidate-reason'></p><h3>Actual six VLM input images</h3><div class='input-grid' id='candidate-inputs'></div><h3>Candidate-level visualization groups</h3><details open><summary>Group 1: candidate versus best-overlapping GT on H&amp;E</summary><div class='detail-body'><figure><img id='candidate-group1'><figcaption>Blue: candidate only. Green: shared area. Magenta: closest GT only.</figcaption></figure></div></details><details><summary>Group 2: mask-only comparison</summary><div class='detail-body'><figure><img id='candidate-group2'><figcaption>Candidate mask on the left; closest official GT region on the right.</figcaption></figure></div></details><details><summary>Group 3: complete registered ROI context</summary><div class='detail-body'><figure><img id='candidate-group3'><figcaption>H&amp;E on the left; raw FICTURE at the same coordinates on the right.</figcaption></figure></div></details></div></div></section>

<section><h2>9. What happens after classification</h2><p>The next stage does not run a second SAM call for each candidate separately. Accepted candidates with the same predicted class are collected on the full TMA, and all of their boxes and inside points are submitted together as one class-specific prompt set. The refined class masks are then returned to their original coordinates and assembled into the final tissue map.</p><div class='flow'><div class='planned'><b>Group accepted regions</b>One group for each predicted tissue class.<span class='stage'>Not run for TMA30</span></div><div class='planned'><b>Create class prompt sets</b>All boxes and points from one class on one registered image.<span class='stage'>Not run for TMA30</span></div><div class='planned'><b>Run second SAM</b>Refine each class from its multi-region prompt set.<span class='stage'>Not run for TMA30</span></div><div class='planned'><b>Return to full TMA</b>Place every refined mask back at its original coordinates.<span class='stage'>Not run for TMA30</span></div><div class='planned'><b>Build final map</b>Combine refined masks by predicted tissue class.<span class='stage'>Not run for TMA30</span></div></div></section>

<section><h2>10. Reproducibility audit</h2>{table(['Item','Verified value'],[['SAM model','SAM3 base'],['Gene module source','Fixed TMA39 120-gene / 9-module definition'],['TMA30 transcript rows used',f"{gene_summary['transcript_rows']:,}"],['GeneMap grid','98 × 92; 4,525 tissue bins'],['First-pass GeneMap regions',str(len(base_prompt_rows))],['Final prompts','45'],['Main SAM3 calls','45 FICTURE + 45 H&amp;E'],['Independent H&amp;E calls','4'],['Final pool','52 unique masks'],['VLM model',html.escape(run_manifest['model'])],['VLM image audit','Six image hashes for every candidate'],['Prompt version',html.escape(run_manifest['prompt_version'])]])}<details><summary>Job and manifest notes</summary><div class='detail-body audit'>GeneMap definition SHA-256: {gene_summary['fixed_module_source_sha256']}<br>Final pool annotation flag: {str(pool_manifest['selection_used_annotation']).lower()}<br>VLM system prompt SHA-256: {run_manifest['system_prompt_sha256']}<br>VLM prompt-template SHA-256: {run_manifest['user_prompt_template_sha256']}<br>Top-score ties: {run_manifest['top_score_tie_candidate_count']}</div></details></section>
<p class='footer'>Corrected TMA30 fixed-workflow report. The older TMA30 sliding-box report is not part of this result.</p>
</main><div class='module-modal' id='module-modal'><button type='button' aria-label='Close module image'>×</button><img alt='Enlarged GeneMap module'></div><script>
const modules={module_json};const candidates={candidate_json};const best={gt_json};
const moduleButtons=document.getElementById('module-buttons');const modal=document.getElementById('module-modal');modules.forEach((src,index)=>{{const b=document.createElement('button');b.innerHTML=`<img src="${{src}}" alt="GeneMap Module ${{index+1}}"><span class="small">Module ${{index+1}}</span>`;b.onclick=()=>{{modal.querySelector('img').src=src;modal.classList.add('open')}};moduleButtons.appendChild(b)}});modal.querySelector('button').onclick=()=>modal.classList.remove('open');modal.onclick=e=>{{if(e.target===modal)modal.classList.remove('open')}};
const bestList=document.getElementById('best-list');const bestFilter=document.getElementById('best-filter');function showBest(){{const rows=best.filter(x=>bestFilter.value==='all'||x.target_class===bestFilter.value);bestList.innerHTML='';rows.forEach((item,index)=>{{const b=document.createElement('button');b.innerHTML=`<b>${{item.region}}</b><small>${{item.candidate}} · Dice ${{item.dice.toFixed(3)}}</small>`;b.onclick=()=>selectBest(item,b);bestList.appendChild(b)}});if(rows.length)selectBest(rows[0],bestList.firstChild)}}function selectBest(item,button){{bestList.querySelectorAll('button').forEach(b=>b.classList.toggle('active',b===button));document.getElementById('best-title').textContent=item.region;document.getElementById('best-subtitle').textContent=item.candidate==='No overlapping candidate'?'None of the 52 frozen candidates overlaps this official region.':`${{item.candidate}} has the largest shared pixel area among all 52 frozen candidates.`;document.getElementById('best-source').textContent=item.source;document.getElementById('best-purity').textContent=item.purity.toFixed(3);document.getElementById('best-coverage').textContent=item.coverage.toFixed(3);document.getElementById('best-iou').textContent=item.iou.toFixed(3);document.getElementById('best-dice').textContent=item.dice.toFixed(3);document.getElementById('best-overlap').src=item.overlap;document.getElementById('best-masks').src=item.masks;document.getElementById('best-context').src=item.context}}bestFilter.onchange=showBest;showBest();
const candidateList=document.getElementById('candidate-list');const sourceFilter=document.getElementById('candidate-source');const evalFilter=document.getElementById('candidate-evaluation');function filteredCandidates(){{return candidates.filter(x=>(sourceFilter.value==='all'||x.source_group===sourceFilter.value)&&(evalFilter.value==='all'||x.evaluation_status===evalFilter.value))}}function showCandidates(){{const rows=filteredCandidates();candidateList.innerHTML='';rows.forEach(item=>{{const b=document.createElement('button');b.innerHTML=`<b>${{item.display_id}}</b><small>${{item.source}} · ${{item.prediction}}</small>`;b.onclick=()=>selectCandidate(item,b);candidateList.appendChild(b)}});if(rows.length)selectCandidate(rows[0],candidateList.firstChild)}}function selectCandidate(item,button){{candidateList.querySelectorAll('button').forEach(b=>b.classList.toggle('active',b===button));document.getElementById('candidate-title').textContent=item.display_id;document.getElementById('candidate-subtitle').textContent=item.source+' · '+(item.has_gt_overlap?item.official_region:'no overlap with an official GT region');document.getElementById('candidate-prediction').textContent=item.prediction;document.getElementById('candidate-gt').textContent=item.official_class;const result=item.evaluation_status==='evaluable'?(item.correct?'Correct':'Incorrect'):'Not evaluated';const resultNode=document.getElementById('candidate-result');resultNode.textContent=result;resultNode.className='badge '+(result==='Correct'?'correct':result==='Incorrect'?'incorrect':'unevaluated');document.getElementById('candidate-purity').textContent=item.purity.toFixed(3);document.getElementById('candidate-iou').textContent=item.iou.toFixed(3);document.getElementById('candidate-dice').textContent=item.dice.toFixed(3);document.getElementById('candidate-reason').textContent=item.reason;document.getElementById('candidate-scores').innerHTML=Object.entries(item.scores).map(([name,value])=>`<div><b>${{value}}</b><span>${{name}}</span></div>`).join('');document.getElementById('candidate-inputs').innerHTML=item.inputs.map(input=>`<figure><h3>${{input.title}}</h3><img src="${{input.src}}" alt="${{input.title}}"></figure>`).join('');document.getElementById('candidate-group1').src=item.group1;document.getElementById('candidate-group2').src=item.group2;document.getElementById('candidate-group3').src=item.group3}}sourceFilter.onchange=showCandidates;evalFilter.onchange=showCandidates;showCandidates();
</script></body></html>"""
    args.output.write_text(content, encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "bytes": args.output.stat().st_size,
                "candidate_count": len(candidate_payload),
                "gt_region_count": len(gt_best_payload),
                "evaluable_candidate_count": len(evaluable),
                "correct_evaluable_candidate_count": len(correct),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
