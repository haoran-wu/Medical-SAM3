#!/usr/bin/env python3
"""Build one presentation-ready, self-contained pre-VLM report for a fixed-workflow TMA."""

from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import html
import io
import json
from collections import Counter
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw
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
    "Same-prompt H&E supplement": "#00A6B4",
    "Independent H&E supplement": "#D81B60",
}
EXPECTED_FACTOR_INFO_SHA256 = (
    "f89e0e599f6451ff37dbad24ff4b1a27b8c99bb66c033f5a40743268711181c6"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--input-audit", type=Path, required=True)
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


def hex_rgb(value: str) -> tuple[int, int, int]:
    value = value.lstrip("#")
    return tuple(int(value[index : index + 2], 16) for index in (0, 2, 4))


def image_uri(image: Image.Image, fmt: str = "JPEG", quality: int = 88) -> str:
    buffer = io.BytesIO()
    if fmt.upper() == "JPEG":
        image.convert("RGB").save(
            buffer,
            format="JPEG",
            quality=quality,
            optimize=True,
            progressive=True,
            subsampling=1,
        )
        mime = "image/jpeg"
    else:
        image.save(buffer, format="PNG", optimize=True)
        mime = "image/png"
    return f"data:{mime};base64,{base64.b64encode(buffer.getvalue()).decode('ascii')}"


def figure_uri(figure: plt.Figure, dpi: int = 165) -> str:
    buffer = io.BytesIO()
    figure.savefig(buffer, format="png", dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    return f"data:image/png;base64,{base64.b64encode(buffer.getvalue()).decode('ascii')}"


def load_binary(path: Path) -> np.ndarray:
    mask = np.asarray(Image.open(path).convert("L"), dtype=np.uint8) > 0
    if not mask.any():
        raise RuntimeError(f"Empty mask: {path}")
    return mask


def resize_for_report(image: Image.Image, max_side: int, *, nearest: bool) -> Image.Image:
    image = image.convert("RGB")
    scale = min(1.0, max_side / max(image.size))
    size = (max(1, round(image.width * scale)), max(1, round(image.height * scale)))
    resampling = Image.Resampling.NEAREST if nearest else Image.Resampling.LANCZOS
    return image.resize(size, resampling)


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


def module_green_image(score: np.ndarray, tissue: np.ndarray, max_side: int = 1100) -> Image.Image:
    value = np.clip(score, 0.0, 1.0)
    rgb = np.zeros((*score.shape, 3), dtype=np.uint8)
    rgb[..., 1] = np.uint8(np.round(value * 255))
    rgb[..., 0] = np.uint8(np.round(np.clip((value - 0.72) / 0.28, 0, 1) * 150))
    rgb[..., 2] = np.uint8(np.round(value * 45))
    rgb[np.logical_and(tissue, value == 0)] = (4, 12, 8)
    image = Image.fromarray(rgb)
    scale = max_side / max(image.size)
    size = (max(1, round(image.width * scale)), max(1, round(image.height * scale)))
    return image.resize(size, Image.Resampling.NEAREST)


def distribution_figure(
    tma: str,
    raw_scores: np.ndarray,
    tissue: np.ndarray,
    thresholds: dict[int, float],
    counts: Counter[int],
) -> str:
    figure, axes = plt.subplots(3, 3, figsize=(15.2, 13.1))
    for module, axis in enumerate(axes.flat, start=1):
        values = raw_scores[module - 1][tissue]
        low, high = np.percentile(values, [0.2, 99.8])
        shown = values[np.logical_and(values >= low, values <= high)]
        axis.hist(shown, bins=52, color="#2E75B6", edgecolor="white", linewidth=0.35)
        threshold = thresholds[module]
        axis.axvline(threshold, color="#E53935", linestyle="--", linewidth=2.2)
        axis.set_title(
            f"Module {module}: red line {threshold:.3f}\n{counts[module]} final prompt regions",
            fontsize=11.5,
            fontweight="bold",
        )
        axis.set_xlabel("Module score at one tissue bin", fontsize=9.5)
        axis.set_ylabel("Number of tissue bins", fontsize=9.5)
        axis.grid(axis="y", color="#DDE5E8", alpha=0.75, linewidth=0.7)
        axis.spines[["top", "right"]].set_visible(False)
    figure.suptitle(f"{tma} GeneMap module-score distributions", fontsize=18, fontweight="bold")
    figure.tight_layout(rect=(0, 0, 1, 0.965))
    return figure_uri(figure)


def prompt_overlay(background: Image.Image, rows: list[dict[str, str]], *, ficture: bool) -> Image.Image:
    shown = resize_for_report(background, 1420, nearest=ficture)
    draw = ImageDraw.Draw(shown)
    sx, sy = shown.width / background.width, shown.height / background.height
    for row in rows:
        color = hex_rgb(MODULE_COLORS[int(row["gene_module"]) - 1])
        box = tuple(
            round(float(row[key]) * scale)
            for key, scale in zip(("box_x1", "box_y1", "box_x2", "box_y2"), (sx, sy, sx, sy))
        )
        draw.rectangle(box, outline="white", width=4)
        draw.rectangle(box, outline=color, width=2)
        point = (round(float(row["point_x"]) * sx), round(float(row["point_y"]) * sy))
        radius = 5
        draw.ellipse(
            (point[0] - radius, point[1] - radius, point[0] + radius, point[1] + radius),
            fill="white",
            outline="#11181B",
            width=2,
        )
        draw.ellipse((point[0] - 2, point[1] - 2, point[0] + 2, point[1] + 2), fill="#11181B")
    return shown


def mask_overlay(
    background: Image.Image,
    rows: list[dict[str, str]],
    mask_resolver: Any,
    *,
    max_side: int = 1420,
) -> Image.Image:
    shown = resize_for_report(background, max_side, nearest=False)
    values = np.asarray(shown, dtype=np.float32).copy()
    outlines: list[tuple[np.ndarray, tuple[int, int, int]]] = []
    for row in rows:
        mask_path = mask_resolver(row)
        mask = Image.open(mask_path).convert("L").resize(shown.size, Image.Resampling.NEAREST)
        binary = np.asarray(mask, dtype=np.uint8) > 0
        if not binary.any():
            continue
        if row.get("source") in SOURCE_COLORS:
            color = hex_rgb(SOURCE_COLORS[row["source"]])
        else:
            color = hex_rgb(MODULE_COLORS[int(row["gene_module"]) - 1])
        values[binary] = values[binary] * 0.57 + np.asarray(color, dtype=np.float32) * 0.43
        boundary = np.logical_and(binary, ~ndi.binary_erosion(binary, iterations=2))
        outlines.append((boundary, color))
    for boundary, color in outlines:
        values[boundary] = color
    return Image.fromarray(np.uint8(np.clip(values, 0, 255)))


def square_bounds(mask: np.ndarray, minimum: int = 820, scale: float = 1.55) -> tuple[int, int, int, int]:
    yy, xx = np.nonzero(mask)
    height, width = mask.shape
    x1, x2 = int(xx.min()), int(xx.max()) + 1
    y1, y2 = int(yy.min()), int(yy.max()) + 1
    center_x, center_y = (x1 + x2) / 2, (y1 + y2) / 2
    side = min(max(minimum, round(max(x2 - x1, y2 - y1) * scale)), width, height)
    left = max(0, min(round(center_x - side / 2), width - side))
    top = max(0, min(round(center_y - side / 2), height - side))
    return left, top, left + side, top + side


def candidate_crop(
    image: Image.Image,
    mask: np.ndarray,
    color: tuple[int, int, int],
    *,
    ficture: bool,
) -> Image.Image:
    bounds = square_bounds(mask)
    x1, y1, x2, y2 = bounds
    base = np.asarray(
        image.crop(bounds).resize(
            (720, 720),
            Image.Resampling.NEAREST if ficture else Image.Resampling.LANCZOS,
        ),
        dtype=np.float32,
    )
    crop_mask = np.asarray(
        Image.fromarray(np.uint8(mask[y1:y2, x1:x2]) * 255).resize(
            (720, 720), Image.Resampling.NEAREST
        )
    ) > 0
    boundary = np.logical_and(crop_mask, ~ndi.binary_erosion(crop_mask, iterations=3))
    base[crop_mask] = base[crop_mask] * 0.76 + np.asarray(color, dtype=np.float32) * 0.24
    base[boundary] = color
    return Image.fromarray(np.uint8(np.clip(base, 0, 255)))


def factor_table(rows: list[dict[str, str]]) -> str:
    body = []
    for row in sorted(rows, key=lambda item: int(item["Factor"])):
        genes = [value.strip() for value in row["TopGene_specific"].split(",") if value.strip()][:5]
        swatch = f"<span class='swatch' style='background:{html.escape(row['hex'])}'></span>"
        body.append(
            [
                html.escape(row["Factor"]),
                f"{swatch}{html.escape(row['hex'].upper())} / {html.escape(row['RGB'])}",
                html.escape(row["Celltype2"]),
                html.escape(", ".join(genes)),
            ]
        )
    return table(["Factor", "Fixed RGB anchor", "Marker-inferred cell type", "Example genes"], body)


def selected_rows(root: Path, source: str) -> list[dict[str, str]]:
    rows = [
        row
        for row in read_csv(root / "sam3_main" / source / "selected_candidates.csv")
        if row["selection_method"] == "genemap_support_f1"
    ]
    rows.sort(key=lambda row: int(row["prompt_index"]))
    return rows


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    gene_summary = json.loads((root / "genemap_scores/summary.json").read_text())
    base_manifest = json.loads((root / "prompts/adaptive_base/adaptive_prompt_manifest.json").read_text())
    prompt_manifest = json.loads((root / "prompts/final_sustained_edge3/adaptive_prompt_manifest.json").read_text())
    ficture_manifest = json.loads((root / "sam3_main/ficture/run_manifest.json").read_text())
    he_manifest = json.loads((root / "sam3_main/he/run_manifest.json").read_text())
    gap_prompt_manifest = json.loads((root / "independent_he_gap_detection/prompt_manifest.json").read_text())
    gap_sam_manifest = json.loads((root / "sam3_independent_he_gap/run_manifest.json").read_text())
    pool_manifest = json.loads((root / "frozen_pool/run_manifest.json").read_text())
    tma = pool_manifest["tma"]

    prompt_rows = read_csv(root / "prompts/final_sustained_edge3/adaptive_box_peak_sustained_edge3_context0.csv")
    threshold_rows = read_csv(root / "prompts/adaptive_base/module_thresholds.csv")
    merge_rows = read_csv(root / "prompts/final_sustained_edge3/region_merge_audit.csv")
    module_rows = read_csv(root / "genemap_scores/gene_modules.csv")
    pool_rows = read_csv(root / "frozen_pool/final_pool.csv")
    pair_rows = read_csv(root / "frozen_pool/same_prompt_he_audit.csv")
    gap_rows = read_csv(root / "independent_he_gap_detection/detected_regions.csv")
    ficture_selected = selected_rows(root, "ficture")
    he_selected = selected_rows(root, "he")

    factor_path = root / "staging/factor_info.csv"
    if sha256(factor_path) != EXPECTED_FACTOR_INFO_SHA256:
        raise RuntimeError("FICTURE factor/RGB legend differs from the frozen TMA39 legend")
    factor_rows = read_csv(factor_path)

    input_audit = json.loads(args.input_audit.read_text())
    matches = [row for row in input_audit["tmas"] if row["tma"] == tma]
    if len(matches) != 1:
        raise RuntimeError(f"{tma} missing or duplicated in the input audit")
    audited_input = matches[0]

    he_path = root / "staging" / f"{tma}_he.png"
    ficture_path = root / "staging" / f"{tma}_ficture.png"
    if sha256(he_path) != audited_input["he_sha256"]:
        raise RuntimeError("H&E hash differs from the audited registered input")
    if sha256(ficture_path) != audited_input["ficture_sha256"]:
        raise RuntimeError("FICTURE hash differs from the audited strict-official raw input")
    he_image = Image.open(he_path).convert("RGB")
    ficture_image = Image.open(ficture_path).convert("RGB")
    if he_image.size != ficture_image.size or list(he_image.size) != audited_input["registered_size"]:
        raise RuntimeError("Registered input geometry differs from the input audit")

    if gene_summary["fixed_gene_count"] != 120 or gene_summary["active_gene_count"] != 120:
        raise RuntimeError("The frozen 120-gene definition was not retained")
    if gene_summary["gene_module_count"] != 9 or prompt_manifest["minimum_shared_edge_bins"] != 3:
        raise RuntimeError("The frozen nine-module prompt rule was not retained")
    if len(prompt_rows) != prompt_manifest["prompt_count"]:
        raise RuntimeError("Prompt count differs from the final prompt manifest")
    if len(pool_rows) != pool_manifest["final_candidate_count"]:
        raise RuntimeError("Final candidate count differs from the pool manifest")

    score_npz = np.load(root / "genemap_scores/gene_module_scores_unclipped.npz")
    raw_scores = score_npz["raw_modules"].astype(np.float32)
    shown_scores = score_npz["modules"].astype(np.float32)
    tissue = score_npz["tissue"].astype(bool)
    if raw_scores.shape[0] != 9 or shown_scores.shape != raw_scores.shape:
        raise RuntimeError("Unexpected GeneMap module shape")
    module_counts: Counter[int] = Counter(int(row["gene_module"]) for row in prompt_rows)
    thresholds = {int(row["gene_module"]): float(row["top5_threshold"]) for row in threshold_rows}

    module_uris = [image_uri(module_green_image(shown_scores[index], tissue), "PNG") for index in range(9)]
    distribution_uri = distribution_figure(tma, raw_scores, tissue, thresholds, module_counts)
    overview_he_uri = image_uri(resize_for_report(he_image, 1420, nearest=False), "JPEG", 90)
    overview_ficture_uri = image_uri(resize_for_report(ficture_image, 1420, nearest=True), "PNG")
    prompt_he_uri = image_uri(prompt_overlay(he_image, prompt_rows, ficture=False), "JPEG", 90)
    prompt_ficture_uri = image_uri(prompt_overlay(ficture_image, prompt_rows, ficture=True), "PNG")

    ficture_map_uri = image_uri(
        mask_overlay(
            he_image,
            ficture_selected,
            lambda row: root / "sam3_main/ficture" / row["selected_mask_path"],
        ),
        "JPEG",
        90,
    )
    he_map_uri = image_uri(
        mask_overlay(
            he_image,
            he_selected,
            lambda row: root / "sam3_main/he" / row["selected_mask_path"],
        ),
        "JPEG",
        90,
    )
    final_pool_uri = image_uri(
        mask_overlay(
            he_image,
            pool_rows,
            lambda row: root / "frozen_pool" / row["mask_path"],
        ),
        "JPEG",
        90,
    )
    gap_overlay_uri = image_uri(
        prompt_overlay(he_image, [], ficture=False) if not gap_rows else
        resize_for_report(
            Image.open(root / "independent_he_gap_detection" / f"{tma}_Raw_HE_Gap_Prompts_G_Figure.png"),
            1800,
            nearest=False,
        ),
        "JPEG",
        89,
    )

    module_table_rows = []
    for module in range(1, 10):
        row = next(item for item in module_rows if int(item["module"]) == module)
        genes = row.get("fixed_genes", row.get("active_genes", ""))
        module_table_rows.append(
            [
                f"<span class='swatch' style='background:{MODULE_COLORS[module - 1]}'></span>Module {module}",
                html.escape(row["fixed_gene_count"]),
                html.escape(", ".join(genes.split(", ")[:6])),
                str(module_counts[module]),
            ]
        )
    module_table_html = table(
        ["Frozen module", "Genes", "Example genes", f"{tma} prompt regions"],
        module_table_rows,
    )

    retained_pairs = [row for row in pair_rows if row["same_prompt_he_retained"].lower() == "true"]
    merged_count = sum(int(row["old_region_count"]) - 1 for row in merge_rows)
    dedup_removed = int(pool_manifest["removed_as_near_duplicate"])
    source_counts = pool_manifest["final_source_counts"]

    candidate_payload = []
    for index, row in enumerate(pool_rows, start=1):
        mask = load_binary(root / "frozen_pool" / row["mask_path"])
        if row["source"] == "FICTURE primary":
            color_hex = MODULE_COLORS[int(row["gene_module"]) - 1]
            source_label = f"FICTURE primary, Module {row['gene_module']}"
        else:
            color_hex = SOURCE_COLORS[row["source"]]
            source_label = row["source"]
        color = hex_rgb(color_hex)
        candidate_payload.append(
            {
                "id": f"{tma} C{index:02d}",
                "candidate_id": row["candidate_id"],
                "source": row["source"],
                "source_label": source_label,
                "module": row["gene_module"] or "none",
                "area": int(row["area_pixels"]),
                "sam_score": float(row["sam_score"]),
                "he": image_uri(candidate_crop(he_image, mask, color, ficture=False), "JPEG", 87),
                "ficture": image_uri(candidate_crop(ficture_image, mask, color, ficture=True), "PNG"),
            }
        )

    factor_table_html = factor_table(factor_rows)
    module_cards = "".join(
        f"<figure class='module-card'><h3>Module {index + 1}</h3><button class='image-button' data-image='{uri}' data-title='{tma} Module {index + 1}'><img src='{uri}' alt='{tma} GeneMap Module {index + 1}'></button><figcaption>{module_counts[index + 1]} final prompt regions</figcaption></figure>"
        for index, uri in enumerate(module_uris)
    )
    module_legend = "".join(
        f"<span><i style='background:{color}'></i>Module {index + 1}</span>"
        for index, color in enumerate(MODULE_COLORS)
    )
    source_filter_options = "".join(
        f"<option value='{html.escape(source)}'>{html.escape('Same-prompt H&E' if source == 'Same-prompt H&E supplement' else 'Independent H&E' if source == 'Independent H&E supplement' else source)}</option>"
        for source in ("FICTURE primary", "Same-prompt H&E supplement", "Independent H&E supplement")
        if int(source_counts[source]) > 0
    )
    present_modules = sorted(
        {int(item["module"]) for item in candidate_payload if item["module"] != "none"}
    )
    module_filter_options = "".join(
        f"<option value='{module}'>Module {module}</option>" for module in present_modules
    )
    if any(item["module"] == "none" for item in candidate_payload):
        module_filter_options += "<option value='none'>No GeneMap module</option>"

    css = """
:root{--ink:#18242a;--muted:#5d6b72;--line:#d7e0e3;--panel:#f6f8f8;--accent:#006b62;--warn:#fff4d6}*{box-sizing:border-box}body{margin:0;background:#fff;color:var(--ink);font-family:Arial,Helvetica,sans-serif;line-height:1.52;letter-spacing:0}main{max-width:1500px;margin:auto;padding:30px 34px 70px}header{border-bottom:3px solid var(--ink);padding-bottom:22px;margin-bottom:24px}.eyebrow{margin:0 0 8px;color:var(--accent);font-weight:800;text-transform:uppercase;font-size:13px}h1{font-size:38px;line-height:1.12;margin:0 0 12px}h2{font-size:25px;margin:0 0 14px}h3{font-size:18px;margin:0 0 10px}.lead{font-size:18px;max-width:1050px;color:#435159}.metrics{display:grid;grid-template-columns:repeat(6,minmax(0,1fr));gap:10px;margin-top:18px}.metric{border-top:4px solid var(--accent);background:var(--panel);padding:13px 12px;min-height:92px}.metric b{display:block;font-size:27px}.metric span{font-size:13px;color:var(--muted)}section{padding:26px 0;border-bottom:1px solid var(--line);min-width:0}.flow{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:10px}.flow div{border:1px solid var(--line);padding:14px;min-height:126px}.flow b{display:block;margin-bottom:6px}.stage{display:block;margin-top:9px;color:var(--accent);font-size:12px;font-weight:bold}.planned{background:#f2f4f5;color:#536067}.grid2{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:16px}.grid3{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:12px}.modules{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:12px}figure{margin:0;border:1px solid var(--line);background:#fff;padding:12px}figure img{display:block;width:100%;height:auto;max-width:100%}.module-card img{image-rendering:pixelated;object-fit:contain;background:#000}.image-button{display:block;width:100%;padding:0;border:0;background:none;cursor:zoom-in}.note{margin-top:14px;padding:13px 15px;border-left:4px solid var(--accent);background:#eef7f5}.note code{overflow-wrap:anywhere;word-break:break-all}.warning{background:var(--warn);border-color:#d59b00}.formula{border:1px solid var(--line);padding:14px;background:var(--panel)}.legend{display:flex;flex-wrap:wrap;gap:8px 16px;margin:12px 0}.legend span{display:flex;align-items:center;font-size:13px}.legend i,.swatch{display:inline-block;width:17px;height:17px;margin-right:7px;border:1px solid #707b80;vertical-align:middle}.table-wrap{overflow:auto;border:1px solid var(--line);max-width:100%}table{border-collapse:collapse;width:100%;font-size:13px}th,td{padding:9px 10px;border-bottom:1px solid var(--line);text-align:left;vertical-align:top}th{background:#edf2f3;position:sticky;top:0}.viewer{display:grid;grid-template-columns:300px 1fr;gap:18px}.viewer aside{border:1px solid var(--line);padding:10px}.controls label{display:block;font-size:12px;font-weight:bold;margin-bottom:8px}.controls select{display:block;width:100%;margin-top:4px;padding:8px}.candidate-list{max-height:690px;overflow:auto}.candidate-button{display:block;width:100%;text-align:left;padding:9px;border:0;border-bottom:1px solid var(--line);background:#fff;cursor:pointer}.candidate-button.active{background:#dff1ed;border-left:4px solid var(--accent)}.candidate-button small{display:block;color:var(--muted)}.candidate-images{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:12px}.candidate-meta{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:8px;margin:10px 0}.candidate-meta div{background:var(--panel);padding:10px}.candidate-meta b,.candidate-meta span{display:block}.candidate-meta span{font-size:12px;color:var(--muted)}details{margin-top:12px;border:1px solid var(--line);max-width:100%;overflow:hidden}summary{cursor:pointer;font-weight:bold;padding:11px 13px;background:var(--panel)}.details-body{padding:13px;min-width:0}.audit{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:12px;overflow-wrap:anywhere}.modal{display:none;position:fixed;z-index:10;inset:0;background:rgba(0,0,0,.82);padding:25px}.modal.open{display:flex;align-items:center;justify-content:center}.modal img{max-width:92vw;max-height:86vh;background:#fff}.modal button{position:absolute;right:20px;top:15px;border:0;background:#fff;font-size:27px;width:42px;height:42px;cursor:pointer}.modal p{position:absolute;left:24px;top:12px;color:#fff;font-weight:bold}.footer{margin-top:24px;color:var(--muted);font-size:12px}@media(max-width:900px){main{padding:20px 14px}.metrics,.flow{grid-template-columns:repeat(2,minmax(0,1fr))}.grid2,.grid3,.modules,.candidate-images{grid-template-columns:1fr}.viewer{grid-template-columns:1fr}.candidate-meta{grid-template-columns:repeat(2,minmax(0,1fr))}}
"""

    content = f"""<!doctype html><html lang='en'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'><title>{tma} Pre-VLM Workflow</title><style>{css}</style></head><body><main>
<header><p class='eyebrow'>Verified pre-VLM result</p><h1>{tma}: Fixed GeneMap-to-SAM3 Workflow</h1><p class='lead'>This report applies the frozen TMA39 method without changing any TMA-specific parameter. It stops after the candidate pool is frozen and before any vision-language model is used.</p><div class='metrics'><div class='metric'><b>120</b><span>fixed genes</span></div><div class='metric'><b>9</b><span>fixed GeneMap modules</span></div><div class='metric'><b>{len(prompt_rows)}</b><span>GeneMap box-plus-point prompts</span></div><div class='metric'><b>{source_counts['FICTURE primary']}</b><span>FICTURE primary candidates</span></div><div class='metric'><b>{source_counts['Same-prompt H&E supplement']} + {source_counts['Independent H&E supplement']}</b><span>two H&amp;E supplement sources</span></div><div class='metric'><b>{len(pool_rows)}</b><span>frozen pre-VLM candidates</span></div></div></header>
<section><h2>1. Complete workflow and current stopping point</h2><div class='flow'><div><b>1. Build GeneMaps</b>Place transcripts on this TMA and score the same nine frozen gene groups.<span class='stage'>Completed</span></div><div><b>2. Place prompts</b>One tight 0% box and one highest-score point for each GeneMap region.<span class='stage'>Completed</span></div><div><b>3. Run paired SAM3</b>Use every prompt once on raw FICTURE and once on registered H&amp;E.<span class='stage'>Completed</span></div><div><b>4. Add H&amp;E evidence</b>Keep same-prompt H&amp;E with at least 40% new area, then add independent H&amp;E gaps.<span class='stage'>Completed</span></div><div><b>5. Freeze the pool</b>Remove only masks with IoU at least 0.90. Do not union masks.<span class='stage'>Completed</span></div><div class='planned'><b>6. VLM classification</b>Classify each frozen candidate from close, medium, and full H&amp;E/FICTURE views.<span class='stage'>Not run in this report</span></div><div class='planned'><b>7. Class-grouped SAM</b>Send same-class boxes and points together for a second refinement.<span class='stage'>Planned after VLM</span></div><div class='planned'><b>8. Assemble the tissue map</b>Return refined class masks to the full TMA coordinates.<span class='stage'>Planned after VLM</span></div></div></section>
<section><h2>2. Registered inputs and the fixed FICTURE color system</h2><p>Both images use the same {he_image.width} by {he_image.height} canvas. The FICTURE image is the K=12 strict-official affine-warped source used by TMA39. It is not smoothed or recolored, and every displayed resize uses nearest-neighbor sampling.</p><div class='grid2'><figure><h3>Registered H&amp;E</h3><img src='{overview_he_uri}' alt='{tma} registered H&E'><figcaption>Morphology input.</figcaption></figure><figure><h3>Registered K=12 strict-official raw FICTURE</h3><img src='{overview_ficture_uri}' alt='{tma} registered raw FICTURE' style='image-rendering:pixelated'><figcaption>Exact stored RGB image used by SAM3. Continuous mixture pixels may fall between the 12 anchor colors; no downstream recoloring was applied.</figcaption></figure></div><h3>One fixed 12-factor RGB legend for every TMA</h3>{factor_table_html}<div class='note'><b>Consistency check:</b> the shared legend SHA-256 is <code>{EXPECTED_FACTOR_INFO_SHA256}</code>. This TMA's H&amp;E and FICTURE hashes match the ten-TMA input audit.</div></section>
<section><h2>3. The same nine GeneMap modules on {tma}</h2><p>The genes are not clustered again for this TMA. Each module score is the average standardized expression of its already assigned genes at one spatial bin. A brighter green bin means that module is stronger at that location.</p><div class='modules'>{module_cards}</div>{module_table_html}<details><summary>How the nine modules were fixed</summary><div class='details-body'><p>TMA39 gene maps were compared with Pearson correlation. Genes bright in the same tissue locations have a small distance, defined as <code>1 - correlation</code>. Average-linkage hierarchical clustering was tested with four through ten groups. Nine groups separated similar and different spatial maps most clearly among those choices, so that membership was frozen and reused here.</p></div></details></section>
<section><h2>4. From module scores to the actual boxes and points</h2><p>Each histogram summarizes one module across this TMA. The horizontal axis is the module score at one tissue bin. The vertical axis is how many tissue bins fall in that score interval. The red dashed line is the module-specific 95th percentile, so the highest-scoring 5% lie to its right and define the possible region extent.</p><figure><img src='{distribution_uri}' alt='{tma} module score distributions'><figcaption>Top 1% bins identify strong centers, connected top 2.5% bins confirm local support, and connected top 5% bins define each box extent.</figcaption></figure><div class='grid3'><div class='formula'><b>Center</b><br>top 1% module-score bins</div><div class='formula'><b>Connection</b><br>connected top 2.5% support</div><div class='formula'><b>Prompt</b><br>tight top 5% extent, 0% expansion, plus one peak point</div></div><p>The first pass found {base_manifest['prompt_count']} regions. The fixed same-module edge rule merged {merged_count} neighboring pieces and produced {len(prompt_rows)} final prompts.</p><div class='grid2'><figure><h3>All prompts on H&amp;E</h3><img src='{prompt_he_uri}' alt='{tma} prompts on H&E'><figcaption>Every rectangle is the actual tight box. Every black-centered white dot is its positive point.</figcaption></figure><figure><h3>The same prompts on raw FICTURE</h3><img src='{prompt_ficture_uri}' alt='{tma} prompts on FICTURE' style='image-rendering:pixelated'><figcaption>Coordinates are identical. Prompt colors identify GeneMap modules, not tissue classes.</figcaption></figure></div><div class='legend'>{module_legend}</div></section>
<section><h2>5. Paired SAM3 and the two H&amp;E supplement branches</h2><p>SAM3 was called once on FICTURE and once on H&amp;E for every GeneMap prompt. Every FICTURE prompt must produce a primary mask. An H&amp;E call contributes at most one mask; if no proposal contains its positive point, that call is recorded without adding an H&amp;E candidate. An available paired H&amp;E mask is retained only when at least 40% of its area lies outside the union of all FICTURE primary masks.</p><div class='metrics'><div class='metric'><b>{ficture_manifest['total_inference_calls']}</b><span>FICTURE SAM3 calls</span></div><div class='metric'><b>{he_manifest['total_inference_calls']}</b><span>paired H&amp;E SAM3 calls</span></div><div class='metric'><b>{ficture_manifest['total_eligible_unique_masks']:,}</b><span>eligible internal FICTURE alternatives audited</span></div><div class='metric'><b>{he_manifest['total_eligible_unique_masks']:,}</b><span>eligible internal H&amp;E alternatives audited</span></div><div class='metric'><b>{he_manifest.get('prompts_without_eligible_mask', 0)}</b><span>H&amp;E calls with no point-consistent mask</span></div><div class='metric'><b>{len(retained_pairs)}</b><span>same-prompt H&amp;E retained by the 40% rule</span></div></div><div class='grid2'><figure><h3>Selected FICTURE primary masks</h3><img src='{ficture_map_uri}' alt='{tma} FICTURE SAM candidates'><figcaption>{len(ficture_selected)} masks, colored by the GeneMap module that placed each prompt.</figcaption></figure><figure><h3>Available paired H&amp;E masks before the 40% rule</h3><img src='{he_map_uri}' alt='{tma} paired H&E SAM candidates'><figcaption>{len(he_selected)} point-consistent H&amp;E masks were available; {len(retained_pairs)} add enough new area to enter the pool.</figcaption></figure></div><h3>Independent H&amp;E-present / FICTURE-low branch</h3><p>The same detector is applied to every TMA: 64 by 64 cells, at least 35% H&amp;E tissue, at most 8% FICTURE signal, and at least six connected cells. Each detected region receives one 10% context box plus one automatic inside positive point.</p><figure><img src='{gap_overlay_uri}' alt='{tma} independent H&E gap prompts'><figcaption>{gap_prompt_manifest['detected_region_count']} regions were detected and {gap_sam_manifest['selected_candidate_count']} independent H&amp;E masks were selected.</figcaption></figure></section>
<section><h2>6. Frozen candidate pool before the VLM</h2><div class='formula'><b>{source_counts['FICTURE primary']} FICTURE primary + {source_counts['Same-prompt H&E supplement']} same-prompt H&amp;E + {source_counts['Independent H&E supplement']} independent H&amp;E = {len(pool_rows)} final candidates</b></div><p>IoU 0.90 cleanup removed {dedup_removed} near-duplicate masks. No masks were unioned.</p><figure><h3>All final candidates on registered H&amp;E</h3><img src='{final_pool_uri}' alt='{tma} final pre-VLM candidate map'><figcaption>Nine fixed colors identify GeneMap modules for FICTURE primary masks. Cyan identifies same-prompt H&amp;E supplements. Magenta identifies independent H&amp;E supplements.</figcaption></figure><div class='legend'>{module_legend}<span><i style='background:{SOURCE_COLORS['Same-prompt H&E supplement']}'></i>Same-prompt H&amp;E</span><span><i style='background:{SOURCE_COLORS['Independent H&E supplement']}'></i>Independent H&amp;E</span></div></section>
<section><h2>7. Interactive inventory of all {len(pool_rows)} frozen candidates</h2><p>Select a source or module, then click one candidate. The two panels show the same mask in local H&amp;E and registered raw FICTURE context. No annotation or tissue label is used here.</p><div class='viewer'><aside><div class='controls'><label>Source<select id='source-filter'><option value='all'>All sources</option>{source_filter_options}</select></label><label>GeneMap module<select id='module-filter'><option value='all'>All modules</option>{module_filter_options}</select></label></div><div class='candidate-list' id='candidate-list'></div></aside><div><h3 id='candidate-title'></h3><p id='candidate-source'></p><div class='candidate-meta'><div><b id='candidate-area'></b><span>mask pixels</span></div><div><b id='candidate-score'></b><span>SAM score</span></div><div><b id='candidate-module'></b><span>GeneMap module</span></div><div><b>Frozen</b><span>pre-VLM status</span></div></div><div class='candidate-images'><figure><h3>Candidate on H&amp;E</h3><img id='candidate-he'><figcaption>Local morphology and the selected mask boundary.</figcaption></figure><figure><h3>Candidate on raw FICTURE</h3><img id='candidate-ficture' style='image-rendering:pixelated'><figcaption>The same candidate coordinates on the unchanged K=12 FICTURE input.</figcaption></figure></div></div></div><details><summary>Complete candidate table</summary><div class='details-body'>{table(['Candidate','Source','Module','Area pixels','SAM score'],[[f'{tma} C{i:02d}',html.escape(row['source']),html.escape(row['gene_module'] or 'none'),f"{int(row['area_pixels']):,}",f"{float(row['sam_score']):.4f}"] for i,row in enumerate(pool_rows,1)])}</div></details></section>
<section><h2>8. Reproducibility audit</h2>{table(['Item','Verified value'],[['Workflow','tma39-fixed-sam3-v1'],['Registered canvas',f'{he_image.width} x {he_image.height}'],['Frozen gene-module SHA-256',html.escape(gene_summary['fixed_module_source_sha256'])],['FICTURE legend SHA-256',EXPECTED_FACTOR_INFO_SHA256],['Transcript rows read',f"{gene_summary['transcript_rows']:,}"],['Transcript rows mapped',f"{gene_summary['transcript_rows_mapped_to_tma_canvas']:,}"],['GeneMap grid',f"{raw_scores.shape[2]} x {raw_scores.shape[1]}"],['Tissue bins',f"{int(tissue.sum()):,}"],['Main prompts',str(len(prompt_rows))],['Main SAM3 calls',f"{ficture_manifest['total_inference_calls']} FICTURE + {he_manifest['total_inference_calls']} H&E"],['Independent H&E calls',str(gap_sam_manifest['total_inference_calls'])],['Final pool',f"{len(pool_rows)} unique masks"],['Annotation files used','0']])}<details><summary>Input hashes</summary><div class='details-body audit'>H&amp;E: {audited_input['he_sha256']}<br>FICTURE: {audited_input['ficture_sha256']}<br>Factor legend: {EXPECTED_FACTOR_INFO_SHA256}</div></details></section>
<p class='footer'>Generated only from the frozen pre-VLM workflow. Classification, annotation comparison, and second-SAM refinement are outside this report.</p>
</main><div class='modal' id='modal'><button id='modal-close' aria-label='Close'>&times;</button><p id='modal-title'></p><img id='modal-image'></div><script>
const candidates={json.dumps(candidate_payload,separators=(',',':'))};
const list=document.getElementById('candidate-list');const sourceFilter=document.getElementById('source-filter');const moduleFilter=document.getElementById('module-filter');let active=null;
function show(item){{active=item.id;document.getElementById('candidate-title').textContent=item.id;document.getElementById('candidate-source').textContent=item.source_label;document.getElementById('candidate-area').textContent=item.area.toLocaleString();document.getElementById('candidate-score').textContent=item.sam_score.toFixed(4);document.getElementById('candidate-module').textContent=item.module==='none'?'none':`Module ${{item.module}}`;document.getElementById('candidate-he').src=item.he;document.getElementById('candidate-ficture').src=item.ficture;renderList();}}
function filtered(){{return candidates.filter(item=>(sourceFilter.value==='all'||item.source===sourceFilter.value)&&(moduleFilter.value==='all'||String(item.module)===moduleFilter.value));}}
function renderList(){{const rows=filtered();list.innerHTML='';for(const item of rows){{const b=document.createElement('button');b.className='candidate-button'+(item.id===active?' active':'');b.innerHTML=`<b>${{item.id}}</b><small>${{item.source_label}}</small>`;b.onclick=()=>show(item);list.appendChild(b);}}if(rows.length&&!rows.some(item=>item.id===active))show(rows[0]);}}
sourceFilter.onchange=renderList;moduleFilter.onchange=renderList;renderList();
const modal=document.getElementById('modal');document.querySelectorAll('.image-button').forEach(button=>button.onclick=()=>{{document.getElementById('modal-image').src=button.dataset.image;document.getElementById('modal-title').textContent=button.dataset.title;modal.classList.add('open');}});document.getElementById('modal-close').onclick=()=>modal.classList.remove('open');modal.onclick=e=>{{if(e.target===modal)modal.classList.remove('open');}};
</script></body></html>"""
    content = content.replace(
        "Every FICTURE prompt must produce a primary mask. An H&amp;E call contributes at most one mask; if no proposal contains its positive point, that call is recorded without adding an H&amp;E candidate.",
        "Each call contributes at most one selected mask. If no proposal survives the same positive-point, foreground, area, and duplicate checks, that call is recorded and adds no mask; the box or point is never moved for one TMA.",
    )
    eligible_ficture_metric = (
        f"<div class='metric'><b>{ficture_manifest['total_eligible_unique_masks']:,}</b>"
        "<span>eligible internal FICTURE alternatives audited</span></div>"
    )
    no_output_metrics = (
        f"<div class='metric'><b>{ficture_manifest.get('prompts_without_eligible_mask', 0)}</b>"
        "<span>FICTURE calls with no valid mask</span></div>"
        f"<div class='metric'><b>{he_manifest.get('prompts_without_eligible_mask', 0)}</b>"
        "<span>H&amp;E calls with no valid mask</span></div>"
    )
    content = content.replace(
        eligible_ficture_metric,
        no_output_metrics + eligible_ficture_metric,
        1,
    )
    content = content.replace(
        f"<div class='metric'><b>{he_manifest.get('prompts_without_eligible_mask', 0)}</b>"
        "<span>H&amp;E calls with no point-consistent mask</span></div>",
        "",
        1,
    )
    args.output.write_text(content, encoding="utf-8")
    print(json.dumps({"tma": tma, "output": str(args.output), "bytes": args.output.stat().st_size, "candidates": len(pool_rows)}, indent=2))


if __name__ == "__main__":
    main()
