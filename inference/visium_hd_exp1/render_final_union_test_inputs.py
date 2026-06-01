#!/usr/bin/env python3
"""Render the six-class final mask inputs used before VLM/CLIP tests.

The purpose of this script is to freeze the rule that tests should see the
final candidate mask for each class: component-unioned where the union has been
validated, otherwise the current single-mask fallback.
"""

from __future__ import annotations

import argparse
import csv
import html
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
ROI_DIR = PROJECT_ROOT / "output/visium_hd_exp1/ficture_official_filtered_candidate_pool_input_roi"
MAY22_DIR = PROJECT_ROOT / "output/visium_hd_exp1/final_deliverables/May22_same_roi_HE_FICTURE_source_image_CLIP_final"
MAY30_UNION_DIR = PROJECT_ROOT / "output/visium_hd_exp1/final_deliverables/May30_he_ficture_merged_fullroi_panels"
DEFAULT_DATA_OUT = PROJECT_ROOT / "data/visium_hd_exp1/current_ficture_vlm_inputs/final_union_test_inputs"
DEFAULT_REPORT_OUT = PROJECT_ROOT / "output/visium_hd_exp1/final_deliverables/Jun01_final_union_test_inputs_preview"


@dataclass(frozen=True)
class FinalClassInput:
    tissue_class: str
    test_mask_rule: str
    source: str
    mask_path: Path
    dice: float
    precision: float
    recall: float
    note: str


FINAL_INPUTS = [
    FinalClassInput(
        tissue_class="bronchiola",
        test_mask_rule="component-aware HE+FICTURE union",
        source="HE+FICTURE union",
        mask_path=MAY30_UNION_DIR / "bronchiola_merged_he_ficture_union_mask.png",
        dice=0.886550,
        precision=0.881295,
        recall=0.891868,
        note="Use union because annotation has several meaningful pieces and the merged union improves recall while keeping precision high.",
    ),
    FinalClassInput(
        tissue_class="alveoli",
        test_mask_rule="single best mask",
        source="HE oracle best",
        mask_path=MAY22_DIR
        / "01_HE_same_ROI_candidate_pool_best/best_candidate_masks/02_lung_alveoli_normal_adjacent_best_candidate_mask.png",
        dice=0.677202,
        precision=0.710824,
        recall=0.646618,
        note="Use one mask because the GT is effectively one main piece; component assembly is not needed.",
    ),
    FinalClassInput(
        tissue_class="vessels",
        test_mask_rule="component-aware HE+FICTURE union",
        source="HE+FICTURE union",
        mask_path=MAY30_UNION_DIR / "vessels_merged_he_ficture_union_mask.png",
        dice=0.890816,
        precision=0.854621,
        recall=0.930212,
        note="Use union because single-best vessels misses multiple meaningful vessel pieces.",
    ),
    FinalClassInput(
        tissue_class="tumor",
        test_mask_rule="precision-aware single-mask fallback",
        source="FICTURE oracle best",
        mask_path=MAY22_DIR
        / "02_FICTURE_official_candidate_pool_best/best_candidate_masks/04_tumor_best_candidate_mask.png",
        dice=0.527418,
        precision=0.365660,
        recall=0.945822,
        note="Do not union every tumor component; current clean fallback is the FICTURE single best until precision-aware component masks are available locally.",
    ),
    FinalClassInput(
        tissue_class="stroma",
        test_mask_rule="precision-aware single-mask fallback",
        source="FICTURE oracle best",
        mask_path=MAY22_DIR
        / "02_FICTURE_official_candidate_pool_best/best_candidate_masks/05_stroma_best_candidate_mask.png",
        dice=0.527584,
        precision=0.371315,
        recall=0.910967,
        note="Do not union every stroma component; broad fragmented tissue is kept as a single-mask fallback for this preview.",
    ),
    FinalClassInput(
        tissue_class="immune_infiltration",
        test_mask_rule="precision-aware single-mask fallback",
        source="FICTURE precision-preferred oracle best",
        mask_path=MAY22_DIR
        / "02_FICTURE_official_candidate_pool_best/best_candidate_masks/06_immune_infiltration_best_candidate_mask.png",
        dice=0.275263,
        precision=0.446625,
        recall=0.198935,
        note="Use the cleaner FICTURE single mask for this preview; later precision-aware subset selection can replace it if component masks are copied from Bouchet.",
    ),
]


ANNOTATION_MASKS = {
    "bronchiola": "01_lung_bronchiola_target_roi.png",
    "alveoli": "04_lung_alveoli_normal_adjacent_target_roi.png",
    "vessels": "05_lung_vessels_target_roi.png",
    "tumor": "08_tumor_target_roi.png",
    "stroma": "07_stroma_target_roi.png",
    "immune_infiltration": "03_immune_infiltration_target_roi.png",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-out", type=Path, default=DEFAULT_DATA_OUT)
    parser.add_argument("--report-out", type=Path, default=DEFAULT_REPORT_OUT)
    parser.add_argument("--max-crop-side", type=int, default=768)
    parser.add_argument("--crop-padding", type=int, default=96)
    return parser.parse_args()


def load_binary_mask(path: Path, expected_size: tuple[int, int]) -> np.ndarray:
    image = Image.open(path).convert("L")
    if image.size != expected_size:
        image = image.resize(expected_size, Image.Resampling.NEAREST)
    return np.array(image) > 0


def save_binary_mask(mask: np.ndarray, path: Path) -> None:
    Image.fromarray((mask.astype(np.uint8) * 255), mode="L").save(path)


def metrics(pred: np.ndarray, gt: np.ndarray) -> dict[str, float]:
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    tp = int(np.logical_and(pred, gt).sum())
    pred_area = int(pred.sum())
    gt_area = int(gt.sum())
    precision = tp / pred_area if pred_area else 0.0
    recall = tp / gt_area if gt_area else 0.0
    dice = (2 * tp) / (pred_area + gt_area) if pred_area + gt_area else 0.0
    return {"dice": dice, "precision": precision, "recall": recall}


def overlay_mask(image: Image.Image, mask: np.ndarray, color: tuple[int, int, int], alpha: int = 132) -> Image.Image:
    base = image.convert("RGBA")
    rgba = np.zeros((*mask.shape, 4), dtype=np.uint8)
    rgba[mask] = (*color, alpha)
    return Image.alpha_composite(base, Image.fromarray(rgba, mode="RGBA")).convert("RGB")


def mask_only(mask: np.ndarray, color: tuple[int, int, int]) -> Image.Image:
    arr = np.full((*mask.shape, 3), 255, dtype=np.uint8)
    arr[mask] = color
    return Image.fromarray(arr, mode="RGB")


def fit_image(image: Image.Image, width: int, height: int) -> Image.Image:
    resized = image.copy()
    resized.thumbnail((width, height), Image.Resampling.BICUBIC)
    canvas = Image.new("RGB", (width, height), "white")
    canvas.paste(resized, ((width - resized.width) // 2, (height - resized.height) // 2))
    return canvas


def render_class_panel(
    tissue_class: str,
    he: Image.Image,
    ficture: Image.Image,
    final_mask: np.ndarray,
    annotation: np.ndarray,
    row: dict[str, object],
    out_path: Path,
) -> None:
    panel_w, panel_h = 430, 455
    title_h = 55
    gap = 18
    margin = 26
    items = [
        ("Annotation on H&E", overlay_mask(he, annotation, (0, 180, 90))),
        ("Final test mask on H&E", overlay_mask(he, final_mask, (0, 90, 255))),
        ("Final test mask only", mask_only(final_mask, (0, 90, 255))),
        ("Annotation mask only", mask_only(annotation, (0, 180, 90))),
        ("H&E ROI", he),
        ("FICTURE ROI", ficture),
    ]
    width = margin * 2 + panel_w * 3 + gap * 2
    height = margin * 2 + 98 + (title_h + panel_h) * 2 + gap
    sheet = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()
    draw.text((margin, margin), f"{tissue_class}: final mask sent to tests", fill=(20, 20, 20), font=font)
    subtitle = (
        f"Rule: {row['test_mask_rule']} | Source: {row['source']} | "
        f"Dice {row['reported_dice']:.3f} | Precision {row['reported_precision']:.3f} | Recall {row['reported_recall']:.3f}"
    )
    draw.text((margin, margin + 24), subtitle, fill=(70, 80, 90), font=font)
    draw.text((margin, margin + 48), str(row["note"]), fill=(70, 80, 90), font=font)
    y0 = margin + 98
    for idx, (label, image) in enumerate(items):
        col = idx % 3
        row_idx = idx // 3
        x = margin + col * (panel_w + gap)
        y = y0 + row_idx * (panel_h + title_h + gap)
        draw.text((x, y), label, fill=(30, 40, 55), font=font)
        thumb = fit_image(image, panel_w, panel_h)
        sheet.paste(thumb, (x, y + title_h))
    sheet.save(out_path)


def render_reverse_blur_crop(
    image: Image.Image,
    mask: np.ndarray,
    out_path: Path,
    padding: int,
    max_side: int,
) -> None:
    if not mask.any():
        raise ValueError(f"Empty mask for {out_path}")
    base = image.convert("RGB")
    sharp = np.array(base)
    gray_blur = np.array(base.convert("L").filter(ImageFilter.GaussianBlur(radius=7)).convert("RGB"))
    composed = gray_blur.copy()
    composed[mask] = sharp[mask]
    ys, xs = np.where(mask)
    x0 = max(0, int(xs.min()) - padding)
    x1 = min(base.width, int(xs.max()) + padding + 1)
    y0 = max(0, int(ys.min()) - padding)
    y1 = min(base.height, int(ys.max()) + padding + 1)
    crop = Image.fromarray(composed, mode="RGB").crop((x0, y0, x1, y1))
    if max(crop.size) > max_side:
        scale = max_side / max(crop.size)
        crop = crop.resize((int(crop.width * scale), int(crop.height * scale)), Image.Resampling.BICUBIC)
    crop.save(out_path, quality=92)


def paired_crop_panel(he_crop: Path, ficture_crop: Path, title: str, out_path: Path) -> None:
    he = Image.open(he_crop).convert("RGB")
    ficture = Image.open(ficture_crop).convert("RGB")
    panel_w, panel_h = 420, 420
    margin = 18
    title_h = 44
    sheet = Image.new("RGB", (margin * 3 + panel_w * 2, margin * 2 + title_h + panel_h), "white")
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()
    draw.text((margin, margin), title, fill=(20, 20, 20), font=font)
    for idx, (label, image) in enumerate([("H&E reverse-blur crop", he), ("FICTURE reverse-blur crop", ficture)]):
        x = margin + idx * (panel_w + margin)
        y = margin + title_h
        draw.text((x, y - 18), label, fill=(50, 60, 75), font=font)
        sheet.paste(fit_image(image, panel_w, panel_h), (x, y))
    sheet.save(out_path)


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_html(report_out: Path, rows: list[dict[str, object]]) -> None:
    overview_rel = Path("assets") / "six_class_paired_crop_overview.png"
    table_head = "".join(
        f"<th>{html.escape(col)}</th>"
        for col in [
            "class",
            "test_mask_rule",
            "source",
            "reported Dice",
            "reported Precision",
            "reported Recall",
            "local Dice",
            "local Precision",
            "local Recall",
        ]
    )
    table_rows = []
    for row in rows:
        panel_rel = Path("assets") / f"{row['class']}_final_test_input_panel.png"
        crop_rel = Path("assets") / f"{row['class']}_paired_reverse_blur_crop.png"
        table_rows.append(
            "<tr>"
            f"<td>{html.escape(str(row['class']))}</td>"
            f"<td>{html.escape(str(row['test_mask_rule']))}</td>"
            f"<td>{html.escape(str(row['source']))}</td>"
            f"<td>{row['reported_dice']:.3f}</td>"
            f"<td>{row['reported_precision']:.3f}</td>"
            f"<td>{row['reported_recall']:.3f}</td>"
            f"<td>{row['local_resized_dice']:.3f}</td>"
            f"<td>{row['local_resized_precision']:.3f}</td>"
            f"<td>{row['local_resized_recall']:.3f}</td>"
            "</tr>"
            f"<tr><td colspan='9'><img src='{panel_rel}'><img src='{crop_rel}'></td></tr>"
        )
    html_text = f"""<!doctype html>
<html lang="zh">
<head>
<meta charset="utf-8">
<title>Final union test inputs preview</title>
<style>
body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 34px; color: #1f2933; }}
p {{ line-height: 1.55; max-width: 1100px; }}
table {{ border-collapse: collapse; width: 100%; font-size: 14px; }}
th, td {{ border-bottom: 1px solid #e5e7eb; text-align: left; padding: 8px 10px; vertical-align: top; }}
th {{ background: #f7f7f8; }}
img {{ max-width: 48%; margin: 8px 14px 22px 0; border: 1px solid #ddd; vertical-align: top; }}
.note {{ background: #f3f4f6; border-left: 4px solid #2563eb; padding: 12px 16px; }}
</style>
</head>
<body>
<h1>Final mask inputs sent to tests</h1>
<p class="note">后续 Test1/Test2 不再直接用原始单块 pool。每个 tissue class 先确定一个最终候选 mask：需要 component-aware union 的类别先 union；不需要 union 或还没有通过 precision-aware component 验证的类别，用当前 single-mask fallback。下面每一行就是送去测试的最终 mask 版本。</p>
<p><img src="{overview_rel}" style="max-width: 100%;"></p>
<table><thead><tr>{table_head}</tr></thead><tbody>{''.join(table_rows)}</tbody></table>
</body>
</html>
"""
    (report_out / "index.html").write_text(html_text)


def write_overview(report_out: Path, rows: list[dict[str, object]]) -> None:
    tile_w, tile_h = 620, 390
    margin = 22
    title_h = 38
    cols = 2
    rows_n = 3
    sheet = Image.new(
        "RGB",
        (margin * (cols + 1) + tile_w * cols, margin * (rows_n + 1) + (tile_h + title_h) * rows_n),
        "white",
    )
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()
    for idx, row in enumerate(rows):
        class_name = str(row["class"])
        image = Image.open(report_out / "assets" / f"{class_name}_paired_reverse_blur_crop.png").convert("RGB")
        col = idx % cols
        row_idx = idx // cols
        x = margin + col * (tile_w + margin)
        y = margin + row_idx * (tile_h + title_h + margin)
        title = f"{class_name} | {row['test_mask_rule']}"
        draw.text((x, y), title, fill=(20, 20, 20), font=font)
        sheet.paste(fit_image(image, tile_w, tile_h), (x, y + title_h))
    sheet.save(report_out / "assets" / "six_class_paired_crop_overview.png")


def main() -> None:
    args = parse_args()
    data_out = args.data_out
    report_out = args.report_out
    for directory in [
        data_out / "masks",
        data_out / "crops",
        report_out / "assets",
    ]:
        directory.mkdir(parents=True, exist_ok=True)

    he = Image.open(ROI_DIR / "he_roi_matching_official_ficture_coverage.png").convert("RGB")
    ficture = Image.open(ROI_DIR / "ficture_official_filtered_roi_rgb.png").convert("RGB")
    expected_size = he.size

    rows: list[dict[str, object]] = []
    for item in FINAL_INPUTS:
        final_mask = load_binary_mask(item.mask_path, expected_size)
        annotation = load_binary_mask(ROI_DIR / "cropped_annotation_masks" / ANNOTATION_MASKS[item.tissue_class], expected_size)
        local = metrics(final_mask, annotation)

        mask_rel = Path("masks") / f"{item.tissue_class}_final_test_mask.png"
        he_crop_rel = Path("crops") / f"{item.tissue_class}_he_reverse_blur.png"
        ficture_crop_rel = Path("crops") / f"{item.tissue_class}_ficture_reverse_blur.png"
        pair_rel = Path("crops") / f"{item.tissue_class}_paired_reverse_blur_crop.png"
        save_binary_mask(final_mask, data_out / mask_rel)
        render_reverse_blur_crop(he, final_mask, data_out / he_crop_rel, args.crop_padding, args.max_crop_side)
        render_reverse_blur_crop(ficture, final_mask, data_out / ficture_crop_rel, args.crop_padding, args.max_crop_side)
        paired_crop_panel(data_out / he_crop_rel, data_out / ficture_crop_rel, item.tissue_class, data_out / pair_rel)

        row = {
            "class": item.tissue_class,
            "test_mask_rule": item.test_mask_rule,
            "source": item.source,
            "reported_dice": item.dice,
            "reported_precision": item.precision,
            "reported_recall": item.recall,
            "local_resized_dice": local["dice"],
            "local_resized_precision": local["precision"],
            "local_resized_recall": local["recall"],
            "mask_rel": str(mask_rel),
            "he_crop_rel": str(he_crop_rel),
            "ficture_crop_rel": str(ficture_crop_rel),
            "paired_crop_rel": str(pair_rel),
            "source_mask_path": str(item.mask_path),
            "note": item.note,
        }
        rows.append(row)

        panel_path = report_out / "assets" / f"{item.tissue_class}_final_test_input_panel.png"
        render_class_panel(item.tissue_class, he, ficture, final_mask, annotation, row, panel_path)
        paired_crop_panel(
            data_out / he_crop_rel,
            data_out / ficture_crop_rel,
            item.tissue_class,
            report_out / "assets" / f"{item.tissue_class}_paired_reverse_blur_crop.png",
        )

    write_csv(data_out / "final_test_input_policy.csv", rows)
    write_csv(data_out / "final_test_requests.csv", rows)
    write_csv(report_out / "final_test_input_policy.csv", rows)
    (data_out / "manifest_final_union_test_inputs.json").write_text(
        json.dumps(
            {
                "status": "preview_current_final_test_inputs",
                "roi_size": {"width": expected_size[0], "height": expected_size[1]},
                "rule": "Tests use the final class mask: component-aware union where validated; otherwise the current single-mask fallback.",
                "data_out": str(data_out),
                "report_out": str(report_out),
                "classes": [row["class"] for row in rows],
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    write_overview(report_out, rows)
    write_html(report_out, rows)
    print(report_out)
    print(data_out)


if __name__ == "__main__":
    main()
