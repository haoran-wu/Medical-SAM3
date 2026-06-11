#!/usr/bin/env python3
"""Append a vectorized FICTURE factor-1 inventory for vessels component 1.

This section answers a narrower failure-analysis question:

* Does the official FICTURE vessel-wall factor contain the missed tiny vessel?
* If yes, why did the semantic proposal/assembly still miss it?
* Would a simple top-k connected-component rule be enough?

The code is intentionally vectorized.  Earlier ad-hoc sweeps that materialized
one mask per component were too slow for this ROI.
"""

from __future__ import annotations

import csv
import html
import re
import zipfile
import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage

import add_jun09_precise_failure_framework_v5 as pack


ROOT = Path("/Users/haoranwu/Desktop/Yan_Lab_Research/Medical-SAM3")
BASE = ROOT / "output/visium_hd_exp1/final_deliverables/Jun09_SkillRanker_ComponentAwareMaskSelection"
OUT = BASE / "vessels_component_oracle_prompt_diagnostic/ficture_factor1_component_inventory"
LOCAL_LOCATOR = BASE / "vessels_component_oracle_prompt_diagnostic/component1_ficture_local_locator/component1_best_ficture_local_locator_mask.png"
ROI_ROOT = ROOT / "output/visium_hd_exp1/ficture_official_filtered_candidate_pool_input_roi"
HE_ROI = ROI_ROOT / "he_roi_matching_official_ficture_coverage.png"
FICTURE_ROI = ROI_ROOT / "ficture_official_filtered_roi_rgb.png"
FACTOR_INDEX = ROI_ROOT / "ficture_official_filtered_roi_factor_index.npy"
ANNOTATION = ROI_ROOT / "cropped_annotation_masks/05_lung_vessels_target_roi.png"
LEGEND = ROOT / "data/visium_hd_exp1/current_ficture_vlm_inputs/ficture_factor_legend_for_prompt.csv"
HTML = BASE / "Jun09_SkillRanker_ComponentAwareMaskSelection.html"
INDEX = BASE / "index.html"


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def table(rows: list[dict[str, object]], cols: list[str]) -> str:
    bits = ["<table><thead><tr>"]
    bits.extend(f"<th>{html.escape(col)}</th>" for col in cols)
    bits.append("</tr></thead><tbody>")
    for row in rows:
        bits.append("<tr>")
        for col in cols:
            bits.append(f"<td>{html.escape(str(row.get(col, '')))}</td>")
        bits.append("</tr>")
    bits.append("</tbody></table>")
    return "".join(bits)


def metrics(pred: np.ndarray, gt: np.ndarray) -> dict[str, float]:
    tp = int(np.logical_and(pred, gt).sum())
    pred_area = int(pred.sum())
    gt_area = int(gt.sum())
    return {
        "dice": 2 * tp / (pred_area + gt_area) if pred_area + gt_area else 0.0,
        "precision": tp / pred_area if pred_area else 0.0,
        "recall": tp / gt_area if gt_area else 0.0,
        "pixels": pred_area,
        "tp": tp,
    }


def bbox_from_mask(mask: np.ndarray) -> tuple[int, int, int, int]:
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return (0, 0, 0, 0)
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def overlay_mask(image: Image.Image, mask: np.ndarray, color: tuple[int, int, int], alpha: int = 135) -> Image.Image:
    base = image.convert("RGBA")
    rgba = np.zeros((*mask.shape, 4), dtype=np.uint8)
    rgba[mask] = (*color, alpha)
    return Image.alpha_composite(base, Image.fromarray(rgba, mode="RGBA")).convert("RGB")


def mask_only(mask: np.ndarray, color: tuple[int, int, int]) -> Image.Image:
    arr = np.full((*mask.shape, 3), 255, dtype=np.uint8)
    arr[mask] = color
    return Image.fromarray(arr, mode="RGB")


def fit_image(image: Image.Image, width: int, height: int) -> Image.Image:
    img = image.convert("RGB")
    img.thumbnail((width, height), Image.Resampling.BICUBIC)
    canvas = Image.new("RGB", (width, height), "white")
    canvas.paste(img, ((width - img.width) // 2, (height - img.height) // 2))
    return canvas


def crop_around(mask: np.ndarray, width: int, height: int, pad: int = 220) -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = bbox_from_mask(mask)
    return max(0, x0 - pad), max(0, y0 - pad), min(width, x1 + pad + 1), min(height, y1 + pad + 1)


def crop_array(mask: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    x0, y0, x1, y1 = box
    return mask[y0:y1, x0:x1]


def render_six_panel(
    out: Path,
    title: str,
    subtitle: str,
    pred: np.ndarray,
    ann: np.ndarray,
    pred_label: str,
    crop_box: tuple[int, int, int, int] | None = None,
) -> None:
    he = Image.open(HE_ROI).convert("RGB")
    ficture = Image.open(FICTURE_ROI).convert("RGB")
    if crop_box is not None:
        he = he.crop(crop_box)
        ficture = ficture.crop(crop_box)
        pred = crop_array(pred, crop_box)
        ann = crop_array(ann, crop_box)
    m = metrics(pred, ann)
    panel_w, panel_h = 260, 275
    gap, margin, header_h, label_h = 18, 26, 110, 24
    sheet = Image.new("RGB", (margin * 2 + panel_w * 6 + gap * 5, margin * 2 + header_h + label_h + panel_h), "white")
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()
    draw.text((margin, margin), title, fill=(20, 20, 20), font=font)
    draw.text(
        (margin, margin + 24),
        f"component-1 Dice {m['dice']:.3f} | Precision {m['precision']:.3f} | Recall {m['recall']:.3f}",
        fill=(70, 80, 90),
        font=font,
    )
    draw.text((margin, margin + 48), subtitle[:210], fill=(70, 80, 90), font=font)
    panels = [
        ("Annotation component on H&E", overlay_mask(he, ann, (0, 180, 90))),
        (pred_label, overlay_mask(he, pred, (0, 90, 255))),
        ("Candidate mask only", mask_only(pred, (0, 90, 255))),
        ("Annotation mask only", mask_only(ann, (0, 180, 90))),
        ("H&E ROI", he),
        ("FICTURE ROI", ficture),
    ]
    y0 = margin + header_h
    for idx, (label, img) in enumerate(panels):
        x = margin + idx * (panel_w + gap)
        draw.text((x, y0), label, fill=(35, 45, 60), font=font)
        sheet.paste(fit_image(img, panel_w, panel_h), (x, y0 + label_h))
    out.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out)


def factor_legend_row() -> dict[str, str]:
    if not LEGEND.exists():
        return {}
    with LEGEND.open(newline="") as handle:
        for row in csv.DictReader(handle):
            if str(row.get("Factor", "")) == "1":
                return row
    return {}


def read_csv(path: Path) -> list[dict[str, object]]:
    with path.open(newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def compute_inventory() -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]], dict[str, np.ndarray]]:
    factor = np.load(FACTOR_INDEX)
    factor1 = factor == 1
    vessels = np.asarray(Image.open(ANNOTATION).convert("L")) > 127
    ann_labels, n_ann = ndimage.label(vessels, structure=ndimage.generate_binary_structure(2, 1))
    comp1 = ann_labels == 1
    f_labels, n_f = ndimage.label(factor1, structure=ndimage.generate_binary_structure(2, 1))

    areas = np.bincount(f_labels.ravel(), minlength=n_f + 1)
    overlap_vessels = np.bincount(f_labels[vessels].ravel(), minlength=n_f + 1)
    overlap_comp1 = np.bincount(f_labels[comp1].ravel(), minlength=n_f + 1)
    ann_areas = np.bincount(ann_labels.ravel(), minlength=n_ann + 1)

    best_comp_id = np.zeros(n_f + 1, dtype=np.int32)
    best_dice = np.zeros(n_f + 1, dtype=float)
    best_precision = np.zeros(n_f + 1, dtype=float)
    best_recall = np.zeros(n_f + 1, dtype=float)
    for comp_id in range(1, n_ann + 1):
        comp_mask = ann_labels == comp_id
        overlaps = np.bincount(f_labels[comp_mask].ravel(), minlength=n_f + 1)
        pred_areas = areas.astype(float)
        comp_area = float(ann_areas[comp_id])
        dice = np.zeros(n_f + 1, dtype=float)
        valid = (pred_areas + comp_area) > 0
        dice[valid] = 2.0 * overlaps[valid] / (pred_areas[valid] + comp_area)
        update = dice > best_dice
        best_dice[update] = dice[update]
        best_comp_id[update] = comp_id
        with np.errstate(divide="ignore", invalid="ignore"):
            precision = np.where(pred_areas > 0, overlaps / pred_areas, 0.0)
            recall = overlaps / comp_area if comp_area else np.zeros(n_f + 1)
        best_precision[update] = precision[update]
        best_recall[update] = recall[update]

    ids = np.arange(1, n_f + 1)
    large_order = ids[np.argsort(-areas[1:])]
    small_order = ids[np.argsort(areas[1:])]
    large_rank = np.zeros(n_f + 1, dtype=np.int32)
    small_rank = np.zeros(n_f + 1, dtype=np.int32)
    large_rank[large_order] = np.arange(1, len(large_order) + 1)
    small_rank[small_order] = np.arange(1, len(small_order) + 1)

    rows: list[dict[str, object]] = []
    for cid in ids:
        if areas[cid] <= 0:
            continue
        mask = f_labels == cid
        x0, y0, x1, y1 = bbox_from_mask(mask)
        rows.append(
            {
                "ficture_component_id": int(cid),
                "area": int(areas[cid]),
                "bbox_xyxy": f"{x0},{y0},{x1},{y1}",
                "overlap_vessels_pixels": int(overlap_vessels[cid]),
                "overlap_component1_pixels": int(overlap_comp1[cid]),
                "best_annotation_component_id": int(best_comp_id[cid]),
                "best_component_dice": float(best_dice[cid]),
                "best_component_precision": float(best_precision[cid]),
                "best_component_recall": float(best_recall[cid]),
                "is_component1_related": bool(overlap_comp1[cid] > 0),
                "area_rank_small_first": int(small_rank[cid]),
                "area_rank_large_first": int(large_rank[cid]),
                "vessel_candidate_size_bucket": "tiny" if areas[cid] < 100 else "small" if areas[cid] < 500 else "medium" if areas[cid] < 3000 else "large",
            }
        )

    rows = sorted(rows, key=lambda r: (int(r["area_rank_large_first"]), -float(r["best_component_dice"])))
    related = [r for r in rows if r["is_component1_related"]]

    topk_rows: list[dict[str, object]] = []
    for k in [12, 24, 31, 40, 60, 100]:
        chosen = large_order[:k]
        pred = np.isin(f_labels, chosen)
        whole = metrics(pred, vessels)
        c1 = metrics(pred, comp1)
        topk_rows.append(
            {
                "rule": f"union largest {k} factor-1 components",
                "whole-vessels D/P/R": f"{whole['dice']:.3f} / {whole['precision']:.3f} / {whole['recall']:.3f}",
                "component-1 D/P/R": f"{c1['dice']:.3f} / {c1['precision']:.3f} / {c1['recall']:.3f}",
                "component-1 included": bool(any(int(r["ficture_component_id"]) in set(map(int, chosen)) for r in related)),
                "selected pixels": int(pred.sum()),
            }
        )

    write_csv(OUT / "ficture_factor1_component_inventory.csv", rows)
    write_csv(OUT / "component1_related_ficture_factor1_components.csv", related)
    write_csv(OUT / "topk_largest_factor1_union_metrics.csv", topk_rows)
    arrays = {"f_labels": f_labels, "vessels": vessels, "comp1": comp1}
    return rows, related, topk_rows, arrays


def compute_topk_from_rows(rows: list[dict[str, object]], related: list[dict[str, object]]) -> list[dict[str, object]]:
    factor = np.load(FACTOR_INDEX)
    f_labels, _ = ndimage.label(factor == 1, structure=ndimage.generate_binary_structure(2, 1))
    vessels = np.asarray(Image.open(ANNOTATION).convert("L")) > 127
    ann_labels, _ = ndimage.label(vessels, structure=ndimage.generate_binary_structure(2, 1))
    comp1 = ann_labels == 1
    large_order = [int(row["ficture_component_id"]) for row in sorted(rows, key=lambda r: int(float(r["area_rank_large_first"])))]
    related_ids = {int(r["ficture_component_id"]) for r in related}
    out: list[dict[str, object]] = []
    for k in [12, 24, 31, 40, 60, 100]:
        chosen = large_order[:k]
        pred = np.isin(f_labels, chosen)
        whole = metrics(pred, vessels)
        c1 = metrics(pred, comp1)
        out.append(
            {
                "rule": f"union largest {k} factor-1 components",
                "whole-vessels D/P/R": f"{whole['dice']:.3f} / {whole['precision']:.3f} / {whole['recall']:.3f}",
                "component-1 D/P/R": f"{c1['dice']:.3f} / {c1['precision']:.3f} / {c1['recall']:.3f}",
                "component-1 included": bool(set(chosen) & related_ids),
                "selected pixels": int(pred.sum()),
            }
        )
    write_csv(OUT / "topk_largest_factor1_union_metrics.csv", out)
    return out


def append_or_replace(section: str) -> None:
    marker = "<h2>17BF. FICTURE Factor-1 Inventory: Why Vessels Still Needs a Tiny Locator</h2>"
    for path in [HTML, INDEX]:
        text = path.read_text()
        if marker in text:
            start = text.index(marker)
            next_match = re.search(r"<h2>17B[G-Z]|<h2>18\.", text[start + len(marker) :])
            end = start + len(marker) + next_match.start() if next_match else text.index("</body>", start)
            text = text[:start] + section + text[end:]
        else:
            insert_before = "<h2>18."
            if insert_before in text:
                idx = text.index(insert_before)
                text = text[:idx] + section + text[idx:]
            else:
                text = text.replace("</body>", section + "</body>")
        path.write_text(text)


def verify(path: Path) -> None:
    text = path.read_text(errors="ignore")
    missing: list[str] = []
    for src in re.findall(r"<img[^>]+src=\"([^\"]+)\"", text):
        if src.startswith(("data:", "http://", "https://")):
            continue
        if not (path.parent / src).exists():
            missing.append(src)
    if missing:
        raise RuntimeError(f"missing images in {path}: {missing[:8]}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--rebuild-zip",
        action="store_true",
        help="Rebuild the large shareable ZIP after updating HTML. Skipped by default because it is slow.",
    )
    parser.add_argument(
        "--refresh-inventory",
        action="store_true",
        help="Recompute connected-component inventory instead of reusing existing CSV outputs.",
    )
    args = parser.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    inventory_csv = OUT / "ficture_factor1_component_inventory.csv"
    related_csv = OUT / "component1_related_ficture_factor1_components.csv"
    topk_csv = OUT / "topk_largest_factor1_union_metrics.csv"
    if args.refresh_inventory or not (inventory_csv.exists() and related_csv.exists()):
        rows, related, topk_rows, arrays = compute_inventory()
    else:
        rows = read_csv(inventory_csv)
        related = read_csv(related_csv)
        topk_rows = read_csv(topk_csv) if topk_csv.exists() else compute_topk_from_rows(rows, related)
        vessels = np.asarray(Image.open(ANNOTATION).convert("L")) > 127
        ann_labels, _ = ndimage.label(vessels, structure=ndimage.generate_binary_structure(2, 1))
        comp1 = ann_labels == 1
        if LOCAL_LOCATOR.exists():
            pred = np.asarray(Image.open(LOCAL_LOCATOR).convert("L")) > 127
        else:
            factor = np.load(FACTOR_INDEX)
            f_labels, _ = ndimage.label(factor == 1, structure=ndimage.generate_binary_structure(2, 1))
            pred = f_labels == int(related[0]["ficture_component_id"])
        arrays = {"vessels": vessels, "comp1": comp1, "pred_component1_factor1": pred}
    if not related:
        raise RuntimeError("No FICTURE factor-1 component overlaps vessels component 1")

    component_row = related[0]
    component_id = int(component_row["ficture_component_id"])
    pred = arrays.get("pred_component1_factor1")
    if pred is None:
        pred = arrays["f_labels"] == component_id
    comp1 = arrays["comp1"]
    width, height = Image.open(HE_ROI).size
    full_fig = OUT / "figures/vessels_component1_factor1_locator_full_roi.png"
    zoom_fig = OUT / "figures/vessels_component1_factor1_locator_zoom.png"
    render_six_panel(
        full_fig,
        "vessels component 1: FICTURE factor-1 locator",
        f"Factor-1 component {component_id}; this is the local FICTURE support before second-SAM.",
        pred,
        comp1,
        "Factor-1 component on H&E",
        crop_box=None,
    )
    render_six_panel(
        zoom_fig,
        "vessels component 1: zoomed FICTURE factor-1 locator",
        "Zoomed view shows that FICTURE finds the neighborhood but includes extra surrounding vessel-wall/stromal signal.",
        pred,
        comp1,
        "Factor-1 component on H&E",
        crop_box=crop_around(np.logical_or(pred, comp1), width, height, pad=180),
    )

    legend = factor_legend_row()
    factor_row = {
        "factor": "1",
        "RGB": legend.get("RGB", "(0, 255, 255)"),
        "major compartment": legend.get("Major Compartment", "Stromal Compartment"),
        "cell type": legend.get("Celltype2", "Fibroblasts & Smooth Muscle Cells"),
        "why relevant": "This is the official current legend row whose marker genes include vessel-wall smooth-muscle/stromal markers.",
    }
    summary_rows = [
        {"question": "How many factor-1 connected components exist?", "answer": f"{len(rows):,}"},
        {"question": "How many overlap any vessels annotation pixel?", "answer": f"{sum(int(r['overlap_vessels_pixels']) > 0 for r in rows):,}"},
        {"question": "Which factor-1 component touches vessels component 1?", "answer": str(component_id)},
        {"question": "Its size rank among factor-1 components", "answer": f"large-first rank {component_row['area_rank_large_first']} / {len(rows)}; small-first rank {component_row['area_rank_small_first']} / {len(rows)}"},
        {"question": "Its component-1 D/P/R before SAM", "answer": f"{float(component_row['best_component_dice']):.3f} / {float(component_row['best_component_precision']):.3f} / {float(component_row['best_component_recall']):.3f}"},
    ]

    section = f"""
<h2>17BF. FICTURE Factor-1 Inventory: Why Vessels Still Needs a Tiny Locator</h2>
<p><b>Goal.</b> Section 17BE showed that FICTURE can roughly locate vessels component 1, but not tightly enough.  This section makes the failure more precise by inventorying every connected component of the official FICTURE vessel-wall factor.</p>
<p><b>Factor used.</b> Factor 1 is the current source-matched FICTURE factor with RGB/cell-type information below.  This is the newer official legend, not the old unmatched HTML legend.</p>
{table([factor_row], ["factor", "RGB", "major compartment", "cell type", "why relevant"])}
<h3>Inventory result</h3>
{table(summary_rows, ["question", "answer"])}
<p><b>Interpretation.</b> The missed tiny vessel is not invisible in FICTURE.  It is inside factor-1 component {component_id}, but that component is broad: component-level D/P/R is {float(component_row['best_component_dice']):.3f}/{float(component_row['best_component_precision']):.3f}/{float(component_row['best_component_recall']):.3f}.  A global keep-top-12 connected-component rule misses it because it is rank {component_row['area_rank_large_first']} by size; increasing keep-top enough to include it also admits many other large factor-1 regions.</p>
{table(topk_rows, ["rule", "whole-vessels D/P/R", "component-1 D/P/R", "component-1 included", "selected pixels"])}
<figure><img src="{html.escape(str(full_fig.relative_to(BASE)))}" alt="full ROI factor-1 component locator"><figcaption>Full-ROI six-panel view.  The true component is small, so the zoomed panel below is the easier visual diagnostic.</figcaption></figure>
<figure><img src="{html.escape(str(zoom_fig.relative_to(BASE)))}" alt="zoomed factor-1 component locator"><figcaption>Zoomed six-panel view.  FICTURE support overlaps the target, but the support region is too broad to be used directly as the final mask or as a loose SAM box.</figcaption></figure>
<div class='callout'><b>Failure split.</b> This is not a pure VLM failure and not a pure SAM failure.  FICTURE gives a useful weak locator; SAM succeeds with a tight locator; the missing skill is converting broad FICTURE vessel-wall support plus local H&amp;E morphology into a tight tiny-vessel prompt.</div>
<p class='small'>Machine-readable outputs: <code>vessels_component_oracle_prompt_diagnostic/ficture_factor1_component_inventory/ficture_factor1_component_inventory.csv</code>, <code>component1_related_ficture_factor1_components.csv</code>, and <code>topk_largest_factor1_union_metrics.csv</code>.</p>
"""
    append_or_replace(section)
    verify(HTML)
    verify(INDEX)
    if args.rebuild_zip:
        pack.rebuild_zip()
        with zipfile.ZipFile(BASE / "Jun09_SkillRanker_ComponentAwareMaskSelection_shareable.zip") as handle:
            bad = handle.testzip()
            if bad:
                raise RuntimeError(f"bad zip entry: {bad}")
    print(HTML)
    print(INDEX)
    print(BASE / "Jun09_SkillRanker_ComponentAwareMaskSelection_shareable.zip")


if __name__ == "__main__":
    main()
