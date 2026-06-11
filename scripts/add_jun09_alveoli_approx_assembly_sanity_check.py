#!/usr/bin/env python3
"""Add an alveoli approximate assembly sanity check to the Jun09 report.

The broad-box branch has useful VLM/display crops locally, but the original
remote binary masks are not all available on this machine.  This script makes a
carefully labeled *approximate* sanity check: it reconstructs candidate masks
from saved gray reverse-blur H&E crops using the known candidate fill ratio,
then checks whether the V2 target-grounding ranking can assemble a reasonable
alveoli union without using hidden Dice as an input.
"""

from __future__ import annotations

import ast
import csv
import html
import re
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont


ROOT = Path("/Users/haoranwu/Desktop/Yan_Lab_Research/Medical-SAM3")
BASE = ROOT / "output/visium_hd_exp1/final_deliverables/Jun09_SkillRanker_ComponentAwareMaskSelection"
OUT = BASE / "alveoli_approx_assembly_sanity"
JUN07 = ROOT / "output/visium_hd_exp1/final_deliverables/Jun07_GPT_manual_alveoli_prompt_test_inputs"
HE_FULL = ROOT / "output/visium_hd_exp1/assets/tissue_hires_image.png"
FICTURE_FULL = ROOT / "output/visium_hd_exp1/ficture_official_filtered_he_aligned/filtered_ficture_official_full_he_canvas.png"
TARGET_MASK = ROOT / "output/visium_hd_exp1/sam3_runs/base_sam3_multipoint_remaining_dev/masks/04_lung_alveoli_normal_adjacent_target.png"
ROI_BBOX = (75, 40, 3219, 3367)
ROI_W = ROI_BBOX[2] - ROI_BBOX[0]
ROI_H = ROI_BBOX[3] - ROI_BBOX[1]


def parse_bbox(value: object) -> tuple[int, int, int, int]:
    return tuple(int(v) for v in ast.literal_eval(str(value)))  # type: ignore[return-value]


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    cols: list[str] = []
    for row in rows:
        for key in row:
            if key not in cols:
                cols.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=cols)
        writer.writeheader()
        writer.writerows(rows)


def table(rows: list[dict[str, object]], cols: list[str]) -> str:
    out = ["<table><thead><tr>"]
    out.extend(f"<th>{html.escape(col)}</th>" for col in cols)
    out.append("</tr></thead><tbody>")
    for row in rows:
        out.append("<tr>")
        for col in cols:
            out.append(f"<td>{html.escape(str(row.get(col, '')))}</td>")
        out.append("</tr>")
    out.append("</tbody></table>")
    return "".join(out)


def font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for path in ["/System/Library/Fonts/Supplemental/Arial.ttf", "/System/Library/Fonts/Helvetica.ttc"]:
        try:
            return ImageFont.truetype(path, size)
        except Exception:
            pass
    return ImageFont.load_default()


def metrics(pred: np.ndarray, target: np.ndarray) -> tuple[float, float, float]:
    pred = pred.astype(bool)
    target = target.astype(bool)
    inter = int(np.logical_and(pred, target).sum())
    p = int(pred.sum())
    t = int(target.sum())
    dice = 2 * inter / (p + t) if p + t else 0.0
    precision = inter / p if p else 0.0
    recall = inter / t if t else 0.0
    return dice, precision, recall


def he_crop_path(candidate_id: int) -> Path:
    return JUN07 / "images" / f"alveoli__he__he_official_12445451__base_box384_s128_m1536__{candidate_id:03d}_he_reverse_blur_gray.png"


def reconstruct_from_reverse_blur_crop(row: pd.Series) -> np.ndarray | None:
    """Approximate a candidate mask from its saved H&E reverse-blur crop.

    The VLM crop shows the candidate region in sharper, more saturated H&E and
    the outside as grayscale/blurred.  We rank crop pixels by saturation plus a
    light edge term, then keep the same fill fraction as the original candidate
    metadata.  The result is resized into the candidate bbox on the ROI canvas.
    This is diagnostic only and never replaces the exact mask when available.
    """

    cid = int(row["candidate_id"])
    path = he_crop_path(cid)
    if not path.exists():
        return None

    crop = np.asarray(Image.open(path).convert("RGB")).astype(np.float32) / 255.0
    mx = crop.max(axis=2)
    mn = crop.min(axis=2)
    saturation = (mx - mn) / (mx + 1e-6)
    gray = 0.299 * crop[:, :, 0] + 0.587 * crop[:, :, 1] + 0.114 * crop[:, :, 2]
    gx = np.zeros_like(gray)
    gy = np.zeros_like(gray)
    gx[:, 1:] = np.abs(gray[:, 1:] - gray[:, :-1])
    gy[1:, :] = np.abs(gray[1:, :] - gray[:-1, :])
    gradient = gx + gy
    score = saturation + 0.35 * (gradient - gradient.mean()) / (gradient.std() + 1e-6)

    x0, y0, x1, y1 = parse_bbox(row["bbox"])
    bbox_area = max(1, (x1 - x0) * (y1 - y0))
    fill_ratio = min(0.98, max(0.02, float(row["candidate_pixels"]) / bbox_area))
    k = int(round(fill_ratio * score.size))
    k = min(score.size - 1, max(1, k))
    threshold = np.partition(score.reshape(-1), score.size - k)[score.size - k]
    crop_mask = score >= threshold

    bbox_w = max(1, x1 - x0)
    bbox_h = max(1, y1 - y0)
    resized = Image.fromarray((crop_mask.astype(np.uint8) * 255), mode="L").resize((bbox_w, bbox_h), Image.Resampling.NEAREST)
    roi_mask = np.zeros((ROI_H, ROI_W), dtype=bool)
    roi_mask[y0:y1, x0:x1] = np.asarray(resized) > 0
    return roi_mask


def overlap_small(a: np.ndarray, b: np.ndarray) -> float:
    aa = int(a.sum())
    bb = int(b.sum())
    if not aa or not bb:
        return 0.0
    inter = int(np.logical_and(a, b).sum())
    return inter / min(aa, bb)


def iou(a: np.ndarray, b: np.ndarray) -> float:
    union = int(np.logical_or(a, b).sum())
    if not union:
        return 0.0
    return int(np.logical_and(a, b).sum()) / union


def overlay(roi: Image.Image, mask: np.ndarray, color: tuple[int, int, int], alpha: float = 0.42) -> Image.Image:
    arr = np.asarray(roi.convert("RGB")).astype(np.float32)
    col = np.array(color, dtype=np.float32)
    m = mask.astype(bool)
    arr[m] = (1 - alpha) * arr[m] + alpha * col
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


def mask_only(mask: np.ndarray, color: tuple[int, int, int]) -> Image.Image:
    arr = np.full((mask.shape[0], mask.shape[1], 3), 255, dtype=np.uint8)
    arr[mask.astype(bool)] = np.array(color, dtype=np.uint8)
    return Image.fromarray(arr)


def fit_panel(im: Image.Image, size: tuple[int, int]) -> Image.Image:
    panel = Image.new("RGB", size, "white")
    x = im.copy()
    x.thumbnail((size[0], size[1] - 34), Image.Resampling.LANCZOS)
    panel.paste(x, ((size[0] - x.width) // 2, 30 + (size[1] - 34 - x.height) // 2))
    return panel


def make_six_panel(union_mask: np.ndarray, target_mask: np.ndarray, metrics_row: dict[str, object]) -> Path:
    he_roi = Image.open(HE_FULL).convert("RGB").crop(ROI_BBOX)
    fic_roi = Image.open(FICTURE_FULL).convert("RGB").crop(ROI_BBOX)
    panels = [
        ("Annotation on H&E", overlay(he_roi, target_mask, (22, 163, 74), 0.42)),
        ("Approx selected union on H&E", overlay(he_roi, union_mask, (37, 99, 235), 0.44)),
        ("Approx selected union mask only", mask_only(union_mask, (37, 99, 235))),
        ("Annotation mask only", mask_only(target_mask, (22, 163, 74))),
        ("H&E ROI", he_roi),
        ("FICTURE ROI", fic_roi),
    ]
    panel_w, panel_h = 310, 360
    top_h = 118
    canvas = Image.new("RGB", (panel_w * len(panels) + 70, panel_h + top_h + 40), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((35, 24), "alveoli: target-grounding V2 approximate assembly sanity check", fill=(17, 24, 39), font=font(24))
    draw.text(
        (35, 58),
        f"Selected candidates: {metrics_row['selected candidates']} | Approx Dice {metrics_row['approx Dice']} | Precision {metrics_row['approx Precision']} | Recall {metrics_row['approx Recall']}",
        fill=(31, 41, 55),
        font=font(16),
    )
    draw.text(
        (35, 83),
        "This uses reconstructed masks from saved reverse-blur crops because the exact broad-box binary masks are not fully local.",
        fill=(107, 114, 128),
        font=font(14),
    )
    for idx, (title, im) in enumerate(panels):
        x = 35 + idx * panel_w
        y = top_h
        panel = fit_panel(im, (panel_w - 18, panel_h))
        canvas.paste(panel, (x, y))
        draw.rectangle([x, y, x + panel.width, y + panel.height], outline=(229, 231, 235), width=1)
        draw.text((x + 8, y + 8), title, fill=(31, 41, 55), font=font(14))
    out = OUT / "figures" / "alveoli_v2_approx_union_13_12_six_panel.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out)
    return out


def replace_or_append(html_path: Path, section: str) -> None:
    text = html_path.read_text()
    marker = "<h2>17AR. Alveoli Approximate Assembly Sanity Check</h2>"
    if marker in text:
        start = text.index(marker)
        next_match = re.search(r"<h2>17A[S-Z]|<h2>18\.", text[start + len(marker):])
        end = start + len(marker) + next_match.start() if next_match else text.index("</body>", start)
        text = text[:start] + section + text[end:]
    else:
        text = text.replace("</body>", section + "</body>")
    html_path.write_text(text)


def verify_html_images(html_path: Path) -> None:
    text = html_path.read_text(errors="ignore")
    missing: list[str] = []
    for src in re.findall(r"<img[^>]+src=\"([^\"]+)\"", text):
        if src.startswith(("data:", "http://", "https://")):
            continue
        if not (html_path.parent / src).exists():
            missing.append(src)
    if missing:
        raise RuntimeError(f"Missing {len(missing)} image assets: {missing[:5]}")


def rebuild_zip() -> None:
    import importlib.util

    script = ROOT / "scripts/add_jun09_precise_failure_framework_v5.py"
    spec = importlib.util.spec_from_file_location("pack", script)
    pack = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(pack)
    pack.rebuild_zip()
    zip_path = BASE / "Jun09_SkillRanker_ComponentAwareMaskSelection_shareable.zip"
    with zipfile.ZipFile(zip_path) as handle:
        bad = handle.testzip()
        names = set(handle.namelist())
    if bad:
        raise RuntimeError(f"bad zip entry: {bad}")
    required = [
        "alveoli_approx_assembly_sanity/approx_assembly_summary.csv",
        "alveoli_approx_assembly_sanity/figures/alveoli_v2_approx_union_13_12_six_panel.png",
    ]
    missing = [name for name in required if name not in names]
    if missing:
        raise RuntimeError(f"zip missing required entries: {missing}")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    selector = pd.read_csv(BASE / "alveoli_target_grounding_selector_v2/alveoli_target_grounding_selector_v2_all_scores.csv")
    hidden = pd.read_csv(JUN07 / "tables/hidden_candidate_truth.csv")
    cols = [
        "candidate_id",
        "candidate_pixels",
        "candidate_bbox_xyxy",
        "component_best_dice",
        "component_best_precision",
        "component_best_recall",
    ]
    data = selector.merge(hidden[cols], on="candidate_id", how="left", suffixes=("", "_hidden"))
    if "candidate_bbox_xyxy" in data.columns:
        data["bbox"] = data["candidate_bbox_xyxy"]

    target_full = np.asarray(Image.open(TARGET_MASK).convert("L")) > 0
    x0, y0, x1, y1 = ROI_BBOX
    target = target_full[y0:y1, x0:x1]
    if target.shape != (ROI_H, ROI_W):
        raise RuntimeError(f"Unexpected target ROI shape {target.shape}")

    masks: dict[int, np.ndarray] = {}
    candidate_rows: list[dict[str, object]] = []
    ranked = data.sort_values("score_target_grounding_v2", ascending=False).reset_index(drop=True)
    for rank, row in ranked.iterrows():
        cid = int(row["candidate_id"])
        mask = reconstruct_from_reverse_blur_crop(row)
        crop_available = mask is not None
        approx_dice = approx_precision = approx_recall = ""
        if mask is not None:
            masks[cid] = mask
            d, p, r = metrics(mask, target)
            approx_dice, approx_precision, approx_recall = f"{d:.3f}", f"{p:.3f}", f"{r:.3f}"
        candidate_rows.append(
            {
                "v2 rank": rank + 1,
                "candidate_id": cid,
                "crop available": crop_available,
                "v2 score": f"{float(row['score_target_grounding_v2']):.3f}",
                "approx Dice": approx_dice,
                "approx Precision": approx_precision,
                "approx Recall": approx_recall,
                "hidden Dice": f"{float(row['component_best_dice']):.3f}",
                "hidden Precision": f"{float(row['component_best_precision']):.3f}",
                "hidden Recall": f"{float(row['component_best_recall']):.3f}",
            }
        )
    write_csv(OUT / "approx_candidate_metrics.csv", candidate_rows)

    overlap_rows: list[dict[str, object]] = []
    key_ids = [13, 11, 12, 24, 37]
    for i, a in enumerate(key_ids):
        for b in key_ids[i + 1:]:
            if a not in masks or b not in masks:
                continue
            overlap_rows.append(
                {
                    "candidate_a": a,
                    "candidate_b": b,
                    "IoU": f"{iou(masks[a], masks[b]):.3f}",
                    "overlap_small": f"{overlap_small(masks[a], masks[b]):.3f}",
                }
            )
    write_csv(OUT / "approx_pairwise_overlap.csv", overlap_rows)

    selected: list[int] = []
    skipped: list[str] = []
    for row in ranked.itertuples(index=False):
        cid = int(getattr(row, "candidate_id"))
        if cid not in masks:
            continue
        if len(selected) >= 2:
            break
        max_overlap = max((overlap_small(masks[cid], masks[s]) for s in selected), default=0.0)
        if max_overlap > 0.55:
            skipped.append(f"{cid} skipped overlap_small={max_overlap:.3f}")
            continue
        selected.append(cid)

    def union_for(ids: list[int]) -> np.ndarray:
        out = np.zeros((ROI_H, ROI_W), dtype=bool)
        for cid in ids:
            out |= masks[cid]
        return out

    scenarios = [
        ("V2 conservative NMS, max 2 pieces", selected),
        ("V2 top1 only", [13] if 13 in masks else []),
        ("V2 naive top3 available", [cid for cid in [13, 11, 12] if cid in masks]),
        ("V2 naive top5 available", [cid for cid in [13, 11, 12, 24, 37] if cid in masks]),
    ]
    assembly_rows: list[dict[str, object]] = []
    for name, ids in scenarios:
        union = union_for(ids)
        d, p, r = metrics(union, target)
        assembly_rows.append(
            {
                "assembly rule": name,
                "selected candidates": ",".join(str(x) for x in ids),
                "approx Dice": f"{d:.3f}",
                "approx Precision": f"{p:.3f}",
                "approx Recall": f"{r:.3f}",
                "predicted pixels": int(union.sum()),
                "note": "; ".join(skipped) if name.startswith("V2 conservative") else "",
            }
        )
    assembly_rows.append(
        {
            "assembly rule": "Known exact Jun07 reference, candidate 12 + 13",
            "selected candidates": "12,13",
            "approx Dice": "not approximate",
            "approx Precision": "not approximate",
            "approx Recall": "not approximate",
            "predicted pixels": "",
            "note": "Exact saved broad-box union D/P/R = 0.748 / 0.653 / 0.875; this is the current precise reference, not recomputed from reverse-blur crops.",
        }
    )
    write_csv(OUT / "approx_assembly_summary.csv", assembly_rows)

    fig = make_six_panel(union_for(selected), target, assembly_rows[0])

    top_candidate_rows = [
        row
        for row in candidate_rows
        if int(row["candidate_id"]) in [13, 11, 12, 24, 37, 48, 22, 90]
    ]
    approx_table_rows = []
    for row in top_candidate_rows:
        approx_table_rows.append(
            {
                "v2 rank": row["v2 rank"],
                "candidate": row["candidate_id"],
                "crop available": row["crop available"],
                "v2 score": row["v2 score"],
                "approx D/P/R": f"{row['approx Dice']} / {row['approx Precision']} / {row['approx Recall']}",
                "hidden D/P/R": f"{row['hidden Dice']} / {row['hidden Precision']} / {row['hidden Recall']}",
            }
        )

    rel_fig = fig.relative_to(BASE)
    section = f"""
<h2>17AR. Alveoli Approximate Assembly Sanity Check</h2>
<p><b>Purpose.</b> Section 17AQ showed that the V2 target-grounding score ranks the useful alveoli candidates above the morphology-like false positive. This section asks a stricter question: if we assemble the V2-ranked candidates with a conservative non-overlap rule, do we get a plausible alveoli mask?</p>
<p><b>Important caveat.</b> This is an approximate sanity check, not the final exact oracle. The exact broad-box binary masks are still remote, so I reconstruct candidate masks from the saved H&amp;E gray reverse-blur crops. The reconstruction uses the visible sharp/saturated candidate region plus the known candidate fill ratio. Hidden Dice/Precision/Recall are used only after reconstruction to check whether the approximation is reasonable.</p>
<h3>Candidate-level approximation check</h3>
{table(approx_table_rows, ['v2 rank', 'candidate', 'crop available', 'v2 score', 'approx D/P/R', 'hidden D/P/R'])}
<h3>Overlap check among top useful candidates</h3>
{table(overlap_rows, ['candidate_a', 'candidate_b', 'IoU', 'overlap_small'])}
<h3>Assembly comparison</h3>
{table(assembly_rows, ['assembly rule', 'selected candidates', 'approx Dice', 'approx Precision', 'approx Recall', 'predicted pixels', 'note'])}
<figure><img src="{rel_fig}" alt="Alveoli V2 approximate selected union six-panel" />
<figcaption>Six-panel view for the conservative V2 assembly. The rule selects candidate 13, skips candidate 11 because it overlaps candidate 13 too much, then selects candidate 12. This gives a high-recall approximate union while avoiding the precision collapse seen when naively adding top 3 or top 5 pieces.</figcaption></figure>
<div class='callout'><b>Decision from this check.</b> The failure is no longer simply \"alveoli morphology is impossible.\" The more precise diagnosis is: H&amp;E morphology alone can find alveoli-like regions but needs target grounding and conservative assembly. V2 fixes the candidate-48 false positive and a max-2 overlap-aware assembly is the next deployable candidate, but exact remote masks are still needed before calling the metric final.</div>
"""
    for name in ["Jun09_SkillRanker_ComponentAwareMaskSelection.html", "index.html"]:
        replace_or_append(BASE / name, section)
        verify_html_images(BASE / name)
    rebuild_zip()
    print(OUT / "approx_assembly_summary.csv")
    print(fig)
    print(BASE / "Jun09_SkillRanker_ComponentAwareMaskSelection.html")


if __name__ == "__main__":
    main()
