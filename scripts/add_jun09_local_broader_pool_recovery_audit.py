#!/usr/bin/env python3
"""Add a local broader-pool recovery audit to the Jun09 report.

The current Jun09 funnel/locator pack has only the compact piece set.  This
diagnostic checks whether older local H&E proposal settings contain pieces for
annotation components that the compact funnel misses.  Masks are resized to the
official ROI for comparison, so this is a recovery diagnostic, not a final
official metric.
"""

from __future__ import annotations

import csv
import html
import re
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage


ROOT = Path("/Users/haoranwu/Desktop/Yan_Lab_Research/Medical-SAM3")
BASE = ROOT / "output/visium_hd_exp1/final_deliverables/Jun09_SkillRanker_ComponentAwareMaskSelection"
OUT = BASE / "local_broader_pool_recovery_audit"
ROI_ROOT = ROOT / "output/visium_hd_exp1/ficture_official_filtered_candidate_pool_input_roi"

HE_ROI = ROI_ROOT / "he_roi_matching_official_ficture_coverage.png"
FICTURE_ROI = ROI_ROOT / "ficture_official_filtered_roi_rgb.png"

SOURCE_ROOT = ROOT / "output/visium_hd_exp1/best_single_candidate_vs_annotation/source_candidate_runs"
SOURCE_DIRS = [
    SOURCE_ROOT / "base_large_boxes_b512_s128/candidate_masks",
    SOURCE_ROOT / "base_large_boxes_b768_s256/candidate_masks",
    SOURCE_ROOT / "base_points_step32_cap1800/candidate_masks",
    SOURCE_ROOT / "dense_boxes_1536_b256_s128/candidate_masks",
    SOURCE_ROOT / "medical_points_step40/candidate_masks",
    SOURCE_ROOT / "medium_boxes_b192_s64/candidate_masks",
]

CLASSES = {
    "bronchiola": ROI_ROOT / "cropped_annotation_masks/01_lung_bronchiola_target_roi.png",
    "alveoli": ROI_ROOT / "cropped_annotation_masks/04_lung_alveoli_normal_adjacent_target_roi.png",
    "vessels": ROI_ROOT / "cropped_annotation_masks/05_lung_vessels_target_roi.png",
    "tumor": ROI_ROOT / "cropped_annotation_masks/08_tumor_target_roi.png",
    "stroma": ROI_ROOT / "cropped_annotation_masks/07_stroma_target_roi.png",
    "immune_infiltration": ROI_ROOT / "cropped_annotation_masks/03_immune_infiltration_target_roi.png",
}

MIN_COMPONENT_PIXELS = 200
SUPPORT_DICE = 0.20
FORCE_FIGURES = {
    "bronchiola": {3},
    "vessels": {1, 4},
}


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


def load_mask(path: Path, shape: tuple[int, int] | None = None) -> np.ndarray:
    im = Image.open(path).convert("L")
    if shape and (im.height, im.width) != shape:
        im = im.resize((shape[1], shape[0]), Image.Resampling.NEAREST)
    return np.asarray(im) > 0


def metrics_vs_components(labels: np.ndarray, component_areas: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    cand_area = int(mask.sum())
    overlaps = np.bincount(labels[mask].reshape(-1), minlength=len(component_areas)) if cand_area else np.zeros(len(component_areas))
    dice = np.zeros(len(component_areas), dtype=float)
    precision = np.zeros(len(component_areas), dtype=float)
    recall = np.zeros(len(component_areas), dtype=float)
    valid = component_areas > 0
    if cand_area:
        dice[valid] = 2 * overlaps[valid] / (cand_area + component_areas[valid])
        precision[valid] = overlaps[valid] / cand_area
        recall[valid] = overlaps[valid] / component_areas[valid]
    return dice, precision, recall


def overlay(roi: Image.Image, mask: np.ndarray, color: tuple[int, int, int], alpha: float = 0.42) -> Image.Image:
    arr = np.asarray(roi.convert("RGB")).astype(np.float32)
    m = mask.astype(bool)
    arr[m] = (1 - alpha) * arr[m] + alpha * np.array(color, dtype=np.float32)
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


def mask_only(mask: np.ndarray, color: tuple[int, int, int]) -> Image.Image:
    arr = np.full((mask.shape[0], mask.shape[1], 3), 255, dtype=np.uint8)
    arr[mask.astype(bool)] = np.array(color, dtype=np.uint8)
    return Image.fromarray(arr)


def fit_panel(im: Image.Image, size: tuple[int, int]) -> Image.Image:
    panel = Image.new("RGB", size, "white")
    x = im.copy()
    x.thumbnail((size[0] - 10, size[1] - 42), Image.Resampling.LANCZOS)
    panel.paste(x, ((size[0] - x.width) // 2, 36 + (size[1] - 42 - x.height) // 2))
    return panel


def make_component_mask(labels: np.ndarray, component_id: int) -> np.ndarray:
    return labels == component_id


def make_six_panel(row: dict[str, object], pred: np.ndarray, target_component: np.ndarray) -> str:
    he = Image.open(HE_ROI).convert("RGB")
    fic = Image.open(FICTURE_ROI).convert("RGB")
    cls = str(row["class"])
    cid = int(row["component_id"])
    panels = [
        ("Target component on H&E", overlay(he, target_component, (22, 163, 74), 0.42)),
        ("Best local candidate on H&E", overlay(he, pred, (37, 99, 235), 0.44)),
        ("Best local candidate mask only", mask_only(pred, (37, 99, 235))),
        ("Target component mask only", mask_only(target_component, (22, 163, 74))),
        ("H&E ROI", he),
        ("FICTURE ROI", fic),
    ]
    panel_w, panel_h = 310, 360
    top_h = 132
    canvas = Image.new("RGB", (panel_w * len(panels) + 70, panel_h + top_h + 40), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((35, 24), f"{cls} component {cid}: local broader-pool recovery diagnostic", fill=(17, 24, 39), font=font(24))
    draw.text(
        (35, 58),
        f"Best source: {row['best_source_setting']} | Dice {row['best_Dice']} | Precision {row['best_Precision']} | Recall {row['best_Recall']}",
        fill=(31, 41, 55),
        font=font(16),
    )
    draw.text(
        (35, 84),
        "Diagnostic only: mask was resized from an older local proposal setting; use this to decide whether to restore/expand proposals.",
        fill=(107, 114, 128),
        font=font(14),
    )
    for idx, (name, im) in enumerate(panels):
        x = 35 + idx * panel_w
        y = top_h
        panel = fit_panel(im, (panel_w - 18, panel_h))
        canvas.paste(panel, (x, y))
        draw.rectangle([x, y, x + panel.width, y + panel.height], outline=(229, 231, 235), width=1)
        draw.text((x + 8, y + 8), name, fill=(31, 41, 55), font=font(14))
    out = OUT / "figures" / f"{cls}_component_{cid}_local_broader_pool_recovery.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out)
    return str(out.relative_to(BASE))


def replace_or_append(html_path: Path, section: str) -> None:
    text = html_path.read_text()
    marker = "<h2>17AV. Local Broader-Pool Recovery Audit</h2>"
    if marker in text:
        start = text.index(marker)
        next_match = re.search(r"<h2>17A[W-Z]|<h2>18\\.", text[start + len(marker) :])
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
        raise RuntimeError(f"Missing {len(missing)} image assets: {missing[:8]}")


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
        "local_broader_pool_recovery_audit/local_broader_pool_recovery_summary.csv",
        "local_broader_pool_recovery_audit/local_broader_pool_recovery_details.csv",
        "local_broader_pool_recovery_audit/source_inventory.csv",
    ]
    missing = [name for name in required if name not in names]
    if missing:
        raise RuntimeError(f"zip missing required local recovery files: {missing}")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    source_rows: list[dict[str, object]] = []
    mask_entries: list[tuple[str, Path]] = []
    for source_dir in SOURCE_DIRS:
        files = sorted(source_dir.glob("*.png"))
        if not files:
            continue
        sample_size = ""
        try:
            im = Image.open(files[0])
            sample_size = f"{im.width}x{im.height}"
        except Exception:
            sample_size = "unreadable"
        setting = source_dir.parent.name
        source_rows.append(
            {
                "source_setting": setting,
                "mask_count": len(files),
                "native_mask_size": sample_size,
                "path": str(source_dir.relative_to(ROOT)),
                "status": "included_as_resized_local_diagnostic",
            }
        )
        for path in files:
            mask_entries.append((setting, path))
    write_csv(OUT / "source_inventory.csv", source_rows)

    detail_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    figure_rows: list[dict[str, object]] = []

    for cls, ann_path in CLASSES.items():
        ann = load_mask(ann_path)
        labels, n_comp = ndimage.label(ann, structure=ndimage.generate_binary_structure(2, 1))
        component_areas = np.bincount(labels.reshape(-1), minlength=n_comp + 1).astype(float)
        component_ids = [cid for cid in range(1, n_comp + 1) if component_areas[cid] >= MIN_COMPONENT_PIXELS]
        best: dict[int, dict[str, object]] = {
            cid: {
                "class": cls,
                "component_id": cid,
                "component_pixels": int(component_areas[cid]),
                "best_source_setting": "",
                "best_mask_file": "",
                "best_Dice": 0.0,
                "best_Precision": 0.0,
                "best_Recall": 0.0,
                "has_supported_local_broader_candidate": False,
            }
            for cid in component_ids
        }
        best_mask_by_component: dict[int, np.ndarray] = {}

        for setting, path in mask_entries:
            mask = load_mask(path, ann.shape)
            dice, precision, recall = metrics_vs_components(labels, component_areas, mask)
            for cid in component_ids:
                if dice[cid] > float(best[cid]["best_Dice"]):
                    best[cid].update(
                        {
                            "best_source_setting": setting,
                            "best_mask_file": path.name,
                            "best_Dice": round(float(dice[cid]), 4),
                            "best_Precision": round(float(precision[cid]), 4),
                            "best_Recall": round(float(recall[cid]), 4),
                            "has_supported_local_broader_candidate": bool(dice[cid] >= SUPPORT_DICE),
                        }
                    )
                    best_mask_by_component[cid] = mask

        rows = list(best.values())
        detail_rows.extend(rows)
        supported = [r for r in rows if r["has_supported_local_broader_candidate"]]
        summary_rows.append(
            {
                "class": cls,
                "components_area_ge_200": len(rows),
                "components_supported_by_local_broader_pool": len(supported),
                "support_rate": f"{len(supported)}/{len(rows)}" if rows else "0/0",
                "best_component_Dice_max": round(max(float(r["best_Dice"]) for r in rows), 4) if rows else 0.0,
                "median_best_component_Dice": round(float(np.median([float(r["best_Dice"]) for r in rows])), 4) if rows else 0.0,
                "top_source_settings": ", ".join(
                    f"{k}:{v}"
                    for k, v in sorted(
                        {
                            str(r["best_source_setting"]): sum(1 for rr in supported if rr["best_source_setting"] == r["best_source_setting"])
                            for r in supported
                        }.items(),
                        key=lambda kv: (-kv[1], kv[0]),
                    )
                    if k
                ),
                "interpretation": (
                    "broader local proposals can recover at least some missed components"
                    if supported
                    else "no evidence of recovery in these local proposal settings"
                ),
            }
        )

        # Figure the largest supported components plus known gap components that
        # explain the current funnel failure.
        figure_candidates: dict[int, dict[str, object]] = {}
        for row in sorted(supported, key=lambda r: int(r["component_pixels"]), reverse=True)[:2]:
            figure_candidates[int(row["component_id"])] = row
        for row in rows:
            cid = int(row["component_id"])
            if cid in FORCE_FIGURES.get(cls, set()):
                figure_candidates[cid] = row
        for row in figure_candidates.values():
            cid = int(row["component_id"])
            if cid in best_mask_by_component:
                rel = make_six_panel(row, best_mask_by_component[cid], make_component_mask(labels, cid))
                figure_rows.append({"class": cls, "component_id": cid, "figure": rel, "caption": f"best Dice {row['best_Dice']} from {row['best_source_setting']}"})

    write_csv(OUT / "local_broader_pool_recovery_details.csv", detail_rows)
    write_csv(OUT / "local_broader_pool_recovery_summary.csv", summary_rows)

    focus = sorted(
        [r for r in detail_rows if r["has_supported_local_broader_candidate"]],
        key=lambda r: (str(r["class"]), -int(r["component_pixels"])),
    )[:80]

    section = f"""
<h2>17AV. Local Broader-Pool Recovery Audit</h2>
<p><b>Goal.</b> The component gap audit showed that the compact funnel misses some annotation components.  This section asks whether older local H&amp;E proposal settings contain pieces for those missed components.  If yes, the failure is probably the funnel/setting choice.  If no, we need a new proposal generator.</p>
<p><b>Important caveat.</b> These masks are local older proposal outputs at downsampled ROI sizes and are resized to the official 3144 x 3327 ROI.  They are useful as a recovery diagnostic, but they are not a final official medpt24 result.</p>
<h3>Candidate sources used</h3>
{table(source_rows, ['source_setting', 'mask_count', 'native_mask_size', 'status', 'path'])}
<h3>Recovery summary by class</h3>
{table(summary_rows, ['class', 'components_area_ge_200', 'components_supported_by_local_broader_pool', 'support_rate', 'best_component_Dice_max', 'median_best_component_Dice', 'top_source_settings', 'interpretation'])}
<h3>Supported components found in the broader local pool</h3>
{table(focus, ['class', 'component_id', 'component_pixels', 'best_source_setting', 'best_mask_file', 'best_Dice', 'best_Precision', 'best_Recall'])}
<h3>Largest recovery examples</h3>
<div class="grid">
{''.join(f'<figure><img src="{html.escape(str(row["figure"]))}" alt="{html.escape(str(row["class"]))} component {html.escape(str(row["component_id"]))} recovery"><figcaption>{html.escape(str(row["class"]))} component {html.escape(str(row["component_id"]))}: {html.escape(str(row["caption"]))}</figcaption></figure>' for row in figure_rows)}
</div>
<p><b>Iteration rule.</b> If a missed component is recovered here, the next real step is to rebuild the compact funnel from the better setting or copy the corresponding full-resolution masks from Bouchet.  If it is still not recovered, the proposal generator itself must change.</p>
"""

    for html_path in [BASE / "Jun09_SkillRanker_ComponentAwareMaskSelection.html", BASE / "index.html"]:
        replace_or_append(html_path, section)
        verify_html_images(html_path)
    rebuild_zip()
    print(f"Wrote {OUT / 'local_broader_pool_recovery_summary.csv'}")
    print(f"Updated {BASE / 'Jun09_SkillRanker_ComponentAwareMaskSelection.html'}")


if __name__ == "__main__":
    main()
