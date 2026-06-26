#!/usr/bin/env python3
"""Append a recovered-proposal hybrid prototype to the Jun09 report.

The local broader-pool audit found proposals that recover missed components.
This script tests whether directly unioning those recovered masks is useful, or
whether they should only be used as prompts/locators for a real second-SAM pass.
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
OUT = BASE / "recovered_proposal_hybrid_prototype"
ROI_ROOT = ROOT / "output/visium_hd_exp1/ficture_official_filtered_candidate_pool_input_roi"
PACK = BASE / "runtime_policy_second_sam_locator_pack/masks"

HE_ROI = ROI_ROOT / "he_roi_matching_official_ficture_coverage.png"
FICTURE_ROI = ROI_ROOT / "ficture_official_filtered_roi_rgb.png"

CLASSES = {
    "bronchiola": {
        "annotation": ROI_ROOT / "cropped_annotation_masks/01_lung_bronchiola_target_roi.png",
        "current": PACK / "bronchiola_selected_piece_union.png",
        "recovered": [
            (
                "medium_boxes_b192_s64 candidate_142",
                ROOT / "output/visium_hd_exp1/best_single_candidate_vs_annotation/source_candidate_runs/medium_boxes_b192_s64/candidate_masks/candidate_142.png",
                "recovers missing bottom bronchiola component",
            )
        ],
    },
    "vessels": {
        "annotation": ROI_ROOT / "cropped_annotation_masks/05_lung_vessels_target_roi.png",
        "current": PACK / "vessels_selected_piece_union.png",
        "recovered": [
            (
                "medium_boxes_b192_s64 candidate_116",
                ROOT / "output/visium_hd_exp1/best_single_candidate_vs_annotation/source_candidate_runs/medium_boxes_b192_s64/candidate_masks/candidate_116.png",
                "recovers vessel component 4 with high recall but low precision",
            )
        ],
    },
    "alveoli": {
        "annotation": ROI_ROOT / "cropped_annotation_masks/04_lung_alveoli_normal_adjacent_target_roi.png",
        "current": PACK / "alveoli_selected_piece_union.png",
        "recovered": [
            (
                "dense_boxes_1536_b256_s128 candidate_025",
                ROOT / "output/visium_hd_exp1/best_single_candidate_vs_annotation/source_candidate_runs/dense_boxes_1536_b256_s128/candidate_masks/candidate_025.png",
                "best local broader-pool diagnostic candidate, included only as a negative control",
            )
        ],
    },
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


def metrics(pred: np.ndarray, target: np.ndarray) -> tuple[float, float, float, int]:
    pred = pred.astype(bool)
    target = target.astype(bool)
    inter = int(np.logical_and(pred, target).sum())
    p = int(pred.sum())
    t = int(target.sum())
    dice = 2 * inter / (p + t) if p + t else 0.0
    precision = inter / p if p else 0.0
    recall = inter / t if t else 0.0
    return dice, precision, recall, p


def refine(mask: np.ndarray, op: str, radius: int) -> np.ndarray:
    st = ndimage.generate_binary_structure(2, 1)
    if op == "identity":
        return mask.copy()
    if op == "dilate":
        return ndimage.binary_dilation(mask, structure=st, iterations=radius)
    if op == "close":
        return ndimage.binary_closing(mask, structure=st, iterations=radius)
    if op == "close_then_dilate":
        closed = ndimage.binary_closing(mask, structure=st, iterations=radius)
        return ndimage.binary_dilation(closed, structure=st, iterations=max(1, radius // 2))
    raise ValueError(op)


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


def make_six_panel(label: str, pred: np.ndarray, target: np.ndarray, row: dict[str, object]) -> str:
    he = Image.open(HE_ROI).convert("RGB")
    fic = Image.open(FICTURE_ROI).convert("RGB")
    panels = [
        ("Annotation on H&E", overlay(he, target, (22, 163, 74), 0.42)),
        ("Hybrid prototype on H&E", overlay(he, pred, (37, 99, 235), 0.44)),
        ("Hybrid prototype mask only", mask_only(pred, (37, 99, 235))),
        ("Annotation mask only", mask_only(target, (22, 163, 74))),
        ("H&E ROI", he),
        ("FICTURE ROI", fic),
    ]
    panel_w, panel_h = 310, 360
    top_h = 132
    canvas = Image.new("RGB", (panel_w * len(panels) + 70, panel_h + top_h + 40), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((35, 24), f"{label}: recovered-proposal hybrid prototype", fill=(17, 24, 39), font=font(24))
    draw.text(
        (35, 58),
        f"{row['best variant']} | Dice {row['Dice']} | Precision {row['Precision']} | Recall {row['Recall']}",
        fill=(31, 41, 55),
        font=font(16),
    )
    draw.text((35, 84), str(row["interpretation"]), fill=(107, 114, 128), font=font(14))
    for idx, (name, im) in enumerate(panels):
        x = 35 + idx * panel_w
        y = top_h
        panel = fit_panel(im, (panel_w - 18, panel_h))
        canvas.paste(panel, (x, y))
        draw.rectangle([x, y, x + panel.width, y + panel.height], outline=(229, 231, 235), width=1)
        draw.text((x + 8, y + 8), name, fill=(31, 41, 55), font=font(14))
    out = OUT / "figures" / f"{label}_recovered_hybrid_prototype_six_panel.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out)
    return str(out.relative_to(BASE))


def replace_or_append(html_path: Path, section: str) -> None:
    text = html_path.read_text()
    marker = "<h2>17AX. Recovered Proposal Hybrid Prototype</h2>"
    if marker in text:
        start = text.index(marker)
        next_match = re.search(r"<h2>17A[Y-Z]|<h2>18\\.", text[start + len(marker) :])
        end = start + len(marker) + next_match.start() if next_match else text.index("</body>", start)
        text = text[:start] + section + text[end:]
    else:
        text = text.replace("</body>", section + "</body>")
    html_path.write_text(text)


def verify_html_images(html_path: Path) -> None:
    text = html_path.read_text(errors="ignore")
    missing = []
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
        "recovered_proposal_hybrid_prototype/recovered_hybrid_metrics.csv",
        "recovered_proposal_hybrid_prototype/recovered_hybrid_decision.csv",
        "recovered_proposal_hybrid_prototype/figures/bronchiola_recovered_hybrid_prototype_six_panel.png",
    ]
    missing = [name for name in required if name not in names]
    if missing:
        raise RuntimeError(f"zip missing required recovered hybrid files: {missing}")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    metric_rows: list[dict[str, object]] = []
    decision_rows: list[dict[str, object]] = []
    figure_rows: list[dict[str, object]] = []

    for label, cfg in CLASSES.items():
        target = load_mask(cfg["annotation"])
        current = load_mask(cfg["current"], target.shape)
        d, p, r, pix = metrics(current, target)
        metric_rows.append(
            {
                "class": label,
                "variant": "current selected union",
                "added recovered proposal": "",
                "Dice": round(d, 4),
                "Precision": round(p, 4),
                "Recall": round(r, 4),
                "pred_pixels": pix,
            }
        )

        hybrid = current.copy()
        recovered_names = []
        notes = []
        for name, path, note in cfg["recovered"]:
            hybrid |= load_mask(path, target.shape)
            recovered_names.append(name)
            notes.append(note)
        d, p, r, pix = metrics(hybrid, target)
        metric_rows.append(
            {
                "class": label,
                "variant": "direct union with recovered proposal",
                "added recovered proposal": "; ".join(recovered_names),
                "Dice": round(d, 4),
                "Precision": round(p, 4),
                "Recall": round(r, 4),
                "pred_pixels": pix,
            }
        )

        best_op = ("identity", 0)
        best_mask = hybrid
        best_vals = metrics(hybrid, target)
        for op in ["dilate", "close", "close_then_dilate"]:
            for radius in [2, 4, 8, 12]:
                pred = refine(hybrid, op, radius)
                vals = metrics(pred, target)
                if vals[0] > best_vals[0]:
                    best_vals = vals
                    best_op = (op, radius)
                    best_mask = pred
        d, p, r, pix = best_vals
        metric_rows.append(
            {
                "class": label,
                "variant": f"hybrid plus local boundary proxy {best_op[0]} r={best_op[1]}",
                "added recovered proposal": "; ".join(recovered_names),
                "Dice": round(d, 4),
                "Precision": round(p, 4),
                "Recall": round(r, 4),
                "pred_pixels": pix,
            }
        )

        current_vals = [row for row in metric_rows if row["class"] == label and row["variant"] == "current selected union"][0]
        direct_vals = [row for row in metric_rows if row["class"] == label and row["variant"] == "direct union with recovered proposal"][0]
        best_variant = f"hybrid + {best_op[0]} r={best_op[1]}"
        if label == "bronchiola":
            interp = "Recovered proposal improves recall but direct union drops precision; use it as a second-SAM prompt, not a final piece."
        elif label == "vessels":
            interp = "Recovered proposal improves recall but direct union lowers Dice; use it as a locator/prompt with precision guard."
        else:
            interp = "This recovered local proposal is a negative-control branch; it does not rescue the medpt24 alveoli path."
        decision = {
            "class": label,
            "current D/P/R": f"{current_vals['Dice']} / {current_vals['Precision']} / {current_vals['Recall']}",
            "direct hybrid D/P/R": f"{direct_vals['Dice']} / {direct_vals['Precision']} / {direct_vals['Recall']}",
            "best proxy D/P/R": f"{round(d, 4)} / {round(p, 4)} / {round(r, 4)}",
            "best variant": best_variant,
            "decision": "use recovered proposal as prompt/locator, not final mask",
            "interpretation": interp,
        }
        fig_row = {
            "best variant": best_variant,
            "Dice": round(d, 4),
            "Precision": round(p, 4),
            "Recall": round(r, 4),
            "interpretation": interp,
        }
        rel = make_six_panel(label, best_mask, target, fig_row)
        decision["six panel figure"] = rel
        decision_rows.append(decision)
        figure_rows.append({"class": label, "figure": rel, "caption": interp})

    write_csv(OUT / "recovered_hybrid_metrics.csv", metric_rows)
    write_csv(OUT / "recovered_hybrid_decision.csv", decision_rows)

    section = f"""
<h2>17AX. Recovered Proposal Hybrid Prototype</h2>
<p><b>Goal.</b> The previous audit found that older local proposal settings can recover some missing bronchiola and vessel components.  This section tests whether those recovered masks can be directly unioned into the final segmentation, or whether they should only be used as prompts for a second SAM pass.</p>
<p><b>Result in plain language.</b> Direct union is not good enough.  It improves recall for bronchiola/vessels, but drops precision enough that the recovered masks should be treated as <b>locators/prompts</b>, not final masks.</p>
{table(decision_rows, ['class', 'current D/P/R', 'direct hybrid D/P/R', 'best proxy D/P/R', 'best variant', 'decision', 'interpretation'])}
<h3>All prototype variants</h3>
{table(metric_rows, ['class', 'variant', 'added recovered proposal', 'Dice', 'Precision', 'Recall', 'pred_pixels'])}
<h3>Six-panel prototype figures</h3>
<div class="grid">
{''.join(f'<figure><img src="{html.escape(str(row["figure"]))}" alt="{html.escape(str(row["class"]))} recovered hybrid"><figcaption>{html.escape(str(row["class"]))}: {html.escape(str(row["caption"]))}</figcaption></figure>' for row in figure_rows)}
</div>
<p><b>Next iteration.</b> Use medium-box recovered proposals to seed true second-SAM / local refinement for bronchiola and vessels.  Do not directly add these boxy recovered masks as final pieces.</p>
"""

    for html_path in [BASE / "Jun09_SkillRanker_ComponentAwareMaskSelection.html", BASE / "index.html"]:
        replace_or_append(html_path, section)
        verify_html_images(html_path)
    rebuild_zip()
    print(f"Wrote {OUT / 'recovered_hybrid_decision.csv'}")


if __name__ == "__main__":
    main()
