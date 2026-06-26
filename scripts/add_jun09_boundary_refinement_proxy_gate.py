#!/usr/bin/env python3
"""Add a boundary-refinement proxy gate to the Jun09 skill-ranker report.

This diagnostic asks a narrow question before launching another expensive SAM
run: if the already selected piece union is slightly expanded/closed, does the
class improve in Dice/Precision/Recall?  If yes, a real second-SAM refinement is
worth testing.  If no, the failure is more likely proposal/selection than mask
boundary refinement.
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
OUT = BASE / "boundary_refinement_proxy_gate"
ROI_ROOT = ROOT / "output/visium_hd_exp1/ficture_official_filtered_candidate_pool_input_roi"
LOCATOR = BASE / "runtime_policy_second_sam_locator_pack"

HE_ROI = ROI_ROOT / "he_roi_matching_official_ficture_coverage.png"
FICTURE_ROI = ROI_ROOT / "ficture_official_filtered_roi_rgb.png"

CLASSES = {
    "bronchiola": {
        "annotation": ROI_ROOT / "cropped_annotation_masks/01_lung_bronchiola_target_roi.png",
        "selected": LOCATOR / "masks/bronchiola_selected_piece_union.png",
    },
    "alveoli": {
        "annotation": ROI_ROOT / "cropped_annotation_masks/04_lung_alveoli_normal_adjacent_target_roi.png",
        "selected": LOCATOR / "masks/alveoli_selected_piece_union.png",
    },
    "vessels": {
        "annotation": ROI_ROOT / "cropped_annotation_masks/05_lung_vessels_target_roi.png",
        "selected": LOCATOR / "masks/vessels_selected_piece_union.png",
    },
    "tumor": {
        "annotation": ROI_ROOT / "cropped_annotation_masks/08_tumor_target_roi.png",
        "selected": LOCATOR / "masks/tumor_selected_piece_union.png",
    },
    "stroma": {
        "annotation": ROI_ROOT / "cropped_annotation_masks/07_stroma_target_roi.png",
        "selected": LOCATOR / "masks/stroma_selected_piece_union.png",
    },
    "immune_infiltration": {
        "annotation": ROI_ROOT / "cropped_annotation_masks/03_immune_infiltration_target_roi.png",
        "selected": LOCATOR / "masks/immune_infiltration_selected_piece_union.png",
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
    if not path.exists():
        raise FileNotFoundError(path)
    im = Image.open(path).convert("L")
    if shape and (im.height, im.width) != shape:
        im = im.resize((shape[1], shape[0]), Image.Resampling.NEAREST)
    return np.asarray(im) > 0


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


def structure() -> np.ndarray:
    return ndimage.generate_binary_structure(2, 1)


def refine(mask: np.ndarray, op: str, radius: int) -> np.ndarray:
    st = structure()
    if op == "identity":
        return mask.copy()
    if op == "dilate":
        return ndimage.binary_dilation(mask, structure=st, iterations=radius)
    if op == "erode":
        return ndimage.binary_erosion(mask, structure=st, iterations=radius)
    if op == "close":
        return ndimage.binary_closing(mask, structure=st, iterations=radius)
    if op == "close_then_dilate":
        closed = ndimage.binary_closing(mask, structure=st, iterations=radius)
        return ndimage.binary_dilation(closed, structure=st, iterations=max(1, radius // 2))
    raise ValueError(op)


def overlay(roi: Image.Image, mask: np.ndarray, color: tuple[int, int, int], alpha: float = 0.42) -> Image.Image:
    arr = np.asarray(roi.convert("RGB")).astype(np.float32)
    m = mask.astype(bool)
    col = np.array(color, dtype=np.float32)
    arr[m] = (1 - alpha) * arr[m] + alpha * col
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
        ("Refined union on H&E", overlay(he, pred, (37, 99, 235), 0.44)),
        ("Refined union mask only", mask_only(pred, (37, 99, 235))),
        ("Annotation mask only", mask_only(target, (22, 163, 74))),
        ("H&E ROI", he),
        ("FICTURE ROI", fic),
    ]
    panel_w, panel_h = 310, 360
    top_h = 128
    canvas = Image.new("RGB", (panel_w * len(panels) + 70, panel_h + top_h + 40), "white")
    draw = ImageDraw.Draw(canvas)
    title = f"{label}: boundary-refinement proxy gate"
    draw.text((35, 24), title, fill=(17, 24, 39), font=font(24))
    draw.text(
        (35, 58),
        f"Best proxy op: {row['best proxy op']} | Dice {row['best Dice']} | Precision {row['best Precision']} | Recall {row['best Recall']}",
        fill=(31, 41, 55),
        font=font(16),
    )
    draw.text(
        (35, 84),
        "Diagnostic only: this is a local morphology proxy to decide whether real second-SAM boundary refinement is worth running.",
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
    out = OUT / "figures" / f"{label}_boundary_refinement_proxy_six_panel.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out)
    return str(out.relative_to(BASE))


def replace_or_append(html_path: Path, section: str) -> None:
    text = html_path.read_text()
    marker = "<h2>17AT. Boundary Refinement Proxy Gate</h2>"
    if marker in text:
        start = text.index(marker)
        next_match = re.search(r"<h2>17A[U-Z]|<h2>18\\.", text[start + len(marker) :])
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
        "boundary_refinement_proxy_gate/boundary_refinement_proxy_summary.csv",
        "boundary_refinement_proxy_gate/boundary_refinement_proxy_all_ops.csv",
        "boundary_refinement_proxy_gate/next_iteration_after_boundary_proxy.csv",
        "boundary_refinement_proxy_gate/figures/bronchiola_boundary_refinement_proxy_six_panel.png",
        "boundary_refinement_proxy_gate/figures/vessels_boundary_refinement_proxy_six_panel.png",
    ]
    missing = [name for name in required if name not in names]
    if missing:
        raise RuntimeError(f"zip missing required boundary proxy files: {missing}")


def classify_signal(base: dict[str, float], best: dict[str, float]) -> str:
    dice_gain = best["dice"] - base["dice"]
    precision_drop = base["precision"] - best["precision"]
    recall_gain = best["recall"] - base["recall"]
    if dice_gain >= 0.03 and precision_drop <= 0.10:
        return "yes: boundary refinement likely worth real second-SAM"
    if dice_gain >= 0.02 and recall_gain >= 0.04 and precision_drop <= 0.10:
        return "maybe: boundary refinement has signal; verify with real second-SAM"
    if recall_gain >= 0.08 and precision_drop <= 0.15 and best["dice"] >= base["dice"] - 0.01:
        return "maybe: recall can improve, but must verify with real SAM"
    return "no: boundary proxy does not fix the earliest failure"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    if not HE_ROI.exists() or not FICTURE_ROI.exists():
        raise FileNotFoundError("official H&E/FICTURE ROI images are missing")

    operations = [("identity", 0)]
    for r in [2, 4, 8, 12]:
        operations.extend([("dilate", r), ("close", r), ("close_then_dilate", r)])

    all_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    figure_rows: list[dict[str, object]] = []
    action_rows: list[dict[str, object]] = []

    for label, paths in CLASSES.items():
        target = load_mask(paths["annotation"])
        selected = load_mask(paths["selected"], target.shape)
        base_d, base_p, base_r = metrics(selected, target)
        base_values = {"dice": base_d, "precision": base_p, "recall": base_r}
        best_row: dict[str, object] | None = None
        best_mask = selected

        for op, radius in operations:
            pred = refine(selected, op, radius)
            d, p, r = metrics(pred, target)
            row = {
                "class": label,
                "operation": op,
                "radius_px": radius,
                "Dice": round(d, 4),
                "Precision": round(p, 4),
                "Recall": round(r, 4),
                "delta_Dice_vs_identity": round(d - base_d, 4),
                "delta_Precision_vs_identity": round(p - base_p, 4),
                "delta_Recall_vs_identity": round(r - base_r, 4),
                "pred_pixels": int(pred.sum()),
                "annotation_pixels": int(target.sum()),
            }
            all_rows.append(row)
            if best_row is None or float(row["Dice"]) > float(best_row["Dice"]):
                best_row = row
                best_mask = pred

        assert best_row is not None
        best_values = {
            "dice": float(best_row["Dice"]),
            "precision": float(best_row["Precision"]),
            "recall": float(best_row["Recall"]),
        }
        signal = classify_signal(base_values, best_values)
        summary = {
            "class": label,
            "identity Dice": round(base_d, 4),
            "identity Precision": round(base_p, 4),
            "identity Recall": round(base_r, 4),
            "best proxy op": f"{best_row['operation']} r={best_row['radius_px']}",
            "best Dice": best_row["Dice"],
            "best Precision": best_row["Precision"],
            "best Recall": best_row["Recall"],
            "Dice gain": round(best_values["dice"] - base_d, 4),
            "Precision change": round(best_values["precision"] - base_p, 4),
            "Recall gain": round(best_values["recall"] - base_r, 4),
            "refinement signal": signal,
        }
        rel_fig = make_six_panel(label, best_mask, target, summary)
        summary["six panel figure"] = rel_fig
        summary_rows.append(summary)
        figure_rows.append({"class": label, "figure": rel_fig, "caption": signal})

        if label == "bronchiola":
            action = "Run true second-SAM only after checking that the missing bottom bronchiola component has an eligible selected piece; boundary refinement helps the covered pieces but cannot invent the missing component."
            layer = "boundary plus component coverage"
        elif label == "alveoli":
            action = "Do not refine this medpt24 selected union. Return to the broad H&E box384 target-grounding branch and recover exact candidate masks; the boundary proxy has too little signal."
            layer = "proposal / target grounding"
        elif label == "vessels":
            action = "Run true second-SAM from selected vessel pieces with a precision guard; the proxy shows recall can improve while precision remains high."
            layer = "boundary / recall refinement"
        elif label == "tumor":
            action = "Treat as secondary. Boundary refinement has some signal, but the main issue is diffuse tumor-vs-stroma proposal quality and false-positive veto."
            layer = "proposal quality plus false-positive veto"
        elif label == "stroma":
            action = "Test true second-SAM or controlled expansion, but require a precision veto because the absolute precision remains low even after the proxy."
            layer = "boundary refinement with precision guard"
        else:
            action = "Do not spend more effort on boundary refinement first. Improve immune piece selection / density-quality skill and validate by components."
            layer = "selection / validation stability"
        action_rows.append(
            {
                "class": label,
                "refinement signal": signal,
                "earliest failure after this gate": layer,
                "next allowed iteration": action,
            }
        )

    write_csv(OUT / "boundary_refinement_proxy_all_ops.csv", all_rows)
    write_csv(OUT / "boundary_refinement_proxy_summary.csv", summary_rows)
    write_csv(OUT / "next_iteration_after_boundary_proxy.csv", action_rows)

    section = f"""
<h2>17AT. Boundary Refinement Proxy Gate</h2>
<p><b>Goal.</b> This section makes the failure analysis more precise.  It tests whether the current selected-piece union fails mainly because the boundary is too tight/fragmented, or whether the earlier candidate selection is the real problem.</p>
<p><b>Paper-guided reason.</b> SaLIP-style cascades suggest using a scorer to find useful regions and then prompting SAM again; region-aware methods such as RegionCLIP and Alpha-CLIP warn that region scoring and mask boundary quality are different problems.  Therefore, before running another expensive second-SAM job, this proxy checks whether a small local boundary refinement would even help.</p>
<p><b>What this is.</b> For each class, I take the current selected-piece union and apply small-radius dilation/closing operations.  Annotation is hidden from the scoring step and used only here to evaluate Dice, Precision, and Recall.  This is <b>diagnostic only</b>; it is not a final segmentation method.</p>
<p><b>How to read the table.</b> Identity means the current selected union before any proxy refinement.  Best proxy is the local operation that gives the highest Dice.  If the best proxy improves Dice or Recall without a major precision drop, real second-SAM refinement is worth running for that class.  If it does not help, the skill must go back to proposal/selection rather than boundary repair.</p>
{table(summary_rows, ['class', 'identity Dice', 'identity Precision', 'identity Recall', 'best proxy op', 'best Dice', 'best Precision', 'best Recall', 'Dice gain', 'Precision change', 'Recall gain', 'refinement signal'])}
<h3>Next action after this gate</h3>
{table(action_rows, ['class', 'refinement signal', 'earliest failure after this gate', 'next allowed iteration'])}
<h3>Six-panel proxy figures</h3>
<div class="grid">
{''.join(f'<figure><img src="{html.escape(str(row["figure"]))}" alt="{html.escape(str(row["class"]))} boundary refinement proxy"><figcaption>{html.escape(str(row["class"]))}: {html.escape(str(row["caption"]))}</figcaption></figure>' for row in figure_rows)}
</div>
<p><b>Next iteration rule.</b> Classes with a positive signal should move to true second-SAM from selected boxes/points.  Classes with no signal should not get more prompt tuning; their earliest failure is candidate/proposal selection or target grounding.</p>
"""

    for html_path in [BASE / "Jun09_SkillRanker_ComponentAwareMaskSelection.html", BASE / "index.html"]:
        replace_or_append(html_path, section)
        verify_html_images(html_path)
    rebuild_zip()
    print(f"Wrote {OUT / 'boundary_refinement_proxy_summary.csv'}")
    print(f"Updated {BASE / 'Jun09_SkillRanker_ComponentAwareMaskSelection.html'}")


if __name__ == "__main__":
    main()
