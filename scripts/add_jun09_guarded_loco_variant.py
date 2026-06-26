#!/usr/bin/env python3
"""Build a practical guarded LOCO assembly variant for Jun09."""

from __future__ import annotations

import csv
import html
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


ROOT = Path("/Users/haoranwu/Desktop/Yan_Lab_Research/Medical-SAM3")
BASE = ROOT / "output/visium_hd_exp1/final_deliverables/Jun09_SkillRanker_ComponentAwareMaskSelection"
OUT = BASE / "guarded_loco_variant"
ANNOTATION_DIR = ROOT / "output/visium_hd_exp1/ficture_official_filtered_candidate_pool_input_roi/cropped_annotation_masks"
HE_ROI = ROOT / "output/visium_hd_exp1/ficture_official_filtered_candidate_pool_input_roi/he_roi_matching_official_ficture_coverage.png"
FICTURE_ROI = ROOT / "output/visium_hd_exp1/ficture_official_filtered_candidate_pool_input_roi/ficture_official_filtered_roi_rgb.png"

CLASS_KEYS = ["bronchiola", "alveoli", "vessels", "tumor", "stroma", "immune_infiltration"]
ANNOTATION_FILES = {
    "bronchiola": "01_lung_bronchiola_target_roi.png",
    "alveoli": "04_lung_alveoli_normal_adjacent_target_roi.png",
    "vessels": "05_lung_vessels_target_roi.png",
    "tumor": "08_tumor_target_roi.png",
    "stroma": "07_stroma_target_roi.png",
    "immune_infiltration": "03_immune_infiltration_target_roi.png",
}
POLICIES = {
    "bronchiola": {"min_score": 35, "min_margin": 4, "max_overlap": 0.75, "max_pieces": 24, "he_guard": 0},
    "alveoli": {"min_score": 45, "min_margin": 5, "max_overlap": 0.75, "max_pieces": 10, "he_guard": 0},
    "vessels": {"min_score": 20, "min_margin": -15, "max_overlap": 0.75, "max_pieces": 24, "he_guard": 75},
    "tumor": {"min_score": 45, "min_margin": 5, "max_overlap": 0.75, "max_pieces": 10, "he_guard": 0},
    "stroma": {"min_score": 45, "min_margin": 5, "max_overlap": 0.75, "max_pieces": 10, "he_guard": 0},
    "immune_infiltration": {"min_score": 35, "min_margin": 4, "max_overlap": 0.75, "max_pieces": 24, "he_guard": 0},
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


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


def load_mask(path: Path, size: tuple[int, int]) -> np.ndarray:
    image = Image.open(path).convert("L")
    if image.size != size:
        image = image.resize(size, Image.Resampling.NEAREST)
    return np.array(image) > 0


def metrics(pred: np.ndarray, gt: np.ndarray) -> tuple[float, float, float]:
    tp = int(np.logical_and(pred, gt).sum())
    pa = int(pred.sum())
    ga = int(gt.sum())
    p = tp / pa if pa else 0.0
    r = tp / ga if ga else 0.0
    d = 2 * tp / (pa + ga) if pa + ga else 0.0
    return d, p, r


def predicted(scores: dict[str, int]) -> tuple[str, int, int, bool]:
    ordered = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    tie = len([v for _, v in ordered if v == ordered[0][1]]) > 1
    second = ordered[1][1] if len(ordered) > 1 else 0
    return ordered[0][0], ordered[0][1], ordered[0][1] - second, tie


def overlap_fraction(mask: np.ndarray, union: np.ndarray) -> float:
    return int(np.logical_and(mask, union).sum()) / max(1, int(mask.sum()))


def overlay_mask(image: Image.Image, mask: np.ndarray, color: tuple[int, int, int], alpha: int = 130) -> Image.Image:
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


def render_six(path: Path, cls: str, union: np.ndarray, ann: np.ndarray, title: str) -> None:
    he = Image.open(HE_ROI).convert("RGB")
    ficture = Image.open(FICTURE_ROI).convert("RGB")
    d, p, r = metrics(union, ann)
    panel_w, panel_h = 260, 275
    gap, margin, header_h, label_h = 18, 26, 98, 24
    sheet = Image.new("RGB", (margin * 2 + panel_w * 6 + gap * 5, margin * 2 + header_h + label_h + panel_h), "white")
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()
    draw.text((margin, margin), title, fill=(20, 20, 20), font=font)
    draw.text((margin, margin + 24), f"{cls} | Dice {d:.3f} | Precision {p:.3f} | Recall {r:.3f}", fill=(70, 80, 90), font=font)
    draw.text((margin, margin + 48), "Practical guarded LOCO variant: no annotation is used in the score, but this threshold was diagnosed on the current ROI.", fill=(70, 80, 90), font=font)
    items = [
        ("Annotation on H&E", overlay_mask(he, ann, (0, 180, 90))),
        ("Guarded union on H&E", overlay_mask(he, union, (0, 90, 255))),
        ("Guarded union mask only", mask_only(union, (0, 90, 255))),
        ("Annotation mask only", mask_only(ann, (0, 180, 90))),
        ("H&E ROI", he),
        ("FICTURE ROI", ficture),
    ]
    y0 = margin + header_h
    for idx, (label, image) in enumerate(items):
        x = margin + idx * (panel_w + gap)
        draw.text((x, y0), label, fill=(35, 45, 60), font=font)
        sheet.paste(fit_image(image, panel_w, panel_h), (x, y0 + label_h))
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path)


def html_table(rows: list[dict[str, object]], cols: list[str]) -> str:
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


def main() -> None:
    size = Image.open(HE_ROI).size
    public = {row["row_key"] if row.get("row_key") else "__".join([row.get("target_label", ""), row.get("source", ""), row.get("run", ""), row.get("setting", ""), row.get("candidate_id", "")]): row for row in read_csv(BASE / "corrected_pool/public_vlm_requests.csv")}
    features = {row["row_key"]: row for row in read_csv(BASE / "candidate_skill_features.csv")}
    loco = read_csv(BASE / "score_outputs/loco_component_logistic_validation/per_candidate_predictions.csv")
    annotations = {c: load_mask(ANNOTATION_DIR / ANNOTATION_FILES[c], size) for c in CLASS_KEYS}
    mask_cache: dict[str, np.ndarray] = {}

    def get_mask(row: dict[str, str]) -> np.ndarray:
        path = public[row["row_key"]]["mask_path"]
        if path not in mask_cache:
            mask_cache[path] = load_mask(Path(path), size)
        return mask_cache[path]

    summary: list[dict[str, object]] = []
    selected_rows: list[dict[str, object]] = []
    for c in CLASS_KEYS:
        policy = POLICIES[c]
        ranked = []
        for row in loco:
            scores = {cls: int(float(row[cls])) for cls in CLASS_KEYS}
            pred, score, margin, tie = predicted(scores)
            if pred != c or tie:
                continue
            he_guard = float(policy["he_guard"])
            he_score = float(features[row["row_key"]].get(f"he_clip_large_{c}", 0) or 0)
            if he_score < he_guard:
                continue
            ranked.append((score, margin, he_score, row))
        ranked.sort(key=lambda item: (item[0], item[1], item[2]), reverse=True)
        union = np.zeros((size[1], size[0]), dtype=bool)
        selected = []
        for score, margin, he_score, row in ranked:
            if score < int(policy["min_score"]):
                break
            if margin < int(policy["min_margin"]):
                continue
            mask = get_mask(row)
            if overlap_fraction(mask, union) > float(policy["max_overlap"]):
                continue
            selected.append((score, margin, he_score, row))
            union |= mask
            if len(selected) >= int(policy["max_pieces"]):
                break
        d, p, r = metrics(union, annotations[c])
        fig = OUT / "figures" / f"{c}_guarded_loco_union.png"
        render_six(fig, c, union, annotations[c], f"{c}: guarded LOCO assembly")
        summary.append(
            {
                "class": c,
                "selected_piece_count": len(selected),
                "dice": f"{d:.3f}",
                "precision": f"{p:.3f}",
                "recall": f"{r:.3f}",
                "policy": f"score>={policy['min_score']}, margin>={policy['min_margin']}, overlap<={policy['max_overlap']}, max_pieces={policy['max_pieces']}, he_guard>={policy['he_guard']}",
                "figure_rel": str(fig.relative_to(BASE)),
            }
        )
        for rank, (score, margin, he_score, row) in enumerate(selected, 1):
            selected_rows.append({"class": c, "rank": rank, "candidate_uid": row["candidate_uid"], "target_score": score, "margin": margin, "he_guard_score": he_score, "true_class": row["true_label"]})
    write_csv(OUT / "guarded_loco_summary.csv", summary)
    write_csv(OUT / "guarded_loco_selected_pieces.csv", selected_rows)
    figures = "".join(
        f"<section class='figure-block'><h3>{html.escape(row['class'])}</h3><img src='{html.escape(row['figure_rel'])}' alt='{html.escape(row['class'])} guarded LOCO'></section>"
        for row in summary
    )
    section = f"""
<h2>13. Practical Guarded LOCO Variant</h2>
<p>After LOCO exposed a vessels precision drop, I tested a practical guard: keep LOCO component-ranker scores, but require a high H&E morphology score for vessels before assembly. This is closer to a deployable rule than the annotation-calibrated refined policy, although the vessel threshold is still diagnosed on this ROI.</p>
{html_table(summary, ['class', 'selected_piece_count', 'dice', 'precision', 'recall', 'policy'])}
{figures}
"""
    for html_path in [BASE / "Jun09_SkillRanker_ComponentAwareMaskSelection.html", BASE / "index.html"]:
        text = html_path.read_text()
        marker = "<h2>13. Practical Guarded LOCO Variant</h2>"
        if marker in text:
            start = text.index(marker)
            end = text.index("<h2>13. Interpretation</h2>", start) if "<h2>13. Interpretation</h2>" in text[start:] else text.index("</body>", start)
            text = text[:start] + section + text[end:]
        elif "<h2>13. Interpretation</h2>" in text:
            text = text.replace("<h2>13. Interpretation</h2>", section + "<h2>14. Interpretation</h2>")
        else:
            text = text.replace("</body>", section + "</body>")
        html_path.write_text(text)
    print(OUT / "guarded_loco_summary.csv")


if __name__ == "__main__":
    main()
