#!/usr/bin/env python3
"""Fine-grained diagnostics and policy refinement for Jun09 skill ranker.

The goal is to identify the failing layer more precisely:
1. candidate/proposal coverage,
2. piece recognition and calibration,
3. selection/NMS/assembly policy.

Any policy tuned with annotation is reported as diagnostic/oracle-calibrated,
not as a deployable no-annotation method.
"""

from __future__ import annotations

import csv
import html
import json
import math
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage as ndi
from sklearn.metrics import average_precision_score, roc_auc_score
from skimage import measure


ROOT = Path("/Users/haoranwu/Desktop/Yan_Lab_Research/Medical-SAM3")
BASE = ROOT / "output/visium_hd_exp1/final_deliverables/Jun09_SkillRanker_ComponentAwareMaskSelection"
ANNOTATION_DIR = ROOT / "output/visium_hd_exp1/ficture_official_filtered_candidate_pool_input_roi/cropped_annotation_masks"
HE_ROI = ROOT / "output/visium_hd_exp1/ficture_official_filtered_candidate_pool_input_roi/he_roi_matching_official_ficture_coverage.png"
FICTURE_ROI = ROOT / "output/visium_hd_exp1/ficture_official_filtered_candidate_pool_input_roi/ficture_official_filtered_roi_rgb.png"

SCORE_DIR = BASE / "score_outputs/combined_skill_ranker_rf_plus_he"
REFINE_DIR = BASE / "refined_diagnostics"

CLASS_KEYS = ["bronchiola", "alveoli", "vessels", "tumor", "stroma", "immune_infiltration"]
DISPLAY = {
    "bronchiola": "bronchiola",
    "alveoli": "alveoli",
    "vessels": "vessels",
    "tumor": "tumor",
    "stroma": "stroma",
    "immune_infiltration": "immune infiltration",
}
LONG_TO_CLASS = {
    "lung_bronchiola": "bronchiola",
    "lung_alveoli_normal_adjacent": "alveoli",
    "lung_vessels": "vessels",
    "tumor": "tumor",
    "stroma": "stroma",
    "immune_infiltration": "immune_infiltration",
}
ANNOTATION_FILES = {
    "bronchiola": "01_lung_bronchiola_target_roi.png",
    "alveoli": "04_lung_alveoli_normal_adjacent_target_roi.png",
    "vessels": "05_lung_vessels_target_roi.png",
    "tumor": "08_tumor_target_roi.png",
    "stroma": "07_stroma_target_roi.png",
    "immune_infiltration": "03_immune_infiltration_target_roi.png",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: Iterable[dict[str, object]], fieldnames: list[str] | None = None) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = []
        for row in rows:
            for key in row:
                if key not in fieldnames:
                    fieldnames.append(key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def class_key(value: str) -> str:
    return LONG_TO_CLASS.get(value, value)


def row_key(row: dict[str, str]) -> str:
    if row.get("row_key"):
        return row["row_key"]
    return "__".join([row.get("target_label", ""), row.get("source", ""), row.get("run", ""), row.get("setting", ""), row.get("candidate_id", "")])


def load_mask(path: Path, size: tuple[int, int]) -> np.ndarray:
    image = Image.open(path).convert("L")
    if image.size != size:
        image = image.resize(size, Image.Resampling.NEAREST)
    return np.array(image) > 0


def load_mask_small(path: Path, size: tuple[int, int]) -> np.ndarray:
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


def overlap_fraction(candidate: np.ndarray, union: np.ndarray) -> float:
    area = int(candidate.sum())
    if area == 0:
        return 1.0
    return int(np.logical_and(candidate, union).sum()) / area


def cover_components(pred: np.ndarray, label_map: np.ndarray, areas: np.ndarray, threshold: float = 0.10) -> tuple[int, int]:
    total = len(areas) - 1
    if total <= 0:
        return 0, 0
    overlaps = np.bincount(label_map[pred].ravel(), minlength=total + 1)
    covered = int(np.sum(((overlaps / np.maximum(1, areas)) >= threshold) & (np.arange(total + 1) > 0)))
    return covered, total


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


def render_six_panel(path: Path, tissue_class: str, selected: np.ndarray, annotation: np.ndarray, title: str) -> None:
    he = Image.open(HE_ROI).convert("RGB")
    ficture = Image.open(FICTURE_ROI).convert("RGB")
    panel_w, panel_h = 260, 275
    gap, margin, header_h, label_h = 18, 26, 98, 24
    width = margin * 2 + 6 * panel_w + 5 * gap
    height = margin * 2 + header_h + label_h + panel_h
    sheet = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()
    d, p, r = metrics(selected, annotation)
    draw.text((margin, margin), title, fill=(20, 20, 20), font=font)
    draw.text((margin, margin + 24), f"{tissue_class} | Dice {d:.3f} | Precision {p:.3f} | Recall {r:.3f}", fill=(70, 80, 90), font=font)
    draw.text((margin, margin + 48), "Diagnostic/oracle-calibrated assembly: annotation used only to find what the current pool could support.", fill=(70, 80, 90), font=font)
    items = [
        ("Annotation on H&E", overlay_mask(he, annotation, (0, 180, 90))),
        ("Refined union on H&E", overlay_mask(he, selected, (0, 90, 255))),
        ("Refined union mask only", mask_only(selected, (0, 90, 255))),
        ("Annotation mask only", mask_only(annotation, (0, 180, 90))),
        ("H&E ROI", he),
        ("FICTURE ROI", ficture),
    ]
    y0 = margin + header_h
    for idx, (label, img) in enumerate(items):
        x = margin + idx * (panel_w + gap)
        draw.text((x, y0), label, fill=(35, 45, 60), font=font)
        sheet.paste(fit_image(img, panel_w, panel_h), (x, y0 + label_h))
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path)


def html_table(rows: Iterable[dict[str, object]], cols: list[str]) -> str:
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


def diagnose_layer(row: dict[str, object]) -> str:
    policy_d = float(row["best_policy_dice"])
    current_d = float(row["current_dice"])
    recog = float(row["target_vs_rest_auc"])
    comp_gap = float(row["missed_component_fraction"])
    component_count = int(row["annotation_components"])
    if comp_gap > 0.40:
        return "candidate/component coverage gap"
    if policy_d < 0.45 and component_count > 10:
        return "candidate pool upper-bound weak"
    if recog < 0.80:
        return "recognition / calibration"
    if policy_d - current_d > 0.08:
        return "selection / NMS / assembly"
    return "mostly acceptable but still needs external validation"


def main() -> None:
    REFINE_DIR.mkdir(parents=True, exist_ok=True)
    size = Image.open(HE_ROI).size
    small_size = (512, int(round(512 * size[1] / size[0])))
    hidden_rows = read_csv(BASE / "corrected_pool/hidden_candidate_truth.csv")
    public_rows = read_csv(BASE / "corrected_pool/public_vlm_requests.csv")
    pred_rows = read_csv(SCORE_DIR / "per_candidate_predictions.csv")
    pred_by_key = {row["row_key"]: row for row in pred_rows}
    hidden_by_key = {row_key(row): row for row in hidden_rows}
    public_by_key = {row_key(row): row for row in public_rows}
    current_summary = read_csv(BASE / "assembly_summary_all_methods.csv")

    annotations = {c: load_mask(ANNOTATION_DIR / ANNOTATION_FILES[c], size) for c in CLASS_KEYS}
    label_maps = {c: measure.label(annotations[c]) for c in CLASS_KEYS}
    label_areas = {c: np.bincount(label_maps[c].ravel(), minlength=int(label_maps[c].max()) + 1) for c in CLASS_KEYS}

    # Per-component proposal coverage: one bincount per candidate per class.
    component_rows: list[dict[str, object]] = []
    candidate_mask_cache: dict[str, np.ndarray] = {}
    for c in CLASS_KEYS:
        labels = label_maps[c]
        areas = label_areas[c]
        best: dict[int, dict[str, object]] = {
            comp_id: {"component_id": comp_id, "component_area": int(areas[comp_id]), "best_dice": 0.0, "best_precision": 0.0, "best_recall": 0.0, "best_candidate_uid": "", "best_candidate_true_class": "", "best_candidate_pred_class": "", "best_candidate_score": 0}
            for comp_id in range(1, len(areas))
        }
        for row in hidden_rows:
            key = row_key(row)
            mask_path = row["mask_path"]
            if mask_path not in candidate_mask_cache:
                candidate_mask_cache[mask_path] = load_mask(Path(mask_path), size)
            mask = candidate_mask_cache[mask_path]
            overlaps = np.bincount(labels[mask].ravel(), minlength=len(areas))
            cand_area = int(mask.sum())
            scores = {cls: int(float(pred_by_key[key][cls])) for cls in CLASS_KEYS}
            pred_class, top_score, _, _ = predicted(scores)
            for comp_id in np.flatnonzero(overlaps > 0):
                if comp_id == 0:
                    continue
                tp = int(overlaps[comp_id])
                precision = tp / cand_area if cand_area else 0.0
                recall = tp / max(1, int(areas[comp_id]))
                dice = 2 * tp / max(1, cand_area + int(areas[comp_id]))
                if dice > float(best[comp_id]["best_dice"]):
                    best[comp_id].update(
                        {
                            "best_dice": dice,
                            "best_precision": precision,
                            "best_recall": recall,
                            "best_candidate_uid": row["candidate_uid"],
                            "best_candidate_true_class": class_key(row["classification_true_label"]),
                            "best_candidate_pred_class": pred_class,
                            "best_candidate_score": top_score,
                        }
                    )
        for comp_id, item in best.items():
            component_rows.append({"class": c, **item})
    write_csv(REFINE_DIR / "component_level_proposal_coverage.csv", component_rows)

    # Recognition diagnostics.
    recognition_rows: list[dict[str, object]] = []
    for c in CLASS_KEYS:
        y_true = []
        y_score = []
        margins_true = []
        target_scores_true = []
        target_scores_false = []
        ranks_true: list[int] = []
        for key, pred in pred_by_key.items():
            true = class_key(hidden_by_key[key]["classification_true_label"])
            scores = {cls: int(float(pred[cls])) for cls in CLASS_KEYS}
            y_true.append(1 if true == c else 0)
            y_score.append(scores[c])
            sorted_scores = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
            if true == c:
                ranks_true.append([name for name, _ in sorted_scores].index(c) + 1)
                target_scores_true.append(scores[c])
                margins_true.append(scores[c] - max(v for k, v in scores.items() if k != c))
            else:
                target_scores_false.append(scores[c])
        auc = roc_auc_score(y_true, y_score) if len(set(y_true)) == 2 else float("nan")
        ap = average_precision_score(y_true, y_score) if len(set(y_true)) == 2 else float("nan")
        recognition_rows.append(
            {
                "class": c,
                "target_vs_rest_auc": f"{auc:.3f}",
                "average_precision": f"{ap:.3f}",
                "true_piece_count": int(sum(y_true)),
                "median_true_target_score": f"{float(np.median(target_scores_true)):.1f}" if target_scores_true else "",
                "median_false_target_score": f"{float(np.median(target_scores_false)):.1f}" if target_scores_false else "",
                "median_true_margin": f"{float(np.median(margins_true)):.1f}" if margins_true else "",
                "median_rank_of_true_class": f"{float(np.median(ranks_true)):.1f}" if ranks_true else "",
                "rank1_true_pieces": f"{sum(1 for rank in ranks_true if rank == 1)}/{len(ranks_true)}",
            }
        )
    write_csv(REFINE_DIR / "recognition_calibration_by_class.csv", recognition_rows)

    # Assembly policy search over combined ranker scores.
    small_mask_cache: dict[str, np.ndarray] = {}
    full_mask_cache: dict[str, np.ndarray] = {}

    def small_mask(row: dict[str, str]) -> np.ndarray:
        path = row["mask_path"]
        if path not in small_mask_cache:
            small_mask_cache[path] = load_mask_small(Path(path), small_size)
        return small_mask_cache[path]

    def full_mask(row: dict[str, str]) -> np.ndarray:
        path = row["mask_path"]
        if path not in full_mask_cache:
            full_mask_cache[path] = load_mask(Path(path), size)
        return full_mask_cache[path]

    score_thresholds = [20, 30, 40, 50, 60, 70, 80]
    margin_thresholds = [-15, -5, 0, 5, 10, 20]
    overlap_thresholds = [0.45, 0.60, 0.75, 0.90]
    max_piece_options = [2, 4, 6, 8, 12, 16, 24, 36]
    policy_rows: list[dict[str, object]] = []
    refined_summary: list[dict[str, object]] = []
    selected_rows: list[dict[str, object]] = []
    for c in CLASS_KEYS:
        ranked = []
        for key, pred in pred_by_key.items():
            scores = {cls: int(float(pred[cls])) for cls in CLASS_KEYS}
            pred_class, _, margin, tie = predicted(scores)
            # Keep strict top-predicted target as deployable behavior; this
            # prevents the same ambiguous piece from being selected for multiple classes.
            if pred_class != c or tie:
                continue
            row = public_by_key[key]
            ranked.append((scores[c], margin, key, row, scores))
        ranked.sort(key=lambda item: (item[0], item[1]), reverse=True)
        best = None
        best_union = None
        best_selected: list[tuple[int, int, str, dict[str, str], dict[str, int]]] = []
        for min_score in score_thresholds:
            for min_margin in margin_thresholds:
                for max_overlap in overlap_thresholds:
                    for max_pieces in max_piece_options:
                        union_small = np.zeros((small_size[1], small_size[0]), dtype=bool)
                        selected: list[tuple[int, int, str, dict[str, str], dict[str, int]]] = []
                        for item in ranked:
                            score, margin, key, row, scores = item
                            if score < min_score:
                                break
                            if margin < min_margin:
                                continue
                            mask_s = small_mask(row)
                            if overlap_fraction(mask_s, union_small) > max_overlap:
                                continue
                            selected.append(item)
                            union_small |= mask_s
                            if len(selected) >= max_pieces:
                                break
                        union = np.zeros((size[1], size[0]), dtype=bool)
                        for _, _, _, row, _ in selected:
                            union |= full_mask(row)
                        d, p, r = metrics(union, annotations[c])
                        cov, total = cover_components(union, label_maps[c], label_areas[c])
                        # Objective is class-specific: broad classes require precision
                        # protection; fragmented classes can trade precision for recall.
                        if c in {"bronchiola", "vessels", "immune_infiltration"}:
                            objective = d + 0.08 * r - (0.06 if p < 0.45 else 0.0)
                        else:
                            objective = d + 0.05 * p - (0.08 if p < 0.40 else 0.0)
                        row_out = {
                            "class": c,
                            "min_score": min_score,
                            "min_margin": min_margin,
                            "max_overlap": max_overlap,
                            "max_pieces": max_pieces,
                            "selected_piece_count": len(selected),
                            "dice": d,
                            "precision": p,
                            "recall": r,
                            "covered_components": cov,
                            "total_components": total,
                            "objective": objective,
                        }
                        policy_rows.append(row_out)
                        if best is None or objective > float(best["objective"]):
                            best = row_out
                            best_union = union
                            best_selected = selected
        assert best is not None and best_union is not None
        fig_path = REFINE_DIR / "figures" / f"{c}_refined_policy_union.png"
        render_six_panel(fig_path, c, best_union, annotations[c], f"{c}: refined diagnostic policy union")
        current = next(row for row in current_summary if row["method"] == "combined_skill_ranker_rf_plus_he" and row["class"] == c)
        comp_rows = [row for row in component_rows if row["class"] == c]
        missed = [row for row in comp_rows if float(row["best_recall"]) < 0.10]
        rec_row = next(row for row in recognition_rows if row["class"] == c)
        refined = {
            "class": c,
            "diagnosed_limiting_layer": "",  # filled below
            "true_candidate_count": sum(1 for row in hidden_rows if class_key(row["classification_true_label"]) == c),
            "annotation_components": len(comp_rows),
            "pool_components_with_best_recall_ge_0.10": len(comp_rows) - len(missed),
            "missed_component_fraction": len(missed) / max(1, len(comp_rows)),
            "target_vs_rest_auc": rec_row["target_vs_rest_auc"],
            "piece_top1": rec_row["rank1_true_pieces"],
            "current_dice": current["dice"],
            "current_precision": current["precision"],
            "current_recall": current["recall"],
            "refined_dice": f"{float(best['dice']):.3f}",
            "refined_precision": f"{float(best['precision']):.3f}",
            "refined_recall": f"{float(best['recall']):.3f}",
            "refined_selected_piece_count": best["selected_piece_count"],
            "refined_policy": f"score>={best['min_score']}, margin>={best['min_margin']}, overlap<={best['max_overlap']}, max_pieces={best['max_pieces']}",
            "best_policy_dice": f"{float(best['dice']):.3f}",
            "figure_rel": str(fig_path.relative_to(BASE)),
        }
        refined["diagnosed_limiting_layer"] = diagnose_layer(refined)
        refined_summary.append(refined)
        for rank, (score, margin, key, row, scores) in enumerate(best_selected, 1):
            selected_rows.append(
                {
                    "class": c,
                    "rank": rank,
                    "candidate_uid": row["candidate_uid"],
                    "target_score": score,
                    "margin": margin,
                    "true_class": class_key(hidden_by_key[key]["classification_true_label"]),
                    "component_dice": hidden_by_key[key].get("component_dice", ""),
                    "component_precision": hidden_by_key[key].get("component_precision", ""),
                    "component_recall": hidden_by_key[key].get("component_recall", ""),
                }
            )
    write_csv(REFINE_DIR / "assembly_policy_grid_all.csv", policy_rows)
    write_csv(REFINE_DIR / "refined_policy_summary_by_class.csv", refined_summary)
    write_csv(REFINE_DIR / "refined_policy_selected_pieces.csv", selected_rows)

    # Insert/replace the detailed refinement section in the main report.
    figure_html = "".join(
        f"<section class='figure-block'><h3>{html.escape(DISPLAY[row['class']])}</h3><img src='{html.escape(row['figure_rel'])}' alt='{html.escape(row['class'])} refined diagnostic union'></section>"
        for row in refined_summary
    )
    section = f"""
<h2>10. Refined Failure Analysis and Iteration</h2>
<p>This adds a stricter, paper-style decomposition inspired by proposal-rerank pipelines such as SaLIP, region-aware scoring in RegionCLIP/Alpha-CLIP, and guided-cropping analyses. The question is no longer only "what is the final Dice?", but which stage is blocking improvement.</p>
<ul>
<li><b>Proposal coverage:</b> for every annotation component, find the best candidate piece that overlaps it.</li>
<li><b>Recognition/calibration:</b> measure whether the target-class score separates true pieces from other pieces.</li>
<li><b>Selection/NMS:</b> grid-search score, margin, overlap, and max-piece thresholds to see whether assembly can be improved from the same ranked pieces.</li>
</ul>
{html_table(refined_summary, ['class', 'diagnosed_limiting_layer', 'annotation_components', 'pool_components_with_best_recall_ge_0.10', 'piece_top1', 'target_vs_rest_auc', 'current_dice', 'current_precision', 'current_recall', 'refined_dice', 'refined_precision', 'refined_recall', 'refined_selected_piece_count', 'refined_policy'])}
<div class='callout warn'><b>Important caveat.</b> The refined policy below is annotation-calibrated, so it is a diagnostic upper bound for this ROI, not yet a deployable no-annotation rule. It tells us what should be refined next: candidate generation, scoring, or assembly.</div>
{figure_html}
"""
    for html_path in [BASE / "Jun09_SkillRanker_ComponentAwareMaskSelection.html", BASE / "index.html"]:
        text = html_path.read_text()
        marker = "<h2>10. Refined Failure Analysis and Iteration</h2>"
        if marker in text:
            start = text.index(marker)
            next_marker = "<h2>10. Interpretation</h2>"
            alt_marker = "<h2>11. Interpretation</h2>"
            if alt_marker in text[start:]:
                end = text.index(alt_marker, start)
                replacement_next = alt_marker
            elif next_marker in text[start:]:
                end = text.index(next_marker, start)
                replacement_next = "<h2>11. Interpretation</h2>"
            else:
                end = text.rindex("</body>")
                replacement_next = ""
            text = text[:start] + section + replacement_next + text[end + len(replacement_next):]
        elif "<h2>10. Interpretation</h2>" in text:
            text = text.replace("<h2>10. Interpretation</h2>", section + "<h2>11. Interpretation</h2>")
        else:
            text = text.replace("</body>", section + "</body>")
        html_path.write_text(text)

    print(REFINE_DIR / "refined_policy_summary_by_class.csv")
    print(BASE / "Jun09_SkillRanker_ComponentAwareMaskSelection.html")


if __name__ == "__main__":
    main()
