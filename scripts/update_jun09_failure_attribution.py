#!/usr/bin/env python3
"""Add step-level failure attribution to the Jun09 skill-ranker report."""

from __future__ import annotations

import csv
import html
import re
from pathlib import Path

import numpy as np
from PIL import Image
from skimage import measure


ROOT = Path("/Users/haoranwu/Desktop/Yan_Lab_Research/Medical-SAM3")
BASE = ROOT / "output/visium_hd_exp1/final_deliverables/Jun09_SkillRanker_ComponentAwareMaskSelection"
ANNOTATION_DIR = ROOT / "output/visium_hd_exp1/ficture_official_filtered_candidate_pool_input_roi/cropped_annotation_masks"
HE_ROI = ROOT / "output/visium_hd_exp1/ficture_official_filtered_candidate_pool_input_roi/he_roi_matching_official_ficture_coverage.png"

CLASS_KEYS = ["bronchiola", "alveoli", "vessels", "tumor", "stroma", "immune_infiltration"]
CLASS_DISPLAY = {
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
CLASS_TO_ANNOTATION = {
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


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def class_key(value: str) -> str:
    return LONG_TO_CLASS.get(value, value)


def load_mask(path: Path, size: tuple[int, int]) -> np.ndarray:
    image = Image.open(path).convert("L")
    if image.size != size:
        image = image.resize(size, Image.Resampling.NEAREST)
    return np.array(image) > 0


def metrics(pred: np.ndarray, gt: np.ndarray) -> tuple[float, float, float]:
    tp = int(np.logical_and(pred, gt).sum())
    pred_area = int(pred.sum())
    gt_area = int(gt.sum())
    precision = tp / pred_area if pred_area else 0.0
    recall = tp / gt_area if gt_area else 0.0
    dice = (2 * tp) / (pred_area + gt_area) if pred_area + gt_area else 0.0
    return dice, precision, recall


def component_coverage(pred: np.ndarray, gt: np.ndarray) -> tuple[int, int]:
    labels = measure.label(gt)
    total = int(labels.max())
    if total == 0:
        return 0, 0
    areas = np.bincount(labels.ravel(), minlength=total + 1)
    overlaps = np.bincount(labels[pred].ravel(), minlength=total + 1)
    covered = int(np.sum(((overlaps / np.maximum(1, areas)) >= 0.10) & (np.arange(total + 1) > 0)))
    return covered, total


def greedy_oracle(rows: list[dict[str, str]], gt: np.ndarray, size: tuple[int, int]) -> tuple[np.ndarray, list[str]]:
    mask_cache: dict[str, np.ndarray] = {}

    def get_mask(row: dict[str, str]) -> np.ndarray:
        path = row["mask_path"]
        if path not in mask_cache:
            mask_cache[path] = load_mask(Path(path), size)
        return mask_cache[path]

    union = np.zeros((size[1], size[0]), dtype=bool)
    chosen: list[str] = []
    current_dice = metrics(union, gt)[0]
    while len(chosen) < 30:
        best_gain = 0.0
        best_row = None
        best_union = None
        for row in rows:
            uid = row["candidate_uid"]
            if uid in chosen:
                continue
            candidate_union = union | get_mask(row)
            candidate_dice = metrics(candidate_union, gt)[0]
            gain = candidate_dice - current_dice
            if gain > best_gain:
                best_gain = gain
                best_row = row
                best_union = candidate_union
        if best_row is None or best_union is None or best_gain <= 1e-6:
            break
        chosen.append(best_row["candidate_uid"])
        union = best_union
        current_dice = metrics(union, gt)[0]
    return union, chosen


def diagnosis_for(c: str, greedy: tuple[float, float, float], combined: tuple[float, float, float], top1: tuple[int, int]) -> str:
    gd, gp, gr = greedy
    cd, cp, cr = combined
    top1_rate = top1[0] / max(1, top1[1])
    if gd < 0.35:
        return "Pool/funnel limit: even oracle union from the 167 pieces is weak; need better candidates before ranking."
    if top1_rate < 0.70:
        return "Recognition/ranker limit: candidates exist, but the score model does not reliably assign the right tissue class."
    if gd - cd > 0.12 or gr - cr > 0.18:
        return "Assembly limit: pieces are mostly recognized, but selection/union is too conservative or misses useful pieces."
    if c == "stroma":
        return "Broad-tissue limit: stroma is diffuse; piece-first assembly keeps precision high but recall remains very low."
    return "Mostly working: remaining gap is precision/recall tradeoff or local boundary mismatch."


def html_table(rows: list[dict[str, object]], columns: list[str]) -> str:
    out = ["<table><thead><tr>"]
    out.extend(f"<th>{html.escape(col)}</th>" for col in columns)
    out.append("</tr></thead><tbody>")
    for row in rows:
        out.append("<tr>")
        for col in columns:
            out.append(f"<td>{html.escape(str(row.get(col, '')))}</td>")
        out.append("</tr>")
    out.append("</tbody></table>")
    return "".join(out)


def main() -> None:
    size = Image.open(HE_ROI).size
    hidden = read_csv(BASE / "corrected_pool/hidden_candidate_truth.csv")
    assembly = read_csv(BASE / "assembly_summary_all_methods.csv")
    piece = read_csv(BASE / "piece_top1_all_methods.csv")
    rows_out: list[dict[str, object]] = []
    for c in CLASS_KEYS:
        gt = load_mask(ANNOTATION_DIR / CLASS_TO_ANNOTATION[c], size)
        cands = [row for row in hidden if class_key(row["classification_true_label"]) == c]
        oracle_mask, oracle_chosen = greedy_oracle(cands, gt, size)
        oracle = metrics(oracle_mask, gt)
        oracle_cov = component_coverage(oracle_mask, gt)
        combined_row = next(row for row in assembly if row["method"] == "combined_skill_ranker_rf_plus_he" and row["class"] == c)
        combined = (float(combined_row["dice"]), float(combined_row["precision"]), float(combined_row["recall"]))
        piece_rows = [row for row in piece if row["method"] == "combined_skill_ranker_rf_plus_he" and row["true_class"] == c]
        top1 = (sum(row["correct"] == "True" for row in piece_rows), len(piece_rows))
        rows_out.append(
            {
                "class": CLASS_DISPLAY[c],
                "true_candidates": len(cands),
                "combined_piece_top1": f"{top1[0]}/{top1[1]}",
                "oracle_greedy_dice_precision_recall": f"{oracle[0]:.3f} / {oracle[1]:.3f} / {oracle[2]:.3f}",
                "oracle_components_covered": f"{oracle_cov[0]}/{oracle_cov[1]}",
                "combined_final_dice_precision_recall": f"{combined[0]:.3f} / {combined[1]:.3f} / {combined[2]:.3f}",
                "combined_components_covered": f"{combined_row['covered_components']}/{combined_row['total_components']}",
                "diagnosis": diagnosis_for(c, oracle, combined, top1),
                "oracle_selected_candidates": ", ".join(oracle_chosen[:12]) + (" ..." if len(oracle_chosen) > 12 else ""),
            }
        )
    write_csv(BASE / "failure_attribution_by_class.csv", rows_out)
    section = f"""
<h2>9. Failure Attribution: Which Step Failed?</h2>
<p>This section separates three possible failure points. <b>Pool upper bound</b> asks whether the 167 candidate pieces contain useful masks at all. <b>Piece Top1</b> asks whether the score model recognizes each piece's hidden tissue class. <b>Assembly</b> asks whether the recognized pieces are selected and unioned into a good final mask.</p>
{html_table(rows_out, ['class', 'true_candidates', 'combined_piece_top1', 'oracle_greedy_dice_precision_recall', 'oracle_components_covered', 'combined_final_dice_precision_recall', 'combined_components_covered', 'diagnosis'])}
<div class='callout'><b>Takeaway.</b> Bronchiola and vessels have enough good pieces and the combined ranker gets close to the oracle. Immune infiltration has usable pieces but still loses recall. Stroma is the clearest downstream failure: the ranker recognizes pieces, but the final assembly covers only a tiny fraction of the diffuse annotation. Alveoli is limited mostly by the small number of available candidates in this 167-piece funnel.</div>
"""
    for html_path in [BASE / "Jun09_SkillRanker_ComponentAwareMaskSelection.html", BASE / "index.html"]:
        text = html_path.read_text()
        if "<h2>9. Failure Attribution: Which Step Failed?</h2>" in text:
            start = text.index("<h2>9. Failure Attribution: Which Step Failed?</h2>")
            next_heading = re.search(r"<h2[^>]*>", text[start + 1 :])
            end = start + 1 + next_heading.start() if next_heading else text.index("</body>", start)
            text = text[:start] + section + text[end:]
        elif "<h2>9. Interpretation</h2>" in text:
            text = text.replace("<h2>9. Interpretation</h2>", section + "<h2>10. Interpretation</h2>")
        else:
            text = text.replace("</body>", section + "</body>")
        html_path.write_text(text)
    print(BASE / "failure_attribution_by_class.csv")
    print(BASE / "Jun09_SkillRanker_ComponentAwareMaskSelection.html")


if __name__ == "__main__":
    main()
