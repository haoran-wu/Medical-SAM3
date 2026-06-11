#!/usr/bin/env python3
"""Append a component-level coverage gap audit to the Jun09 report.

Whole-class Dice can hide the real failure.  This audit splits each annotation
mask into connected components, then asks three questions per component:

1. Is there any candidate piece in the current funnel that overlaps it well?
2. Was such a piece selected by the current runtime policy for the target class?
3. If selected, is the remaining issue mostly boundary/refinement?
"""

from __future__ import annotations

import csv
import html
import re
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from scipy import ndimage


ROOT = Path("/Users/haoranwu/Desktop/Yan_Lab_Research/Medical-SAM3")
BASE = ROOT / "output/visium_hd_exp1/final_deliverables/Jun09_SkillRanker_ComponentAwareMaskSelection"
OUT = BASE / "component_coverage_gap_audit"
ROI_ROOT = ROOT / "output/visium_hd_exp1/ficture_official_filtered_candidate_pool_input_roi"
PACK = BASE / "runtime_policy_second_sam_locator_pack"
PIECE_DIR = PACK / "piece_masks"
SELECTED_CSV = PACK / "runtime_policy_selected_pieces_for_second_sam.csv"

CLASSES = {
    "bronchiola": ROI_ROOT / "cropped_annotation_masks/01_lung_bronchiola_target_roi.png",
    "alveoli": ROI_ROOT / "cropped_annotation_masks/04_lung_alveoli_normal_adjacent_target_roi.png",
    "vessels": ROI_ROOT / "cropped_annotation_masks/05_lung_vessels_target_roi.png",
    "tumor": ROI_ROOT / "cropped_annotation_masks/08_tumor_target_roi.png",
    "stroma": ROI_ROOT / "cropped_annotation_masks/07_stroma_target_roi.png",
    "immune_infiltration": ROI_ROOT / "cropped_annotation_masks/03_immune_infiltration_target_roi.png",
}

MIN_COMPONENT_PIXELS = 200
CANDIDATE_SUPPORT_DICE = 0.20
SELECTED_SUPPORT_DICE = 0.20
HIGH_RECALL_SUPPORT = 0.50
LOW_PRECISION_CUTOFF = 0.20
PARTIAL_RECALL_SUPPORT = 0.20


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


def load_mask(path: Path, shape: tuple[int, int] | None = None) -> np.ndarray:
    im = Image.open(path).convert("L")
    if shape and (im.height, im.width) != shape:
        im = im.resize((shape[1], shape[0]), Image.Resampling.NEAREST)
    return np.asarray(im) > 0


def candidate_metrics_for_components(
    labels: np.ndarray,
    component_areas: np.ndarray,
    candidate_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    cand_area = int(candidate_mask.sum())
    if cand_area == 0:
        n = len(component_areas) - 1
        return np.zeros(n + 1), np.zeros(n + 1), np.zeros(n + 1)
    overlaps = np.bincount(labels[candidate_mask].reshape(-1), minlength=len(component_areas))
    dice = np.zeros_like(component_areas, dtype=float)
    precision = np.zeros_like(component_areas, dtype=float)
    recall = np.zeros_like(component_areas, dtype=float)
    valid = component_areas > 0
    dice[valid] = 2 * overlaps[valid] / (cand_area + component_areas[valid])
    precision[valid] = overlaps[valid] / cand_area
    recall[valid] = overlaps[valid] / component_areas[valid]
    return dice, precision, recall


def selected_map() -> dict[str, set[str]]:
    df = pd.read_csv(SELECTED_CSV)
    out: dict[str, set[str]] = {}
    for cls, sub in df.groupby("class"):
        out[str(cls)] = set(str(v) for v in sub["candidate_uid"])
    return out


def replace_or_append(html_path: Path, section: str) -> None:
    text = html_path.read_text()
    marker = "<h2>17AU. Component Coverage Gap Audit</h2>"
    if marker in text:
        start = text.index(marker)
        next_match = re.search(r"<h2>17A[V-Z]|<h2>18\\.", text[start + len(marker) :])
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
        "component_coverage_gap_audit/component_coverage_gap_summary.csv",
        "component_coverage_gap_audit/component_coverage_gap_details.csv",
    ]
    missing = [name for name in required if name not in names]
    if missing:
        raise RuntimeError(f"zip missing required component gap files: {missing}")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    piece_paths = sorted(PIECE_DIR.glob("*.png"))
    if not piece_paths:
        raise FileNotFoundError(PIECE_DIR)
    selected_by_class = selected_map()

    detail_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []

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
                "best_candidate_uid": "",
                "best_candidate_Dice": 0.0,
                "best_candidate_Precision": 0.0,
                "best_candidate_Recall": 0.0,
                "best_selected_uid": "",
                "best_selected_Dice": 0.0,
                "best_selected_Precision": 0.0,
                "best_selected_Recall": 0.0,
            }
            for cid in component_ids
        }

        for path in piece_paths:
            uid = path.stem
            mask = load_mask(path, ann.shape)
            dice, precision, recall = candidate_metrics_for_components(labels, component_areas, mask)
            for cid in component_ids:
                if dice[cid] > float(best[cid]["best_candidate_Dice"]):
                    best[cid].update(
                        {
                            "best_candidate_uid": uid,
                            "best_candidate_Dice": round(float(dice[cid]), 4),
                            "best_candidate_Precision": round(float(precision[cid]), 4),
                            "best_candidate_Recall": round(float(recall[cid]), 4),
                        }
                    )
                if uid in selected_by_class.get(cls, set()) and dice[cid] > float(best[cid]["best_selected_Dice"]):
                    best[cid].update(
                        {
                            "best_selected_uid": uid,
                            "best_selected_Dice": round(float(dice[cid]), 4),
                            "best_selected_Precision": round(float(precision[cid]), 4),
                            "best_selected_Recall": round(float(recall[cid]), 4),
                        }
                    )

        rows = []
        for cid in component_ids:
            row = best[cid]
            best_dice = float(row["best_candidate_Dice"])
            best_precision = float(row["best_candidate_Precision"])
            best_recall = float(row["best_candidate_Recall"])
            selected_dice = float(row["best_selected_Dice"])
            selected_precision = float(row["best_selected_Precision"])
            selected_recall = float(row["best_selected_Recall"])
            has_candidate = best_dice >= CANDIDATE_SUPPORT_DICE
            has_selected = selected_dice >= SELECTED_SUPPORT_DICE
            if has_selected:
                failure = "covered_by_current_selection"
            elif has_candidate:
                failure = "selection_gap_candidate_exists_not_selected"
            elif selected_recall >= HIGH_RECALL_SUPPORT and selected_precision < LOW_PRECISION_CUTOFF:
                failure = "selected_but_too_broad_quality_gap"
            elif best_recall >= HIGH_RECALL_SUPPORT and best_precision < LOW_PRECISION_CUTOFF:
                failure = "candidate_exists_but_too_broad_quality_gap"
            elif best_recall >= PARTIAL_RECALL_SUPPORT:
                failure = "partial_candidate_gap"
            else:
                failure = "proposal_gap_no_local_piece"
            row["has_supported_candidate"] = has_candidate
            row["has_supported_selected_piece"] = has_selected
            row["failure_type"] = failure
            rows.append(row)
            detail_rows.append(row)

        n = len(rows)
        n_candidate = sum(1 for r in rows if r["has_supported_candidate"])
        n_selected = sum(1 for r in rows if r["has_supported_selected_piece"])
        n_selection_gap = sum(1 for r in rows if r["failure_type"] == "selection_gap_candidate_exists_not_selected")
        n_broad_gap = sum(
            1
            for r in rows
            if r["failure_type"] in {"candidate_exists_but_too_broad_quality_gap", "selected_but_too_broad_quality_gap"}
        )
        n_partial_gap = sum(1 for r in rows if r["failure_type"] == "partial_candidate_gap")
        n_no_piece_gap = sum(1 for r in rows if r["failure_type"] == "proposal_gap_no_local_piece")
        n_proposal_gap = n_broad_gap + n_partial_gap + n_no_piece_gap
        largest_gaps = sorted(
            [r for r in rows if r["failure_type"] != "covered_by_current_selection"],
            key=lambda r: int(r["component_pixels"]),
            reverse=True,
        )[:5]
        summary_rows.append(
            {
                "class": cls,
                "components_area_ge_200": n,
                "components_with_supported_candidate": n_candidate,
                "components_covered_by_current_selection": n_selected,
                "selection_gap_components": n_selection_gap,
                "too_broad_quality_gap_components": n_broad_gap,
                "partial_candidate_gap_components": n_partial_gap,
                "no_local_piece_gap_components": n_no_piece_gap,
                "proposal_or_quality_gap_components": n_proposal_gap,
                "largest_gap_component_ids": ", ".join(str(r["component_id"]) for r in largest_gaps),
                "interpretation": (
                    "selection is the bottleneck"
                    if n_selection_gap > n_proposal_gap
                    else "quality/boundary is the bottleneck"
                    if n_broad_gap > max(n_partial_gap, n_no_piece_gap)
                    else "proposal/funnel coverage is the bottleneck"
                    if n_partial_gap or n_no_piece_gap
                    else "mostly boundary/assembly after selection"
                ),
            }
        )

    write_csv(OUT / "component_coverage_gap_details.csv", detail_rows)
    write_csv(OUT / "component_coverage_gap_summary.csv", summary_rows)

    focus_rows = []
    for row in detail_rows:
        if row["failure_type"] != "covered_by_current_selection":
            focus_rows.append(row)
    focus_rows = sorted(focus_rows, key=lambda r: (str(r["class"]), -int(r["component_pixels"])))[:80]

    section = f"""
<h2>17AU. Component Coverage Gap Audit</h2>
<p><b>Goal.</b> This is the more precise failure microscope.  Instead of asking only whether the whole class mask has a good Dice score, it splits each annotation into connected components and checks whether each component has a candidate piece and whether that piece was selected.</p>
<p><b>Definitions.</b> A <b>proposal gap</b> means the current 167-piece funnel does not contain a good piece for that annotation component.  A <b>selection gap</b> means a good piece exists in the funnel, but the runtime skill/ranker did not select it.  A <b>boundary/assembly gap</b> means the component is selected, but the final shape still needs boundary refinement or second-SAM.</p>
<p><b>Hidden ground truth use.</b> Annotation components are used only in this audit after scoring, never in the model prompt or runtime selection rule.</p>
{table(summary_rows, ['class', 'components_area_ge_200', 'components_with_supported_candidate', 'components_covered_by_current_selection', 'selection_gap_components', 'too_broad_quality_gap_components', 'partial_candidate_gap_components', 'no_local_piece_gap_components', 'proposal_or_quality_gap_components', 'largest_gap_component_ids', 'interpretation'])}
<h3>Largest uncovered or unselected components</h3>
{table(focus_rows, ['class', 'component_id', 'component_pixels', 'failure_type', 'best_candidate_uid', 'best_candidate_Dice', 'best_candidate_Precision', 'best_candidate_Recall', 'best_selected_uid', 'best_selected_Dice', 'best_selected_Precision', 'best_selected_Recall'])}
<p><b>Iteration rule.</b> If a component has a supported candidate but no selected piece, refine the skill/ranker.  If no supported candidate exists, refine the funnel/proposal generator.  If it is already selected, move to boundary refinement or second-SAM.</p>
"""

    for html_path in [BASE / "Jun09_SkillRanker_ComponentAwareMaskSelection.html", BASE / "index.html"]:
        replace_or_append(html_path, section)
        verify_html_images(html_path)
    rebuild_zip()
    print(f"Wrote {OUT / 'component_coverage_gap_summary.csv'}")
    print(f"Updated {BASE / 'Jun09_SkillRanker_ComponentAwareMaskSelection.html'}")


if __name__ == "__main__":
    main()
