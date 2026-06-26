#!/usr/bin/env python3
"""Add a decision matrix that chooses the next refinement target by class."""

from __future__ import annotations

import csv
import html
import zipfile
from pathlib import Path


ROOT = Path("/Users/haoranwu/Desktop/Yan_Lab_Research/Medical-SAM3")
BASE = ROOT / "output/visium_hd_exp1/final_deliverables/Jun09_SkillRanker_ComponentAwareMaskSelection"
REFINE_DIR = BASE / "refined_diagnostics"

CLASS_KEYS = ["bronchiola", "alveoli", "vessels", "tumor", "stroma", "immune_infiltration"]


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


def fmt_dpr(row: dict[str, str], prefix: str = "") -> str:
    if prefix:
        return f"{row[prefix + 'dice']} / {row[prefix + 'precision']} / {row[prefix + 'recall']}"
    return f"{float(row['dice']):.3f} / {float(row['precision']):.3f} / {float(row['recall']):.3f}"


def classify_limitation(c: str, current: dict[str, str], refined: dict[str, str], loco: dict[str, str]) -> tuple[str, str]:
    cur_d, cur_p, cur_r = float(current["dice"]), float(current["precision"]), float(current["recall"])
    ref_d, ref_p, ref_r = float(refined["refined_dice"]), float(refined["refined_precision"]), float(refined["refined_recall"])
    loc_d, loc_p, loc_r = float(loco["dice"]), float(loco["precision"]), float(loco["recall"])
    if c == "alveoli":
        return ("candidate scarcity / validation unavailable", "Increase alveoli candidate diversity before trusting a trained ranker; current 167 pool has only 5 alveoli pieces.")
    if ref_d - cur_d > 0.12 and loc_d >= cur_d - 0.03:
        return ("assembly policy is the main fix", "Replace the old conservative policy with class-specific score/margin/overlap settings and keep candidate-level audit.")
    if ref_d - cur_d > 0.12 and loc_d < cur_d - 0.08:
        return ("ranker stability + assembly", "The pool can do better, but LOCO drops; use interpretable morphology filters or more annotated ROIs before claiming a learned ranker.")
    if ref_d < 0.45 and ref_r < 0.45:
        return ("candidate pool / tissue definition", "The current pieces do not represent this broad tissue well enough; generate broader candidates or treat this class separately.")
    if loc_p < 0.45 and ref_p > 0.80:
        return ("generalization precision loss", "The scoring signal exists but precision is unstable; add a morphology guard before assembly.")
    return ("usable but not final", "Keep this as a working baseline, then validate on another ROI or sample before using it as the method.")


def main() -> None:
    current_rows = [row for row in read_csv(BASE / "assembly_summary_all_methods.csv") if row["method"] == "combined_skill_ranker_rf_plus_he"]
    refined_rows = read_csv(REFINE_DIR / "refined_policy_summary_by_class.csv")
    loco_rows = read_csv(BASE / "assembly_outputs/loco_component_logistic_validation/assembly_summary.csv")
    current = {row["class"]: row for row in current_rows}
    refined = {row["class"]: row for row in refined_rows}
    loco = {row["class"]: row for row in loco_rows}
    matrix: list[dict[str, object]] = []
    for c in CLASS_KEYS:
        limit, action = classify_limitation(c, current[c], refined[c], loco[c])
        matrix.append(
            {
                "class": c,
                "current in-sample D/P/R": fmt_dpr(current[c]),
                "refined diagnostic D/P/R": fmt_dpr(refined[c], "refined_"),
                "LOCO validation D/P/R": fmt_dpr(loco[c]),
                "limiting step": limit,
                "next refinement": action,
            }
        )
    write_csv(REFINE_DIR / "decision_matrix_by_class.csv", matrix)
    section = f"""
<h2>12. Decision Matrix: What Should Change Next?</h2>
<p>This table compares three views: the original in-sample skill ranker, an annotation-calibrated diagnostic policy, and leave-one-component-out validation. A class is considered usable only if the improvement does not disappear under LOCO.</p>
{html_table(matrix, ['class', 'current in-sample D/P/R', 'refined diagnostic D/P/R', 'LOCO validation D/P/R', 'limiting step', 'next refinement'])}
<div class='callout'><b>Current conclusion.</b> Bronchiola is close to usable after policy refinement. Vessels has strong in-sample and diagnostic potential, but LOCO shows precision instability. Immune infiltration has a real signal but needs recall-oriented assembly. Alveoli needs more candidate diversity. Stroma and tumor should be treated as broad/diffuse classes, not as the main proof point for piece-first assembly.</div>
<p class='small'>Paper framing used for this diagnosis: SaLIP supports proposal-rerank-second-SAM thinking; RegionCLIP and Alpha-CLIP motivate region-aware rather than whole-image scoring; guided cropping work motivates separating object/candidate focus from surrounding context.</p>
"""
    for html_path in [BASE / "Jun09_SkillRanker_ComponentAwareMaskSelection.html", BASE / "index.html"]:
        text = html_path.read_text()
        marker = "<h2>12. Decision Matrix: What Should Change Next?</h2>"
        if marker in text:
            start = text.index(marker)
            end = text.index("<h2>12. Interpretation</h2>", start) if "<h2>12. Interpretation</h2>" in text[start:] else text.index("</body>", start)
            text = text[:start] + section + text[end:]
        elif "<h2>12. Interpretation</h2>" in text:
            text = text.replace("<h2>12. Interpretation</h2>", section + "<h2>13. Interpretation</h2>")
        else:
            text = text.replace("</body>", section + "</body>")
        html_path.write_text(text)
    zip_path = BASE / "Jun09_SkillRanker_ComponentAwareMaskSelection_shareable.zip"
    include = []
    for rel in [
        "Jun09_SkillRanker_ComponentAwareMaskSelection.html",
        "index.html",
        "candidate_skill_features.csv",
        "piece_top1_all_methods.csv",
        "assembly_summary_all_methods.csv",
        "selected_pieces_all_methods.csv",
        "failure_attribution_by_class.csv",
        "run_config.json",
    ]:
        include.append(BASE / rel)
    for root in [BASE / "corrected_pool", BASE / "assembly_outputs/combined_skill_ranker_rf_plus_he/figures", REFINE_DIR]:
        for p in root.rglob("*"):
            if p.is_file():
                include.append(p)
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as handle:
        for p in include:
            handle.write(p, p.relative_to(BASE))
    print(REFINE_DIR / "decision_matrix_by_class.csv")
    print(zip_path)


if __name__ == "__main__":
    main()
