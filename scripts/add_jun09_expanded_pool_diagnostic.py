#!/usr/bin/env python3
"""Add expanded-pool diagnostic notes to Jun09 report."""

from __future__ import annotations

import csv
import html
import zipfile
from collections import defaultdict
from pathlib import Path


ROOT = Path("/Users/haoranwu/Desktop/Yan_Lab_Research/Medical-SAM3")
BASE = ROOT / "output/visium_hd_exp1/final_deliverables/Jun09_SkillRanker_ComponentAwareMaskSelection"
CLEAN = Path("/Users/haoranwu/Desktop/Yan_Lab_Research/VisiumHD-Segmentation-MaskSelection")
FULL_CLUSTER = CLEAN / "output/visium_hd_exp1/final_deliverables/Jun03_medpt24_clustered_iou_component_aware_union/cluster_component_scores/tables/candidate_component_scores.csv"
FULL_UNION = CLEAN / "output/visium_hd_exp1/final_deliverables/Jun03_medpt24_clustered_iou_component_aware_union/component_aware_union/tables/chosen_union_summary.csv"

CLASS_MAP = {
    "lung_bronchiola": "bronchiola",
    "lung_alveoli_normal_adjacent": "alveoli",
    "lung_vessels": "vessels",
    "tumor": "tumor",
    "stroma": "stroma",
    "immune_infiltration": "immune infiltration",
}
LONG = {
    "lung_bronchiola": "bronchiola",
    "lung_alveoli_normal_adjacent": "alveoli",
    "lung_vessels": "vessels",
    "tumor": "tumor",
    "stroma": "stroma",
    "immune_infiltration": "immune infiltration",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    keys = []
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
    out.extend(f"<th>{html.escape(c)}</th>" for c in cols)
    out.append("</tr></thead><tbody>")
    for row in rows:
        out.append("<tr>")
        for col in cols:
            out.append(f"<td>{html.escape(str(row.get(col, '')))}</td>")
        out.append("</tr>")
    out.append("</tbody></table>")
    return "".join(out)


def main() -> None:
    compact = read_csv(BASE / "corrected_pool/hidden_candidate_truth.csv")
    full = read_csv(FULL_CLUSTER)
    union = {LONG[row["label"]]: row for row in read_csv(FULL_UNION)}

    compact_by_class: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in compact:
        compact_by_class[LONG[row["classification_true_label"]]].append(row)
    full_by_class: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in full:
        full_by_class[LONG[row["label"]]].append(row)

    rows = []
    for label, display in CLASS_MAP.items():
        compact_rows = compact_by_class[display]
        full_rows = full_by_class[display]
        top_full = sorted(full_rows, key=lambda r: float(r["component_dice"]), reverse=True)[:5]
        rows.append(
            {
                "class": display,
                "compact167 true pieces": len(compact_rows),
                "full clustered candidates": len({r["candidate_uid"] for r in full_rows}),
                "compact max component Dice": f"{max(float(r['component_dice']) for r in compact_rows):.3f}" if compact_rows else "0.000",
                "full max component Dice": f"{max(float(r['component_dice']) for r in full_rows):.3f}" if full_rows else "0.000",
                "full oracle union D/P/R": f"{float(union[display]['dice']):.3f} / {float(union[display]['precision']):.3f} / {float(union[display]['recall']):.3f}" if display in union else "",
                "top full candidates": "; ".join(f"{r['candidate_uid']} D={float(r['component_dice']):.3f} P={float(r['component_precision']):.3f} R={float(r['component_recall']):.3f}" for r in top_full),
            }
        )
    out_csv = BASE / "refined_diagnostics/expanded_pool_diagnostic.csv"
    write_csv(out_csv, rows)
    section = f"""
<h2>14. Expanded Pool Diagnostic: Do We Need More Than 167 Pieces?</h2>
<p>The 167-piece funnel is useful for fast VLM/ranker experiments, but it is not the final candidate universe. The full clustered <code>medical_official_points_step24</code> pool has about 4,961 clustered candidates. This diagnostic checks whether the compact pool lost important pieces, and whether the next iteration should expand the funnel before more prompt/ranker tuning.</p>
{html_table(rows, ['class', 'compact167 true pieces', 'full clustered candidates', 'compact max component Dice', 'full max component Dice', 'full oracle union D/P/R'])}
<div class='callout'><b>Interpretation.</b> The compact pool kept the best single component for most classes, so the failure is not simply "no good piece exists." The harder problem is ranking and assembling multiple pieces without annotation. However, 167 is still too narrow for robust learning and validation, especially for alveoli and broad/diffuse classes. The next useful branch is an expanded skill-funnel of roughly 500-800 candidates from the full clustered pool, with diversity constraints and no raw FICTURE-image prompting.</div>
<p class='small'>Expanded-pool candidate details are saved in <code>refined_diagnostics/expanded_pool_diagnostic.csv</code>.</p>
"""
    for html_path in [BASE / "Jun09_SkillRanker_ComponentAwareMaskSelection.html", BASE / "index.html"]:
        text = html_path.read_text()
        marker = "<h2>14. Expanded Pool Diagnostic: Do We Need More Than 167 Pieces?</h2>"
        if marker in text:
            start = text.index(marker)
            end = text.index("<h2>14. Interpretation</h2>", start) if "<h2>14. Interpretation</h2>" in text[start:] else text.index("</body>", start)
            text = text[:start] + section + text[end:]
        elif "<h2>14. Interpretation</h2>" in text:
            text = text.replace("<h2>14. Interpretation</h2>", section + "<h2>15. Interpretation</h2>")
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
    for root in [BASE / "corrected_pool", BASE / "assembly_outputs/combined_skill_ranker_rf_plus_he/figures", BASE / "refined_diagnostics", BASE / "guarded_loco_variant"]:
        for p in root.rglob("*"):
            if p.is_file():
                include.append(p)
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as handle:
        for p in include:
            handle.write(p, p.relative_to(BASE))
    print(out_csv)
    print(zip_path)


if __name__ == "__main__":
    main()
