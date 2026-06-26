#!/usr/bin/env python3
"""Add a class-specific iteration decision section to the Jun09 report."""

from __future__ import annotations

import csv
import html
import zipfile
from pathlib import Path


ROOT = Path("/Users/haoranwu/Desktop/Yan_Lab_Research/Medical-SAM3")
BASE = ROOT / "output/visium_hd_exp1/final_deliverables/Jun09_SkillRanker_ComponentAwareMaskSelection"
CLEAN = Path("/Users/haoranwu/Desktop/Yan_Lab_Research/VisiumHD-Segmentation-MaskSelection")
ALL48_UNION = ROOT / "output/visium_hd_exp1/final_deliverables/Jun07_best_setting_matrix/all48_union.csv"
if not ALL48_UNION.exists():
    ALL48_UNION = CLEAN / "output/visium_hd_exp1/final_deliverables/Jun03_mainline_all48_component_aware_union/all48_union.csv"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    cols: list[str] = []
    for row in rows:
        for key in row:
            if key not in cols:
                cols.append(key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=cols)
        writer.writeheader()
        writer.writerows(rows)


def table(rows: list[dict[str, object]], cols: list[str]) -> str:
    out = ["<table><thead><tr>"]
    for col in cols:
        out.append(f"<th>{html.escape(col)}</th>")
    out.append("</tr></thead><tbody>")
    for row in rows:
        out.append("<tr>")
        for col in cols:
            out.append(f"<td>{html.escape(str(row.get(col, '')))}</td>")
        out.append("</tr>")
    out.append("</tbody></table>")
    return "".join(out)


def dpr(row: dict[str, str], d: str = "dice", p: str = "precision", r: str = "recall") -> str:
    if not row:
        return ""
    return f"{float(row[d]):.3f} / {float(row[p]):.3f} / {float(row[r]):.3f}"


def main() -> None:
    expanded = {r["class"]: r for r in read_csv(BASE / "refined_diagnostics/expanded_pool_diagnostic.csv")}
    guarded = {r["class"].replace("_", " "): r for r in read_csv(BASE / "guarded_loco_variant/guarded_loco_summary.csv")}
    all48 = {}
    if ALL48_UNION.exists():
        for row in read_csv(ALL48_UNION):
            all48[row["class"]] = row

    rows = []
    specs = [
        (
            "bronchiola",
            "assembly / multi-component selection",
            "medpt24 has excellent pieces, but a deployable selector must choose the right 2-3 components without annotation.",
            "Keep skill assembly, but use class-specific thresholds plus a second-SAM/local-refinement option.",
        ),
        (
            "alveoli",
            "candidate generator, not ranker",
            "medpt24 full pool max component Dice is only 0.477 while all-48 has a 0.677 H&E mask.",
            "Do not keep tuning VLM/ranker for alveoli on medpt24; switch to a broader H&E box-style candidate generator.",
        ),
        (
            "vessels",
            "generalization precision",
            "Good vessel pieces exist, but LOCO selection admits non-vessel pieces unless an H&E morphology guard is used.",
            "Keep component-first scoring, add morphology/shape guards, and consider second-SAM from selected piece boxes.",
        ),
        (
            "tumor",
            "broad tissue mask mismatch",
            "Tumor is not a few clean small pieces; piece-first selection under-recovers the broad tumor area.",
            "Treat tumor separately with broader candidates or a semantic compartment mask instead of small-piece union only.",
        ),
        (
            "stroma",
            "diffuse class / assembly coverage",
            "High-Dice local stroma pieces exist, but selected unions cover only a tiny fraction of the diffuse annotation.",
            "Use stroma as a broad compartment problem; combine stromal prior with larger candidates and conservative false-positive control.",
        ),
        (
            "immune infiltration",
            "many-small-components coverage",
            "The signal is usable, but recall depends on selecting many small pieces without adding tumor/stroma false positives.",
            "Use expanded all-48/full-pool recall-push assembly with class-specific NMS and candidate audit.",
        ),
    ]
    for klass, limiting, evidence, next_step in specs:
        g = guarded.get(klass, {})
        a = all48.get(klass, {})
        e = expanded.get(klass, {})
        rows.append(
            {
                "class": klass,
                "medpt24 full max component Dice": e.get("full max component Dice", ""),
                "medpt24 guarded LOCO D/P/R": dpr(g) if g else "",
                "current all-48 best D/P/R": f"{float(a['Dice']):.3f} / {float(a['Precision']):.3f} / {float(a['Recall']):.3f}" if a else "",
                "limiting step": limiting,
                "evidence": evidence,
                "next iteration": next_step,
            }
        )

    out_csv = BASE / "refined_diagnostics/class_specific_iteration_plan.csv"
    write_csv(out_csv, rows)

    section = f"""
<h2>15. Class-Specific Iteration Plan: Stop Treating All Classes the Same</h2>
<p>This is the key failure-analysis conclusion. A single prompt, single model, or single assembly rule is not the right abstraction. The six tissue classes fail for different reasons, so the next useful method is class-specific: use a point/component strategy for small disconnected structures, but use broader candidates for diffuse tissue regions.</p>
{table(rows, ['class', 'medpt24 full max component Dice', 'medpt24 guarded LOCO D/P/R', 'current all-48 best D/P/R', 'limiting step', 'next iteration'])}
<div class='callout'><b>Paper-inspired interpretation.</b> SaLIP-style methods use SAM proposals as an intermediate step, then rerun/refine SAM around selected regions; RegionCLIP/Alpha-CLIP-style methods emphasize region-aware scoring rather than isolated crop classification. Our data agrees: VLM-only piece recognition is not stable enough, but skill features plus class-specific assembly can identify useful pieces for bronchiola/vessels/immune. Alveoli, tumor, and stroma need different candidate generators or broader compartment masks.</div>
<p class='small'>Decision table saved to <code>refined_diagnostics/class_specific_iteration_plan.csv</code>. This section is a decision rule for the next iteration, not a claim that the current skill-ranker is final.</p>
"""

    for html_path in [BASE / "Jun09_SkillRanker_ComponentAwareMaskSelection.html", BASE / "index.html"]:
        text = html_path.read_text()
        marker = "<h2>15. Class-Specific Iteration Plan: Stop Treating All Classes the Same</h2>"
        if marker in text:
            start = text.index(marker)
            end = text.index("<h2>15. Interpretation</h2>", start) if "<h2>15. Interpretation</h2>" in text[start:] else text.index("</body>", start)
            text = text[:start] + section + text[end:]
        elif "<h2>15. Interpretation</h2>" in text:
            text = text.replace("<h2>15. Interpretation</h2>", section + "<h2>16. Interpretation</h2>")
        elif "<h2>14. Interpretation</h2>" in text:
            text = text.replace("<h2>14. Interpretation</h2>", section + "<h2>16. Interpretation</h2>")
        else:
            text = text.replace("</body>", section + "</body>")
        html_path.write_text(text)

    zip_path = BASE / "Jun09_SkillRanker_ComponentAwareMaskSelection_shareable.zip"
    include: list[Path] = []
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
    for root in [
        BASE / "corrected_pool",
        BASE / "assembly_outputs/combined_skill_ranker_rf_plus_he/figures",
        BASE / "refined_diagnostics",
        BASE / "guarded_loco_variant",
    ]:
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
