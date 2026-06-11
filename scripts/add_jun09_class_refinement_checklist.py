#!/usr/bin/env python3
"""Append a class-specific refinement checklist to the Jun09 report."""

from __future__ import annotations

import csv
import html
import re
import zipfile
from pathlib import Path


ROOT = Path("/Users/haoranwu/Desktop/Yan_Lab_Research/Medical-SAM3")
BASE = ROOT / "output/visium_hd_exp1/final_deliverables/Jun09_SkillRanker_ComponentAwareMaskSelection"
REFINE = BASE / "refined_diagnostics"
PACK = BASE / "region_aware_piece_context_locator_pack"
RESULTS = BASE / "region_aware_vlm_results"

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
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
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


def replace_section(text: str, title: str, section: str) -> str:
    marker = f"<h2>{title}</h2>"
    if marker in text:
        start = text.index(marker)
        next_heading = re.search(r"<h2[^>]*>", text[start + 1 :])
        end = start + 1 + next_heading.start() if next_heading else text.index("</body>", start)
        return text[:start] + section + text[end:]
    return text.replace("</body>", section + "</body>")


def fmt_refined(row: dict[str, str]) -> str:
    return f"{row.get('refined_dice', '')} / {row.get('refined_precision', '')} / {row.get('refined_recall', '')}"


def main() -> None:
    refined = {row["class"]: row for row in read_csv(REFINE / "refined_policy_summary_by_class.csv")}
    decision_path = REFINE / "decision_matrix_by_class.csv"
    decision = {row["class"]: row for row in read_csv(decision_path)} if decision_path.exists() else {}
    plan_path = REFINE / "class_specific_iteration_plan.csv"
    plan = {row["class"]: row for row in read_csv(plan_path)} if plan_path.exists() else {}

    manual = {
        "bronchiola": {
            "next_change": "Keep component pieces but replace the old conservative assembly with class-specific score/margin/overlap thresholds; then test second-SAM from selected component boxes.",
            "pass_condition": "D/P/R stays around 0.76/0.89/0.66 or better under LOCO and region-aware VLM does not demote the two main components.",
        },
        "alveoli": {
            "next_change": "Stop tuning VLM on the 5-piece medpt24 funnel; add broader H&E box-style candidates first.",
            "pass_condition": "Candidate-generator upper bound reaches at least the earlier all-48 alveoli best range before any ranker/VLM claim.",
        },
        "vessels": {
            "next_change": "Keep component-first scoring, but add lumen/wall/elongation morphology guard before assembly and compare with region-aware all6 VLM.",
            "pass_condition": "LOCO precision no longer collapses while recall remains near current in-sample recall.",
        },
        "tumor": {
            "next_change": "Treat tumor as broad tissue, not only small pieces; test broader candidates or semantic compartment masks.",
            "pass_condition": "Broad-candidate upper bound improves over the piece-only refined D/P/R without a major precision loss.",
        },
        "stroma": {
            "next_change": "Use broader stromal candidates plus structured FICTURE/shape prior; piece-only assembly is too fragmented for diffuse stroma.",
            "pass_condition": "Recall rises substantially above the old conservative assembly while precision stays above the current broad-class guard.",
        },
        "immune_infiltration": {
            "next_change": "Use many-small-component recall-push assembly with class-specific NMS; audit selected and rejected pieces to prevent tumor/stroma leakage.",
            "pass_condition": "D/P/R moves toward the all-48/full-pool range and selected pieces cover more immune components without tumor collapse.",
        },
    }

    rows: list[dict[str, object]] = []
    for c in CLASS_KEYS:
        r = refined.get(c, {})
        d = decision.get(c, {})
        p = plan.get(c, {})
        rows.append(
            {
                "class": c,
                "diagnosed layer": r.get("diagnosed_limiting_layer") or d.get("limiting step", ""),
                "current evidence": (
                    f"refined D/P/R {fmt_refined(r)}; "
                    f"components covered {r.get('pool_components_with_best_recall_ge_0.10', '')}/{r.get('annotation_components', '')}; "
                    f"piece Top1 {r.get('piece_top1', '')}"
                ),
                "next change": manual[c]["next_change"],
                "pass condition": manual[c]["pass_condition"],
                "do not do": {
                    "bronchiola": "Do not score only a union mask; the useful evidence is component-level.",
                    "alveoli": "Do not spend API/VLM runs before candidate upper bound is repaired.",
                    "vessels": "Do not allow any high vessel score without a morphology/context guard.",
                    "tumor": "Do not use small-piece union as the only tumor method.",
                    "stroma": "Do not judge diffuse stroma only by isolated pieces.",
                    "immune_infiltration": "Do not select every immune-colored or high-score speckle without NMS/audit.",
                }[c],
                "existing plan note": p.get("next iteration", ""),
            }
        )
    out_csv = REFINE / "class_specific_refinement_checklist.csv"
    write_csv(out_csv, rows)

    section = f"""
<h2>17D. Class-Specific Refinement Checklist</h2>
<p>This checklist is the current working rule for the next iteration. It prevents the project from repeatedly changing only prompts when the actual failing layer is different for each tissue class.</p>
{table(rows, ['class', 'diagnosed layer', 'current evidence', 'next change', 'pass condition', 'do not do'])}
<div class='callout'><b>Use this as the iteration gate.</b> A class can move forward only after its pass condition is checked with figures and D/P/R. If a class fails, change the layer named in <code>diagnosed layer</code>, not a random prompt.</div>
"""
    for html_path in [BASE / "Jun09_SkillRanker_ComponentAwareMaskSelection.html", BASE / "index.html"]:
        text = html_path.read_text()
        html_path.write_text(replace_section(text, "17D. Class-Specific Refinement Checklist", section))

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
        p = BASE / rel
        if p.exists():
            include.append(p)
    for root in [
        BASE / "corrected_pool",
        BASE / "assembly_outputs/combined_skill_ranker_rf_plus_he/figures",
        REFINE,
        BASE / "guarded_loco_variant",
        PACK,
        RESULTS,
        ROOT / "scripts/hpc/jun09_region_aware_multiimage_vlm.sbatch",
        ROOT / "scripts/hpc/jun09_region_aware_piece_assembly_cpu.sbatch",
        ROOT / "scripts/collect_jun09_region_aware_vlm_results.py",
        ROOT / "scripts/add_jun09_class_refinement_checklist.py",
    ]:
        if root.is_file():
            include.append(root)
        elif root.exists():
            include.extend(p for p in root.rglob("*") if p.is_file())
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as handle:
        seen: set[Path] = set()
        for p in include:
            if p in seen:
                continue
            seen.add(p)
            try:
                arcname = p.relative_to(BASE)
            except ValueError:
                arcname = Path("support_scripts") / p.relative_to(ROOT)
            handle.write(p, arcname)

    print(out_csv)
    print(BASE / "Jun09_SkillRanker_ComponentAwareMaskSelection.html")
    print(zip_path)


if __name__ == "__main__":
    main()
