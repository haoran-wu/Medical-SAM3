#!/usr/bin/env python3
"""Add a class-specific generator gate to the Jun09 skill-ranker report.

The skill should not keep tuning prompts when the candidate generator is the
limiting layer.  This section converts the current evidence into executable
per-class generator decisions.
"""

from __future__ import annotations

import csv
import html
import re
import zipfile
from pathlib import Path


ROOT = Path("/Users/haoranwu/Desktop/Yan_Lab_Research/Medical-SAM3")
BASE = ROOT / "output/visium_hd_exp1/final_deliverables/Jun09_SkillRanker_ComponentAwareMaskSelection"
REF = BASE / "refined_diagnostics"

CLASS_ORDER = ["bronchiola", "alveoli", "vessels", "tumor", "stroma", "immune_infiltration"]


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


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
    parts = ["<table><thead><tr>"]
    parts.extend(f"<th>{html.escape(col)}</th>" for col in cols)
    parts.append("</tr></thead><tbody>")
    for row in rows:
        parts.append("<tr>")
        for col in cols:
            parts.append(f"<td>{html.escape(str(row.get(col, '')))}</td>")
        parts.append("</tr>")
    parts.append("</tbody></table>")
    return "".join(parts)


def fnum(value: str | float | int, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def by_class(rows: list[dict[str, str]], key: str = "class") -> dict[str, list[dict[str, str]]]:
    out: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        out.setdefault(row.get(key, ""), []).append(row)
    return out


def first_by_class(rows: list[dict[str, str]], key: str = "class") -> dict[str, dict[str, str]]:
    return {row[key]: row for row in rows if row.get(key)}


def build_gate_rows() -> list[dict[str, object]]:
    top = by_class(read_csv(REF / "minimal_setting_top_by_class.csv"))
    expanded = first_by_class(read_csv(REF / "expanded_pool_diagnostic.csv"))
    iteration = first_by_class(read_csv(REF / "class_specific_iteration_plan.csv"))
    microscope = first_by_class(read_csv(REF / "failure_microscope_by_class.csv"))

    manual = {
        "bronchiola": {
            "generator decision": "keep medpt24 point pieces; run second-SAM focus",
            "candidate settings to test next": "medical_official_points_step24",
            "why": "Best pieces already strong; failure is assembly/refinement, not generator search.",
            "pass condition": "second-SAM or refined assembly keeps D/P/R near 0.77/0.89/0.67 or improves recall without precision collapse.",
            "if fails": "keep selected-piece union; tune assembly/NMS before adding more settings.",
        },
        "alveoli": {
            "generator decision": "switch to broad H&E box generator",
            "candidate settings to test next": "base_box384_s128_m1536, base_official_boxes_b384_s96",
            "why": "point pieces cap at low Dice; broad alveolar region needs box-style candidate.",
            "pass condition": "component or union Dice reaches the all-48 alveoli range around 0.67 with acceptable precision.",
            "if fails": "try 512-box setting or class-specific alveoli morphology, not VLM prompt changes.",
        },
        "vessels": {
            "generator decision": "keep component-first point pieces; add morphology guard and second-SAM",
            "candidate settings to test next": "medical_official_points_step24, medical_box512_s160_m2048",
            "why": "good vessel pieces exist, but held-out precision is unstable.",
            "pass condition": "LOCO precision stabilizes while recall remains near current; second-SAM does not leak from vessel walls.",
            "if fails": "add lumen/wall shape guard or use box512 only for large vessel components.",
        },
        "tumor": {
            "generator decision": "treat as broad tissue, not small-piece union only",
            "candidate settings to test next": "base_official_points_step24 plus semantic tumor compartment mask / broader boxes",
            "why": "small pieces can be recognized but do not cover enough tumor annotation.",
            "pass condition": "broad-candidate union improves recall and Dice over 0.41 without precision falling below the current broad guard.",
            "if fails": "separate tumor as a compartment prior task rather than a piece-first VLM task.",
        },
        "stroma": {
            "generator decision": "treat as diffuse compartment with broader candidates",
            "candidate settings to test next": "base_official_points_step24, medical_official_points_step24, stromal semantic compartment prior",
            "why": "high-quality local stroma pieces exist but union coverage is too fragmented.",
            "pass condition": "recall rises substantially above 0.34 while precision stays near or above 0.45.",
            "if fails": "use stroma as a tissue-background compartment, not an object-like mask.",
        },
        "immune_infiltration": {
            "generator decision": "recall-push full/expanded component pool with strict NMS audit",
            "candidate settings to test next": "base_official_points_step24, medical_official_points_step24, all-48 recall-push reference",
            "why": "many small components; medpt24 compact pool under-covers components.",
            "pass condition": "move toward all-48 immune D/P/R around 0.60/0.65/0.56 without selecting tumor/stroma leakage.",
            "if fails": "increase candidate diversity and audit rejected/selected small components before changing VLM.",
        },
    }

    rows: list[dict[str, object]] = []
    for cls in CLASS_ORDER:
        top_rows = top.get(cls if cls != "immune_infiltration" else "immune infiltration", []) or top.get(cls, [])
        top3 = "; ".join(
            f"{r.get('setting')} ({r.get('best component D/P/R')}, {r.get('source')})"
            for r in top_rows[:3]
        )
        ex = expanded.get(cls if cls != "immune_infiltration" else "immune infiltration", {}) or expanded.get(cls, {})
        it = iteration.get(cls, {})
        mi = microscope.get(cls, {})
        row = {
            "class": cls,
            "limiting layer": mi.get("root cause", it.get("limiting step", "")),
            "current evidence": (
                f"compact max {ex.get('compact max component Dice', '')}; "
                f"full max {ex.get('full max component Dice', '')}; "
                f"current class plan: {it.get('limiting step', '')}"
            ),
            "top available settings": top3,
            **manual[cls],
        }
        rows.append(row)
    return rows


def build_remote_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    return [
        {
            "priority": 1,
            "job": "minimal three-setting component oracle",
            "script": "scripts/hpc/jun09_minimal_three_setting_component_oracle.sbatch",
            "purpose": "Verify a deployable small generator set before all-48 fallback.",
            "run when": "Bouchet SSH works; CPU/day_amd or faster CPU partition.",
        },
        {
            "priority": 2,
            "job": "second-SAM focus",
            "script": "scripts/hpc/jun09_selected_piece_second_sam_refine.sbatch",
            "purpose": "Test whether bronchiola/vessels selected pieces are good prompts for SAM3.",
            "run when": "GPU available; use LABELS=bronchiola,vessels,immune_infiltration first.",
        },
        {
            "priority": 3,
            "job": "alveoli broad box oracle",
            "script": "componentwise_candidate_oracle.py with base_box384_s128_m1536 and base_official_boxes_b384_s96 roots",
            "purpose": "Repair alveoli generator upper bound before more VLM/ranker work.",
            "run when": "After minimal-three result confirms medpt24 alone remains weak for alveoli.",
        },
        {
            "priority": 4,
            "job": "broad compartment branch for tumor/stroma",
            "script": "new branch to combine broader boxes plus structured FICTURE composition prior",
            "purpose": "Do not force diffuse classes into object-like small-piece assembly.",
            "run when": "After second-SAM/component branches establish piece-first limits.",
        },
    ]


def replace_section(text: str, title: str, section: str) -> str:
    marker = f"<h2>{title}</h2>"
    if marker in text:
        start = text.index(marker)
        match = re.search(r"<h2[^>]*>", text[start + 1 :])
        end = start + 1 + match.start() if match else text.index("</body>", start)
        return text[:start] + section + text[end:]
    return text.replace("</body>", section + "</body>")


def update_zip() -> None:
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
        BASE / "refined_diagnostics",
        BASE / "guarded_loco_variant",
        BASE / "region_aware_piece_context_locator_pack",
        BASE / "region_aware_vlm_results",
        BASE / "second_sam_refinement",
        BASE / "run_bouchet_jun09_next_generator_jobs.sh",
        ROOT / "scripts/add_jun09_class_specific_generator_gate.py",
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


def main() -> None:
    rows = build_gate_rows()
    remote_rows = build_remote_rows(rows)
    write_csv(REF / "class_specific_generator_gate.csv", rows)
    write_csv(REF / "next_remote_generator_jobs.csv", remote_rows)
    section = f"""
<h2>17G. Class-Specific Generator Gate</h2>
<p>This is the next practical branch after the failure microscope. The skill should not use one candidate generator for every tissue type. Some classes are object-like components, while others are broad or diffuse compartments. This gate says which generator to test next for each class and what would count as success.</p>
{table(rows, ['class', 'limiting layer', 'current evidence', 'top available settings', 'generator decision', 'candidate settings to test next', 'pass condition', 'if fails'])}
<h3>Next executable jobs</h3>
{table(remote_rows, ['priority', 'job', 'script', 'purpose', 'run when'])}
<div class='callout'><b>Decision rule.</b> Do not tune VLM prompts for a class until the generator gate passes. For alveoli, tumor, and stroma, the next change is the proposal generator. For bronchiola and vessels, the next change is assembly/second-SAM refinement. For immune infiltration, the next change is expanded recall-push assembly with stricter NMS/audit.</div>
<p class='small'>Machine-readable tables: <code>refined_diagnostics/class_specific_generator_gate.csv</code> and <code>refined_diagnostics/next_remote_generator_jobs.csv</code>.</p>
"""
    for html_path in [BASE / "Jun09_SkillRanker_ComponentAwareMaskSelection.html", BASE / "index.html"]:
        html_path.write_text(replace_section(html_path.read_text(), "17G. Class-Specific Generator Gate", section))
    update_zip()
    print(REF / "class_specific_generator_gate.csv")
    print(REF / "next_remote_generator_jobs.csv")


if __name__ == "__main__":
    main()
