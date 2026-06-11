#!/usr/bin/env python3
"""Controller update after oracle coverage upper-bound diagnosis."""

from __future__ import annotations

import csv
import html
import zipfile
from pathlib import Path

import pandas as pd


ROOT = Path("/Users/haoranwu/Desktop/Yan_Lab_Research/Medical-SAM3")
BASE = ROOT / "output/visium_hd_exp1/final_deliverables/Jun09_SkillRanker_ComponentAwareMaskSelection"
OUT = BASE / "controller_after_oracle_coverage"


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


def parse_dpr(value: str) -> tuple[float, float, float]:
    a, b, c = str(value).split("/")
    return float(a), float(b), float(c)


def rebuild_zip() -> None:
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
        BASE / "second_sam_refinement",
        BASE / "structured_veto_microscope",
        BASE / "quality_skill_diagnostic",
        BASE / "quality_adjusted_ranker",
        BASE / "proxy_quality_ranker",
        BASE / "failure_driven_proxy_refinement",
        BASE / "alveoli_generator_gate",
        BASE / "paper_informed_failure_engine",
        BASE / "remote_gate_preflight",
        BASE / "failure_gate_matrix",
        BASE / "skill_iteration_controller",
        BASE / "feature_space_upper_bound",
        BASE / "runtime_policy_prototype",
        BASE / "runtime_policy_second_sam_locator_pack",
        BASE / "alveoli_broad_box_branch",
        BASE / "failure_debugger_v2",
        BASE / "runtime_candidate_error_auditor",
        BASE / "one_class_assignment_test",
        BASE / "soft_assignment_veto_test",
        BASE / "class_specific_assembly_grid",
        BASE / "iteration_state_after_grid",
        BASE / "structured_quality_v2",
        BASE / "failure_taxonomy_v3_after_quality_v2",
        BASE / "oracle_coverage_upper_bound",
        OUT,
    ]:
        if root.exists():
            include.extend(path for path in root.rglob("*") if path.is_file())
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as handle:
        seen: set[Path] = set()
        for path in include:
            if path in seen:
                continue
            seen.add(path)
            handle.write(path, path.relative_to(BASE))


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    oracle = pd.read_csv(BASE / "oracle_coverage_upper_bound/oracle_coverage_summary.csv")

    rows: list[dict[str, object]] = []
    for _, row in oracle.iterrows():
        cls = str(row["class"])
        rt_d, rt_p, rt_r = parse_dpr(row["runtime D/P/R"])
        hidden_d, hidden_p, hidden_r = parse_dpr(row["hidden true-class oracle D/P/R"])
        all_d, all_p, all_r = parse_dpr(row["all-candidate oracle D/P/R"])
        gap = all_d - rt_d
        if cls in {"bronchiola", "vessels"} and gap <= 0.04:
            state = "selector essentially saturated on this pool"
            next_action = "freeze selector; run second-SAM/MedSAM locator refinement and external validation"
            success = "boundary/recall improves without losing the current high precision"
            stop = "do not spend more cycles on prompt, hand-crafted quality, or threshold grids"
        elif cls == "alveoli":
            state = "pool lacks strong broad alveoli proposal"
            next_action = "switch to broad H&E proposal generator / minimal-three gate before any scorer work"
            success = "oracle on new proposal pool exceeds current 0.514 and approaches the earlier broad-box evidence"
            stop = "do not tune scorer on medpt24-only 167 pool"
        elif cls in {"tumor", "stroma"}:
            state = "piece-first pool cannot assemble a strong diffuse compartment"
            next_action = "test diffuse-compartment generator or pathology encoder on larger coherent regions"
            success = "oracle and deployable runtime both exceed 0.55 Dice with acceptable precision"
            stop = "do not treat diffuse tumor/stroma as independent small-object pieces"
        else:
            state = "many-component pool has partial upper bound but quality/coverage still weak"
            next_action = "test density/region-growing or pathology encoder branch; keep structured FICTURE only as auxiliary prior"
            success = "hidden oracle and deployable runtime both improve above current 0.50-ish ceiling"
            stop = "do not select immune pieces by tissue probability alone"
        rows.append(
            {
                "class": cls,
                "runtime D/P/R": row["runtime D/P/R"],
                "oracle D/P/R": row["all-candidate oracle D/P/R"],
                "oracle-runtime Dice gap": f"{gap:+.3f}",
                "controller state": state,
                "next action": next_action,
                "success criterion": success,
                "stop doing": stop,
            }
        )

    priority_rows = [
        {
            "priority": 1,
            "branch": "second-SAM locator refinement for bronchiola/vessels",
            "reason": "selector is saturated and high precision; only refinement can plausibly improve current pool",
            "needs Bouchet": "yes",
        },
        {
            "priority": 2,
            "branch": "alveoli broad-proposal generator gate",
            "reason": "current pool oracle is too weak; scorer changes cannot fix absent proposal",
            "needs Bouchet": "yes",
        },
        {
            "priority": 3,
            "branch": "diffuse-compartment / pathology encoder for tumor/stroma",
            "reason": "piece-first oracle itself is weak for diffuse classes",
            "needs Bouchet": "likely yes for real encoder; local design can continue",
        },
        {
            "priority": 4,
            "branch": "immune density/region-growing quality branch",
            "reason": "hidden true oracle partially improves but still has many-component quality/coverage limits",
            "needs Bouchet": "maybe",
        },
    ]

    write_csv(OUT / "controller_after_oracle_coverage.csv", rows)
    write_csv(OUT / "next_branch_priority_after_oracle_coverage.csv", priority_rows)

    section = f"""
<h2>17AE. Controller After Oracle Coverage: What Is Actually Still Worth Trying?</h2>
<p><b>Goal.</b> 17AD gives the cleanest diagnostic so far: it tells us whether the current 167-piece pool can be solved by a better selector. This section converts that into a concrete controller state for each tissue class.</p>
<h3>Per-class controller</h3>
{table(rows, ['class', 'runtime D/P/R', 'oracle D/P/R', 'oracle-runtime Dice gap', 'controller state', 'next action', 'success criterion', 'stop doing'])}
<h3>Next branch priority</h3>
{table(priority_rows, ['priority', 'branch', 'reason', 'needs Bouchet'])}
<div class='callout'><b>17AE decision.</b> The skill should now become a cascade. For bronchiola/vessels, use the current skill as a high-precision locator and refine with SAM/MedSAM. For alveoli/tumor/stroma, change the proposal/generator level before more ranking. For immune infiltration, test a density/region-growing quality branch instead of selecting by class probability alone.</div>
"""
    marker = "<h2>17AE. Controller After Oracle Coverage: What Is Actually Still Worth Trying?</h2>"
    for html_path in [BASE / "Jun09_SkillRanker_ComponentAwareMaskSelection.html", BASE / "index.html"]:
        text = html_path.read_text()
        if marker in text:
            start = text.index(marker)
            end = text.index("</body>", start)
            text = text[:start] + section + text[end:]
        else:
            text = text.replace("</body>", section + "</body>")
        html_path.write_text(text)
    rebuild_zip()
    print(OUT / "controller_after_oracle_coverage.csv")
    print(OUT / "next_branch_priority_after_oracle_coverage.csv")
    print(BASE / "Jun09_SkillRanker_ComponentAwareMaskSelection.html")
    print(BASE / "Jun09_SkillRanker_ComponentAwareMaskSelection_shareable.zip")


if __name__ == "__main__":
    main()
