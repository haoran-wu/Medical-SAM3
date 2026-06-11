#!/usr/bin/env python3
"""Update Jun09 controller state after assignment/veto/assembly-grid tests."""

from __future__ import annotations

import csv
import html
import zipfile
from pathlib import Path

import pandas as pd


ROOT = Path("/Users/haoranwu/Desktop/Yan_Lab_Research/Medical-SAM3")
BASE = ROOT / "output/visium_hd_exp1/final_deliverables/Jun09_SkillRanker_ComponentAwareMaskSelection"
OUT = BASE / "iteration_state_after_grid"


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
        OUT,
    ]:
        if root.exists():
            include.extend(p for p in root.rglob("*") if p.is_file())
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as handle:
        seen: set[Path] = set()
        for path in include:
            if path in seen:
                continue
            seen.add(path)
            handle.write(path, path.relative_to(BASE))


def parse_dice(dpr: str) -> float:
    return float(str(dpr).split("/")[0])


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    runtime_audit = pd.read_csv(BASE / "runtime_candidate_error_auditor/runtime_selection_summary_by_class.csv")
    hard = pd.read_csv(BASE / "one_class_assignment_test/one_class_assignment_runtime_comparison.csv")
    soft = pd.read_csv(BASE / "soft_assignment_veto_test/soft_veto_runtime_comparison.csv")
    grid = pd.read_csv(BASE / "class_specific_assembly_grid/class_specific_assembly_grid_best_by_class.csv")

    hard_by = hard.set_index("class").to_dict("index")
    soft_by = soft.set_index("class").to_dict("index")
    grid_by = grid.set_index("class").to_dict("index")

    controller_rows: list[dict[str, object]] = []
    for _, row in runtime_audit.iterrows():
        cls = str(row["class"])
        runtime_dpr = hard_by[cls]["runtime D/P/R"]
        hard_dpr = hard_by[cls]["one-class fixed 0.05 D/P/R"]
        soft_dpr = soft_by[cls]["soft-veto ratio 0.55 D/P/R"]
        grid_dpr = grid_by[cls]["grid-best D/P/R"]
        runtime_d = parse_dice(runtime_dpr)
        soft_d = parse_dice(soft_dpr)
        grid_d = parse_dice(grid_dpr)
        if cls == "alveoli":
            next_layer = "proposal generator"
            next_action = "run minimal-three proposal gate and add broad H&E box384 branch before scorer tuning"
            stop = "stop tuning VLM/ranker/assembly for alveoli until broad candidates are present"
        elif cls in {"bronchiola", "vessels"} and runtime_d >= 0.70:
            next_layer = "refinement / boundary"
            next_action = "run second-SAM locator refinement on selected pieces; keep current runtime selector fixed"
            stop = "stop global threshold search and hard one-class assignment"
        elif cls in {"tumor", "stroma", "immune infiltration"}:
            next_layer = "quality feature / encoder"
            next_action = "build deployable quality feature or pathology-encoder branch; use structured FICTURE only as prior"
            stop = "stop selecting by tissue probability or raw FICTURE color alone"
        else:
            next_layer = "validation"
            next_action = "freeze current selector and validate on new proposal/sample"
            stop = "do not add more prompt variants before validation"
        controller_rows.append(
            {
                "class": cls,
                "runtime D/P/R": runtime_dpr,
                "hard one-class result": hard_dpr,
                "soft veto result": soft_dpr,
                "class-specific grid result": grid_dpr,
                "best local delta after 17X-17Z": f"{max(soft_d, grid_d) - runtime_d:+.3f}",
                "what this proves": conclusion(cls, runtime_d, soft_d, grid_d),
                "next layer to change": next_layer,
                "exact next action": next_action,
                "stop doing": stop,
            }
        )

    stop_rows = [
        {
            "direction tested": "hard one-class assignment",
            "evidence": "17X: vessels/tumor/stroma/immune recall collapses",
            "decision": "do not use as main rule; only possible as narrow veto",
        },
        {
            "direction tested": "soft cross-class veto",
            "evidence": "17Y: vessels/tumor improve only by about 0.004 Dice; other classes neutral",
            "decision": "not enough to fix the skill",
        },
        {
            "direction tested": "class-specific assembly grid",
            "evidence": "17Z: no class gets a meaningful improvement over runtime policy",
            "decision": "stop broad threshold/NMS search on the same scores",
        },
        {
            "direction tested": "more prompt/VLM scoring on same medpt24 pieces",
            "evidence": "proposal and quality gates, not prompt format, are the earliest failures for weak classes",
            "decision": "do not spend API/GPU on this until proposal/quality gates change",
        },
    ]

    next_rows = [
        {
            "priority": 1,
            "experiment": "minimal-three proposal gate",
            "why": "alveoli cannot be solved by scorer/assembly in current medpt24 pool",
            "blocked by": "Bouchet SSH currently requires keyboard-interactive/Duo",
            "success evidence": "broad H&E candidate or union near local 17U evidence, around 0.748/0.653/0.875 for alveoli",
        },
        {
            "priority": 2,
            "experiment": "second-SAM locator refinement for bronchiola/vessels",
            "why": "current locators are high precision and close to usable; refinement may improve boundary/recall",
            "blocked by": "Bouchet compute until SSH session is restored",
            "success evidence": "improves recall without precision collapse below class-specific floor",
        },
        {
            "priority": 3,
            "experiment": "deployable quality proxy or pathology encoder",
            "why": "tumor/stroma/immune/vessels show quality failure, not just tissue recognition failure",
            "blocked by": "local feature branch can be prepared; larger encoders may need Bouchet/GPU",
            "success evidence": "score-Dice correlation improves and selected false positives drop without hidden Dice",
        },
    ]

    write_csv(OUT / "controller_state_after_assignment_and_grid.csv", controller_rows)
    write_csv(OUT / "directions_to_stop_after_17x_17y_17z.csv", stop_rows)
    write_csv(OUT / "next_experiments_after_grid.csv", next_rows)

    section = f"""
<h2>17AA. Controller State After 17X-17Z: What To Stop And What To Try Next</h2>
<p><b>Goal.</b> The last three sections tested concrete refinements rather than only describing failure: hard one-class assignment, soft cross-class veto, and a class-specific assembly grid. This section updates the controller so the next iteration changes the right layer instead of looping back to prompt or threshold tweaks.</p>
<h3>Per-class controller update</h3>
{table(controller_rows, ['class', 'runtime D/P/R', 'hard one-class result', 'soft veto result', 'class-specific grid result', 'best local delta after 17X-17Z', 'what this proves', 'next layer to change', 'exact next action', 'stop doing'])}
<h3>Directions now deprioritized</h3>
{table(stop_rows, ['direction tested', 'evidence', 'decision'])}
<h3>Next evidence-backed experiments</h3>
{table(next_rows, ['priority', 'experiment', 'why', 'blocked by', 'success evidence'])}
<div class='callout'><b>17AA decision.</b> The skill is not failing because we forgot one more global threshold or one more VLM prompt. The current evidence says: proposal branch first for alveoli, second-SAM refinement for high-precision localizers, and quality/pathology-encoder features for the diffuse or mixed classes.</div>
"""
    marker = "<h2>17AA. Controller State After 17X-17Z: What To Stop And What To Try Next</h2>"
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
    print(OUT / "controller_state_after_assignment_and_grid.csv")
    print(OUT / "directions_to_stop_after_17x_17y_17z.csv")
    print(OUT / "next_experiments_after_grid.csv")
    print(BASE / "Jun09_SkillRanker_ComponentAwareMaskSelection.html")
    print(BASE / "Jun09_SkillRanker_ComponentAwareMaskSelection_shareable.zip")


def conclusion(cls: str, runtime_d: float, soft_d: float, grid_d: float) -> str:
    best_delta = max(soft_d, grid_d) - runtime_d
    if cls == "alveoli":
        return "same-score refinements cannot fix absent broad proposal"
    if best_delta > 0.03:
        return "same-score refinement has some value but needs validation"
    if cls in {"bronchiola", "vessels"} and runtime_d >= 0.70:
        return "selector is already strong; refinement should target boundaries/components"
    return "same-score refinement is exhausted; change quality feature or proposal"


if __name__ == "__main__":
    main()
