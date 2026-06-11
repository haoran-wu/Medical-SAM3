#!/usr/bin/env python3
"""Add a class-wise failure debugger to the Jun09 skill-ranker report.

This is the "do not just run it" layer.  It turns the current metrics into a
debugging contract: for each tissue class, identify the earliest failing layer,
change exactly one thing next, and define the evidence needed before advancing.
"""

from __future__ import annotations

import csv
import html
import json
import zipfile
from pathlib import Path

import pandas as pd


ROOT = Path("/Users/haoranwu/Desktop/Yan_Lab_Research/Medical-SAM3")
BASE = ROOT / "output/visium_hd_exp1/final_deliverables/Jun09_SkillRanker_ComponentAwareMaskSelection"
OUT = BASE / "failure_debugger_v2"


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


def class_debug_rows() -> list[dict[str, object]]:
    gate = pd.read_csv(BASE / "failure_gate_matrix/quantitative_failure_gate_matrix.csv")
    runtime = pd.read_csv(BASE / "runtime_policy_prototype/runtime_policy_summary.csv").set_index("class")
    rows: list[dict[str, object]] = []
    for _, row in gate.iterrows():
        cls = row["class"]
        cls_runtime = runtime.loc[cls] if cls in runtime.index else None
        runtime_dpr = (
            f"{float(cls_runtime['dice']):.3f}/{float(cls_runtime['precision']):.3f}/{float(cls_runtime['recall']):.3f}"
            if cls_runtime is not None
            else ""
        )
        failing = row["primary failing layer"]
        if cls == "bronchiola":
            next_change = "keep current runtime selector; run second-SAM locator refinement and then tune max-piece/NMS only if refinement misses components"
            exact_artifact = "runtime_policy_second_sam_locator_pack + scripts/local/sync_and_submit_jun09_runtime_second_sam.sh"
            success = "post-refinement Dice improves over 0.799 or recall rises without precision dropping below 0.85"
            switch = "if second-SAM worsens boundary, keep raw runtime union and focus on external validation"
        elif cls == "alveoli":
            next_change = "change generator first: add H&E base_box384_s128_m1536 broad candidates, then rerun component oracle before scoring"
            exact_artifact = "scripts/local/sync_and_submit_jun09_minimal_three_gate.sh; 17U candidate 12+13 is local evidence"
            success = "minimal-three oracle recovers a broad alveoli candidate or union near 0.748/0.653/0.875"
            switch = "if minimal-three still fails, test a second broad-box setting rather than prompt/VLM changes"
        elif cls == "vessels":
            next_change = "keep proposal; add deployable quality proxy and run second-SAM locator refinement for selected vessel pieces"
            exact_artifact = "runtime_policy_second_sam_locator_pack; quality_skill_diagnostic; proxy_quality_ranker"
            success = "precision stays above 0.90 and recall improves beyond 0.588"
            switch = "if quality proxy cannot improve score-Dice correlation, try pathology/mask-aware H&E encoder"
        elif cls == "tumor":
            next_change = "do not chase VLM prompt; add broad compartment proposal and deployable quality filter because many selected pieces are mixed"
            exact_artifact = "new tumor/stroma broad-compartment branch; use structured FICTURE prior only as support"
            success = "final union precision remains above 0.55 while recall rises above 0.40"
            switch = "if broad compartment branch merges stroma/tumor, split tumor into epithelial-core vs mixed-boundary candidates"
        elif cls == "stroma":
            next_change = "change assembly rule, not recognition: use diffuse-compartment assembly with area cap and quality filter"
            exact_artifact = "class-specific assembly grid over fig_shape/quality scores"
            success = "oracle-runtime gap drops below 0.10 while precision does not collapse below 0.35"
            switch = "if diffuse assembly fails, treat stroma as broad compartment segmentation rather than object-piece union"
        else:
            next_change = "improve quality filter for immune pieces; avoid selecting by immune score alone"
            exact_artifact = "proxy_quality_ranker + structured_veto_microscope"
            success = "recall exceeds 0.56 or Dice exceeds 0.50 without precision dropping below 0.40"
            switch = "if high-score pieces remain mixed, add local-density / immune-cluster morphology features"
        rows.append(
            {
                "class": cls,
                "runtime D/P/R": runtime_dpr,
                "first failing layer": failing,
                "evidence for failure": row.get(f"{failing} gate", ""),
                "next single change": next_change,
                "artifact to run or inspect": exact_artifact,
                "success evidence": success,
                "switch direction if": switch,
                "do not do next": row["do not do"],
            }
        )
    return rows


def decision_loop_rows() -> list[dict[str, object]]:
    return [
        {
            "step": "1. data/proposal gate",
            "question": "does a usable candidate exist before scoring?",
            "metric": "component-aware best Dice/Precision/Recall by class",
            "if fail": "change proposal generator or add class-specific candidate family",
            "if pass": "freeze candidate family and move to scoring",
        },
        {
            "step": "2. recognition gate",
            "question": "can the scorer identify tissue type on candidates?",
            "metric": "Piece Top1, target-vs-rest margin, class-collapse rate",
            "if fail": "change input representation or encoder; do not tune assembly",
            "if pass": "move to quality scoring",
        },
        {
            "step": "3. quality gate",
            "question": "can the skill distinguish good masks from bad masks within the same tissue?",
            "metric": "score-Dice correlation and selected-piece precision/recall",
            "if fail": "add deployable shape/context/quality features or pathology encoder",
            "if pass": "move to assembly",
        },
        {
            "step": "4. assembly gate",
            "question": "does the selected set union correctly?",
            "metric": "final union Dice/Precision/Recall and oracle-runtime gap",
            "if fail": "change per-class threshold, overlap NMS, max pieces, or compartment policy",
            "if pass": "try second-SAM refinement or external validation",
        },
        {
            "step": "5. refinement/validation gate",
            "question": "does second-SAM or another sample improve robustness?",
            "metric": "post-refinement D/P/R and held-out ROI/component stability",
            "if fail": "return to the earliest failing gate shown by metrics",
            "if pass": "freeze method as candidate final skill",
        },
    ]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    debug_rows = class_debug_rows()
    loop_rows = decision_loop_rows()
    action_queue = [
        {
            "priority": 1,
            "action": "Run minimal-three component oracle once Bouchet SSH works",
            "reason": "Alveoli is blocked at proposal gate; no scorer can recover a mask absent from the pool.",
            "proof needed": "six-class component oracle summary and six-panel figures",
        },
        {
            "priority": 2,
            "action": "Run focused second-SAM only for bronchiola/vessels",
            "reason": "These already have high-precision runtime locators; refinement may improve recall/boundary.",
            "proof needed": "post-refinement D/P/R vs 17T prompt-preview D/P/R",
        },
        {
            "priority": 3,
            "action": "Build deployable quality proxy for vessels/tumor/immune",
            "reason": "Those classes pass proposal/recognition but fail quality or remain partial.",
            "proof needed": "score-Dice correlation and selected union improvement without hidden compact_funnel_score",
        },
        {
            "priority": 4,
            "action": "Try pathology/mask-aware H&E encoder if quality proxy plateaus",
            "reason": "Natural-image/VLM features are structurally weak for masked pathology regions.",
            "proof needed": "same candidate pool, same assembly rule, improved quality/ranking metrics",
        },
    ]

    write_csv(OUT / "class_failure_debugger.csv", debug_rows)
    write_csv(OUT / "skill_decision_loop.csv", loop_rows)
    write_csv(OUT / "next_experiment_priority_queue.csv", action_queue)
    (OUT / "failure_debugger_state.json").write_text(
        json.dumps(
            {
                "status": "active_not_complete",
                "principle": "change the earliest failing layer; do not tune later layers until earlier gates pass",
                "current_top_priority": action_queue[0],
                "class_count": len(debug_rows),
            },
            indent=2,
            ensure_ascii=False,
        )
    )

    section = f"""
<h2>17V. Failure Debugger V2: Iterate The Skill By Earliest Failing Layer</h2>
<p>This section is the guardrail against a merely runnable but unreliable skill. The skill must improve by debugging the first failing layer for each tissue class. If proposal generation fails, do not tune a VLM prompt. If recognition passes but quality fails, do not change the pool. If quality passes but assembly fails, change only the class-specific assembly rule.</p>
<h3>Decision loop used after every experiment</h3>
{table(loop_rows, ['step', 'question', 'metric', 'if fail', 'if pass'])}
<h3>Per-class next change</h3>
{table(debug_rows, ['class', 'runtime D/P/R', 'first failing layer', 'evidence for failure', 'next single change', 'artifact to run or inspect', 'success evidence', 'switch direction if', 'do not do next'])}
<h3>Immediate priority queue</h3>
{table(action_queue, ['priority', 'action', 'reason', 'proof needed'])}
<div class='callout'><b>17V rule.</b> The next experiment must produce evidence for one gate only. For the current state, the first gate is proposal availability for alveoli. Once that is proven or disproven, the controller moves to second-SAM refinement for bronchiola/vessels or to deployable quality proxies for vessels/tumor/immune. This is how the skill is iterated toward a usable segmentation method rather than a collection of prompt trials.</div>
"""

    marker = "<h2>17V. Failure Debugger V2: Iterate The Skill By Earliest Failing Layer</h2>"
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
    print(OUT / "class_failure_debugger.csv")
    print(OUT / "skill_decision_loop.csv")
    print(OUT / "next_experiment_priority_queue.csv")
    print(BASE / "Jun09_SkillRanker_ComponentAwareMaskSelection.html")
    print(BASE / "Jun09_SkillRanker_ComponentAwareMaskSelection_shareable.zip")


if __name__ == "__main__":
    main()
