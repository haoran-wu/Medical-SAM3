#!/usr/bin/env python3
"""Paper-guided failure taxonomy after structured quality-v2.

This section makes the controller more precise: each tissue class is assigned
to the earliest failing layer, with explicit evidence and the next experiment
that is still worth running.  It is meant to prevent endless prompt/threshold
cycling once a layer has been exhausted.
"""

from __future__ import annotations

import csv
import html
import zipfile
from pathlib import Path

import pandas as pd


ROOT = Path("/Users/haoranwu/Desktop/Yan_Lab_Research/Medical-SAM3")
BASE = ROOT / "output/visium_hd_exp1/final_deliverables/Jun09_SkillRanker_ComponentAwareMaskSelection"
OUT = BASE / "failure_taxonomy_v3_after_quality_v2"


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
            value = str(row.get(col, ""))
            if value.startswith("http"):
                out.append(f"<td><a href='{html.escape(value)}'>{html.escape(value)}</a></td>")
            else:
                out.append(f"<td>{html.escape(value)}</td>")
        out.append("</tr>")
    out.append("</tbody></table>")
    return "".join(out)


def key(label: str) -> str:
    return str(label).replace("immune infiltration", "immune_infiltration")


def parse_delta(value: str) -> float:
    return float(str(value).replace("+", ""))


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
    gate = pd.read_csv(BASE / "failure_gate_matrix/quantitative_failure_gate_matrix.csv")
    qv2 = pd.read_csv(BASE / "structured_quality_v2/structured_quality_v2_best_by_class.csv")
    runtime = pd.read_csv(BASE / "runtime_policy_prototype/runtime_policy_summary.csv")
    audit = pd.read_csv(BASE / "runtime_candidate_error_auditor/runtime_selection_summary_by_class.csv")
    quality = pd.read_csv(BASE / "quality_skill_diagnostic/quality_skill_summary_by_class.csv")

    qv2_by = {key(r["class"]): r for _, r in qv2.iterrows()}
    runtime_by = {key(r["class"]): r for _, r in runtime.iterrows()}
    audit_by = {key(r["class"]): r for _, r in audit.iterrows()}
    quality_by = {key(r["class"]): r for _, r in quality.iterrows()}

    paper_rows = [
        {
            "paper": "SaLIP",
            "failure lesson": "Use a proposal-rerank-prompt cascade; do not assume the first SAM mask is final.",
            "how it changes this skill": "For high-precision locators, feed selected bbox/point back to SAM/Medical-SAM for refinement.",
            "source": "https://openaccess.thecvf.com/content/CVPR2024W/DEF-AI-MIA/html/Aleem_Test-Time_Adaptation_with_SaLIP_A_Cascade_of_SAM_and_CLIP_CVPRW_2024_paper.html",
        },
        {
            "paper": "Alpha-CLIP / Region-aware CLIP",
            "failure lesson": "A marked region needs context; isolated crops lose the surrounding structure that defines the object.",
            "how it changes this skill": "Treat piece score as locator evidence; add context/locator view or numeric context features before trusting small pieces.",
            "source": "https://arxiv.org/abs/2312.03818",
        },
        {
            "paper": "RegionGPT / set-of-mark grounding",
            "failure lesson": "Detailed region understanding is weak unless the queried region is explicitly grounded.",
            "how it changes this skill": "Do not interpret unmarked FICTURE colors as direct tissue labels; use candidate-specific region features or marked context.",
            "source": "https://arxiv.org/abs/2403.02330",
        },
        {
            "paper": "Pathology multi-scale models",
            "failure lesson": "Small pathology patches often need larger tissue context.",
            "how it changes this skill": "For bronchiola/vessels/alveoli, local lumen/septa context is a necessary skill, not optional decoration.",
            "source": "https://www.nature.com/articles/s41586-024-07441-w",
        },
    ]

    taxonomy_rows: list[dict[str, object]] = []
    for _, row in gate.iterrows():
        cls_display = str(row["class"])
        cls = key(cls_display)
        qrow = qv2_by[cls]
        rrow = runtime_by[cls]
        arow = audit_by[cls]
        qu = quality_by[cls]
        delta = parse_delta(qrow["delta Dice vs runtime"])
        if cls == "alveoli":
            earliest = "proposal"
            evidence = "medpt24 max component Dice is only 0.477; quality-v2 also drops; broad H&E box branch already shows much stronger local evidence"
            next_test = "run broad-proposal/minimal-three generator gate before any VLM/ranker work"
            stop_rule = "stop tuning scorer on medpt24-only alveoli pieces"
        elif cls in {"bronchiola", "vessels"} and float(rrow["precision"]) >= 0.90:
            earliest = "refinement after good locator"
            evidence = f"runtime precision {float(rrow['precision']):.3f}; quality-v2 delta {delta:+.3f}; selected pieces locate target but boundaries/recall remain incomplete"
            next_test = "second-SAM locator refinement from selected piece bbox/point; keep selector fixed"
            stop_rule = "stop more prompt variants and broad hand-crafted quality formula search"
        elif cls in {"tumor", "stroma"}:
            earliest = "diffuse-compartment proposal/quality"
            evidence = f"selected false positives remain high ({arow.get('selected false positives', 'NA')}); quality-v2 delta {delta:+.3f}; small-piece object-style selection is unstable"
            next_test = "try broad/diffuse compartment generator or pathology/mask-aware encoder; evaluate whether it ranks large coherent regions"
            stop_rule = "stop treating diffuse tumor/stroma as many independent object-like tiny pieces"
        else:
            earliest = "mask quality / many-component assembly"
            evidence = f"quality score-Dice signal {qu.get('Spearman(score, Dice)', 'NA')}; quality-v2 delta {delta:+.3f}; many small true and false pieces remain hard to separate"
            next_test = "use pathology encoder or density/region-growing quality branch; keep structured FICTURE as auxiliary prior"
            stop_rule = "stop selecting by tissue probability alone"
        taxonomy_rows.append(
            {
                "class": cls_display,
                "earliest failing layer now": earliest,
                "evidence from current run": evidence,
                "quality-v2 result": qrow["structured quality-v2 D/P/R"],
                "runtime result": qrow["runtime D/P/R"],
                "next experiment that can still improve it": next_test,
                "do not spend time on": stop_rule,
            }
        )

    stop_rows = [
        {
            "direction": "more prompt wording on same 167 piece crops",
            "why stop": "recognition is mostly not the earliest failing layer; proposal/quality/refinement fail first",
        },
        {
            "direction": "more hand-crafted inside/ring/shape quality formulas",
            "why stop": "17AB quality-v2 gives no meaningful Dice gain; it is not the missing signal",
        },
        {
            "direction": "hard one-class assignment",
            "why stop": "17X caused severe recall collapse for several classes",
        },
        {
            "direction": "raw FICTURE image as a VLM picture",
            "why stop": "previous ablations showed FICTURE-only and H&E+FICTURE raw views can bias predictions rather than help",
        },
    ]

    active_rows = [
        {
            "priority": 1,
            "branch": "alveoli broad proposal generator",
            "why it is first": "a scorer cannot recover a mask that is absent from the pool",
            "success metric": "broad proposal/union near the local evidence around Dice 0.65-0.75 without precision collapse",
            "current blocker": "Bouchet SSH requires interactive Duo before submission",
        },
        {
            "priority": 2,
            "branch": "second-SAM refinement for bronchiola/vessels",
            "why it is first": "these are already high-precision locators; SaLIP/MedSAM-style cascade may improve boundaries and recall",
            "success metric": "recall improves while precision stays near current high level",
            "current blocker": "Bouchet SSH requires interactive Duo before submission",
        },
        {
            "priority": 3,
            "branch": "pathology/mask-aware encoder for tumor/stroma/immune",
            "why it is first": "manual quality proxies did not add the missing quality signal",
            "success metric": "selected false positives decrease and component Dice ranking improves without hidden Dice",
            "current blocker": "can prepare locally; real encoder inference needs GPU/Bouchet",
        },
    ]

    write_csv(OUT / "paper_guided_failure_taxonomy_v3.csv", taxonomy_rows)
    write_csv(OUT / "directions_to_stop_after_quality_v2.csv", stop_rows)
    write_csv(OUT / "active_next_branches_after_quality_v2.csv", active_rows)
    write_csv(OUT / "paper_lessons_used.csv", paper_rows)

    section = f"""
<h2>17AC. Paper-Guided Failure Taxonomy V3: Stop Cycling, Change The Right Layer</h2>
<p><b>Goal.</b> This section answers the practical question: after many VLM, CLIP, assignment, assembly, and quality-proxy trials, which part of the pipeline is actually failing for each tissue class? The point is to keep iterating, but only on the first layer that still has evidence of failure.</p>
<p><b>How to read this.</b> The method checks layers in order: proposal availability, region/tissue recognition, mask-quality ranking, assembly, and final refinement. A later layer is not tuned until the earlier failing layer has been fixed. Annotation is used here only for diagnosis and evaluation.</p>
<h3>Paper lessons turned into tests</h3>
{table(paper_rows, ['paper', 'failure lesson', 'how it changes this skill', 'source'])}
<h3>Per-class failure taxonomy after 17AB</h3>
{table(taxonomy_rows, ['class', 'earliest failing layer now', 'evidence from current run', 'runtime result', 'quality-v2 result', 'next experiment that can still improve it', 'do not spend time on'])}
<h3>Directions stopped by evidence</h3>
{table(stop_rows, ['direction', 'why stop'])}
<h3>Active next branches</h3>
{table(active_rows, ['priority', 'branch', 'why it is first', 'success metric', 'current blocker'])}
<div class='callout'><b>17AC decision.</b> The next usable method is not another single prompt. It should be a gated cascade: first ensure the right proposal exists, then use the skill ranker as a locator/quality scorer, then assemble or prompt SAM again. For alveoli this means proposal generation first; for bronchiola/vessels it means second-SAM refinement; for tumor/stroma/immune it means a stronger pathology/mask-aware quality encoder or diffuse-compartment proposal.</div>
"""
    marker = "<h2>17AC. Paper-Guided Failure Taxonomy V3: Stop Cycling, Change The Right Layer</h2>"
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
    print(OUT / "paper_guided_failure_taxonomy_v3.csv")
    print(OUT / "active_next_branches_after_quality_v2.csv")
    print(BASE / "Jun09_SkillRanker_ComponentAwareMaskSelection.html")
    print(BASE / "Jun09_SkillRanker_ComponentAwareMaskSelection_shareable.zip")


if __name__ == "__main__":
    main()
