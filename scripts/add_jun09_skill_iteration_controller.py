#!/usr/bin/env python3
"""Add a paper-expanded executable iteration controller to the Jun09 report."""

from __future__ import annotations

import csv
import html
import json
import zipfile
from pathlib import Path


ROOT = Path("/Users/haoranwu/Desktop/Yan_Lab_Research/Medical-SAM3")
BASE = ROOT / "output/visium_hd_exp1/final_deliverables/Jun09_SkillRanker_ComponentAwareMaskSelection"
OUT = BASE / "skill_iteration_controller"


PAPER_BRANCH_ROWS = [
    {
        "paper / method family": "OVSeg / mask-adapted CLIP",
        "source": "https://openaccess.thecvf.com/content/CVPR2023/papers/Liang_Open-Vocabulary_Semantic_Segmentation_With_Mask-Adapted_CLIP_CVPR_2023_paper.pdf",
        "what it says for us": "Two-stage open-vocabulary segmentation is proposal first, classifier second; ordinary CLIP on masked crops is a bottleneck.",
        "branch to test": "Do not trust natural-image CLIP as final. Try pathology encoder or mask-aware feature extraction on candidate pieces.",
        "failure layer": "recognition / quality",
    },
    {
        "paper / method family": "MaskCLIP / Open-Vocabulary SAM",
        "source": "https://arxiv.org/abs/2401.02955",
        "what it says for us": "Naively combining SAM and CLIP is weaker than transferring recognition knowledge into the segmentation pipeline.",
        "branch to test": "Use ranker as a region-prompt selector and refine with SAM/Medical-SAM rather than accepting raw piece masks.",
        "failure layer": "refinement",
    },
    {
        "paper / method family": "LISA reasoning segmentation",
        "source": "https://arxiv.org/abs/2308.00692",
        "what it says for us": "A multimodal model can reason from language and output segmentation masks, but it needs a segmentation interface, not just a score.",
        "branch to test": "Change VLM role from class scorer to locator generator: emit target bbox/keypoints for SAM.",
        "failure layer": "recognition / refinement",
    },
    {
        "paper / method family": "GenSeg-R1 reason-then-segment",
        "source": "https://arxiv.org/abs/2602.09701",
        "what it says for us": "Decoupled reasoning-to-spatial-prompts plus frozen SAM improves fine-grained referring segmentation.",
        "branch to test": "Use candidate/context pack to ask for structured spatial prompts, then run second-SAM.",
        "failure layer": "refinement",
    },
    {
        "paper / method family": "UNI pathology foundation model",
        "source": "https://www.nature.com/articles/s41591-024-02857-3",
        "what it says for us": "Histology morphology benefits from pathology-pretrained encoders rather than natural-image encoders.",
        "branch to test": "Replace or augment CLIP-large H&E morphology with UNI/CONCH/PLIP embeddings if model access is available.",
        "failure layer": "recognition / quality",
    },
    {
        "paper / method family": "CONCH / PLIP pathology VLM encoders",
        "source": "https://www.nature.com/articles/s41591-024-02856-4",
        "what it says for us": "Histopathology-specific image-text pretraining can support classification, retrieval, and tissue segmentation.",
        "branch to test": "Use pathology image-text similarity for H&E morphology and candidate tissue prompts, not generic VLM color reasoning.",
        "failure layer": "recognition",
    },
    {
        "paper / method family": "WSI-SAM / multi-resolution histology SAM",
        "source": "https://proceedings.mlr.press/v254/liu24a.html",
        "what it says for us": "Whole-slide histology needs multi-resolution patches; one local crop can lose context.",
        "branch to test": "Use multi-scale candidate context: piece crop, local context, full ROI locator; feed only selected locator to final mask step.",
        "failure layer": "recognition / refinement",
    },
]


ACCEPTANCE_ROWS = [
    {
        "gate": "data integrity",
        "must be true before success": "PASS_OFFICIAL same-ROI FICTURE and every candidate root has matching report/metadata/mask counts.",
        "current status": "local report ready; Bouchet account-disabled blocks live remote audit",
    },
    {
        "gate": "proposal availability",
        "must be true before success": "every class has at least one plausible candidate/component proposal before the ranker is tuned.",
        "current status": "alveoli fails in medpt24-only pool; minimal-three gate is next",
    },
    {
        "gate": "recognition signal",
        "must be true before success": "target pieces can be separated from non-target pieces without tumor/stroma collapse.",
        "current status": "works in-sample for compact167 but needs validation on new/minimal-three pool",
    },
    {
        "gate": "deployable quality score",
        "must be true before success": "mask quality can be predicted without using annotation-derived compact_funnel_score.",
        "current status": "quality head proves value but is not deployable; proxy quality remains partial",
    },
    {
        "gate": "assembly/refinement",
        "must be true before success": "selected pieces or second-SAM refinement produce stable final Dice/Precision/Recall for all six classes.",
        "current status": "bronchiola/stroma assembly and vessels/immune quality still need iteration",
    },
    {
        "gate": "stability",
        "must be true before success": "rules pass leave-one-component-out or another ROI/sample; one-ROI oracle is not enough.",
        "current status": "not yet proven",
    },
]


def read_csv(path: Path) -> list[dict[str, str]]:
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


def table(rows: list[dict[str, object]], cols: list[str], link_col: str | None = None) -> str:
    out = ["<table><thead><tr>"]
    out.extend(f"<th>{html.escape(col)}</th>" for col in cols)
    out.append("</tr></thead><tbody>")
    for row in rows:
        out.append("<tr>")
        for col in cols:
            val = str(row.get(col, ""))
            if link_col and col == link_col and val.startswith("http"):
                out.append(f"<td><a href='{html.escape(val)}'>{html.escape(val)}</a></td>")
            else:
                out.append(f"<td>{html.escape(val)}</td>")
        out.append("</tr>")
    out.append("</tbody></table>")
    return "".join(out)


def first_gate_text(value: str) -> str:
    return str(value).split("(", 1)[0].strip()


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    gate_rows = read_csv(BASE / "failure_gate_matrix/quantitative_failure_gate_matrix.csv")

    class_queue: list[dict[str, object]] = []
    for row in gate_rows:
        cls = row["class"]
        first_layer = row["primary failing layer"]
        if first_layer == "proposal":
            branch = "minimal-three proposal gate"
            exact_next = "Run scripts/local/sync_and_submit_jun09_minimal_three_gate.sh after Bouchet access is restored."
            stop_rule = "Stop all VLM/prompt/ranker tuning for this class until a stronger proposal exists."
            success_rule = "Proceed only if minimal-three component oracle improves class proposal upper bound."
        elif first_layer == "quality":
            branch = "deployable mask-quality proxy + pathology/mask-aware encoder"
            exact_next = "Use current 167/full candidate features to train/test a no-annotation quality proxy; if access allows, add UNI/CONCH/PLIP H&E embedding."
            stop_rule = "Stop selecting pieces by tissue score alone."
            success_rule = "Proceed if score-Dice correlation becomes partial/pass and final union improves without compact_funnel_score."
        elif first_layer == "assembly":
            branch = "class-specific assembly grid + optional second-SAM"
            exact_next = "Search per-class score/margin/IoU/max-piece rules; for component classes, test selected-piece boxes as second-SAM prompts."
            stop_rule = "Stop using one global assembly rule."
            success_rule = "Proceed if oracle-runtime Dice gap falls below 0.07 or second-SAM improves recall without precision collapse."
        elif first_layer == "recognition":
            branch = "marked/context region scorer"
            exact_next = "Use piece + local context + full ROI locator and require predicted_class=target before assembly."
            stop_rule = "Stop using isolated tiny crops as the only visual evidence."
            success_rule = "Proceed if Piece Top1 and target-vs-rest ranking improve without class collapse."
        else:
            branch = "external validation"
            exact_next = "Freeze interpretable rules and test on another ROI/sample when available."
            stop_rule = "Stop overclaiming one-ROI in-sample success."
            success_rule = "Proceed only if held-out/spatial validation remains stable."

        class_queue.append(
            {
                "class": cls,
                "first failing layer": first_layer,
                "current evidence": row.get(f"{first_layer} gate", ""),
                "controller branch": branch,
                "exact next action": exact_next,
                "stop rule": stop_rule,
                "success rule": success_rule,
            }
        )

    global_queue = [
        {
            "priority": 1,
            "work item": "restore Bouchet access and run strict minimal-three proposal gate",
            "why first": "alveoli cannot be solved by scorer tuning without a broad proposal; strict preflight prevents bad-pool conclusions",
            "local/remote": "remote after account access",
            "status": "blocked by Duo account-disabled, heartbeat will retry when access changes",
        },
        {
            "priority": 2,
            "work item": "pathology/mask-aware H&E encoder branch",
            "why first": "OVSeg shows ordinary CLIP on masked crops is the bottleneck; pathology encoders are more appropriate for H&E",
            "local/remote": "GPU/model access needed",
            "status": "not run yet",
        },
        {
            "priority": 3,
            "work item": "second-SAM prompt refinement from selected locators",
            "why first": "LISA/GenSeg-R1/SaLIP-style logic suggests ranker should choose prompts, not final masks",
            "local/remote": "remote GPU/SAM",
            "status": "packaged but not executed in this continuation",
        },
        {
            "priority": 4,
            "work item": "marked/context region scorer",
            "why first": "small pieces lack lumen/wall/septa context and need explicit region grounding",
            "local/remote": "local pack exists, remote/model run needed for new variants",
            "status": "partial previous runs; needs rerun only after proposal gate or for specific weak classes",
        },
    ]

    state = {
        "status": "active_not_complete",
        "current_blocker": "Bouchet Duo returned account disabled; local controller/report work continues.",
        "not_success_until": [row["must be true before success"] for row in ACCEPTANCE_ROWS],
        "class_queue": class_queue,
        "global_queue": global_queue,
        "paper_branch_count": len(PAPER_BRANCH_ROWS),
    }

    write_csv(OUT / "paper_expanded_branch_catalog.csv", PAPER_BRANCH_ROWS)
    write_csv(OUT / "class_iteration_controller.csv", class_queue)
    write_csv(OUT / "global_iteration_queue.csv", global_queue)
    write_csv(OUT / "acceptance_gates_not_yet_satisfied.csv", ACCEPTANCE_ROWS)
    (OUT / "skill_iteration_state.json").write_text(json.dumps(state, indent=2, ensure_ascii=False))

    section = f"""
<h2>17Q. Skill Iteration Controller: What To Change Next, And What Not To Change</h2>
<p>This section turns the failure gates into an executable controller. The skill is not considered finished just because it runs. For each class, the controller identifies the first failing layer, selects one branch to change, and defines a stop rule so the next iteration does not drift into random prompt tuning.</p>
<h3>Paper-expanded branch catalog</h3>
{table(PAPER_BRANCH_ROWS, ['paper / method family', 'source', 'what it says for us', 'branch to test', 'failure layer'], link_col='source')}
<h3>Class-level controller</h3>
{table(class_queue, ['class', 'first failing layer', 'current evidence', 'controller branch', 'exact next action', 'stop rule', 'success rule'])}
<h3>Global execution queue</h3>
{table(global_queue, ['priority', 'work item', 'why first', 'local/remote', 'status'])}
<h3>Not yet satisfied acceptance gates</h3>
{table(ACCEPTANCE_ROWS, ['gate', 'must be true before success', 'current status'])}
<div class='callout'><b>Controller decision.</b> The skill is still active. The immediate scientific gate remains proposal availability, because alveoli fails before scoring. In parallel, the next non-prompt branch to prepare is a pathology/mask-aware H&amp;E encoder or second-SAM locator refinement, because literature suggests ordinary VLM/CLIP classification of isolated masked crops is structurally weak.</div>
"""

    for html_path in [BASE / "Jun09_SkillRanker_ComponentAwareMaskSelection.html", BASE / "index.html"]:
        text = html_path.read_text()
        marker = "<h2>17Q. Skill Iteration Controller: What To Change Next, And What Not To Change</h2>"
        if marker in text:
            start = text.index(marker)
            end = text.index("<h2>18.", start) if "<h2>18." in text[start:] else text.index("</body>", start)
            text = text[:start] + section + text[end:]
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
        OUT,
    ]:
        if root.exists():
            include.extend(path for path in root.rglob("*") if path.is_file())
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as handle:
        for path in include:
            handle.write(path, path.relative_to(BASE))

    print(OUT / "skill_iteration_state.json")


if __name__ == "__main__":
    main()
