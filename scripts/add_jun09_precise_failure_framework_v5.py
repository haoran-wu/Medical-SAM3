#!/usr/bin/env python3
"""Append a precise, paper-informed failure framework to the Jun09 report.

This section is deliberately operational.  It turns the current evidence into
small diagnostic gates so the next experiment changes the earliest failing
layer instead of repeatedly changing a VLM prompt.
"""

from __future__ import annotations

import csv
import html
import re
import zipfile
from pathlib import Path


ROOT = Path("/Users/haoranwu/Desktop/Yan_Lab_Research/Medical-SAM3")
BASE = ROOT / "output/visium_hd_exp1/final_deliverables/Jun09_SkillRanker_ComponentAwareMaskSelection"
OUT = BASE / "precise_failure_framework_v5"


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


def replace_or_append(html_path: Path, section: str) -> None:
    text = html_path.read_text()
    marker = "<h2>17AP. Precise Failure Framework V5: Diagnose Before Refining</h2>"
    if marker in text:
        start = text.index(marker)
        next_match = re.search(r"<h2>17A[Q-Z]|<h2>18\.", text[start + len(marker) :])
        end = start + len(marker) + next_match.start() if next_match else text.index("</body>", start)
        text = text[:start] + section + text[end:]
    else:
        text = text.replace("</body>", section + "</body>")
    html_path.write_text(text)


def verify_html_images(html_path: Path) -> None:
    text = html_path.read_text(errors="ignore")
    missing: list[str] = []
    for src in re.findall(r"<img[^>]+src=\"([^\"]+)\"", text):
        if src.startswith(("data:", "http://", "https://")):
            continue
        if not (html_path.parent / src).exists():
            missing.append(src)
    if missing:
        raise RuntimeError(f"Missing {len(missing)} image assets, first examples: {missing[:5]}")


def rebuild_zip() -> None:
    zip_path = BASE / "Jun09_SkillRanker_ComponentAwareMaskSelection_shareable.zip"
    include_roots = [
        OUT,
        BASE / "corrected_pool",
        BASE / "assembly_outputs/combined_skill_ranker_rf_plus_he",
        BASE / "refined_diagnostics",
        BASE / "second_sam_refinement",
        BASE / "structured_veto_microscope",
        BASE / "runtime_policy_prototype",
        BASE / "runtime_policy_second_sam_locator_pack",
        BASE / "runtime_candidate_error_auditor",
        BASE / "oracle_coverage_upper_bound",
        BASE / "feature_space_upper_bound",
        BASE / "failure_driven_proxy_refinement",
        BASE / "one_class_assignment_test",
        BASE / "class_specific_assembly_grid",
        BASE / "region_aware_piece_context_locator_pack",
        BASE / "region_aware_vlm_results",
        BASE / "semantic_ficture_proposal_generator",
        BASE / "failure_driven_hybrid_controller",
        BASE / "failure_hierarchy_v4",
        BASE / "immune_he_morphology_branch",
        BASE / "immune_loco_validation",
        BASE / "alveoli_broad_box_branch",
        BASE / "alveoli_deployable_selector_gate",
        BASE / "alveoli_feature_sufficiency_audit",
        BASE / "alveoli_bbox_he_morphology_audit",
        BASE / "alveoli_target_grounding_audit",
        BASE / "alveoli_target_grounding_selector_v2",
        BASE / "alveoli_approx_assembly_sanity",
        BASE / "stage_failure_iteration_controller_v6",
        BASE / "boundary_refinement_proxy_gate",
        BASE / "component_coverage_gap_audit",
        BASE / "local_broader_pool_recovery_audit",
        BASE / "recovered_proposal_hybrid_prototype",
        BASE / "recovered_second_sam_prompt_pack",
        BASE / "skill_failure_controller_v7",
        BASE / "vessels_component_oracle_prompt_diagnostic",
        BASE / "soft_assignment_veto_test",
        BASE / "structured_quality_v2",
        BASE / "failure_analysis_protocol_v2",
        BASE / "paper_informed_failure_engine",
    ]
    include_files = [
        BASE / "Jun09_SkillRanker_ComponentAwareMaskSelection.html",
        BASE / "index.html",
        BASE / "candidate_skill_features.csv",
        BASE / "piece_top1_all_methods.csv",
        BASE / "assembly_summary_all_methods.csv",
        BASE / "selected_pieces_all_methods.csv",
        BASE / "failure_attribution_by_class.csv",
        BASE / "run_config.json",
    ]
    # The report contains many already-compressed PNG/JPG assets.  Deflating the
    # whole bundle is very slow and saves little space, so use ZIP_STORED for
    # fast, repeatable shareable-package refreshes.
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_STORED) as handle:
        seen: set[Path] = set()
        for p in include_files:
            if p.exists() and p not in seen:
                handle.write(p, p.relative_to(BASE))
                seen.add(p)
        for root in include_roots:
            if not root.exists():
                continue
            for p in root.rglob("*"):
                if p.is_file() and p not in seen:
                    handle.write(p, p.relative_to(BASE))
                    seen.add(p)
    with zipfile.ZipFile(zip_path) as handle:
        bad = handle.testzip()
    if bad:
        raise RuntimeError(f"Bad zip entry: {bad}")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    paper_rows = [
        {
            "paper idea": "SaLIP cascade",
            "source": "https://openaccess.thecvf.com/content/CVPR2024W/DEF-AI-MIA/html/Aleem_Test-Time_Adaptation_with_SaLIP_A_Cascade_of_SAM_and_CLIP_CVPRW_2024_paper.html",
            "failure lesson": "Do not expect one scorer to directly produce the final mask. Use scoring to find useful regions, then prompt SAM again.",
            "our diagnostic rule": "If a selected piece has high tissue confidence but weak boundary/recall, test second-SAM from its box or point before tuning the scorer.",
        },
        {
            "paper idea": "Region-aware CLIP/VLM",
            "source": "https://arxiv.org/abs/2112.09106",
            "failure lesson": "Whole-image image-text matching is not the same as fine region-level judgment.",
            "our diagnostic rule": "Separate tissue recognition from target-region grounding; do not call a morphology-like false positive a tissue-recognition failure.",
        },
        {
            "paper idea": "Alpha-CLIP / mask-indicated context",
            "source": "https://arxiv.org/abs/2312.03818",
            "failure lesson": "The model needs to know which region is the target while still seeing surrounding context.",
            "our diagnostic rule": "Prefer piece + local context + full-ROI locator, or structured region features, over isolated small crops.",
        },
        {
            "paper idea": "Pathology multi-scale context",
            "source": "https://www.nature.com/articles/s41586-024-07441-w",
            "failure lesson": "Pathology models use local tissue texture and broader tissue context together.",
            "our diagnostic rule": "A candidate score should include local H&E morphology, surrounding tissue context, and structured FICTURE prior, not raw color alone.",
        },
    ]

    gate_rows = [
        {
            "gate": "0. Official data and alignment",
            "question": "Are H&E and FICTURE the official same-ROI inputs?",
            "pass evidence": "PASS_OFFICIAL and no missing image assets",
            "if fail": "stop; fix data root/alignment before any model/ranker conclusion",
        },
        {
            "gate": "1. Proposal existence",
            "question": "Does the candidate pool contain a plausible piece or broad region?",
            "pass evidence": "component-aware oracle or diagnostic proposal reaches near-current best",
            "if fail": "change SAM/FICTURE proposal generator; do not tune VLM/ranker",
        },
        {
            "gate": "2. Tissue recognition",
            "question": "Can the method tell what tissue a candidate looks like?",
            "pass evidence": "piece top1 by class is non-collapse and examples visually match the predicted tissue",
            "if fail": "change visual input, prompt, model, or structured prior",
        },
        {
            "gate": "3. Target grounding",
            "question": "Does the candidate correspond to the target region/component, not just similar-looking tissue elsewhere?",
            "pass evidence": "known useful candidates outrank morphology-similar false positives such as alveoli candidate 48",
            "if fail": "add full-ROI locator, proposal-family constraints, or landmark/context features",
        },
        {
            "gate": "4. Mask-quality ranking",
            "question": "Among correct-tissue candidates, does the score prefer better masks?",
            "pass evidence": "score correlates with component Dice or selected pieces have good P/R tradeoff",
            "if fail": "add shape/boundary/context quality skill or second-SAM refinement",
        },
        {
            "gate": "5. Assembly",
            "question": "Can selected pieces be unioned without precision collapse?",
            "pass evidence": "class-specific NMS/margin/max-piece rule improves D/P/R with readable six-panel figures",
            "if fail": "change assembly policy, not tissue-recognition prompt",
        },
        {
            "gate": "6. Validation",
            "question": "Does the rule survive held-out components or a new ROI?",
            "pass evidence": "component-fold validation or second ROI remains plausible",
            "if fail": "mark as diagnostic, not deployable",
        },
    ]

    class_rows = [
        {
            "class": "bronchiola",
            "earliest unresolved failure": "boundary/refinement after proposal success",
            "evidence now": "semantic airway FICTURE proposal is strong; piece recognition is not the bottleneck",
            "next refinement": "test second-SAM from selected airway components; accept only if boundary/recall improves without precision loss",
        },
        {
            "class": "alveoli",
            "earliest unresolved failure": "target grounding after broad proposal success",
            "evidence now": "H&E morphology ranks true 12/13 high, but candidate 48 is morphology-like and hidden Dice 0",
            "next refinement": "recover broad-box masks, then add locator/context grounding so candidate 48 falls below 12/13 before union",
        },
        {
            "class": "vessels",
            "earliest unresolved failure": "recall and boundary refinement",
            "evidence now": "runtime selector is high precision but misses components",
            "next refinement": "use selected vessel pieces as point/box prompts for second-SAM; tune recall with strict vessel-wall quality gates",
        },
        {
            "class": "tumor",
            "earliest unresolved failure": "diffuse compartment quality control",
            "evidence now": "semantic tumor proposal is better than small-piece VLM assembly, but needs false-positive veto",
            "next refinement": "combine FICTURE tumor prior with H&E morphology veto for necrosis/stroma-like regions",
        },
        {
            "class": "stroma",
            "earliest unresolved failure": "semantic overlap with vessel wall/smooth muscle",
            "evidence now": "stromal/smooth-muscle FICTURE helps, but conflates stroma and vascular smooth muscle",
            "next refinement": "split stroma quality skill from vessel-wall skill before final assembly",
        },
        {
            "class": "immune infiltration",
            "earliest unresolved failure": "validation and density-quality stability",
            "evidence now": "full-annotation optimized branch improved, but component-fold validation is weak",
            "next refinement": "replace annotation-optimized rule with fixed nuclei-density / immune-support features and revalidate by held-out components",
        },
    ]

    stop_rows = [
        {
            "do not keep doing": "global prompt sweeps on the same 167 isolated crops",
            "why": "multiple runs show raw VLM recognition is not the earliest failure for most classes",
        },
        {
            "do not keep doing": "treating raw FICTURE color images as visual evidence by themselves",
            "why": "raw FICTURE often behaves like a misleading color picture; structured composition prior is safer",
        },
        {
            "do not keep doing": "accepting an in-sample annotation-trained quality head as deployable",
            "why": "one ROI can memorize target location; every rule needs a no-hidden-label gate or held-out component gate",
        },
        {
            "do not keep doing": "unioning more pieces only to raise recall",
            "why": "assembly must pass precision and visual checks; otherwise it only dilutes the mask",
        },
    ]

    iteration_rows = [
        {
            "iteration step": "diagnose earliest failing gate",
            "artifact required": "CSV table plus visual examples of true positive, false positive, and missed component",
            "move on only if": "the failing layer is identified without relying on vague model blame",
        },
        {
            "iteration step": "change one layer",
            "artifact required": "before/after table with Dice, Precision, Recall, and selected-piece count",
            "move on only if": "the changed layer improves the exact failure it targeted",
        },
        {
            "iteration step": "stress test",
            "artifact required": "held-out component, false-positive challenge set, or no-hidden-label selector gate",
            "move on only if": "the result is not just full-annotation overfitting",
        },
        {
            "iteration step": "package as usable branch",
            "artifact required": "six-panel figure per class, definitions, method provenance, and known caveats",
            "move on only if": "a collaborator can understand and reproduce the decision",
        },
    ]

    write_csv(OUT / "paper_to_diagnostic_gates_v5.csv", paper_rows)
    write_csv(OUT / "precise_failure_gates_v5.csv", gate_rows)
    write_csv(OUT / "class_next_refinement_v5.csv", class_rows)
    write_csv(OUT / "directions_to_stop_v5.csv", stop_rows)
    write_csv(OUT / "iteration_contract_v5.csv", iteration_rows)

    section = f"""
<h2>17AP. Precise Failure Framework V5: Diagnose Before Refining</h2>
<p><b>Purpose.</b> This section makes the failure analysis stricter. A method is not accepted because it runs, and a failure is not blamed on the VLM until earlier layers are ruled out. The goal is to iterate the skill toward a usable mask-selection system.</p>
<h3>Paper ideas converted into diagnostic gates</h3>
{table(paper_rows, ['paper idea', 'source', 'failure lesson', 'our diagnostic rule'])}
<h3>Exact failure gates</h3>
{table(gate_rows, ['gate', 'question', 'pass evidence', 'if fail'])}
<h3>Current class-specific refinement target</h3>
{table(class_rows, ['class', 'earliest unresolved failure', 'evidence now', 'next refinement'])}
<h3>Directions to stop for now</h3>
{table(stop_rows, ['do not keep doing', 'why'])}
<h3>Iteration contract</h3>
{table(iteration_rows, ['iteration step', 'artifact required', 'move on only if'])}
<div class='callout'><b>V5 decision.</b> The next work is failure-driven, not prompt-driven. Alveoli now needs target grounding and broad-box mask recovery; bronchiola/vessels need second-SAM refinement from high-confidence locators; tumor/stroma need diffuse-compartment quality vetoes; immune needs validation-stable density features. If a branch fails these gates, switch the layer being changed instead of continuing the same direction.</div>
"""

    replace_or_append(BASE / "Jun09_SkillRanker_ComponentAwareMaskSelection.html", section)
    replace_or_append(BASE / "index.html", section)
    verify_html_images(BASE / "Jun09_SkillRanker_ComponentAwareMaskSelection.html")
    rebuild_zip()
    print(OUT / "class_next_refinement_v5.csv")
    print(BASE / "Jun09_SkillRanker_ComponentAwareMaskSelection.html")
    print(BASE / "Jun09_SkillRanker_ComponentAwareMaskSelection_shareable.zip")


if __name__ == "__main__":
    main()
