#!/usr/bin/env python3
"""Add the current failure-driven class-specific controller to Jun09 HTML."""

from __future__ import annotations

import csv
import html
import zipfile
from pathlib import Path


ROOT = Path("/Users/haoranwu/Desktop/Yan_Lab_Research/Medical-SAM3")
BASE = ROOT / "output/visium_hd_exp1/final_deliverables/Jun09_SkillRanker_ComponentAwareMaskSelection"
OUT = BASE / "failure_driven_hybrid_controller"


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


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    controller_rows = [
        {
            "class": "bronchiola",
            "chosen branch": "semantic FICTURE airway proposal",
            "current D/P/R": "0.898 / 0.868 / 0.930",
            "why this branch": "official airway factor recovers all three disconnected bronchiola components better than the 167-piece selector",
            "deployability status": "promising; biology-only factor rule also reaches 0.898 / 0.881 / 0.915",
            "figure_rel": "semantic_ficture_proposal_generator/figures/bronchiola_semantic_ficture_best.png",
        },
        {
            "class": "alveoli",
            "chosen branch": "broad H&E box384 proposal union",
            "current D/P/R": "0.748 / 0.653 / 0.875",
            "why this branch": "medpt24 pieces and semantic FICTURE both miss broad alveolar parenchyma; broad H&E box candidates 12+13 solve the proposal gap",
            "deployability status": "partially validated by an annotation-free diagnostic selector: candidate 13 and 12 rank top3, but broad false positives remain and the rule still has one-ROI spatial bias",
            "figure_rel": "alveoli_broad_box_branch/figures/alveoli_box384_union_12_13_six_panel.png",
        },
        {
            "class": "vessels",
            "chosen branch": "runtime piece selector",
            "current D/P/R": "0.718 / 0.921 / 0.588",
            "why this branch": "semantic FICTURE smooth-muscle/stroma map over-segments vessel-like tissue; current piece selector keeps precision high",
            "deployability status": "usable locator; next improvement should be boundary/recall refinement, not more VLM prompt tuning",
            "figure_rel": "runtime_policy_prototype/figures/vessels_runtime_policy_union.png",
        },
        {
            "class": "tumor",
            "chosen branch": "semantic FICTURE tumor-epithelial proposal",
            "current D/P/R": "0.732 / 0.718 / 0.746",
            "why this branch": "tumor is a diffuse compartment; small-piece VLM/ranker assembly capped near 0.41 Dice, while tumor factor proposal forms a coherent compartment mask",
            "deployability status": "promising; biology-only tumor factor rule gives the same 0.732 / 0.718 / 0.746",
            "figure_rel": "semantic_ficture_proposal_generator/figures/tumor_semantic_ficture_best.png",
        },
        {
            "class": "stroma",
            "chosen branch": "semantic FICTURE stromal/smooth-muscle proposal",
            "current D/P/R": "0.604 / 0.534 / 0.697",
            "why this branch": "stroma is also broad/diffuse; point-piece assembly misses coverage, while the stromal factor captures the compartment",
            "deployability status": "promising but needs validation because factor 1 also overlaps vessel wall biology",
            "figure_rel": "semantic_ficture_proposal_generator/figures/stroma_semantic_ficture_best.png",
        },
        {
            "class": "immune infiltration",
            "chosen branch": "H&E nuclear-density + official immune-support morphology branch",
            "current D/P/R": "0.554 / 0.586 / 0.524",
            "why this branch": "multi-objective full-resolution guardrail found a better immune rule than the first objective-only sweep; it balances RF+H&E precision and runtime+semantic recall",
            "deployability status": "diagnostic only for now; 5-fold held-out component validation was weak at 0.184 / 0.116 / 0.475, so it needs a fixed deployable rule or external validation",
            "figure_rel": "immune_he_morphology_branch/figures/immune_infiltration_he_morphology_best.png",
        },
    ]
    write_csv(OUT / "current_best_class_specific_controller.csv", controller_rows)

    biology_rows = [
        {"rule": "bronchiola: airway factor 7, top3 components", "class": "bronchiola", "D/P/R": "0.898 / 0.881 / 0.915", "interpretation": "not just annotation tuning; a clean airway-factor rule is enough"},
        {"rule": "bronchiola: airway factor 7 plus rare airway/neuroendocrine factor 9, top3", "class": "bronchiola", "D/P/R": "0.898 / 0.868 / 0.930", "interpretation": "slightly higher recall but slightly lower precision"},
        {"rule": "tumor: tumor epithelial factor 0", "class": "tumor", "D/P/R": "0.732 / 0.718 / 0.746", "interpretation": "strong diffuse tumor proposal from official semantic FICTURE"},
        {"rule": "tumor: factors 0 and 2", "class": "tumor", "D/P/R": "0.742 / 0.709 / 0.779", "interpretation": "higher Dice/recall, slightly lower precision"},
        {"rule": "stroma: stromal/smooth-muscle factor 1", "class": "stroma", "D/P/R": "0.604 / 0.534 / 0.697", "interpretation": "useful broad stroma proposal, but vessel-wall overlap must be handled"},
        {"rule": "vessels: factor 1 top12", "class": "vessels", "D/P/R": "0.391 / 0.262 / 0.769", "interpretation": "bad precision; factor 1 is too broad for vessels alone"},
        {"rule": "immune: immune factors top60", "class": "immune infiltration", "D/P/R": "0.378 / 0.268 / 0.641", "interpretation": "good recall but low precision; raw immune factors are not enough"},
        {"rule": "alveoli: AT2/alveolar factors 3 and 5", "class": "alveoli", "D/P/R": "0.102 / 0.059 / 0.360", "interpretation": "FICTURE semantic map does not solve alveoli proposal"},
    ]
    write_csv(OUT / "biology_only_semantic_rule_check.csv", biology_rows)

    immune_rows = [
        {"variant": "runtime piece selector", "D/P/R": "0.448 / 0.393 / 0.522", "decision": "baseline, moderate recall but low precision"},
        {"variant": "RF + H&E precision option", "D/P/R": "0.462 / 0.543 / 0.402", "decision": "better precision and Dice, lower recall"},
        {"variant": "semantic immune factors", "D/P/R": "0.378 / 0.268 / 0.641", "decision": "recall only; too many false positives"},
        {"variant": "runtime plus supported semantic components", "D/P/R": "0.465 / 0.350 / 0.694", "decision": "best recall, precision still too low"},
        {"variant": "smoothed immune density map", "D/P/R": "0.431 / 0.332 / 0.614", "decision": "does not beat simpler hybrid; stop threshold-only FICTURE density tuning"},
        {"variant": "H&E nuclear-density + official immune support", "D/P/R": "0.554 / 0.586 / 0.524", "decision": "best full-annotation immune branch, but component-fold validation is weak; keep as diagnostic"},
    ]
    write_csv(OUT / "immune_branch_probe.csv", immune_rows)

    failure_rows = [
        {"class": "bronchiola", "failure layer now": "selector was too fragmented; semantic airway proposal fixes components", "next action": "freeze airway-factor proposal rule and validate; optional second-SAM only for boundary cleanup"},
        {"class": "alveoli", "failure layer now": "proposal family failure plus deployable-selection gap", "next action": "keep broad H&E box proposals; replace the diagnostic location-heavy selector with a broad-candidate H&E morphology/context encoder"},
        {"class": "vessels", "failure layer now": "semantic FICTURE too broad; locator already high precision", "next action": "keep piece selector; improve recall with second-SAM/MedSAM or boundary refinement"},
        {"class": "tumor", "failure layer now": "wrong object granularity", "next action": "treat as diffuse semantic-FICTURE compartment, not small-piece union"},
        {"class": "stroma", "failure layer now": "wrong object granularity plus vessel-wall overlap", "next action": "use stromal semantic proposal, then add vessel-wall veto/refinement"},
        {"class": "immune infiltration", "failure layer now": "full-annotation result improved, but held-out component stability is weak", "next action": "treat H&E-density plus immune-support as a diagnostic clue; next try fixed nuclei-density features, nuclei segmentation, or pathology encoder validation"},
    ]
    write_csv(OUT / "failure_driven_next_iteration.csv", failure_rows)

    section = build_section(controller_rows, biology_rows, immune_rows, failure_rows)
    for html_path in [BASE / "Jun09_SkillRanker_ComponentAwareMaskSelection.html", BASE / "index.html"]:
        text = html_path.read_text()
        marker = "<h2>17AG. Failure-Driven Hybrid Controller</h2>"
        if marker in text:
            start = text.index(marker)
            next_pos = text.find("<h2>17AH.", start + len(marker))
            end = next_pos if next_pos != -1 else text.index("</body>", start)
            text = text[:start] + section + text[end:]
        else:
            text = text.replace("</body>", section + "</body>")
        html_path.write_text(text)
    rebuild_zip()
    verify_html_images(BASE / "Jun09_SkillRanker_ComponentAwareMaskSelection.html")
    print(OUT / "current_best_class_specific_controller.csv")
    print(BASE / "Jun09_SkillRanker_ComponentAwareMaskSelection.html")


def build_section(controller_rows, biology_rows, immune_rows, failure_rows) -> str:
    figs = []
    for row in controller_rows:
        rel = html.escape(row["figure_rel"])
        cls = html.escape(row["class"])
        figs.append(f"<figure class='wide-figure'><img src='{rel}' alt='{cls} current branch'><figcaption>{cls}: {html.escape(row['chosen branch'])}</figcaption></figure>")
    return f"""
<h2>17AG. Failure-Driven Hybrid Controller</h2>
<p><b>Purpose.</b> The previous sections show that one global VLM prompt, one global ranker, or one global FICTURE rule is not enough. The better design is a class-specific controller: first diagnose which layer failed, then route each tissue class to the proposal/scoring branch that actually matches its biology and mask geometry.</p>
<p><b>Plain-language definition.</b> A controller here is not an agent. It is a decision table: bronchiola may use an airway semantic proposal, vessels may use a high-precision piece selector, and alveoli may need a broad H&amp;E box proposal. The hidden annotation is used to evaluate this development-stage controller; the deployable version still needs fixed rules and validation.</p>
<h3>Current strongest branch by class</h3>
{table(controller_rows, ['class', 'chosen branch', 'current D/P/R', 'why this branch', 'deployability status'])}
<h3>Biology-only semantic rule sanity check</h3>
{table(biology_rows, ['rule', 'class', 'D/P/R', 'interpretation'])}
<h3>Immune branch probe</h3>
{table(immune_rows, ['variant', 'D/P/R', 'decision'])}
<h3>Failure layer and next iteration</h3>
{table(failure_rows, ['class', 'failure layer now', 'next action'])}
<div class='callout'><b>Main conclusion.</b> The useful method is not “VLM judges everything.” It is a skill-based, failure-driven segmentation controller: use FICTURE semantic proposals for bronchiola/tumor/stroma, broad H&amp;E box proposals for alveoli, high-precision piece selection for vessels, and keep the H&amp;E nuclear-density plus immune-support branch as an immune diagnostic clue rather than a frozen deployable rule.</div>
{''.join(figs)}
"""


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
        BASE / "assembly_outputs/combined_skill_ranker_rf_plus_he",
        BASE / "runtime_policy_prototype",
        BASE / "alveoli_broad_box_branch",
        BASE / "semantic_ficture_proposal_generator",
        BASE / "oracle_coverage_upper_bound",
        BASE / "controller_after_oracle_coverage",
        BASE / "failure_taxonomy_v3_after_quality_v2",
        BASE / "immune_he_morphology_branch",
        BASE / "immune_loco_validation",
        BASE / "failure_analysis_protocol_v2",
        BASE / "alveoli_deployable_selector_gate",
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
    with zipfile.ZipFile(zip_path) as handle:
        bad = handle.testzip()
    if bad is not None:
        raise RuntimeError(f"Corrupt zip member: {bad}")


def verify_html_images(html_path: Path) -> None:
    import re

    text = html_path.read_text()
    missing = []
    for src in re.findall(r"<img[^>]+src=['\"]([^'\"]+)['\"]", text):
        if src.startswith(("http://", "https://", "data:")):
            continue
        if not (BASE / src).exists():
            missing.append(src)
    if missing:
        raise RuntimeError(f"Missing HTML images: {missing[:20]}")


if __name__ == "__main__":
    main()
