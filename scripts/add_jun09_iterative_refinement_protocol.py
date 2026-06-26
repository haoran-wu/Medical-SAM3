#!/usr/bin/env python3
"""Add an explicit iterative failure-analysis protocol to Jun09 report."""

from __future__ import annotations

import csv
import html
import zipfile
from pathlib import Path


ROOT = Path("/Users/haoranwu/Desktop/Yan_Lab_Research/Medical-SAM3")
BASE = ROOT / "output/visium_hd_exp1/final_deliverables/Jun09_SkillRanker_ComponentAwareMaskSelection"


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
    out += [f"<th>{html.escape(col)}</th>" for col in cols]
    out.append("</tr></thead><tbody>")
    for row in rows:
        out.append("<tr>")
        for col in cols:
            out.append(f"<td>{html.escape(str(row.get(col, '')))}</td>")
        out.append("</tr>")
    out.append("</tbody></table>")
    return "".join(out)


def main() -> None:
    gates = [
        {
            "gate": "1. Data / alignment gate",
            "what to check": "Official PASS_OFFICIAL FICTURE; H&E and FICTURE ROI dimensions match; mask/report/metadata counts are complete.",
            "failure means": "Any downstream ranking result is invalid.",
            "next action": "Repair candidate pool or alignment before running ranker/VLM.",
        },
        {
            "gate": "2. Proposal coverage gate",
            "what to check": "For every annotation component, ask whether at least one candidate reaches useful component Dice/Recall.",
            "failure means": "The generator missed the tissue; no prompt/ranker can recover it.",
            "next action": "Change SAM setting, add a broader box setting, or use second-SAM from selected regions.",
        },
        {
            "gate": "3. Recognition/calibration gate",
            "what to check": "Can the score separate target-class pieces from other pieces? Use Piece Top1, AUC/AP, score distribution, and class-collapse checks.",
            "failure means": "The scoring signal is weak or biased.",
            "next action": "Remove raw FICTURE image if it biases the model; use H&E morphology, structured FICTURE composition, shape/location, and local context as separate skills.",
        },
        {
            "gate": "4. Assembly gate",
            "what to check": "Given good pieces and useful scores, does selected union achieve acceptable Dice/Precision/Recall?",
            "failure means": "NMS/threshold/margin/max-piece policy is wrong.",
            "next action": "Tune class-specific assembly: strict for broad classes, recall-push for multi-component structures, morphology guard for vessels.",
        },
        {
            "gate": "5. Stability gate",
            "what to check": "Leave-one-component-out or spatial holdout. Does the rule still work when a component is held out?",
            "failure means": "The model/ranker is memorizing this ROI.",
            "next action": "Freeze only interpretable rules, collect another annotated ROI, or report as diagnostic rather than deployable.",
        },
    ]
    directions = [
        {
            "branch": "Minimal three-setting pool",
            "why": "medpt24 alone misses alveoli; all-48 is too expensive.",
            "concrete test": "Run component oracle over base_official_points_step24 + medical_official_points_step24 + base_box384_s128_m1536.",
            "pass condition": "Close to all-48 best for bronchiola, alveoli, vessels, and immune, without a huge candidate count.",
        },
        {
            "branch": "Structured FICTURE skill",
            "why": "Raw FICTURE color images bias VLM toward tumor or color-text shortcuts.",
            "concrete test": "Use mask-inside and context-ring RGB/cell-type composition as numeric prior, not as a raw image.",
            "pass condition": "Improves at least one class over H&E morphology without causing tumor/stroma collapse.",
        },
        {
            "branch": "Region-aware VLM/locator prompt",
            "why": "Small pieces lack lumen/wall/septa context.",
            "concrete test": "Show piece crop + local context + full ROI locator, but score only the marked piece.",
            "pass condition": "Piece Top1 improves for bronchiola/vessels without lowering H&E-only performance.",
        },
        {
            "branch": "Second-SAM refinement",
            "why": "SaLIP-style methods use selected proposals to prompt a second segmentation pass.",
            "concrete test": "Convert selected piece boxes/points into SAM prompts and compare refined mask to piece union.",
            "pass condition": "Improves recall or boundary quality without precision collapse.",
        },
        {
            "branch": "Class-specific generator",
            "why": "Alveoli/tumor/stroma are broad/diffuse; bronchiola/vessels/immune are component-like.",
            "concrete test": "Use box-style candidates for alveoli and broader tissue; use point/component candidates for vessels/bronchiola/immune.",
            "pass condition": "Each class has a plausible generator upper bound before ranker tuning.",
        },
    ]
    out_dir = BASE / "refined_diagnostics"
    write_csv(out_dir / "iterative_failure_gates.csv", gates)
    write_csv(out_dir / "next_refinement_branches.csv", directions)

    section = f"""
<h2>17. Iterative Failure-Analysis Protocol</h2>
<p>This section is the working protocol for turning the skill-ranker into a usable segmentation method. The rule is: do not move to a more complex model until the previous gate is proven. A failed gate decides what to change next.</p>
{table(gates, ['gate', 'what to check', 'failure means', 'next action'])}
<h3>Next branches to test</h3>
{table(directions, ['branch', 'why', 'concrete test', 'pass condition'])}
<div class='callout'><b>Current decision.</b> The next concrete branch is the minimal three-setting pool. It directly addresses the strongest current failure: medpt24 alone is not enough for alveoli, while all-48 is too expensive for a practical workflow.</div>
<p class='small'>Protocol tables saved to <code>refined_diagnostics/iterative_failure_gates.csv</code> and <code>refined_diagnostics/next_refinement_branches.csv</code>. The Bouchet submission script for the next branch is <code>scripts/hpc/jun09_minimal_three_setting_component_oracle.sbatch</code>.</p>
"""

    for html_path in [BASE / "Jun09_SkillRanker_ComponentAwareMaskSelection.html", BASE / "index.html"]:
        text = html_path.read_text()
        marker = "<h2>17. Iterative Failure-Analysis Protocol</h2>"
        if marker in text:
            start = text.index(marker)
            end = text.index("<h2>17. Interpretation</h2>", start) if "<h2>17. Interpretation</h2>" in text[start:] else text.index("</body>", start)
            text = text[:start] + section + text[end:]
        elif "<h2>17. Interpretation</h2>" in text:
            text = text.replace("<h2>17. Interpretation</h2>", section + "<h2>18. Interpretation</h2>")
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

    print(out_dir / "iterative_failure_gates.csv")
    print(zip_path)


if __name__ == "__main__":
    main()
