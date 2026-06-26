#!/usr/bin/env python3
"""Add a current failure-analysis protocol section to the Jun09 report.

This section is intentionally different from a result table.  It records the
diagnostic logic that decides what to try next, so the skill can iterate toward
a usable segmentation method instead of repeatedly running prompts or models.
"""

from __future__ import annotations

import csv
import html
import re
import zipfile
from pathlib import Path


ROOT = Path("/Users/haoranwu/Desktop/Yan_Lab_Research/Medical-SAM3")
BASE = ROOT / "output/visium_hd_exp1/final_deliverables/Jun09_SkillRanker_ComponentAwareMaskSelection"
OUT = BASE / "failure_analysis_protocol_v2"


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
                label = str(row.get("paper", val))
                out.append(f"<td><a href='{html.escape(val)}'>{html.escape(label)}</a></td>")
            else:
                out.append(f"<td>{html.escape(val)}</td>")
        out.append("</tr>")
    out.append("</tbody></table>")
    return "".join(out)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    class_rows = [
        {
            "class": "bronchiola",
            "current best branch": "official FICTURE airway semantic proposal",
            "current D/P/R": "0.898 / 0.868 / 0.930",
            "main failure before this": "small-piece/VLM scoring fragmented disconnected airway components",
            "current failure layer": "mostly solved at proposal-prior layer",
            "next refinement": "freeze the airway-factor rule and validate; use second-SAM only for boundary cleanup",
            "what not to tune": "do not keep changing generic VLM prompts for bronchiola",
        },
        {
            "class": "alveoli",
            "current best branch": "broad H&E box384 proposal union",
            "current D/P/R": "0.748 / 0.653 / 0.875",
            "main failure before this": "medpt24 point pieces do not contain a broad alveolar proposal",
            "current failure layer": "proposal generator / object scale",
            "next refinement": "build a deployable broad H&E proposal ranker; then add tissue scoring",
            "what not to tune": "do not spend more VLM/API calls if the broad proposal is absent",
        },
        {
            "class": "vessels",
            "current best branch": "runtime piece selector",
            "current D/P/R": "0.718 / 0.921 / 0.588",
            "main failure before this": "raw FICTURE smooth-muscle/stroma prior over-segments vessel-like tissue",
            "current failure layer": "boundary/recall refinement",
            "next refinement": "keep high-precision locators; try second-SAM/MedSAM or vessel-wall boundary expansion",
            "what not to tune": "do not replace this with broad semantic factor masks",
        },
        {
            "class": "tumor",
            "current best branch": "official FICTURE tumor-epithelial semantic proposal",
            "current D/P/R": "0.732 / 0.718 / 0.746",
            "main failure before this": "small piece union treats diffuse tumor as many disconnected objects",
            "current failure layer": "wrong object granularity",
            "next refinement": "treat tumor as a diffuse compartment; validate semantic proposal and boundary cleanup",
            "what not to tune": "do not force tumor into small-piece component assembly",
        },
        {
            "class": "stroma",
            "current best branch": "official FICTURE stromal/smooth-muscle semantic proposal",
            "current D/P/R": "0.604 / 0.534 / 0.697",
            "main failure before this": "point-piece assembly misses broad stromal compartment",
            "current failure layer": "diffuse compartment plus vessel-wall overlap",
            "next refinement": "add vessel-wall veto or shape/location split after stromal proposal",
            "what not to tune": "do not rely on VLM color interpretation of raw FICTURE RGB",
        },
        {
            "class": "immune infiltration",
            "current best branch": "H&E hematoxylin-density constrained by official immune FICTURE support",
            "current D/P/R": "0.554 / 0.586 / 0.524",
            "main failure before this": "immune colors gave recall with false positives; first objective-only H&E sweep falsely looked too low-recall",
            "current failure layer": "full-annotation result improved, but component-fold validation is weak",
            "next refinement": "treat as diagnostic clue; test fixed nuclei-density features, nuclei segmentation, or pathology encoder",
            "what not to tune": "do not conclude immune failed from only objective-top parameter sweeps",
        },
    ]

    layer_rows = [
        {
            "failure layer": "official data gate",
            "question": "Are H&E and FICTURE aligned and from PASS_OFFICIAL roots?",
            "diagnostic": "check official summary, ROI size, factor-index shape, bbox, mask/report completeness",
            "if failed": "stop all scoring and fix data root",
        },
        {
            "failure layer": "proposal coverage",
            "question": "Does the candidate pool contain a plausible mask/component before any model scoring?",
            "diagnostic": "component-level oracle and class-specific upper bound",
            "if failed": "change SAM prompt setting/generator, not VLM prompt",
        },
        {
            "failure layer": "region recognition",
            "question": "Can the scorer tell this marked region belongs to a tissue class?",
            "diagnostic": "Piece Top1, predicted-class collapse, target-vs-rest score distribution",
            "if failed": "use marked/context views, pathology encoder, or structured features",
        },
        {
            "failure layer": "mask quality",
            "question": "Among same-class candidates, can the method prefer the cleaner mask?",
            "diagnostic": "score-vs-Dice correlation, high-score false positives, component quality table",
            "if failed": "add shape/quality features or second-SAM refinement",
        },
        {
            "failure layer": "structured modality prior",
            "question": "Does FICTURE help as biology, or only confuse the visual model as colors?",
            "diagnostic": "semantic factor proposal, RGB/cell-type composition, H&E-only vs FICTURE-only ablations",
            "if failed": "use FICTURE as numeric prior or drop it for that class",
        },
        {
            "failure layer": "assembly/refinement",
            "question": "Do selected pieces combine into a usable final mask?",
            "diagnostic": "selected-piece count, NMS/overlap, union D/P/R, second-SAM D/P/R",
            "if failed": "change assembly policy or prompt SAM again from selected locators",
        },
        {
            "failure layer": "deployability",
            "question": "Was the rule selected using annotation?",
            "diagnostic": "mark branch as diagnostic/oracle-selected versus fixed rule",
            "if failed": "do not freeze; convert to annotation-free rule or validate on new ROI",
        },
    ]

    paper_rows = [
        {
            "paper": "SaLIP",
            "source": "https://openaccess.thecvf.com/content/CVPR2024W/DEF-AI-MIA/html/Aleem_Test-Time_Adaptation_with_SaLIP_A_Cascade_of_SAM_and_CLIP_CVPRW_2024_paper.html",
            "lesson used here": "Use a cascade: generate proposals, retrieve/score promising regions, then prompt SAM again rather than accepting the first mask.",
            "our concrete use": "For high-precision bronchiola/vessel locators, test second-SAM/MedSAM for boundary and recall.",
        },
        {
            "paper": "RegionCLIP",
            "source": "https://arxiv.org/abs/2112.09106",
            "lesson used here": "Whole-image CLIP/VLM behavior does not automatically transfer to fine-grained region recognition.",
            "our concrete use": "Separate piece classification from final assembly and require marked/local-context region evidence.",
        },
        {
            "paper": "Alpha-CLIP",
            "source": "https://arxiv.org/abs/2312.03818",
            "lesson used here": "Region focus should preserve context while telling the model which mask matters.",
            "our concrete use": "Use candidate mask/context/shape features instead of isolated tiny crops.",
        },
        {
            "paper": "Prov-GigaPath",
            "source": "https://www.nature.com/articles/s41586-024-07441-w",
            "lesson used here": "Pathology benefits from local-tile plus broader-slide/context modelling, not only isolated texture.",
            "our concrete use": "For alveoli/vessels/bronchiola, decide with local architecture and not only the candidate interior.",
        },
        {
            "paper": "HoVer-Net",
            "source": "https://arxiv.org/abs/1812.06499",
            "lesson used here": "Immune-like histology needs nuclei-aware segmentation/classification rather than only color thresholds.",
            "our concrete use": "If the immune density rule is not stable, next branch should use explicit nuclei features or a pathology encoder.",
        },
    ]

    iteration_rows = [
        {
            "rule": "Change exactly one layer at a time",
            "why": "Otherwise an improvement cannot be assigned to proposal, recognition, quality, or assembly.",
        },
        {
            "rule": "A scorer cannot fix a missing proposal",
            "why": "If component oracle is low, regenerate candidates before using VLM/CLIP/API.",
        },
        {
            "rule": "A high-recall branch with very low precision is not a final mask",
            "why": "It can be useful as support/recall prior, but needs veto/quality/assembly constraints.",
        },
        {
            "rule": "A high-precision low-recall branch is a locator",
            "why": "It should feed second-SAM or expansion, not be called final segmentation.",
        },
        {
            "rule": "Annotation-selected parameters are diagnostic",
            "why": "They can guide method design, but must be converted to fixed rules before being called deployable.",
        },
    ]

    write_csv(OUT / "current_failure_by_class.csv", class_rows)
    write_csv(OUT / "failure_layers_and_gates.csv", layer_rows)
    write_csv(OUT / "paper_lessons_for_failure_analysis.csv", paper_rows)
    write_csv(OUT / "iteration_rules.csv", iteration_rows)

    section = f"""
<h2>17AI. Failure-Analysis Protocol v2</h2>
<p><b>Purpose.</b> This is the current controller logic after the recall-oriented immune check. The skill should not stop at “the code ran.” It must decide which layer failed, change only that layer, and rerun the relevant gate. Hidden annotation is used here for development diagnostics and evaluation only.</p>
<h3>Current failure state by class</h3>
{table(class_rows, ['class', 'current best branch', 'current D/P/R', 'main failure before this', 'current failure layer', 'next refinement', 'what not to tune'])}
<h3>Failure layers and gates</h3>
{table(layer_rows, ['failure layer', 'question', 'diagnostic', 'if failed'])}
<h3>Paper lessons translated into this skill</h3>
{table(paper_rows, ['paper', 'source', 'lesson used here', 'our concrete use'], link_col='source')}
<h3>Iteration rules</h3>
{table(iteration_rows, ['rule', 'why'])}
<div class='callout'><b>Updated decision.</b> The best current design is class-specific. Bronchiola/tumor/stroma use semantic FICTURE proposals, alveoli uses broad H&amp;E proposals, vessels use high-precision piece locators, and immune has a promising but not yet stable H&amp;E density plus immune-support diagnostic branch. The remaining work is not another generic prompt sweep; it is deployable rule validation, proposal gates, and refinement for weak boundaries/recall.</div>
"""

    for html_path in [BASE / "Jun09_SkillRanker_ComponentAwareMaskSelection.html", BASE / "index.html"]:
        text = html_path.read_text()
        marker = "<h2>17AI. Failure-Analysis Protocol v2</h2>"
        if marker in text:
            start = text.index(marker)
            next_match = re.search(r"<h2>17A[J-Z]|<h2>18\\.", text[start + len(marker):])
            end = start + len(marker) + next_match.start() if next_match else text.index("</body>", start)
            text = text[:start] + section + text[end:]
        else:
            text = text.replace("</body>", section + "</body>")
        html_path.write_text(text)

    rebuild_zip()
    verify_html_images(BASE / "Jun09_SkillRanker_ComponentAwareMaskSelection.html")
    print(OUT / "current_failure_by_class.csv")
    print(BASE / "Jun09_SkillRanker_ComponentAwareMaskSelection.html")
    print(BASE / "Jun09_SkillRanker_ComponentAwareMaskSelection_shareable.zip")


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
        BASE / "failure_driven_hybrid_controller",
        BASE / "immune_he_morphology_branch",
        BASE / "immune_loco_validation",
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
