#!/usr/bin/env python3
"""Append the Jun09 V7 class-specific failure controller.

The V7 controller consolidates the many Jun09 diagnostics into one operational
table.  Its job is to prevent another broad prompt/model sweep when the
earliest failing layer is actually candidate generation, mask-quality ranking,
assembly, or targeted boundary refinement.
"""

from __future__ import annotations

import csv
import html
import re
import zipfile
from pathlib import Path

import add_jun09_precise_failure_framework_v5 as pack


ROOT = Path("/Users/haoranwu/Desktop/Yan_Lab_Research/Medical-SAM3")
BASE = ROOT / "output/visium_hd_exp1/final_deliverables/Jun09_SkillRanker_ComponentAwareMaskSelection"
OUT = BASE / "skill_failure_controller_v7"
HTML = BASE / "Jun09_SkillRanker_ComponentAwareMaskSelection.html"
INDEX = BASE / "index.html"


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


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def table(rows: list[dict[str, object]], cols: list[str]) -> str:
    bits = ["<table><thead><tr>"]
    bits.extend(f"<th>{html.escape(col)}</th>" for col in cols)
    bits.append("</tr></thead><tbody>")
    for row in rows:
        bits.append("<tr>")
        for col in cols:
            bits.append(f"<td>{html.escape(str(row.get(col, '')))}</td>")
        bits.append("</tr>")
    bits.append("</tbody></table>")
    return "".join(bits)


def fmt_dpr(dice: float, precision: float, recall: float) -> str:
    return f"{dice:.3f} / {precision:.3f} / {recall:.3f}"


def append_or_replace(section: str) -> None:
    marker = "<h2>17BC. V7 Skill Failure Controller: Refine the Earliest Failing Layer</h2>"
    for path in [HTML, INDEX]:
        text = path.read_text()
        if marker in text:
            start = text.index(marker)
            next_match = re.search(r"<h2>17B[D-Z]|<h2>18\.", text[start + len(marker) :])
            end = start + len(marker) + next_match.start() if next_match else text.index("</body>", start)
            text = text[:start] + section + text[end:]
        else:
            insert_before = "<h2>18."
            if insert_before in text:
                idx = text.index(insert_before)
                text = text[:idx] + section + text[idx:]
            else:
                text = text.replace("</body>", section + "</body>")
        path.write_text(text)


def verify(path: Path) -> None:
    text = path.read_text(errors="ignore")
    missing: list[str] = []
    for src in re.findall(r"<img[^>]+src=\"([^\"]+)\"", text):
        if src.startswith(("data:", "http://", "https://")):
            continue
        if not (path.parent / src).exists():
            missing.append(src)
    if missing:
        raise RuntimeError(f"missing images in {path}: {missing[:8]}")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    # These rows deliberately include the latest class-specific diagnostics
    # rather than only the original combined-ranker table.
    current_routes = [
        {
            "class": "bronchiola",
            "current best route": "semantic airway proposal + targeted second-SAM only for the missed bottom component",
            "current best D/P/R": fmt_dpr(0.819462, 0.791608, 0.849347),
            "earliest failing layer": "component-specific boundary/refinement, not tissue recognition",
            "why this is the route": "The main selected pieces already work; replacing every piece with second-SAM lowers Dice, but refining only the recovered bottom component improves Dice and Recall.",
            "next refinement": "Freeze the targeted rule and rerun on Bouchet GPU / held-out ROI; do not run another generic VLM prompt sweep first.",
            "status": "usable in-sample; needs validation",
        },
        {
            "class": "alveoli",
            "current best route": "class-specific broad H&E box384 proposal, candidates 12 + 13",
            "current best D/P/R": fmt_dpr(0.748, 0.653, 0.875),
            "earliest failing layer": "proposal family: medpt24 small pieces are the wrong granularity",
            "why this is the route": "The broad-box branch recovers the alveolar region; medpt24 piece scoring was not the main bottleneck.",
            "next refinement": "Recover/recompute exact broad-box masks in the clean pipeline and add target-grounding veto for broad false positives.",
            "status": "promising but separate generator branch",
        },
        {
            "class": "vessels",
            "current best route": "current vessel piece selector; annotation-free H&E/shape veto as optional precision guard",
            "current best D/P/R": fmt_dpr(0.718038, 0.920785, 0.588464),
            "earliest failing layer": "clean component recovery / recall, not only false-positive selection",
            "why this is the route": "A deployable veto raises precision to 0.974 but drops Dice/Recall, so it cannot replace missing vessel components.",
            "next refinement": "Generate cleaner local vessel-wall/lumen proposals for missed components 1 and 4; then test targeted refinement only if the proposal is a good locator.",
            "status": "high precision; recall not solved",
        },
        {
            "class": "tumor",
            "current best route": "semantic FICTURE tumor compartment proposal + H&E false-positive veto",
            "current best D/P/R": fmt_dpr(0.7317, 0.7179, 0.7461),
            "earliest failing layer": "object granularity: tumor is a diffuse compartment, not a small-piece class",
            "why this is the route": "The semantic FICTURE proposal beats the piece-pool runtime and oracle evidence by a large margin.",
            "next refinement": "Keep structured FICTURE as a compartment proposal, then add H&E morphology veto for necrosis/stroma-like leakage.",
            "status": "strong diagnostic branch; needs validation",
        },
        {
            "class": "stroma",
            "current best route": "semantic FICTURE stromal/smooth-muscle proposal with vessel-wall split/veto",
            "current best D/P/R": fmt_dpr(0.6043, 0.5336, 0.6966),
            "earliest failing layer": "broad compartment definition and vessel-wall overlap",
            "why this is the route": "Small-piece assembly under-covers diffuse stroma; semantic FICTURE recovers more of the broad compartment but overlaps vessel biology.",
            "next refinement": "Split stromal matrix from vascular smooth-muscle/vessel-wall components before final assembly.",
            "status": "better than piece-first, not final",
        },
        {
            "class": "immune_infiltration",
            "current best route": "H&E nuclear-density / immune-support branch plus recall-push assembly",
            "current best D/P/R": fmt_dpr(0.554, 0.586, 0.524),
            "earliest failing layer": "component stability and small-object quality, not raw tissue recognition",
            "why this is the route": "The full-annotation optimized immune rule improves the mask, but component-fold validation is weak.",
            "next refinement": "Use deployable nuclei-density or pathology-encoder features and validate by held-out annotation components before freezing.",
            "status": "diagnostic only; not deployable yet",
        },
    ]

    failure_lenses = [
        {
            "lens": "proposal upper bound",
            "what it asks": "Does any candidate/proposal contain the structure at all?",
            "paper motivation": "SaLIP-style cascades first need a useful proposal before reranking or second-SAM can help.",
            "action if failed": "change SAM/FICTURE proposal generator or class granularity",
        },
        {
            "lens": "region grounding",
            "what it asks": "Is the score about the marked candidate, not surrounding tissue?",
            "paper motivation": "RegionCLIP and Alpha-CLIP show that whole-image CLIP/VLM is not reliable for marked-region recognition without region focus.",
            "action if failed": "add piece+context+locator or structured mask/location features",
        },
        {
            "lens": "mask-quality skill",
            "what it asks": "Among correct-tissue candidates, does the ranker prefer the better mask?",
            "paper motivation": "A region classifier is not automatically a segmentation-quality scorer.",
            "action if failed": "train/calibrate quality score or use targeted second-SAM only for the bad component",
        },
        {
            "lens": "class granularity",
            "what it asks": "Is the tissue object-like or a broad/diffuse compartment?",
            "paper motivation": "Pathology models often need local-global context; broad compartments should not be assembled as many tiny independent objects.",
            "action if failed": "switch to broad H&E box or semantic FICTURE compartment proposal",
        },
        {
            "lens": "validation stability",
            "what it asks": "Does the rule survive held-out components/ROI, not just the training annotation?",
            "paper motivation": "One annotated ROI can overfit; the failure gate must mark diagnostic-only rules.",
            "action if failed": "do not freeze as method; collect another ROI or use component-fold validation",
        },
    ]

    stop_continue = [
        {
            "candidate change": "another generic VLM prompt/model sweep",
            "decision": "stop for now",
            "reason": "current evidence shows several classes fail at proposal granularity or mask-quality ranking, not six-class recognition.",
        },
        {
            "candidate change": "blanket second-SAM replacement",
            "decision": "stop",
            "reason": "true second-SAM smoke reduced or barely changed Dice unless applied only to the missing bronchiola component.",
        },
        {
            "candidate change": "targeted second-SAM for proven locator components",
            "decision": "continue",
            "reason": "bronchiola bottom component improved from component D/P/R 0.504/0.403/0.671 to 0.622/0.474/0.905.",
        },
        {
            "candidate change": "raw FICTURE RGB as VLM image",
            "decision": "stop as primary input",
            "reason": "earlier ablations showed raw FICTURE image often misleads VLM; use structured composition/semantic factors instead.",
        },
        {
            "candidate change": "class-specific proposal families",
            "decision": "continue",
            "reason": "alveoli, tumor, and stroma improve only after changing candidate granularity/source.",
        },
    ]

    queue = [
        {
            "priority": 1,
            "class": "bronchiola",
            "next run": "validate targeted second-SAM hybrid on Bouchet GPU and package it as the current best bronchiola rule",
            "acceptance gate": "D/P/R stays near 0.819/0.792/0.849 and visual six-panel shows all three components",
        },
        {
            "priority": 2,
            "class": "alveoli",
            "next run": "recover exact broad H&E box384 masks inside the clean pipeline and rerun target-grounding assembly",
            "acceptance gate": "exact D/P/R remains near 0.748/0.653/0.875 and false broad candidates are vetoed",
        },
        {
            "priority": 3,
            "class": "vessels",
            "next run": "search/generate cleaner local vessel component proposals for components 1 and 4",
            "acceptance gate": "raise recall above current 0.588 without dropping precision below about 0.90",
        },
        {
            "priority": 4,
            "class": "tumor/stroma",
            "next run": "turn semantic FICTURE compartment proposals into deployable rules with H&E veto",
            "acceptance gate": "keep tumor/stroma Dice gains while removing obvious false positives",
        },
        {
            "priority": 5,
            "class": "immune_infiltration",
            "next run": "replace ad hoc immune score with fixed nuclei-density/pathology-encoder features and component-fold validation",
            "acceptance gate": "held-out component Dice improves beyond the current weak 0.184 mean",
        },
    ]

    write_csv(OUT / "current_best_route_by_class.csv", current_routes)
    write_csv(OUT / "failure_lenses_v7.csv", failure_lenses)
    write_csv(OUT / "stop_continue_decisions_v7.csv", stop_continue)
    write_csv(OUT / "next_iteration_queue_v7.csv", queue)

    figure_rows = [
        {
            "class": "bronchiola",
            "figure": "recovered_second_sam_prompt_pack/targeted_component_second_sam_hybrid/bronchiola_targeted_hybrid_six_panel.png",
            "caption": "Targeted component hybrid: only the missed bottom bronchiola component is refined with second-SAM.",
        },
        {
            "class": "alveoli",
            "figure": "alveoli_broad_box_branch/figures/alveoli_box384_union_12_13_six_panel.png",
            "caption": "Alveoli broad H&E box branch: the current fix is a generator/granularity change, not another VLM prompt.",
        },
        {
            "class": "vessels",
            "figure": "recovered_second_sam_prompt_pack/vessels_annotation_free_veto_diagnostic/vessels_annotation_free_he_veto_six_panel.png",
            "caption": "Vessels annotation-free veto: precision guard helps but recall still needs cleaner component proposals.",
        },
        {
            "class": "tumor",
            "figure": "semantic_ficture_proposal_generator/figures/tumor_semantic_ficture_best.png",
            "caption": "Tumor semantic FICTURE compartment proposal.",
        },
        {
            "class": "stroma",
            "figure": "semantic_ficture_proposal_generator/figures/stroma_semantic_ficture_best.png",
            "caption": "Stroma semantic FICTURE compartment proposal with known vessel-wall overlap.",
        },
        {
            "class": "immune_infiltration",
            "figure": "immune_he_morphology_branch/figures/immune_infiltration_he_morphology_best.png",
            "caption": "Immune H&E morphology branch: useful diagnostic direction, but held-out stability is weak.",
        },
    ]

    figure_html = []
    for row in figure_rows:
        fig = BASE / row["figure"]
        if fig.exists():
            figure_html.append(
                f"<figure><img src=\"{html.escape(row['figure'])}\" alt=\"{html.escape(row['class'])} V7 route figure\"><figcaption>{html.escape(row['caption'])}</figcaption></figure>"
            )

    section = f"""
<h2>17BC. V7 Skill Failure Controller: Refine the Earliest Failing Layer</h2>
<p><b>Goal.</b> This controller turns the current Jun09 evidence into a class-specific refinement plan.  The skill should not keep changing one giant prompt.  It should first decide which layer failed, then only modify that layer.</p>
<p><b>Paper-guided principle.</b> RegionCLIP and Alpha-CLIP motivate region-focused scoring instead of whole-image scoring.  SaLIP motivates proposal scoring followed by a second SAM step, but our true SAM smoke shows this must be targeted, not blanket.  Pathology foundation-model work motivates multi-scale H&amp;E context and structured priors rather than isolated tiny crops.</p>
<h3>Current best route by class</h3>
{table(current_routes, ["class", "current best route", "current best D/P/R", "earliest failing layer", "why this is the route", "next refinement", "status"])}
<h3>Failure lenses used by the controller</h3>
{table(failure_lenses, ["lens", "what it asks", "paper motivation", "action if failed"])}
<h3>Stop / continue decisions</h3>
{table(stop_continue, ["candidate change", "decision", "reason"])}
<h3>Next iteration queue</h3>
{table(queue, ["priority", "class", "next run", "acceptance gate"])}
<h3>Route figures</h3>
{''.join(figure_html)}
<div class='callout'><b>V7 decision.</b> The skill is now a controller, not a single model prompt.  Bronchiola should use targeted component refinement; alveoli should use a broad H&amp;E proposal branch; vessels need cleaner component proposals plus an optional precision guard; tumor and stroma should use semantic FICTURE compartment proposals with H&amp;E veto; immune needs a validated small-object/nuclear-density skill.  The next iteration should follow this queue instead of running another generic VLM sweep.</div>
<p class='small'>Machine-readable outputs: <code>skill_failure_controller_v7/current_best_route_by_class.csv</code>, <code>skill_failure_controller_v7/failure_lenses_v7.csv</code>, <code>skill_failure_controller_v7/stop_continue_decisions_v7.csv</code>, and <code>skill_failure_controller_v7/next_iteration_queue_v7.csv</code>.</p>
"""

    append_or_replace(section)
    verify(HTML)
    verify(INDEX)
    pack.rebuild_zip()
    with zipfile.ZipFile(BASE / "Jun09_SkillRanker_ComponentAwareMaskSelection_shareable.zip") as handle:
        bad = handle.testzip()
        if bad:
            raise RuntimeError(f"bad zip entry: {bad}")
    print(HTML)
    print(INDEX)
    print(BASE / "Jun09_SkillRanker_ComponentAwareMaskSelection_shareable.zip")


if __name__ == "__main__":
    main()
