#!/usr/bin/env python3
"""Append a stricter failure hierarchy and iteration controller to Jun09.

This section is intentionally decision-oriented.  It turns the current Jun09
experiments into a checklist that decides which layer to change next for each
tissue class, so the project does not keep cycling through prompt edits after
the real failure layer has already been identified.
"""

from __future__ import annotations

import csv
import html
import re
import zipfile
from pathlib import Path

import pandas as pd


ROOT = Path("/Users/haoranwu/Desktop/Yan_Lab_Research/Medical-SAM3")
BASE = ROOT / "output/visium_hd_exp1/final_deliverables/Jun09_SkillRanker_ComponentAwareMaskSelection"
OUT = BASE / "failure_hierarchy_v4"


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


def read_csv(rel: str) -> pd.DataFrame:
    path = BASE / rel
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    controller = read_csv("failure_driven_hybrid_controller/current_best_class_specific_controller.csv")
    alveoli_gate = read_csv("alveoli_deployable_selector_gate/alveoli_deployable_selector_summary.csv")
    immune_fold = read_csv("immune_loco_validation/immune_component_fold_validation_summary.csv")

    # Keep the exact gate evidence in short strings so the report remains
    # readable to collaborators.
    alveoli_note = "not run"
    if len(alveoli_gate):
        combined = alveoli_gate[alveoli_gate["selector"] == "combined diagnostic selector"].iloc[0]
        alveoli_note = (
            f"candidate 12 rank {combined['rank of candidate 12']}, "
            f"candidate 13 rank {combined['rank of candidate 13']}; "
            f"82/90 ranks {combined['rank of airway FP 82']}/{combined['rank of immune/stroma FP 90']}"
        )
    immune_note = "not run"
    if len(immune_fold):
        row = immune_fold.iloc[0]
        immune_note = f"held-out component D/P/R {row['mean heldout D/P/R']}"

    layer_rows = [
        {
            "layer": "1. Proposal coverage",
            "question": "Does the candidate pool contain a good-enough mask or region before scoring?",
            "failure sign": "oracle/component upper bound is low, or true anatomy needs a broader proposal family",
            "next change": "change SAM prompt setting or class-specific proposal generator; do not tune VLM yet",
        },
        {
            "layer": "2. Region recognition",
            "question": "Given a reasonable candidate, can the scorer identify the tissue type?",
            "failure sign": "Piece Top1 collapse, tumor bias, raw FICTURE image misleading, or no score diversity",
            "next change": "use mask/context/locator views or structured priors; stop if repeated prompt variants fail",
        },
        {
            "layer": "3. Mask quality ranking",
            "question": "Among candidates for the right tissue, can the method prefer the better mask?",
            "failure sign": "good tissue type but bad boundary/fragmentation; score does not correlate with Dice",
            "next change": "add morphology/shape/context quality features or second-SAM refinement",
        },
        {
            "layer": "4. Assembly",
            "question": "Can multiple selected pieces be combined without precision collapse?",
            "failure sign": "same piece selected by multiple classes, false positives enter top union, recall/precision tradeoff unstable",
            "next change": "class-specific NMS, one-class assignment, target-vs-second margin, and precision/recall gates",
        },
        {
            "layer": "5. Validation",
            "question": "Does the rule survive held-out components or another ROI?",
            "failure sign": "full-annotation optimum is good but component-fold validation is weak",
            "next change": "mark as diagnostic only; freeze only simple rules that survive validation",
        },
    ]

    class_rows = [
        {
            "class": "bronchiola",
            "current best": "semantic FICTURE airway proposal, D/P/R 0.898/0.868/0.930",
            "failure layer now": "validation / boundary cleanup",
            "evidence": "airway factor recovers the disconnected components; proposal and tissue prior are strong",
            "next iteration": "freeze airway-factor proposal; optional second-SAM only for boundary cleanup",
            "stop doing": "generic VLM prompt sweeps on isolated pieces",
        },
        {
            "class": "alveoli",
            "current best": "broad H&E box384 union 12+13, D/P/R 0.748/0.653/0.875",
            "failure layer now": "deployable selector after proposal fix",
            "evidence": f"broad proposal works; annotation-free selector gate: {alveoli_note}; broad false positives still appear in top15",
            "next iteration": "replace location-heavy diagnostic selector with broad H&E morphology/context encoder, then recompute selected union",
            "stop doing": "using raw FICTURE alveolar color percentage as the main alveoli score",
        },
        {
            "class": "vessels",
            "current best": "runtime piece selector, D/P/R 0.718/0.921/0.588",
            "failure layer now": "recall/refinement after high-precision locator",
            "evidence": "precision is high; FICTURE stromal/smooth-muscle proposal is too broad and reduces precision",
            "next iteration": "use selected vessel pieces as prompts for second-SAM or boundary recall refinement",
            "stop doing": "turning broad stromal/smooth-muscle FICTURE into a vessel mask directly",
        },
        {
            "class": "tumor",
            "current best": "semantic FICTURE tumor proposal, D/P/R 0.732/0.718/0.746",
            "failure layer now": "diffuse compartment proposal validation",
            "evidence": "small-piece VLM/ranker assembly capped much lower; semantic tumor factors make coherent region",
            "next iteration": "validate tumor factor rule and add H&E morphology veto for necrosis/stroma-like false positives",
            "stop doing": "forcing tumor to be selected as many tiny independent pieces",
        },
        {
            "class": "stroma",
            "current best": "semantic FICTURE stromal/smooth-muscle proposal, D/P/R 0.604/0.534/0.697",
            "failure layer now": "semantic proposal overlap / validation",
            "evidence": "stroma factor captures broad compartment but overlaps vessel wall biology",
            "next iteration": "add vessel-wall veto or split stroma versus vessel smooth muscle before assembly",
            "stop doing": "treating all factor-1 smooth-muscle-like tissue as one final stroma mask",
        },
        {
            "class": "immune infiltration",
            "current best": "H&E nuclear-density + immune support, full D/P/R 0.554/0.586/0.524",
            "failure layer now": "validation",
            "evidence": f"full annotation optimum improved, but {immune_note}; rule is not stable enough",
            "next iteration": "try fixed nuclei-density / nuclei-segmentation features and validate by held-out components before freezing",
            "stop doing": "calling the full-annotation optimized immune rule deployable",
        },
    ]

    acceptance_rows = [
        {
            "gate": "usable proposal",
            "minimum evidence": "candidate family has oracle/diagnostic union near current best and false-positive examples are inspectable",
            "if not met": "change proposal generator first",
        },
        {
            "gate": "deployable selector",
            "minimum evidence": "selection uses no hidden annotation and ranks true candidates above known false positives",
            "if not met": "add morphology/context features or collect more annotated examples",
        },
        {
            "gate": "stable rule",
            "minimum evidence": "survives component-fold validation or another ROI; not just best-on-full-annotation",
            "if not met": "label as diagnostic only",
        },
        {
            "gate": "final mask",
            "minimum evidence": "six-panel figure is visually plausible and D/P/R tradeoff is class-appropriate",
            "if not met": "adjust assembly/refinement, not upstream prompt wording",
        },
    ]

    write_csv(OUT / "failure_layers_v4.csv", layer_rows)
    write_csv(OUT / "class_iteration_decisions_v4.csv", class_rows)
    write_csv(OUT / "acceptance_gates_v4.csv", acceptance_rows)

    section = f"""
<h2>17AL. Failure Hierarchy V4: Iterate Until The Right Layer Is Fixed</h2>
<p><b>Purpose.</b> This section is the operating rule for the next iterations. A result is not accepted just because code ran or a table exists. Each class must pass the earliest failing layer before later layers are tuned.</p>
<h3>Failure layers</h3>
{table(layer_rows, ['layer', 'question', 'failure sign', 'next change'])}
<h3>Class-specific decisions after the latest gates</h3>
{table(class_rows, ['class', 'current best', 'failure layer now', 'evidence', 'next iteration', 'stop doing'])}
<h3>Acceptance gates</h3>
{table(acceptance_rows, ['gate', 'minimum evidence', 'if not met'])}
<div class='callout'><b>Current decision.</b> The skill is no longer a single VLM prompt. It is a class-specific failure-driven controller. Bronchiola/tumor/stroma have useful semantic-FICTURE proposals; alveoli has a strong broad-H&amp;E proposal but still needs a deployable selector; vessels needs recall/refinement after a high-precision locator; immune remains diagnostic until validation improves.</div>
"""

    for html_path in [BASE / "Jun09_SkillRanker_ComponentAwareMaskSelection.html", BASE / "index.html"]:
        text = html_path.read_text()
        marker = "<h2>17AL. Failure Hierarchy V4: Iterate Until The Right Layer Is Fixed</h2>"
        if marker in text:
            start = text.index(marker)
            next_match = re.search(r"<h2>17A[M-Z]|<h2>18\\.", text[start + len(marker) :])
            end = start + len(marker) + next_match.start() if next_match else text.index("</body>", start)
            text = text[:start] + section + text[end:]
        else:
            text = text.replace("</body>", section + "</body>")
        html_path.write_text(text)

    rebuild_zip()
    verify_html_images(BASE / "Jun09_SkillRanker_ComponentAwareMaskSelection.html")
    print(OUT / "class_iteration_decisions_v4.csv")
    print(BASE / "Jun09_SkillRanker_ComponentAwareMaskSelection.html")


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
        BASE / "alveoli_deployable_selector_gate",
        BASE / "semantic_ficture_proposal_generator",
        BASE / "failure_driven_hybrid_controller",
        BASE / "immune_he_morphology_branch",
        BASE / "immune_loco_validation",
        BASE / "failure_analysis_protocol_v2",
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
