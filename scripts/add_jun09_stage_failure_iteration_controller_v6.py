#!/usr/bin/env python3
"""Append a stage-by-stage failure controller to the Jun09 report.

The goal is to prevent the skill-ranker line from becoming a pile of model
scores.  For each tissue class we name the earliest failing layer and the next
allowed refinement.  This makes the workflow iterative: fix the earliest
failed gate first, then rerun only the necessary downstream step.
"""

from __future__ import annotations

import csv
import html
import re
import zipfile
from pathlib import Path


ROOT = Path("/Users/haoranwu/Desktop/Yan_Lab_Research/Medical-SAM3")
BASE = ROOT / "output/visium_hd_exp1/final_deliverables/Jun09_SkillRanker_ComponentAwareMaskSelection"
OUT = BASE / "stage_failure_iteration_controller_v6"


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
    marker = "<h2>17AS. Stage-by-Stage Failure Controller V6</h2>"
    if marker in text:
        start = text.index(marker)
        next_match = re.search(r"<h2>17A[T-Z]|<h2>18\.", text[start + len(marker):])
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
        raise RuntimeError(f"Missing {len(missing)} image assets: {missing[:5]}")


def rebuild_zip() -> None:
    import importlib.util

    script = ROOT / "scripts/add_jun09_precise_failure_framework_v5.py"
    spec = importlib.util.spec_from_file_location("pack", script)
    pack = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(pack)
    pack.rebuild_zip()
    zip_path = BASE / "Jun09_SkillRanker_ComponentAwareMaskSelection_shareable.zip"
    with zipfile.ZipFile(zip_path) as handle:
        bad = handle.testzip()
        names = set(handle.namelist())
    if bad:
        raise RuntimeError(f"bad zip entry: {bad}")
    required = ["stage_failure_iteration_controller_v6/stage_failure_controller_v6.csv"]
    missing = [name for name in required if name not in names]
    if missing:
        raise RuntimeError(f"zip missing required entries: {missing}")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    stage_rows = [
        {
            "class": "bronchiola",
            "proposal pool": "PASS: useful airway pieces exist",
            "tissue recognition": "PASS: piece Top1 14/14",
            "target grounding": "mostly pass",
            "mask quality / assembly": "FAIL: combined final 0.614/0.545/0.703 vs oracle greedy 0.807/0.902/0.730",
            "validation": "not final",
            "earliest failing layer": "mask quality / boundary refinement",
            "next allowed change": "second-SAM from selected airway pieces or stricter boundary-quality skill; do not spend first effort on another VLM prompt",
        },
        {
            "class": "alveoli",
            "proposal pool": "medpt24 weak, broad H&E box384 branch passes",
            "tissue recognition": "not the main issue in broad branch",
            "target grounding": "V2 improves: ranks 13/11/12 top and demotes morphology-like FP 48",
            "mask quality / assembly": "PARTIAL: approximate V2 assembly 13+12 gives 0.687/0.583/0.835; exact saved 12+13 is 0.748/0.653/0.875",
            "validation": "blocked by missing exact broad-box masks locally",
            "earliest failing layer": "exact mask recovery + target-grounded assembly confirmation",
            "next allowed change": "recover exact broad-box masks; recompute V2-selected union exactly; only then test second-SAM or more features",
        },
        {
            "class": "vessels",
            "proposal pool": "PASS: useful vessel pieces exist",
            "tissue recognition": "PASS: piece Top1 24/24",
            "target grounding": "mostly pass",
            "mask quality / assembly": "FAIL: high precision but recall misses components; combined 0.716/0.871/0.608 vs oracle 0.750/0.956/0.617",
            "validation": "not final",
            "earliest failing layer": "recall/boundary refinement",
            "next allowed change": "second-SAM from selected vessel wall pieces; add rule that expands local wall/lumen boundaries only if precision remains high",
        },
        {
            "class": "tumor",
            "proposal pool": "PASS under semantic FICTURE proposal; weak under isolated small pieces",
            "tissue recognition": "piece Top1 is high but not enough for diffuse tumor coverage",
            "target grounding": "mixed",
            "mask quality / assembly": "FAIL: diffuse tissue coverage and false-positive veto; combined 0.401/0.660/0.288",
            "validation": "not final",
            "earliest failing layer": "diffuse compartment mask-quality selection",
            "next allowed change": "use broad semantic FICTURE tumor proposal plus H&E false-positive veto; do not rely on isolated pieces alone",
        },
        {
            "class": "stroma",
            "proposal pool": "pieces exist but class is broad",
            "tissue recognition": "piece Top1 32/32, so label recognition is not the bottleneck",
            "target grounding": "mixed because stroma overlaps vessel wall and smooth muscle",
            "mask quality / assembly": "FAIL: combined 0.054/0.830/0.028, oracle 0.397/0.503/0.328",
            "validation": "not final",
            "earliest failing layer": "class definition + assembly recall",
            "next allowed change": "split stroma into stromal matrix vs vascular smooth-muscle context; then reassemble with class-specific recall rule",
        },
        {
            "class": "immune infiltration",
            "proposal pool": "PASS: many immune-like pieces exist",
            "tissue recognition": "mostly pass: piece Top1 68/70",
            "target grounding": "mostly pass but small aggregates are noisy",
            "mask quality / assembly": "PARTIAL: combined 0.462/0.543/0.402; oracle 0.518/0.523/0.512",
            "validation": "FAIL/weak: component-fold validation remains unstable",
            "earliest failing layer": "validation stability",
            "next allowed change": "freeze annotation-free nuclear-density and immune-support features; test held-out components before accepting the rule",
        },
    ]

    iteration_rows = [
        {
            "rule": "Fix the earliest failed gate first",
            "meaning": "If the proposal pool lacks a useful mask, do not tune VLM or ranker. If recognition passes but union fails, tune assembly or second-SAM instead.",
        },
        {
            "rule": "One change per iteration",
            "meaning": "Change one threshold, feature family, or proposal generator at a time, then compare Dice/Precision/Recall and candidate-level figures.",
        },
        {
            "rule": "Hidden annotation stays after scoring",
            "meaning": "Annotation can diagnose and evaluate, but the selector feature itself must be computable without annotation.",
        },
        {
            "rule": "Exact-mask gate before final claim",
            "meaning": "Approximate crop reconstruction can guide debugging, but final numbers require exact binary masks.",
        },
        {
            "rule": "Stop bad directions explicitly",
            "meaning": "Do not continue raw FICTURE-as-image VLM prompting, global prompt sweeps, or adding more union pieces only to increase recall.",
        },
    ]

    write_csv(OUT / "stage_failure_controller_v6.csv", stage_rows)
    write_csv(OUT / "iteration_rules_v6.csv", iteration_rows)

    section = f"""
<h2>17AS. Stage-by-Stage Failure Controller V6</h2>
<p><b>Purpose.</b> This section turns the skill-ranker into an iterative diagnostic system. Instead of asking only “which model scored higher,” it asks where each tissue class first fails: candidate proposal, tissue recognition, target grounding, mask-quality ranking, assembly, or validation. The next experiment is only allowed to change the earliest failed layer.</p>
<h3>Class-level failure location</h3>
{table(stage_rows, ['class', 'proposal pool', 'tissue recognition', 'target grounding', 'mask quality / assembly', 'validation', 'earliest failing layer', 'next allowed change'])}
<h3>Iteration rules</h3>
{table(iteration_rows, ['rule', 'meaning'])}
<div class='callout'><b>Current conclusion.</b> The skill line is not failing at one universal place. Bronchiola and vessels need boundary/second-SAM refinement. Alveoli needs exact broad-box mask recovery plus target-grounded assembly confirmation. Tumor and stroma need broader compartment-specific proposal/quality control. Immune needs validation stability. This is why another global VLM prompt sweep is not the right next move.</div>
"""
    for name in ["Jun09_SkillRanker_ComponentAwareMaskSelection.html", "index.html"]:
        replace_or_append(BASE / name, section)
        verify_html_images(BASE / name)
    rebuild_zip()
    print(OUT / "stage_failure_controller_v6.csv")
    print(BASE / "Jun09_SkillRanker_ComponentAwareMaskSelection.html")


if __name__ == "__main__":
    main()
