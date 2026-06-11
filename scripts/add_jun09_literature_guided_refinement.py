#!/usr/bin/env python3
"""Add paper-guided refinement branches to the Jun09 report."""

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
    out.extend(f"<th>{html.escape(col)}</th>" for col in cols)
    out.append("</tr></thead><tbody>")
    for row in rows:
        out.append("<tr>")
        for col in cols:
            val = str(row.get(col, ""))
            if col == "source":
                out.append(f"<td><a href='{html.escape(val)}'>{html.escape(row.get('paper', val))}</a></td>")
            else:
                out.append(f"<td>{html.escape(val)}</td>")
        out.append("</tr>")
    out.append("</tbody></table>")
    return "".join(out)


def main() -> None:
    rows = [
        {
            "idea": "Region-aware scoring, not isolated crop scoring",
            "paper": "RegionCLIP",
            "source": "https://arxiv.org/abs/2112.09106",
            "failure addressed": "Small bronchiola/vessel wall pieces lack lumen/context, so VLM/CLIP mistakes them for tumor or stroma.",
            "implementation in this project": "Score each candidate with piece crop + local context + full ROI locator; keep the mask fixed but give the scorer surrounding structure.",
            "pass/fail test": "Piece Top1 and target-vs-rest AUC must improve for bronchiola/vessels without causing tumor collapse.",
        },
        {
            "idea": "Mask-indicated region focus with context preserved",
            "paper": "Alpha-CLIP",
            "source": "https://arxiv.org/abs/2312.03818",
            "failure addressed": "A normal image-text model can attend to the whole crop and ignore which small mask is the target.",
            "implementation in this project": "Use piece/local/locator views and structured mask features to mark the target while preserving surrounding anatomy.",
            "pass/fail test": "Target score should change when the marked piece changes inside the same local context; otherwise the scorer is reading background context only.",
        },
        {
            "idea": "Proposal-rerank-second-SAM cascade",
            "paper": "SaLIP",
            "source": "https://openaccess.thecvf.com/content/CVPR2024W/DEF-AI-MIA/html/Aleem_Test-Time_Adaptation_with_SaLIP_A_Cascade_of_SAM_and_CLIP_CVPRW_2024_paper.html",
            "failure addressed": "A selected small piece may identify the right region but be too fragmented as a final mask.",
            "implementation in this project": "Use selected piece bbox/center/edge points as a second SAM prompt; compare second-SAM mask against raw piece union.",
            "pass/fail test": "Second SAM must improve recall/boundary quality while keeping precision above the raw selected-piece union.",
        },
        {
            "idea": "Pathology multi-scale context",
            "paper": "Prov-GigaPath",
            "source": "https://www.nature.com/articles/s41586-024-07441-w",
            "failure addressed": "Pathology tissue identity often depends on larger tissue architecture, not only local texture.",
            "implementation in this project": "Use H&E morphology skill from local crop plus larger-window context features; treat FICTURE as structured prior, not as a raw RGB image.",
            "pass/fail test": "Context features should reduce bronchiola/vessel/stroma confusion and improve LOCO stability.",
        },
        {
            "idea": "Generator upper bound before scorer tuning",
            "paper": "Project diagnostic rule derived from proposal-based segmentation papers",
            "source": "https://openaccess.thecvf.com/content/CVPR2024W/DEF-AI-MIA/html/Aleem_Test-Time_Adaptation_with_SaLIP_A_Cascade_of_SAM_and_CLIP_CVPRW_2024_paper.html",
            "failure addressed": "If the candidate pool misses a tissue component, no VLM/ranker prompt can recover it.",
            "implementation in this project": "Run minimal three-setting component oracle before more prompt/API work; use class-specific generators when the upper bound fails.",
            "pass/fail test": "Each class must have a plausible component-level oracle before we call a scoring method bad or good.",
        },
    ]
    experiments = [
        {
            "priority": 1,
            "branch": "Minimal three-setting component oracle",
            "why now": "Separates generator failure from ranker failure and avoids the impractical all-48 pool.",
            "decision if good": "Use the smaller generator set as the main proposal pool.",
            "decision if bad": "Class-specific generator is required; do not keep tuning VLM on a missing candidate pool.",
        },
        {
            "priority": 2,
            "branch": "Region-aware scoring ablation",
            "why now": "Directly tests the small-piece context problem that caused tumor/stroma bias.",
            "decision if good": "Use locator/context views for VLM/CLIP/ranker features.",
            "decision if bad": "Stop spending API on region VLM; move to structured features and second SAM.",
        },
        {
            "priority": 3,
            "branch": "Second-SAM refinement",
            "why now": "A piece can be a good locator even when its mask is incomplete.",
            "decision if good": "Final method becomes proposal scorer plus SAM refinement.",
            "decision if bad": "Assembly must stay piece-union based with stricter morphology guards.",
        },
        {
            "priority": 4,
            "branch": "Structured FICTURE prior only",
            "why now": "Raw FICTURE RGB hurt VLM; numeric composition may still help.",
            "decision if good": "Keep FICTURE as a skill feature.",
            "decision if bad": "Demote FICTURE to audit/context only for this ROI.",
        },
    ]

    out_dir = BASE / "refined_diagnostics"
    write_csv(out_dir / "literature_guided_failure_branches.csv", rows)
    write_csv(out_dir / "literature_guided_next_experiments.csv", experiments)

    section = f"""
<h2>17A. Literature-Guided Refinement Map</h2>
<p>This section turns related paper ideas into concrete failure tests for this project. The point is not to cite papers as decoration. Each idea is linked to one failure gate and one executable branch.</p>
{table(rows, ['idea', 'source', 'failure addressed', 'implementation in this project', 'pass/fail test'])}
<h3>Next experiment order</h3>
{table(experiments, ['priority', 'branch', 'why now', 'decision if good', 'decision if bad'])}
<div class='callout'><b>Practical rule.</b> Do not keep changing the prompt if the proposal coverage gate fails. Do not run second-SAM until we know whether a candidate can reliably identify the correct local region. Do not use raw FICTURE RGB as a primary VLM image unless it improves recognition rather than increasing tumor/stroma bias.</div>
"""

    for html_path in [BASE / "Jun09_SkillRanker_ComponentAwareMaskSelection.html", BASE / "index.html"]:
        text = html_path.read_text()
        marker = "<h2>17A. Literature-Guided Refinement Map</h2>"
        if marker in text:
            start = text.index(marker)
            end = text.index("<h2>18.", start) if "<h2>18." in text[start:] else text.index("</body>", start)
            text = text[:start] + section + text[end:]
        elif "<h2>17. Iterative Failure-Analysis Protocol</h2>" in text:
            insert_at = text.index("</body>")
            text = text[:insert_at] + section + text[insert_at:]
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
    ]:
        if root.exists():
            include.extend(p for p in root.rglob("*") if p.is_file())
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as handle:
        seen: set[Path] = set()
        for p in include:
            if p in seen:
                continue
            seen.add(p)
            handle.write(p, p.relative_to(BASE))

    print(out_dir / "literature_guided_next_experiments.csv")
    print(zip_path)


if __name__ == "__main__":
    main()
