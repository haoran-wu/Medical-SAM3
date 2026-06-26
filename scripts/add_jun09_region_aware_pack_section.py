#!/usr/bin/env python3
"""Append the Jun09 region-aware input pack to the skill-ranker report."""

from __future__ import annotations

import csv
import html
import zipfile
from pathlib import Path


ROOT = Path("/Users/haoranwu/Desktop/Yan_Lab_Research/Medical-SAM3")
BASE = ROOT / "output/visium_hd_exp1/final_deliverables/Jun09_SkillRanker_ComponentAwareMaskSelection"
PACK = BASE / "region_aware_piece_context_locator_pack"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def count_by(rows: list[dict[str, str]], key: str) -> dict[str, int]:
    out: dict[str, int] = {}
    for row in rows:
        out[row.get(key, "")] = out.get(row.get(key, ""), 0) + 1
    return out


def table(rows: list[dict[str, object]], cols: list[str]) -> str:
    parts = ["<table><thead><tr>"]
    parts.extend(f"<th>{html.escape(col)}</th>" for col in cols)
    parts.append("</tr></thead><tbody>")
    for row in rows:
        parts.append("<tr>")
        for col in cols:
            parts.append(f"<td>{html.escape(str(row.get(col, '')))}</td>")
        parts.append("</tr>")
    parts.append("</tbody></table>")
    return "".join(parts)


def main() -> None:
    rows = read_csv(PACK / "public_region_aware_requests.csv")
    by_class = count_by(rows, "true_class")
    summary_rows = [
        {"item": "candidate pieces", "value": len(rows)},
        {"item": "view files", "value": len(list((PACK / "views").glob("*")))},
        {"item": "views per candidate", "value": 6},
        {"item": "HTML", "value": str((PACK / "index.html").relative_to(BASE))},
        {"item": "prompt with FICTURE legend", "value": str((PACK / "prompt_region_aware_he_ficture.txt").relative_to(BASE))},
        {"item": "prompt H&E only", "value": str((PACK / "prompt_region_aware_he_only.txt").relative_to(BASE))},
        {"item": "hidden evaluation table", "value": str((PACK / "hidden_candidate_truth.csv").relative_to(BASE))},
        {"item": "Bouchet commands", "value": str((PACK / "run_bouchet_region_aware_vlm_commands.sh").relative_to(BASE))},
    ]
    class_rows = [{"class": k, "candidate pieces": v} for k, v in sorted(by_class.items())]
    example_rows = []
    for row in rows[:6]:
        example_rows.append(
            {
                "candidate": row["candidate_uid"],
                "true class": row["true_class"],
                "component": row["matched_annotation_component_id"],
                "D/P/R": f"{row['component_dice']} / {row['component_precision']} / {row['component_recall']}",
                "bbox": row["bbox_xyxy"],
            }
        )

    section = f"""
<h2>17B. Region-Aware Piece Scoring Input Pack</h2>
<p>This is the next scoring ablation prepared after the literature-guided failure map. The candidate mask is unchanged, but each piece now has three visual scales: isolated piece crop, larger local context, and full ROI locator. H&amp;E and FICTURE versions are both included. This directly tests whether the VLM/CLIP failure came from showing small pieces without tissue architecture.</p>
{table(summary_rows, ['item', 'value'])}
<h3>Exact image order sent to the local VLM</h3>
<ol>
<li>H&amp;E piece reverse-blur crop.</li>
<li>FICTURE piece reverse-blur crop.</li>
<li>H&amp;E local context crop.</li>
<li>FICTURE local context crop.</li>
<li>H&amp;E full ROI locator.</li>
<li>FICTURE full ROI locator.</li>
</ol>
<p>The prompt file has been checked against this CSV order. This matters because a previous draft had H&amp;E context and FICTURE piece images described in the wrong order, which would make any result uninterpretable.</p>
<h3>Candidate distribution</h3>
{table(class_rows, ['class', 'candidate pieces'])}
<h3>Example rows</h3>
{table(example_rows, ['candidate', 'true class', 'component', 'D/P/R', 'bbox'])}
<p><a href="{html.escape(str((PACK / 'index.html').relative_to(BASE)))}">Open the full region-aware input pack</a>. The full pack contains every candidate and all six views per candidate.</p>
<h3>Execution and failure gates</h3>
<table><thead><tr><th>Gate</th><th>Pass condition</th><th>If it fails</th></tr></thead><tbody>
<tr><td>18-row smoke</td><td>All rows parse into the six required JSON keys; no all-zero/all-tie output.</td><td>Fix image order, prompt file, parser, or model loader before any full run.</td></tr>
<tr><td>Score distribution</td><td>No dominant class collapse such as most candidates predicted tumor.</td><td>Compare H&amp;E-only, FICTURE-only, and all6; downgrade FICTURE if it causes collapse.</td></tr>
<tr><td>Piece Top1 by class</td><td>Bronchiola/vessels/immune improve versus piece-only or at least do not collapse.</td><td>Use class-specific prompts or structured features instead of raw VLM scoring.</td></tr>
<tr><td>Assembly</td><td>Selected pieces produce interpretable six-panel union figures and D/P/R per class.</td><td>Adjust score threshold, margin, NMS/overlap, or selected-piece cap one variable at a time.</td></tr>
</tbody></table>
<div class='callout'><b>Next test.</b> Run the same model/prompt comparison on piece-only versus piece+local-context+locator. If recognition improves for bronchiola/vessels without increasing tumor/stroma collapse, this becomes the default VLM/CLIP input style. If not, stop spending effort on raw VLM context and move to structured skill features plus second-SAM refinement.</div>
"""

    for html_path in [BASE / "Jun09_SkillRanker_ComponentAwareMaskSelection.html", BASE / "index.html"]:
        text = html_path.read_text()
        marker = "<h2>17B. Region-Aware Piece Scoring Input Pack</h2>"
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
        PACK,
        ROOT / "scripts/hpc/jun09_region_aware_multiimage_vlm.sbatch",
        ROOT / "scripts/hpc/jun09_region_aware_piece_assembly_cpu.sbatch",
    ]:
        if root.is_file():
            include.append(root)
        elif root.exists():
            include.extend(p for p in root.rglob("*") if p.is_file())
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as handle:
        seen: set[Path] = set()
        for p in include:
            if p in seen:
                continue
            seen.add(p)
            try:
                arcname = p.relative_to(BASE)
            except ValueError:
                arcname = Path("support_scripts") / p.relative_to(ROOT)
            handle.write(p, arcname)
    print(BASE / "Jun09_SkillRanker_ComponentAwareMaskSelection.html")
    print(zip_path)


if __name__ == "__main__":
    main()
