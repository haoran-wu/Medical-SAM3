#!/usr/bin/env python3
"""Add the Jun09 selected-piece second-SAM refinement gate to the report."""

from __future__ import annotations

import csv
import html
import re
import shutil
import zipfile
from pathlib import Path


ROOT = Path("/Users/haoranwu/Desktop/Yan_Lab_Research/Medical-SAM3")
BASE = ROOT / "output/visium_hd_exp1/final_deliverables/Jun09_SkillRanker_ComponentAwareMaskSelection"
OUT = BASE / "second_sam_refinement"


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
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


def table(rows: list[dict[str, object]], cols: list[str]) -> str:
    if not rows:
        return "<p class='small'>No rows yet.</p>"
    parts = ["<table><thead><tr>"]
    parts.extend(f"<th>{html.escape(col)}</th>" for col in cols)
    parts.append("</tr></thead><tbody>")
    for row in rows:
        parts.append("<tr>")
        for col in cols:
            val = row.get(col, "")
            parts.append(f"<td>{html.escape(str(val))}</td>")
        parts.append("</tr>")
    parts.append("</tbody></table>")
    return "".join(parts)


def replace_section(text: str, title: str, section: str) -> str:
    marker = f"<h2>{title}</h2>"
    if marker in text:
        start = text.index(marker)
        match = re.search(r"<h2[^>]*>", text[start + 1 :])
        end = start + 1 + match.start() if match else text.index("</body>", start)
        return text[:start] + section + text[end:]
    return text.replace("</body>", section + "</body>")


def summarize_pending() -> tuple[list[dict[str, object]], str]:
    selected = read_csv(BASE / "refined_diagnostics/refined_policy_selected_pieces.csv")
    by_class: dict[str, int] = {}
    for row in selected:
        by_class[row["class"]] = by_class.get(row["class"], 0) + 1
    rows = [
        {
            "class": cls,
            "selected pieces to prompt SAM": count,
            "planned prompt": "box from selected piece mask",
            "planned clamp": "keep SAM component near original piece context",
        }
        for cls, count in sorted(by_class.items())
    ]
    return rows, "pending"


def summarize_completed() -> tuple[list[dict[str, object]], str]:
    preview = read_csv(OUT / "second_sam_prompt_preview_summary.csv")
    summary = read_csv(OUT / "second_sam_summary.csv")
    if not summary:
        if preview:
            rows = [
                {
                    "class": row.get("class", ""),
                    "selected pieces": row.get("selected_piece_count", ""),
                    "selected-piece preview D/P/R": (
                        f"{float(row.get('prompt_preview_dice', 0)):.3f} / "
                        f"{float(row.get('prompt_preview_precision', 0)):.3f} / "
                        f"{float(row.get('prompt_preview_recall', 0)):.3f}"
                    ),
                    "second-SAM status": "not run yet",
                }
                for row in preview
            ]
            return rows, "prompt_preview_ready"
        return summarize_pending()
    rows: list[dict[str, object]] = []
    for row in summary:
        rows.append(
            {
                "class": row.get("class", ""),
                "selected pieces": row.get("selected_piece_count", ""),
                "piece union D/P/R": (
                    f"{float(row.get('prior_union_dice', 0)):.3f} / "
                    f"{float(row.get('prior_union_precision', 0)):.3f} / "
                    f"{float(row.get('prior_union_recall', 0)):.3f}"
                ),
                "second-SAM D/P/R": (
                    f"{float(row.get('second_sam_dice', 0)):.3f} / "
                    f"{float(row.get('second_sam_precision', 0)):.3f} / "
                    f"{float(row.get('second_sam_recall', 0)):.3f}"
                ),
                "delta D/P/R": (
                    f"{float(row.get('delta_dice', 0)):+.3f} / "
                    f"{float(row.get('delta_precision', 0)):+.3f} / "
                    f"{float(row.get('delta_recall', 0)):+.3f}"
                ),
            }
        )
    return rows, "completed"


def copy_remote_results(src: Path) -> None:
    if not src.exists():
        raise SystemExit(f"Missing second-SAM result source: {src}")
    if OUT.exists():
        shutil.rmtree(OUT)
    shutil.copytree(src, OUT)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--copy-results-from", type=Path)
    args = parser.parse_args()

    if args.copy_results_from:
        copy_remote_results(args.copy_results_from)

    rows, status = summarize_completed()
    write_csv(BASE / "refined_diagnostics/second_sam_refinement_gate_status.csv", rows)

    figure_parts: list[str] = []
    if status in {"completed", "prompt_preview_ready"}:
        figure_source = (
            read_csv(OUT / "second_sam_summary.csv")
            if status == "completed"
            else read_csv(OUT / "second_sam_prompt_preview_summary.csv")
        )
        for row in figure_source:
            rel = row.get("figure_rel", "")
            if not rel:
                continue
            fig = OUT / rel
            if fig.exists():
                label = "second SAM refinement" if status == "completed" else "second SAM prompt preview"
                figure_parts.append(
                    "<section class='class-block'>"
                    f"<h3>{html.escape(row.get('class', ''))}</h3>"
                    f"<img src='{html.escape(str(fig.relative_to(BASE)))}' alt='{label} {html.escape(row.get('class', ''))}'>"
                    "</section>"
                )

    section = f"""
<h2>17E. Selected-Piece Second-SAM Refinement Gate</h2>
<p>This branch tests a SaLIP-style idea: a selected small candidate piece may locate the right tissue structure, even if the piece mask itself is too fragmented. In this gate, the selected piece is converted into a local box prompt for SAM3/Medical-SAM3. The annotation is still hidden during prompting and used only afterward for Dice, Precision, and Recall.</p>
<table><thead><tr><th>Step</th><th>What happens</th><th>Why it matters</th></tr></thead><tbody>
<tr><td>1. selected piece</td><td>Use the current skill/ranker-selected component piece as a spatial locator.</td><td>Tests whether the ranker found the right local tissue region.</td></tr>
<tr><td>2. second SAM prompt</td><td>Turn the piece mask into a bounding box and run SAM3 on the official H&amp;E ROI.</td><td>Lets SAM recover a cleaner or more complete local mask.</td></tr>
<tr><td>3. prior clamp</td><td>Keep the SAM component near the original piece context.</td><td>Prevents a box prompt from leaking into unrelated tissue.</td></tr>
<tr><td>4. union and evaluate</td><td>Union refined pieces per class and compute Dice / Precision / Recall.</td><td>Decides whether second-SAM improves the final segmentation or only adds noise.</td></tr>
</tbody></table>
<h3>Status</h3>
<p><b>{html.escape(status)}</b>. If pending, this section lists the pieces that will be used as prompts. If prompt preview is ready, the figures show the selected pieces and prompt boxes before SAM3 is run. If completed, it compares the original selected-piece union against second-SAM refined union.</p>
{table(rows, list(rows[0].keys()) if rows else [])}
<h3>Failure interpretation</h3>
<table><thead><tr><th>Observed result</th><th>Interpretation</th><th>Next refinement</th></tr></thead><tbody>
<tr><td>Dice/Recall improves and Precision stays acceptable</td><td>The piece was a good locator but the raw piece mask was too fragmented.</td><td>Use second-SAM for that class.</td></tr>
<tr><td>Recall improves but Precision collapses</td><td>The box prompt leaks outside the target structure.</td><td>Use a tighter prior radius, smaller box margin, or negative/tissue guard.</td></tr>
<tr><td>Second-SAM is worse than piece union</td><td>The selected piece is already better than a box-refined SAM mask, or SAM is not aligned to this tissue boundary.</td><td>Keep component union and improve assembly/score calibration instead.</td></tr>
<tr><td>Only some classes improve</td><td>The failure is class-specific, not a global VLM/SAM failure.</td><td>Use second-SAM only for those classes and keep separate policies for broad tissues.</td></tr>
</tbody></table>
{''.join(figure_parts)}
<p class='small'>Executable script: <code>inference/visium_hd_exp1/run_jun09_selected_piece_second_sam_refine.py</code>. Bouchet script: <code>scripts/hpc/jun09_selected_piece_second_sam_refine.sbatch</code>.</p>
"""

    for html_path in [BASE / "Jun09_SkillRanker_ComponentAwareMaskSelection.html", BASE / "index.html"]:
        text = html_path.read_text()
        html_path.write_text(replace_section(text, "17E. Selected-Piece Second-SAM Refinement Gate", section))

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
        BASE / "region_aware_piece_context_locator_pack",
        BASE / "region_aware_vlm_results",
        OUT,
        ROOT / "inference/visium_hd_exp1/run_jun09_selected_piece_second_sam_refine.py",
        ROOT / "scripts/hpc/jun09_selected_piece_second_sam_refine.sbatch",
        ROOT / "scripts/collect_jun09_second_sam_refinement_results.py",
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
    print(BASE / "refined_diagnostics/second_sam_refinement_gate_status.csv")
    print(zip_path)


if __name__ == "__main__":
    main()
