#!/usr/bin/env python3
"""Append the Jun09 recovered-piece second-SAM smoke results to the report."""

from __future__ import annotations

import csv
import html
import re
import zipfile
from pathlib import Path

import add_jun09_precise_failure_framework_v5 as pack


ROOT = Path("/Users/haoranwu/Desktop/Yan_Lab_Research/Medical-SAM3")
BASE = ROOT / "output/visium_hd_exp1/final_deliverables/Jun09_SkillRanker_ComponentAwareMaskSelection"
PACK = BASE / "recovered_second_sam_prompt_pack"
HTML = BASE / "Jun09_SkillRanker_ComponentAwareMaskSelection.html"
INDEX = BASE / "index.html"
OUT = PACK / "second_sam_smoke_report"


RUNS = {
    "dry_run": PACK / "dry_run_prompt_preview",
    "bronchiola1_default_box_r96": PACK / "local_cpu_sam3_base_smoke_bronchiola1_conda",
    "bronchiola1_tight_box_r32": PACK / "local_cpu_sam3_base_smoke_bronchiola1_tightbox_r32",
    "bronchiola1_mid_box_r64": PACK / "local_cpu_sam3_base_smoke_bronchiola1_midbox_r64",
    "bronchiola_all_mid_box_r64": PACK / "local_cpu_sam3_base_bronchiola_all_midbox_r64",
    "vessels1_mid_box_r64": PACK / "local_cpu_sam3_base_smoke_vessels1_midbox_r64",
}


def read_csv(path: Path) -> list[dict[str, str]]:
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


def fmt3(value: str | float | None) -> str:
    if value is None or value == "":
        return ""
    try:
        return f"{float(value):.3f}"
    except Exception:
        return str(value)


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


def append_or_replace(section: str) -> None:
    marker = "<h2>17AY. True Second-SAM Smoke After Recovered-Piece Prompt Pack</h2>"
    for path in [HTML, INDEX]:
        text = path.read_text()
        if marker in text:
            start = text.index(marker)
            next_match = re.search(r"<h2>17A[Z]|<h2>17B|<h2>18\.", text[start + len(marker) :])
            end = start + len(marker) + next_match.start() if next_match else text.index("</body>", start)
            text = text[:start] + section + text[end:]
        else:
            text = text.replace("</body>", section + "</body>")
        path.write_text(text)


def verify_html_images(path: Path) -> None:
    text = path.read_text(errors="ignore")
    missing: list[str] = []
    for src in re.findall(r"<img[^>]+src=\"([^\"]+)\"", text):
        if src.startswith(("data:", "http://", "https://")):
            continue
        if not (path.parent / src).exists():
            missing.append(src)
    if missing:
        raise RuntimeError(f"Missing image assets in {path.name}: {missing[:8]}")


def collect_rows() -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    dry_rows: list[dict[str, object]] = []
    smoke_rows: list[dict[str, object]] = []
    piece_rows: list[dict[str, object]] = []

    for row in read_csv(RUNS["dry_run"] / "second_sam_prompt_preview_summary.csv"):
        dry_rows.append(
            {
                "class": row["class"],
                "selected pieces": row["selected_piece_count"],
                "selected-piece union Dice / Precision / Recall": (
                    f"{fmt3(row['prompt_preview_dice'])} / {fmt3(row['prompt_preview_precision'])} / "
                    f"{fmt3(row['prompt_preview_recall'])}"
                ),
                "meaning": "before second-SAM; selected/recovered pieces are directly unioned",
            }
        )

    for name, run_dir in RUNS.items():
        if name == "dry_run":
            continue
        summary_path = run_dir / "second_sam_summary.csv"
        if not summary_path.exists():
            continue
        row = read_csv(summary_path)[0]
        smoke_rows.append(
            {
                "run": name,
                "class": row["class"],
                "pieces": row["selected_piece_count"],
                "prior union D/P/R": (
                    f"{fmt3(row['prior_union_dice'])} / {fmt3(row['prior_union_precision'])} / "
                    f"{fmt3(row['prior_union_recall'])}"
                ),
                "second-SAM D/P/R": (
                    f"{fmt3(row['second_sam_dice'])} / {fmt3(row['second_sam_precision'])} / "
                    f"{fmt3(row['second_sam_recall'])}"
                ),
                "delta D/P/R": (
                    f"{fmt3(row['delta_dice'])} / {fmt3(row['delta_precision'])} / "
                    f"{fmt3(row['delta_recall'])}"
                ),
                "decision": (
                    "reject automatic replacement"
                    if float(row["delta_dice"]) < 0
                    else "tiny diagnostic gain only"
                ),
            }
        )
        piece_path = run_dir / "second_sam_piece_results.csv"
        if piece_path.exists():
            for piece in read_csv(piece_path):
                piece_rows.append(
                    {
                        "run": name,
                        "class": piece["class"],
                        "candidate": piece["candidate_uid"],
                        "prior D/P/R": (
                            f"{fmt3(piece['prior_dice'])} / {fmt3(piece['prior_precision'])} / "
                            f"{fmt3(piece['prior_recall'])}"
                        ),
                        "second-SAM D/P/R": (
                            f"{fmt3(piece['second_sam_dice'])} / {fmt3(piece['second_sam_precision'])} / "
                            f"{fmt3(piece['second_sam_recall'])}"
                        ),
                    }
                )

    decision_rows = [
        {
            "question": "Did second-SAM fix bronchiola?",
            "answer": "No as an automatic replacement.",
            "evidence": "All 4 bronchiola pieces: direct prior union 0.800/0.786/0.814; second-SAM midbox 0.789/0.765/0.815.",
            "next action": "Keep the selected-piece union as the current bronchiola mask; use recovered component proposals only as optional locators.",
        },
        {
            "question": "Did second-SAM fix vessels?",
            "answer": "No clear evidence.",
            "evidence": "One top vessel piece improved only from 0.135/1.000/0.072 to 0.142/1.000/0.077.",
            "next action": "Do not spend a full local run first; use proposal recovery and class-specific assembly before second-SAM.",
        },
        {
            "question": "What failed now?",
            "answer": "Boundary refinement is not the earliest failing layer.",
            "evidence": "Proxy dilation suggested possible recall gains, but true SAM3 box prompts did not improve Dice on smoke tests.",
            "next action": "Refine the skill by component coverage, proposal generator choice, and precision-aware assembly; second-SAM stays optional.",
        },
    ]

    write_csv(OUT / "second_sam_dry_run_summary.csv", dry_rows)
    write_csv(OUT / "second_sam_smoke_summary.csv", smoke_rows)
    write_csv(OUT / "second_sam_smoke_piece_results.csv", piece_rows)
    write_csv(OUT / "second_sam_smoke_decision.csv", decision_rows)
    return dry_rows, smoke_rows, decision_rows


def build_section() -> str:
    dry_rows, smoke_rows, decision_rows = collect_rows()
    rel = lambda p: html.escape(str(p.relative_to(BASE)))
    figures = [
        (
            "Dry-run prompt pack: bronchiola selected/recovered pieces before second-SAM",
            RUNS["dry_run"] / "figures/bronchiola_selected_piece_prompt_preview.png",
        ),
        (
            "Dry-run prompt pack: vessels selected/recovered pieces before second-SAM",
            RUNS["dry_run"] / "figures/vessels_selected_piece_prompt_preview.png",
        ),
        (
            "True second-SAM smoke: bronchiola all selected pieces, mid box/radius",
            RUNS["bronchiola_all_mid_box_r64"] / "figures/bronchiola_second_sam_union.png",
        ),
        (
            "True second-SAM smoke: top vessel piece, mid box/radius",
            RUNS["vessels1_mid_box_r64"] / "figures/vessels_second_sam_union.png",
        ),
    ]
    figure_html = "\n".join(
        f"<figure><img src=\"{rel(path)}\" alt=\"{html.escape(title)}\"><figcaption>{html.escape(title)}</figcaption></figure>"
        for title, path in figures
        if path.exists()
    )

    return f"""
<h2>17AY. True Second-SAM Smoke After Recovered-Piece Prompt Pack</h2>
<p><b>Goal.</b> Earlier sections showed a possible boundary-refinement signal: small local dilation could raise recall for bronchiola and vessels.  This section tests that idea with a real SAM3/Medical-SAM3 box-prompt smoke run instead of treating the morphology proxy as final evidence.</p>
<p><b>Design.</b> Selected candidate pieces are used only as spatial prompts.  For each selected piece, the script builds a local box prompt, asks SAM3 to segment inside that box, keeps the component near the original piece, and then unions the refined pieces.  Annotation is hidden from the prompt and used only after prediction to compute Dice, Precision, and Recall.</p>
<p><b>Execution note.</b> Bouchet SSH was not available in batch mode during this pass, so I ran a local conda <code>medsam3</code> smoke.  The plain Python environment failed before inference because <code>iopath</code> was missing; the conda environment loaded SAM3 successfully.  MPS loaded the model but failed on an unsupported PyTorch MPS box-prompt operator, so the completed smoke runs used CPU.  This is acceptable for validation, but full second-SAM should be run on Bouchet GPU once SSH/Duo is available.</p>
<h3>Prompt-pack preview before second-SAM</h3>
{table(dry_rows, ["class", "selected pieces", "selected-piece union Dice / Precision / Recall", "meaning"])}
<h3>Actual SAM3 smoke results</h3>
{table(smoke_rows, ["run", "class", "pieces", "prior union D/P/R", "second-SAM D/P/R", "delta D/P/R", "decision"])}
<h3>Decision from this refinement gate</h3>
{table(decision_rows, ["question", "answer", "evidence", "next action"])}
<h3>Six-panel figures</h3>
<div class="grid">
{figure_html}
</div>
<div class='callout'><b>Updated failure diagnosis.</b> The second-SAM smoke makes the failure more precise: the current bronchiola/vessel issue is not solved by automatically replacing selected pieces with SAM3 box-prompt outputs.  The safer next iteration is component-aware proposal recovery plus precision-aware assembly.  Second-SAM should remain a controlled optional refinement step, not the main skill-ranker decision.</div>
"""


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    section = build_section()
    append_or_replace(section)
    verify_html_images(HTML)
    verify_html_images(INDEX)
    pack.rebuild_zip()
    zip_path = BASE / "Jun09_SkillRanker_ComponentAwareMaskSelection_shareable.zip"
    with zipfile.ZipFile(zip_path) as handle:
        bad = handle.testzip()
        if bad:
            raise RuntimeError(f"bad zip entry: {bad}")
    print(HTML)
    print(zip_path)


if __name__ == "__main__":
    main()
