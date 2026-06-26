#!/usr/bin/env python3
"""Render a shareable June6 Clip report from score/assembly outputs."""

from __future__ import annotations

import argparse
import csv
import html
import json
from pathlib import Path


CLASS_ORDER = [
    "bronchiola",
    "alveoli",
    "vessels",
    "tumor",
    "stroma",
    "immune_infiltration",
]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def table(rows: list[dict[str, object]], cols: list[str]) -> str:
    out = ["<table><thead><tr>"]
    out.extend(f"<th>{html.escape(col)}</th>" for col in cols)
    out.append("</tr></thead><tbody>")
    for row in rows:
        out.append("<tr>")
        for col in cols:
            val = row.get(col, "")
            cls = " class='num'" if col.lower() in {"dice", "precision", "recall", "piece top1", "selected pieces"} else ""
            out.append(f"<td{cls}>{html.escape(str(val))}</td>")
        out.append("</tr>")
    out.append("</tbody></table>")
    return "".join(out)


def fmt3(value: object) -> str:
    try:
        return f"{float(value):.3f}"
    except Exception:
        return str(value)


def method_group(root: Path, method: str) -> str:
    text = str(root)
    if "model_openai_clip-vit-large-patch14" in text:
        return "CLIP-large"
    if "model_vinid_plip" in text:
        return "PLIP"
    if method.startswith("FICTURE_composition_prior"):
        return "FICTURE marker prior"
    return "CLIP-base"


def collect(root: Path) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    summary_rows: list[dict[str, object]] = []
    accuracy_rows: list[dict[str, object]] = []
    assembly_roots = [root / "assembly_outputs", root / "model_openai_clip-vit-large-patch14" / "assembly_outputs", root / "model_vinid_plip" / "assembly_outputs"]
    score_roots = [root / "score_outputs", root / "marker_prior_score_outputs", root / "model_openai_clip-vit-large-patch14" / "score_outputs", root / "model_vinid_plip" / "score_outputs"]

    for assembly_root in assembly_roots:
        if not assembly_root.exists():
            continue
        for path in sorted(assembly_root.glob("*/assembly_summary.csv")):
            method = path.parent.name
            group = method_group(assembly_root, method)
            for row in read_csv(path):
                fig = path.parent / row.get("figure_rel", "")
                summary_rows.append(
                    {
                        "Model family": group,
                        "Method": method,
                        "Tissue class": row["class"],
                        "Selected pieces": row["selected_piece_count"],
                        "Dice": fmt3(row["dice"]),
                        "Precision": fmt3(row["precision"]),
                        "Recall": fmt3(row["recall"]),
                        "Policy": row["chosen_policy"],
                        "Figure": str(fig.relative_to(root)) if fig.exists() else "",
                        "_dice": float(row["dice"]),
                        "_precision": float(row["precision"]),
                        "_recall": float(row["recall"]),
                    }
                )

    for score_root in score_roots:
        if not score_root.exists():
            continue
        for path in sorted(score_root.glob("*/overall_accuracy.csv")):
            method = path.parent.name
            group = method_group(score_root, method)
            row = read_csv(path)[0]
            accuracy_rows.append(
                {
                    "Model family": group,
                    "Method": method,
                    "Piece Top1": f"{row['correct']}/{row['total']}",
                    "Accuracy": fmt3(row["accuracy"]),
                }
            )
    return summary_rows, accuracy_rows


def best_by_class(summary_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    for class_name in CLASS_ORDER:
        rows = [row for row in summary_rows if row["Tissue class"] == class_name]
        if not rows:
            continue
        best = max(rows, key=lambda row: (float(row["_dice"]), float(row["_precision"])))
        out.append({key: best[key] for key in ["Tissue class", "Model family", "Method", "Selected pieces", "Dice", "Precision", "Recall", "Policy", "Figure"]})
    return out


def render(root: Path, output: Path) -> None:
    summary_rows, accuracy_rows = collect(root)
    best_rows = best_by_class(summary_rows)
    payload = {
        "root": str(root),
        "n_summary_rows": len(summary_rows),
        "n_accuracy_rows": len(accuracy_rows),
        "best_by_class": best_rows,
    }
    (root / "june6_clip_report_manifest.json").write_text(json.dumps(payload, indent=2))

    figure_sections: list[str] = []
    for row in best_rows:
        figure = row.get("Figure", "")
        if figure:
            figure_sections.append(
                "<section class='figure-block'>"
                f"<h3>{html.escape(str(row['Tissue class']))}: {html.escape(str(row['Model family']))} / {html.escape(str(row['Method']))}</h3>"
                f"<p>Dice / Precision / Recall = {html.escape(str(row['Dice']))} / {html.escape(str(row['Precision']))} / {html.escape(str(row['Recall']))}</p>"
                f"<img src='{html.escape(str(figure))}' alt='{html.escape(str(row['Tissue class']))} six-panel result'>"
                "</section>"
            )

    grouped_sections: list[str] = []
    for group in ["CLIP-base", "CLIP-large", "PLIP", "FICTURE marker prior"]:
        rows = [row for row in summary_rows if row["Model family"] == group]
        if not rows:
            continue
        grouped_sections.append(
            f"<h3>{html.escape(group)}</h3>"
            + table(rows, ["Method", "Tissue class", "Selected pieces", "Dice", "Precision", "Recall", "Policy"])
        )

    html_text = f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8">
<title>June6 Clip: H&E + FICTURE semantic latent piece ranker</title>
<style>
body {{ font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Arial,sans-serif; margin:34px; color:#111827; line-height:1.55; }}
h1 {{ margin-bottom:6px; }}
h2 {{ margin-top:34px; border-top:1px solid #e5e7eb; padding-top:22px; }}
.callout {{ background:#f8fafc; border-left:4px solid #2563eb; padding:14px 18px; margin:18px 0 24px; }}
.warn {{ background:#fff7ed; border-left-color:#f97316; }}
table {{ border-collapse:collapse; width:100%; margin:12px 0 26px; font-size:13px; }}
th,td {{ border-bottom:1px solid #e5e7eb; padding:8px 9px; text-align:left; vertical-align:top; }}
th {{ background:#f3f4f6; }}
td.num {{ font-variant-numeric:tabular-nums; }}
img {{ width:100%; border:1px solid #e5e7eb; display:block; }}
.figure-block {{ margin:26px 0 40px; }}
code {{ background:#f3f4f6; padding:1px 5px; border-radius:5px; }}
</style></head><body>
<h1>June6 Clip: H&E + FICTURE Semantic-Latent Piece Ranker</h1>
<p>This report tests whether a cheaper CLIP-style ranker can replace or support VLM scoring on the Jun5 167-candidate piece-first pool.</p>

<div class="callout">
<b>Input.</b> The model scores individual component/piece candidates, not final union masks. Each candidate comes from the <code>medical_official_points_step24</code> funnel and has paired H&E and official FICTURE gray reverse-blur crops. Union happens only after scoring.
</div>

<h2>What Was Tested</h2>
<table><thead><tr><th>Branch</th><th>What it uses</th><th>Why test it</th></tr></thead><tbody>
<tr><td>H&E CLIP</td><td>H&E crop image + tissue text prompts</td><td>Tests morphology signal: lumen, vessel wall, septa, tumor/stroma texture.</td></tr>
<tr><td>FICTURE image CLIP</td><td>FICTURE false-color crop image + tissue text prompts</td><td>Ablation: checks whether CLIP can use the color map directly.</td></tr>
<tr><td>FICTURE semantic latent</td><td>RGB/cell-type composition converted into CLIP text-embedding space</td><td>Uses FICTURE as cell-type semantics instead of ordinary image color.</td></tr>
<tr><td>FICTURE marker prior</td><td>Marker/cell-type-derived factor-to-class scores weighted by candidate composition</td><td>Non-CLIP semantic baseline grounded in the source-matched FICTURE legend.</td></tr>
<tr><td>Fusion</td><td>Weighted score combinations</td><td>Tests whether H&E morphology and FICTURE semantics complement each other.</td></tr>
</tbody></table>

<h2>Short Takeaway</h2>
<div class="callout warn">
Generic CLIP-base is weak. CLIP-large H&E is useful for vessels and moderately useful for immune/stroma. PLIP gives high precision for a small bronchiola piece and better immune recall, but it weakens vessels. Direct FICTURE-image CLIP is not reliable. FICTURE semantic/marker priors alone do not solve the task.
</div>

<h2>Piece-Level Top1</h2>
<p>Piece Top1 means: for each candidate piece, take the highest of the six class scores and compare it with the hidden component label. This is a diagnostic score, not the final mask score.</p>
{table(accuracy_rows, ["Model family", "Method", "Piece Top1", "Accuracy"])}

<h2>Best Final Assembly Found Per Class</h2>
<p>After each scoring method ranks pieces, the same assembly rule selects high-score, low-overlap pieces and unions them. This table shows the best Dice found per tissue class among all June6 Clip variants.</p>
{table(best_rows, ["Tissue class", "Model family", "Method", "Selected pieces", "Dice", "Precision", "Recall", "Policy"])}

<h2>Best Six-Panel Visual Checks</h2>
{''.join(figure_sections)}

<h2>All Assembly Tables</h2>
{''.join(grouped_sections)}

<h2>Interpretation</h2>
<ul>
<li><b>Use H&E morphology first.</b> The strongest signal came from H&E CLIP-large, especially vessels.</li>
<li><b>Do not trust direct FICTURE-color CLIP as a main branch.</b> It often selects visually colorful but biologically wrong pieces.</li>
<li><b>FICTURE semantics are useful as a prior, not a standalone selector.</b> Composition alone misses morphology and spatial structure.</li>
<li><b>Next improvement.</b> Train a lightweight candidate ranker over H&E embedding + FICTURE composition + shape features, rather than relying on zero-shot CLIP similarity alone.</li>
</ul>
</body></html>
"""
    output.write_text(html_text)
    print(output)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    render(args.root, args.output)


if __name__ == "__main__":
    main()
