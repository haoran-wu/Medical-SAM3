#!/usr/bin/env python3
"""Append annotation-free vessels veto diagnostic to the Jun09 report.

This section is deliberately separate from the oracle false-positive veto:
it asks whether a deployable H&E/shape guard can remove obvious vessel false
positives without using annotation at inference time.
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
OUT = BASE / "recovered_second_sam_prompt_pack/vessels_annotation_free_veto_diagnostic"
HTML = BASE / "Jun09_SkillRanker_ComponentAwareMaskSelection.html"
INDEX = BASE / "index.html"


def read_csv(path: Path) -> list[dict[str, str]]:
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


def append_or_replace(section: str) -> None:
    marker = "<h2>17BB. Annotation-Free Vessels Veto: Deployable Precision Guard</h2>"
    for path in [HTML, INDEX]:
        text = path.read_text()
        if marker in text:
            start = text.index(marker)
            next_match = re.search(r"<h2>17B[C-Z]|<h2>18\.", text[start + len(marker) :])
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
    summary = read_csv(OUT / "vessels_annotation_free_veto_summary.csv")
    features = read_csv(OUT / "vessels_annotation_free_veto_piece_features.csv")

    summary_rows = []
    for row in summary:
        summary_rows.append(
            {
                "rule": row["rule"],
                "kept pieces": row["kept_count"],
                "Dice / Precision / Recall": f"{float(row['dice']):.3f} / {float(row['precision']):.3f} / {float(row['recall']):.3f}",
                "meaning": row["note"],
            }
        )

    feature_rows = []
    for row in features:
        feature_rows.append(
            {
                "candidate": row["candidate_uid"],
                "rank": row["rank"],
                "target score": row["target_score"],
                "H&E vessel / stroma / immune": f"{row['he_vessels']} / {row['he_stroma']} / {row['he_immune']}",
                "area": f"{float(row['area_frac']):.4f}",
                "keep by deployable veto": row["runtime_veto_keep"],
                "hidden truth after evaluation": row["hidden_true_class_after_eval"],
                "hidden component Dice": f"{float(row['hidden_component_dice_after_eval']):.3f}",
            }
        )

    fig = OUT / "vessels_annotation_free_he_veto_six_panel.png"
    section = f"""
<h2>17BB. Annotation-Free Vessels Veto: Deployable Precision Guard</h2>
<p><b>Goal.</b> Section 17BA used hidden labels after scoring to ask whether vessel mistakes were false positives or missing components.  This section removes that oracle: it tests whether a deployable vessels guard can reject obvious non-vessel pieces using only H&amp;E-derived tissue scores, simple shape, and the candidate mask.</p>
<p><b>Method.</b> Start from the eight runtime-selected vessel pieces.  The deployable rule keeps a piece only when its H&amp;E vessel score is high enough and it is not strongly stroma- or immune-dominant.  The annotation is used only after the rule runs to compute Dice / Precision / Recall.</p>
<h3>Annotation-free veto rules</h3>
{table(summary_rows, ["rule", "kept pieces", "Dice / Precision / Recall", "meaning"])}
<h3>Piece-level decision audit</h3>
{table(feature_rows, ["candidate", "rank", "target score", "H&E vessel / stroma / immune", "area", "keep by deployable veto", "hidden truth after evaluation", "hidden component Dice"])}
<figure><img src="{html.escape(str(fig.relative_to(BASE)))}" alt="annotation-free vessels veto six-panel"><figcaption>Annotation-free vessels veto: the H&amp;E/shape guard improves precision by removing two false positives, but it also removes one weak true vessel piece, so recall and Dice drop.</figcaption></figure>
<div class='callout'><b>Updated vessels decision.</b> This guard is useful as a precision safety check, not as the final vessel solution.  It raises precision from 0.921 to 0.974, but recall falls from 0.588 to 0.547.  The next real improvement must recover cleaner vessel components, especially the missing/weak vessel pieces, before assembly.</div>
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
