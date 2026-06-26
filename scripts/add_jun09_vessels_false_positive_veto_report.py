#!/usr/bin/env python3
"""Append vessels false-positive veto diagnostic to Jun09 report."""

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
OUT = PACK / "vessels_false_positive_veto_diagnostic"
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
    marker = "<h2>17BA. Vessels False-Positive Veto Diagnostic</h2>"
    for path in [HTML, INDEX]:
        text = path.read_text()
        if marker in text:
            start = text.index(marker)
            next_match = re.search(r"<h2>17B[B-Z]|<h2>18\.", text[start + len(marker) :])
            end = start + len(marker) + next_match.start() if next_match else text.index("</body>", start)
            text = text[:start] + section + text[end:]
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
    summary = read_csv(OUT / "vessels_false_positive_veto_summary.csv")
    diagnosis = read_csv(OUT / "vessels_selected_piece_diagnosis.csv")
    summary_rows = [
        {
            "variant": row["variant"],
            "selected": row["selected_count"],
            "Dice / Precision / Recall": f"{float(row['dice']):.3f} / {float(row['precision']):.3f} / {float(row['recall']):.3f}",
            "what it tests": row["note"],
        }
        for row in summary
    ]
    diag_rows = [
        {
            "candidate": row["candidate_uid"],
            "rank": row["rank"],
            "VLM/skill score": row["target_score"],
            "hidden true class": row["hidden_true_class"],
            "hidden component D/P/R": row["hidden_component_D/P/R"],
            "vessels-class D/P/R": row["vessels_class_D/P/R"],
            "interpretation": (
                "false positive for vessels"
                if row["hidden_true_class"] in {"stroma", "immune_infiltration"}
                else "vessel-supporting piece"
            ),
        }
        for row in diagnosis
    ]
    fig = OUT / "vessels_oracle_veto_six_panel.png"
    section = f"""
<h2>17BA. Vessels False-Positive Veto Diagnostic</h2>
<p><b>Goal.</b> Section 17AZ showed that second-SAM is not helpful for the recovered vessel component.  This section asks whether vessels fail because the skill selected wrong tissue pieces, or because the proposal pool still lacks clean vessel components.</p>
<p><b>Method.</b> This is an oracle diagnostic only.  I look at the runtime-selected vessel pieces after scoring, then use hidden labels after the fact to remove pieces whose true class is stroma or immune infiltration.  This quantifies whether a false-positive veto would solve vessels.</p>
<h3>Union variants</h3>
{table(summary_rows, ["variant", "selected", "Dice / Precision / Recall", "what it tests"])}
<h3>Selected-piece diagnosis</h3>
{table(diag_rows, ["candidate", "rank", "VLM/skill score", "hidden true class", "hidden component D/P/R", "vessels-class D/P/R", "interpretation"])}
<figure><img src="{html.escape(str(fig.relative_to(BASE)))}" alt="vessels false-positive veto six-panel"><figcaption>Vessels oracle-veto diagnostic: removing selected stroma/immune false positives improves precision but does not recover the missing components.</figcaption></figure>
<div class='callout'><b>Updated vessels diagnosis.</b> A false-positive veto helps precision, but it does not fix recall.  The main vessel bottleneck is still component recovery: the current recovered component-4 proposal is too broad, and second-SAM made it worse.  The next vessel iteration should generate cleaner local vessel-wall/lumen proposals rather than just tuning VLM scores or adding recovered boxes.</div>
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
    print(BASE / "Jun09_SkillRanker_ComponentAwareMaskSelection_shareable.zip")


if __name__ == "__main__":
    main()
