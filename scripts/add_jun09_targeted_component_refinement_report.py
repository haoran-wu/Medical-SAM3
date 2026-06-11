#!/usr/bin/env python3
"""Append class-specific targeted component refinement results to Jun09 report."""

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
OUT = PACK / "targeted_component_second_sam_hybrid"
HTML = BASE / "Jun09_SkillRanker_ComponentAwareMaskSelection.html"
INDEX = BASE / "index.html"


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
    marker = "<h2>17AZ. Targeted Component-Level Refinement Decision</h2>"
    for path in [HTML, INDEX]:
        text = path.read_text()
        if marker in text:
            start = text.index(marker)
            next_match = re.search(r"<h2>17B|<h2>18\.", text[start + len(marker) :])
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
    bron_rows = read_csv(OUT / "bronchiola_targeted_hybrid_metrics.csv")
    component_rows = [
        {
            "class": "bronchiola",
            "component": "bottom component 3",
            "prior component D/P/R": "0.504 / 0.403 / 0.671",
            "second-SAM component D/P/R": "0.622 / 0.474 / 0.905",
            "decision": "use second-SAM only for this missing component",
        },
        {
            "class": "vessels",
            "component": "component 4",
            "prior component D/P/R": "0.408 / 0.259 / 0.968",
            "second-SAM component D/P/R": "0.166 / 0.102 / 0.437",
            "decision": "reject second-SAM for this recovered vessel component",
        },
    ]
    bron_table = [
        {
            "variant": row["variant"],
            "Dice / Precision / Recall": f"{float(row['dice']):.3f} / {float(row['precision']):.3f} / {float(row['recall']):.3f}",
            "interpretation": row["interpretation"],
        }
        for row in bron_rows
    ]
    final_rows = [
        {
            "class": "bronchiola",
            "best current targeted rule": "main selected pieces + second-SAM only for recovered bottom component",
            "result D/P/R": "0.819 / 0.792 / 0.849",
            "why this rule": "It fixes the missed bottom component while keeping the good main pieces unchanged.",
        },
        {
            "class": "vessels",
            "best current targeted rule": "keep existing selected pieces; do not second-SAM the recovered component 4",
            "result D/P/R": "not changed in this section",
            "why this rule": "The recovered vessel component got worse after second-SAM even when evaluated on the matched component.",
        },
    ]
    write_csv(OUT / "component_level_second_sam_decision.csv", component_rows)
    write_csv(OUT / "targeted_refinement_final_decision.csv", final_rows)
    fig = OUT / "bronchiola_targeted_hybrid_six_panel.png"
    section = f"""
<h2>17AZ. Targeted Component-Level Refinement Decision</h2>
<p><b>Goal.</b> Section 17AY showed that replacing every selected piece with a second-SAM output is not reliable.  This section tests the more precise rule: only refine the component that actually failed, and keep already-good pieces unchanged.</p>
<p><b>Why this matters.</b> A global operation can look bad even when one missing component benefits.  The skill should therefore decide at the component level: identify the failed component, choose a recovery proposal, test refinement only there, and accept it only if the component-level metric improves.</p>
<h3>Component-level second-SAM check</h3>
{table(component_rows, ["class", "component", "prior component D/P/R", "second-SAM component D/P/R", "decision"])}
<h3>Bronchiola targeted hybrid variants</h3>
{table(bron_table, ["variant", "Dice / Precision / Recall", "interpretation"])}
<h3>Class-specific refinement decision</h3>
{table(final_rows, ["class", "best current targeted rule", "result D/P/R", "why this rule"])}
<figure><img src="{html.escape(str(fig.relative_to(BASE)))}" alt="bronchiola targeted component hybrid six-panel"><figcaption>Bronchiola targeted hybrid: keep the high-quality main bronchiola pieces, and replace only the recovered bottom component with its second-SAM refined mask.  This gives Dice/Precision/Recall 0.819/0.792/0.849.</figcaption></figure>
<div class='callout'><b>Updated skill rule.</b> Second-SAM is not a global post-processing step.  It is a component-specific repair tool.  Accept it for bronchiola bottom component 3, reject it for vessel component 4, and keep testing other classes only where the component-level evidence supports refinement.</div>
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
