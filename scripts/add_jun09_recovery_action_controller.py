#!/usr/bin/env python3
"""Add an action controller after the local broader-pool recovery audit."""

from __future__ import annotations

import csv
import html
import re
import zipfile
from pathlib import Path

import pandas as pd


ROOT = Path("/Users/haoranwu/Desktop/Yan_Lab_Research/Medical-SAM3")
BASE = ROOT / "output/visium_hd_exp1/final_deliverables/Jun09_SkillRanker_ComponentAwareMaskSelection"
OUT = BASE / "local_broader_pool_recovery_audit"


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    cols: list[str] = []
    for row in rows:
        for key in row:
            if key not in cols:
                cols.append(key)
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
    marker = "<h2>17AW. Iteration Decision After Local Recovery</h2>"
    if marker in text:
        start = text.index(marker)
        next_match = re.search(r"<h2>17A[X-Z]|<h2>18\\.", text[start + len(marker) :])
        end = start + len(marker) + next_match.start() if next_match else text.index("</body>", start)
        text = text[:start] + section + text[end:]
    else:
        text = text.replace("</body>", section + "</body>")
    html_path.write_text(text)


def verify_html_images(html_path: Path) -> None:
    text = html_path.read_text(errors="ignore")
    missing = []
    for src in re.findall(r"<img[^>]+src=\"([^\"]+)\"", text):
        if src.startswith(("data:", "http://", "https://")):
            continue
        if not (html_path.parent / src).exists():
            missing.append(src)
    if missing:
        raise RuntimeError(f"Missing {len(missing)} image assets: {missing[:8]}")


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
    required = ["local_broader_pool_recovery_audit/next_iteration_after_local_recovery.csv"]
    missing = [name for name in required if name not in names]
    if missing:
        raise RuntimeError(f"zip missing required recovery action files: {missing}")


def main() -> None:
    details = pd.read_csv(OUT / "local_broader_pool_recovery_details.csv")
    summary = pd.read_csv(OUT / "local_broader_pool_recovery_summary.csv")

    def row_for(cls: str) -> pd.Series:
        return summary[summary["class"] == cls].iloc[0]

    rows = [
        {
            "class": "bronchiola",
            "new evidence": "all 3 components are recoverable in local broader proposals; missing bottom component comes from medium_boxes_b192_s64 candidate_142",
            "failure layer now": "compact funnel / setting choice, then boundary refinement",
            "next iteration": "restore or rerun medium_boxes_b192_s64-style local proposals for bronchiola, then use the recovered bottom piece as a second-SAM prompt instead of directly unioning the boxy mask",
        },
        {
            "class": "vessels",
            "new evidence": "5/7 components recoverable; component 4 has high recall from medium_boxes_b192_s64 candidate_116, but component 1 still has no clean local piece",
            "failure layer now": "mixed: funnel recovery for component 4, new local proposal needed for component 1",
            "next iteration": "add medium_boxes_b192_s64-style proposals for vessel recovery and a denser local prompt around the tiny component 1 region; run second-SAM with a precision guard",
        },
        {
            "class": "alveoli",
            "new evidence": f"local broader pool support {row_for('alveoli')['support_rate']} with best component Dice {row_for('alveoli')['best_component_Dice_max']}",
            "failure layer now": "generator/setting choice rather than VLM label recognition",
            "next iteration": "use the broad-box H&E branch for alveoli and recover exact masks; do not use medpt24 selected union as the final alveoli path",
        },
        {
            "class": "immune_infiltration",
            "new evidence": "only 13/189 components pass support threshold; many components have high recall but low precision from too-broad candidates",
            "failure layer now": "quality skill and component granularity, not only missing proposals",
            "next iteration": "build an immune-specific small-object quality skill: nuclei-density/local texture + upper area cap + precision veto; do not simply expand recall",
        },
        {
            "class": "stroma",
            "new evidence": "28/451 components supported; stroma remains broad and heterogeneous",
            "failure layer now": "broad compartment definition plus quality/boundary",
            "next iteration": "split stromal matrix from vessel/smooth-muscle-like pieces before assembly; treat stroma as a lower-priority broad-compartment branch",
        },
        {
            "class": "tumor",
            "new evidence": "13/233 components supported in local broader proposals",
            "failure layer now": "diffuse proposal quality and tumor-vs-stroma false-positive veto",
            "next iteration": "use structured FICTURE tumor prior as candidate generator, but gate with H&E morphology and avoid sending raw FICTURE colors as primary VLM input",
        },
    ]

    write_csv(OUT / "next_iteration_after_local_recovery.csv", rows)

    section = f"""
<h2>17AW. Iteration Decision After Local Recovery</h2>
<p><b>Purpose.</b> This table turns the recovery audit into the next concrete engineering decision.  The key lesson is that different classes fail at different layers; therefore the skill should not keep applying one global VLM prompt or one global union rule.</p>
{table(rows, ['class', 'new evidence', 'failure layer now', 'next iteration'])}
<p><b>Practical next step.</b> The strongest immediate path is to rebuild the compact candidate funnel with recovered medium-box local proposals for bronchiola and vessels, then run true second-SAM on those selected pieces.  Alveoli should branch to the broad H&amp;E box setting, while immune needs a precision-oriented small-object quality skill.</p>
"""
    for html_path in [BASE / "Jun09_SkillRanker_ComponentAwareMaskSelection.html", BASE / "index.html"]:
        replace_or_append(html_path, section)
        verify_html_images(html_path)
    rebuild_zip()
    print(f"Wrote {OUT / 'next_iteration_after_local_recovery.csv'}")


if __name__ == "__main__":
    main()
