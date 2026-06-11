#!/usr/bin/env python3
"""Add a minimal-setting diagnostic to the Jun09 report."""

from __future__ import annotations

import csv
import html
import zipfile
from collections import defaultdict
from pathlib import Path


ROOT = Path("/Users/haoranwu/Desktop/Yan_Lab_Research/Medical-SAM3")
BASE = ROOT / "output/visium_hd_exp1/final_deliverables/Jun09_SkillRanker_ComponentAwareMaskSelection"
CLEAN = Path("/Users/haoranwu/Desktop/Yan_Lab_Research/VisiumHD-Segmentation-MaskSelection")
SETTING_BY_LABEL = CLEAN / "output/visium_hd_exp1/final_deliverables/Jun03_mainline_all48_setting_suitability/tables/setting_suitability_by_label.csv"

LABELS = [
    ("lung_bronchiola", "bronchiola"),
    ("lung_alveoli_normal_adjacent", "alveoli"),
    ("lung_vessels", "vessels"),
    ("tumor", "tumor"),
    ("stroma", "stroma"),
    ("immune_infiltration", "immune infiltration"),
]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


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
    out += [f"<th>{html.escape(col)}</th>" for col in cols]
    out.append("</tr></thead><tbody>")
    for row in rows:
        out.append("<tr>")
        for col in cols:
            out.append(f"<td>{html.escape(str(row.get(col, '')))}</td>")
        out.append("</tr>")
    out.append("</tbody></table>")
    return "".join(out)


def main() -> None:
    rows = read_csv(SETTING_BY_LABEL)
    label_to_display = dict(LABELS)
    best = {
        label: max(float(r["best_component_dice"]) for r in rows if r["label"] == label)
        for label, _ in LABELS
    }

    top_by_class = []
    for label, display in LABELS:
        sub = sorted(
            [r for r in rows if r["label"] == label],
            key=lambda r: float(r["best_component_dice"]),
            reverse=True,
        )[:5]
        for rank, r in enumerate(sub, 1):
            top_by_class.append(
                {
                    "class": display,
                    "rank": rank,
                    "setting": r["setting"],
                    "family": r["setting_family"],
                    "total candidates": r["total_candidates"],
                    "best component D/P/R": f"{float(r['best_component_dice']):.3f} / {float(r['best_component_precision']):.3f} / {float(r['best_component_recall']):.3f}",
                    "source": r["best_component_source"],
                    "qualified candidates": r["qualified_candidate_count"],
                }
            )

    coverage_rows = []
    for cutoff in [0.98, 0.95, 0.90]:
        cover: dict[str, list[str]] = defaultdict(list)
        cand_count: dict[str, int] = defaultdict(int)
        for r in rows:
            label = r["label"]
            if label in best and float(r["best_component_dice"]) >= cutoff * best[label]:
                cover[r["setting"]].append(label_to_display[label])
                cand_count[r["setting"]] = max(cand_count[r["setting"]], int(r["total_candidates"]))
        for setting, labels in sorted(cover.items(), key=lambda kv: (-len(kv[1]), kv[0]))[:12]:
            coverage_rows.append(
                {
                    "near-best cutoff": f"{int(cutoff * 100)}%",
                    "setting": setting,
                    "covered classes": ", ".join(labels),
                    "n classes": len(labels),
                    "setting candidates": cand_count[setting],
                }
            )

    proposed = [
        {
            "setting": "base_official_points_step24",
            "why keep it": "Near-best for bronchiola, tumor, stroma, and immune; much smaller than all-48.",
            "main role": "small/mid component pieces and immune/tumor/stroma alternatives",
        },
        {
            "setting": "medical_official_points_step24",
            "why keep it": "Strongest single official point setting and very good for bronchiola, vessels, and stroma.",
            "main role": "current piece-first skill pool and vessels/bronchiola support",
        },
        {
            "setting": "base_box384_s128_m1536",
            "why keep it": "Best alveoli candidate; point settings miss this broad alveolar mask.",
            "main role": "alveoli broad H&E mask",
        },
    ]

    out_dir = BASE / "refined_diagnostics"
    write_csv(out_dir / "minimal_setting_top_by_class.csv", top_by_class)
    write_csv(out_dir / "minimal_setting_near_best_coverage.csv", coverage_rows)
    write_csv(out_dir / "minimal_setting_proposed_next.csv", proposed)

    section = f"""
<h2>16. Minimal-Setting Diagnostic: Can We Avoid the 48-Setting Pool?</h2>
<p>The original mainline used many SAM settings to maximize the chance of finding a good mask. That is useful for discovery, but not realistic as a deployable workflow. This diagnostic asks whether a small set of complementary settings can recover most of the useful candidate types.</p>
<h3>Top settings by class</h3>
{table(top_by_class, ['class', 'rank', 'setting', 'family', 'total candidates', 'best component D/P/R', 'source', 'qualified candidates'])}
<h3>Near-best coverage</h3>
<p>A setting covers a class if its best component Dice is within the listed percentage of the best setting for that class. At 98%, no single setting covers alveoli; a box setting is still needed.</p>
{table(coverage_rows, ['near-best cutoff', 'setting', 'n classes', 'covered classes', 'setting candidates'])}
<h3>Proposed next generator set</h3>
{table(proposed, ['setting', 'main role', 'why keep it'])}
<div class='callout'><b>Decision.</b> The next branch should not use all 48 settings, and it should not use medpt24 alone. A practical compromise is a three-setting pool: one strong official point setting for component pieces, one base official point setting for complementary FICTURE/immune/tumor/stroma pieces, and one H&E box setting for alveoli.</div>
"""

    for html_path in [BASE / "Jun09_SkillRanker_ComponentAwareMaskSelection.html", BASE / "index.html"]:
        text = html_path.read_text()
        marker = "<h2>16. Minimal-Setting Diagnostic: Can We Avoid the 48-Setting Pool?</h2>"
        if marker in text:
            start = text.index(marker)
            end = text.index("<h2>16. Interpretation</h2>", start) if "<h2>16. Interpretation</h2>" in text[start:] else text.index("</body>", start)
            text = text[:start] + section + text[end:]
        elif "<h2>16. Interpretation</h2>" in text:
            text = text.replace("<h2>16. Interpretation</h2>", section + "<h2>17. Interpretation</h2>")
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
        include.append(BASE / rel)
    for root in [
        BASE / "corrected_pool",
        BASE / "assembly_outputs/combined_skill_ranker_rf_plus_he/figures",
        BASE / "refined_diagnostics",
        BASE / "guarded_loco_variant",
    ]:
        for p in root.rglob("*"):
            if p.is_file():
                include.append(p)
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as handle:
        for p in include:
            handle.write(p, p.relative_to(BASE))

    print(out_dir / "minimal_setting_proposed_next.csv")
    print(zip_path)


if __name__ == "__main__":
    main()
